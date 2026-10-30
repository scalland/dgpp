#pragma once
// The folder backend of the NVMe cold tier (disk-cache phase 2): each cache
// entry is stored as its own folder, `<root>/<session>/`, holding the entry's
// bytes split into files of at most `file_bytes` each, and one CRC-32C per
// page in a sidecar beside every file. The session name is the caller's (the
// hash of prompt_cache_key, or of the token prefix), so finding an entry is a
// direct path with no search.
//
// Writes go to `<root>/.tmp-<session>/` first and are fsynced, then renamed
// into place: the rename is the commit, so a crash mid-write leaves a temp
// folder that the next start removes, never a half-written session. Reads
// fan out one thread per file (up to `io_threads`) and verify each page
// before the bytes are used; a checksum mismatch fails the whole read.
//
// The store knows bytes and files only. Which session holds what, and the
// index that survives a restart, belong to the tier. Capacity is counted in
// bytes; the caller evicts (remove) before a put that would not fit.
//
// Data and the destination buffer are page-aligned multiples of
// `page_bytes`. The files use buffered I/O for now: the slab's direct-I/O
// rule (the page cache would grow into the memory the model plans on) must
// be added before this backend is enabled in a deployment (disk-cache
// phase 3).
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/crc32c.hpp"

namespace dgpp {

class NvmeFolderStore {
 public:
  struct Options {
    std::string root;             // the binding's directory
    int64_t capacity_bytes = 0;   // the bytes the sessions may hold together
    size_t page_bytes = 4096;     // one checksummed page; a multiple of 4 KiB
    size_t file_bytes = 256u << 20;  // the most bytes one file holds
    int io_threads = 4;           // files read or written at once
  };

  explicit NvmeFolderStore(Options o) : o_(std::move(o)) {
    if (o_.root.empty()) throw std::invalid_argument("NvmeFolderStore: an empty root");
    if (o_.page_bytes == 0 || o_.page_bytes % 4096 != 0)
      throw std::invalid_argument("NvmeFolderStore: the page must be a positive 4 KiB multiple");
    if (o_.file_bytes < o_.page_bytes || o_.file_bytes % o_.page_bytes != 0)
      throw std::invalid_argument("NvmeFolderStore: a file must hold a whole number of pages");
    if (o_.capacity_bytes <= 0) throw std::invalid_argument("NvmeFolderStore: the capacity must be positive");
    o_.io_threads = std::max(1, o_.io_threads);
    std::error_code ec;
    std::filesystem::create_directories(o_.root, ec);
    if (ec) throw std::runtime_error("NvmeFolderStore: cannot create " + o_.root + ": " + ec.message());
    rescan();
  }
  NvmeFolderStore(const NvmeFolderStore&) = delete;
  NvmeFolderStore& operator=(const NvmeFolderStore&) = delete;

  // Whether a session's folder exists (a committed entry).
  bool contains(const std::string& session) const {
    std::error_code ec;
    return std::filesystem::is_directory(session_dir(session), ec);
  }

  // Bytes held by committed sessions.
  int64_t bytes_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_;
  }
  int64_t bytes_free() const { return o_.capacity_bytes - bytes_used(); }

  // Stores `bytes` of data as the session's entry, replacing any previous
  // one. False with `why` when it would not fit or a file write fails; the
  // previous entry is left in place on failure.
  bool put(const std::string& session, const void* data, size_t bytes, std::string* why) {
    check_bytes(bytes);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const int64_t old = sizes_.count(session) ? sizes_[session] : 0;
      if (used_ - old + static_cast<int64_t>(bytes) > o_.capacity_bytes) {
        *why = "the cache would hold " + std::to_string(used_ - old + static_cast<int64_t>(bytes)) +
               " bytes, above its capacity of " + std::to_string(o_.capacity_bytes);
        return false;
      }
    }
    const std::string tmp = tmp_dir(session);
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp, ec);
    if (ec) {
      *why = "cannot create " + tmp + ": " + ec.message();
      return false;
    }
    const size_t files = file_count(bytes);
    std::vector<int> ok(files, 0);
    std::vector<std::string> errors(files);
    run(files, [&](size_t f) {
      const size_t off = f * o_.file_bytes;
      const size_t n = std::min(o_.file_bytes, bytes - off);
      ok[f] = write_file(tmp, f, static_cast<const uint8_t*>(data) + off, n, &errors[f]) ? 1 : 0;
    });
    for (size_t f = 0; f < files; ++f)
      if (!ok[f]) {
        std::filesystem::remove_all(tmp, ec);
        *why = errors[f];
        return false;
      }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const int64_t old = sizes_.count(session) ? sizes_[session] : 0;
      if (used_ - old + static_cast<int64_t>(bytes) > o_.capacity_bytes) {
        std::filesystem::remove_all(tmp, ec);
        *why = "the cache would hold " + std::to_string(used_ - old + static_cast<int64_t>(bytes)) +
               " bytes, above its capacity of " + std::to_string(o_.capacity_bytes);
        return false;
      }
      // The commit: the previous folder goes, then the new one takes its name.
      std::filesystem::remove_all(session_dir(session), ec);
      std::filesystem::rename(tmp, session_dir(session), ec);
      if (ec) {
        std::filesystem::remove_all(tmp, ec);
        *why = "cannot commit " + session + ": " + ec.message();
        sizes_.erase(session);
        used_ -= old;
        return false;
      }
      used_ += static_cast<int64_t>(bytes) - old;
      sizes_[session] = static_cast<int64_t>(bytes);
    }
    return true;
  }

  // Reads the session's entry into `dst` (`bytes` long, the size it was put
  // with). False with `why` when the session is absent, a file is missing or
  // short, or a page's checksum does not match.
  bool get(const std::string& session, void* dst, size_t bytes, std::string* why) const {
    int64_t stored = -1;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = sizes_.find(session);
      if (it != sizes_.end()) stored = it->second;
    }
    if (stored < 0) {
      *why = "no entry for session " + session;
      return false;
    }
    if (static_cast<int64_t>(bytes) != stored) {
      *why = "entry " + session + " holds " + std::to_string(stored) + " bytes, not " + std::to_string(bytes);
      return false;
    }
    const size_t files = file_count(bytes);
    std::vector<int> ok(files, 0);
    std::vector<std::string> errors(files);
    run(files, [&](size_t f) {
      const size_t off = f * o_.file_bytes;
      const size_t n = std::min(o_.file_bytes, bytes - off);
      ok[f] = read_file(session_dir(session), f, static_cast<uint8_t*>(dst) + off, n, &errors[f]) ? 1 : 0;
    });
    for (size_t f = 0; f < files; ++f)
      if (!ok[f]) {
        *why = errors[f];
        return false;
      }
    return true;
  }

  // Removes a session's folder and its bytes from the accounting.
  void remove(const std::string& session) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sizes_.find(session);
    if (it == sizes_.end()) return;
    std::error_code ec;
    std::filesystem::remove_all(session_dir(session), ec);
    used_ -= it->second;
    sizes_.erase(it);
  }

  // The folder a session's entry lives in.
  std::string session_dir(const std::string& session) const { return o_.root + "/" + session; }

  // The number of files an entry of `bytes` takes.
  size_t file_count(size_t bytes) const { return (bytes + o_.file_bytes - 1) / o_.file_bytes; }

 private:
  std::string tmp_dir(const std::string& session) const { return o_.root + "/.tmp-" + session; }

  void check_bytes(size_t bytes) const {
    if (bytes == 0 || bytes % o_.page_bytes != 0)
      throw std::invalid_argument("NvmeFolderStore: an entry must be a positive whole number of pages");
  }

  // Sessions already on disk count toward the capacity; leftover temp
  // folders from an interrupted write are removed.
  void rescan() {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(o_.root, ec)) {
      const std::string name = entry.path().filename().string();
      if (!entry.is_directory(ec)) continue;
      if (name.rfind(".tmp-", 0) == 0) {
        std::filesystem::remove_all(entry.path(), ec);
        continue;
      }
      int64_t bytes = 0;
      for (const auto& f : std::filesystem::directory_iterator(entry.path(), ec))
        if (f.is_regular_file(ec) && f.path().extension() == ".bin") bytes += static_cast<int64_t>(f.file_size(ec));
      sizes_[name] = bytes;
      used_ += bytes;
    }
  }

  // Runs f(0..n-1) on up to io_threads threads.
  template <class F>
  void run(size_t n, F&& f) const {
    const size_t threads = std::min<size_t>(static_cast<size_t>(o_.io_threads), n);
    if (threads <= 1) {
      for (size_t i = 0; i < n; ++i) f(i);
      return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (size_t t = 0; t < threads; ++t)
      pool.emplace_back([&] {
        for (size_t i = next++; i < n; i = next++) f(i);
      });
    for (auto& th : pool) th.join();
  }

  static std::string file_name(size_t f, const char* ext) {
    return "part-" + std::to_string(f) + ext;
  }

  // One payload file and its per-page CRC sidecar, fsynced.
  bool write_file(const std::string& dir, size_t f, const uint8_t* data, size_t n, std::string* why) const {
    const std::string path = dir + "/" + file_name(f, ".bin");
    const std::string crc_path = dir + "/" + file_name(f, ".crc");
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
      *why = "cannot create " + path + ": " + std::strerror(errno);
      return false;
    }
    bool good = write_all(fd, data, n, why, path);
    if (good && ::fsync(fd) != 0) {
      *why = "cannot sync " + path + ": " + std::strerror(errno);
      good = false;
    }
    ::close(fd);
    if (!good) return false;
    const size_t pages = n / o_.page_bytes;
    std::vector<uint32_t> crcs(pages);
    for (size_t p = 0; p < pages; ++p) crcs[p] = crc32c(data + p * o_.page_bytes, o_.page_bytes);
    const int cfd = ::open(crc_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (cfd < 0) {
      *why = "cannot create " + crc_path + ": " + std::strerror(errno);
      return false;
    }
    good = write_all(cfd, crcs.data(), pages * sizeof(uint32_t), why, crc_path);
    if (good && ::fsync(cfd) != 0) {
      *why = "cannot sync " + crc_path + ": " + std::strerror(errno);
      good = false;
    }
    ::close(cfd);
    return good;
  }

  bool read_file(const std::string& dir, size_t f, uint8_t* dst, size_t n, std::string* why) const {
    const std::string path = dir + "/" + file_name(f, ".bin");
    const std::string crc_path = dir + "/" + file_name(f, ".crc");
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      *why = "cannot open " + path + ": " + std::strerror(errno);
      return false;
    }
    const bool got = read_all(fd, dst, n, why, path);
    ::close(fd);
    if (!got) return false;
    const size_t pages = n / o_.page_bytes;
    std::vector<uint32_t> want(pages);
    const int cfd = ::open(crc_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (cfd < 0) {
      *why = "cannot open " + crc_path + ": " + std::strerror(errno);
      return false;
    }
    const bool crc_ok = read_all(cfd, want.data(), pages * sizeof(uint32_t), why, crc_path);
    ::close(cfd);
    if (!crc_ok) return false;
    for (size_t p = 0; p < pages; ++p)
      if (crc32c(dst + p * o_.page_bytes, o_.page_bytes) != want[p]) {
        *why = "checksum mismatch on page " + std::to_string(p) + " of " + path;
        return false;
      }
    return true;
  }

  static bool write_all(int fd, const void* buf, size_t n, std::string* why, const std::string& path) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t left = n;
    while (left > 0) {
      const ssize_t w = ::write(fd, p, left);
      if (w < 0) {
        if (errno == EINTR) continue;
        *why = "write of " + path + ": " + std::strerror(errno);
        return false;
      }
      p += w;
      left -= static_cast<size_t>(w);
    }
    return true;
  }

  static bool read_all(int fd, void* buf, size_t n, std::string* why, const std::string& path) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t left = n;
    while (left > 0) {
      const ssize_t r = ::read(fd, p, left);
      if (r < 0) {
        if (errno == EINTR) continue;
        *why = "read of " + path + ": " + std::strerror(errno);
        return false;
      }
      if (r == 0) {
        *why = "short read of " + path;
        return false;
      }
      p += r;
      left -= static_cast<size_t>(r);
    }
    return true;
  }

  Options o_;
  mutable std::mutex mutex_;
  std::map<std::string, int64_t> sizes_;  // committed sessions and their bytes
  int64_t used_ = 0;
};

}  // namespace dgpp
