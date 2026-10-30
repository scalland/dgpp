#pragma once
// The folder backend of the NVMe cold tier (disk-cache phase 2): each cache
// entry is one folder, `<root>/<session>/<entry>/`, holding the entry's bytes
// split into files of at most `file_bytes` each, and one CRC-32C per page in a
// sidecar beside every file. A key names the folder: `<session>/<entry>`,
// where the session is the hash of prompt_cache_key (or of the token prefix)
// and the entry is the hash of the entry's own prefix and cut. Finding an
// entry is a direct path with no search, and no two entries share a file.
//
// Writes go to `<root>/.tmp-<session>.<entry>/` first and are fsynced, then
// renamed into place: the rename is the commit, so a crash mid-write leaves a
// temp folder that the next start removes, never a half-written entry.
// put and get move a whole buffer, one thread per file (up to `io_threads`).
// The tier streams an entry instead (Writer and Reader): its staging chunks
// arrive in order, so the entry is written and read one page run at a time.
// Every page is checked against its CRC before the bytes are used; a mismatch
// fails the read.
//
// The store knows bytes and files only. Which entry holds what, and the index
// that survives a restart, belong to the tier (sched/folder_disk_cache.hpp).
// Capacity is counted in bytes; the caller evicts (remove) before a write
// that would not fit.
//
// Data and the destination buffer are page-aligned multiples of `page_bytes`.
// The files use buffered I/O for now: the slab's direct-I/O rule (the page
// cache would grow into the memory the model plans on) must be added before
// this backend is enabled in a deployment (disk-cache phase 3).
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
    int64_t capacity_bytes = 0;   // the bytes the entries may hold together
    size_t page_bytes = 4096;     // one checksummed page; a multiple of 4 KiB
    size_t file_bytes = 256u << 20;  // the most bytes one file holds
    int io_threads = 4;           // files read or written at once (put and get)
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

  // Streams one entry in, in page runs: the entry commits when every byte
  // has been appended and commit() is called. Destroyed before that, it
  // removes its temp folder and the previous entry stays in place.
  class Writer {
   public:
    // Starts the write. ok() is false, with `why` set, when the entry would
    // not fit or the temp folder cannot be made.
    Writer(NvmeFolderStore& store, const std::string& key, size_t bytes, std::string* why)
        : store_(&store), key_(key), tmp_(store.tmp_dir(key)), bytes_(bytes) {
      check_key(key);
      store.check_bytes(bytes);
      if (!store.fits(key, bytes, why)) return;
      std::error_code ec;
      std::filesystem::remove_all(tmp_, ec);
      std::filesystem::create_directories(tmp_, ec);
      if (ec) {
        *why = "cannot create " + tmp_ + ": " + ec.message();
        return;
      }
      ok_ = true;
    }
    ~Writer() { abort(); }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    bool ok() const { return ok_; }
    size_t written() const { return written_; }

    // Appends the next `n` bytes: a positive whole number of pages, never more
    // than the entry's size. False with `why` on a write failure; the write
    // is then aborted.
    bool append(const void* data, size_t n, std::string* why) {
      if (!ok_ || done_) throw std::logic_error("NvmeFolderStore::Writer: the write is not open");
      if (n == 0 || n % store_->o_.page_bytes != 0 || written_ + n > bytes_)
        throw std::invalid_argument("NvmeFolderStore::Writer: an append must be whole pages within the entry");
      const auto* p = static_cast<const uint8_t*>(data);
      const size_t page = store_->o_.page_bytes;
      while (n > 0) {
        if (fd_ < 0 && !open_part(why)) return fail();
        const size_t room = std::min(n, file_size() - in_file_);
        if (!write_all(fd_, p, room, why, path_)) return fail();
        for (size_t off = 0; off < room; off += page) crcs_.push_back(crc32c(p + off, page));
        p += room;
        n -= room;
        in_file_ += room;
        written_ += room;
        if (in_file_ == file_size() && !close_part(why)) return fail();
      }
      return true;
    }

    // Commits the entry: every byte must have been appended.
    bool commit(std::string* why) {
      if (!ok_ || done_) {
        *why = "the write of " + key_ + " is not open";
        return false;
      }
      if (written_ != bytes_) {
        *why = "entry " + key_ + " is incomplete: " + std::to_string(written_) + " of " + std::to_string(bytes_) + " bytes";
        abort();
        return false;
      }
      done_ = true;
      return store_->commit_tmp(key_, tmp_, static_cast<int64_t>(bytes_), why);
    }

    // Drops the write: the temp folder goes. A no-op once committed.
    void abort() {
      if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
      }
      if (ok_ && !done_) {
        std::error_code ec;
        std::filesystem::remove_all(tmp_, ec);
      }
      done_ = true;
    }

   private:
    // The size of the file being written: a full file, or the entry's last.
    size_t file_size() const {
      return std::min(store_->o_.file_bytes, bytes_ - file_index_ * store_->o_.file_bytes);
    }
    bool open_part(std::string* why) {
      path_ = tmp_ + "/" + file_name(file_index_, ".bin");
      fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
      if (fd_ < 0) {
        *why = "cannot create " + path_ + ": " + std::strerror(errno);
        return false;
      }
      return true;
    }
    // Syncs the file, writes its CRC sidecar and syncs that: the file is
    // whole. The next file starts empty.
    bool close_part(std::string* why) {
      const bool synced = ::fsync(fd_) == 0;
      ::close(fd_);
      fd_ = -1;
      if (!synced) {
        *why = "cannot sync " + path_ + ": " + std::strerror(errno);
        return false;
      }
      const std::string crc_path = tmp_ + "/" + file_name(file_index_, ".crc");
      const int cfd = ::open(crc_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
      if (cfd < 0) {
        *why = "cannot create " + crc_path + ": " + std::strerror(errno);
        return false;
      }
      bool good = write_all(cfd, crcs_.data(), crcs_.size() * sizeof(uint32_t), why, crc_path);
      if (good && ::fsync(cfd) != 0) {
        *why = "cannot sync " + crc_path + ": " + std::strerror(errno);
        good = false;
      }
      ::close(cfd);
      if (!good) return false;
      crcs_.clear();
      in_file_ = 0;
      ++file_index_;
      return true;
    }
    bool fail() {
      abort();
      return false;
    }

    NvmeFolderStore* store_;
    std::string key_, tmp_, path_;
    size_t bytes_ = 0;
    size_t written_ = 0;
    size_t in_file_ = 0;
    size_t file_index_ = 0;
    int fd_ = -1;
    std::vector<uint32_t> crcs_;
    bool ok_ = false;
    bool done_ = false;
  };

  // Streams one entry out, in page runs, checking each page against its CRC
  // before the bytes are returned. ok() is false, with `why` set, when the
  // entry is absent.
  class Reader {
   public:
    Reader(const NvmeFolderStore& store, const std::string& key, std::string* why)
        : store_(&store), key_(key), dir_(store.entry_dir(key)) {
      check_key(key);
      std::lock_guard<std::mutex> lock(store.mutex_);
      const auto it = store.sizes_.find(key);
      if (it == store.sizes_.end()) {
        *why = "no entry " + key;
        return;
      }
      bytes_ = static_cast<size_t>(it->second);
      ok_ = true;
    }
    ~Reader() {
      if (fd_ >= 0) ::close(fd_);
    }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    bool ok() const { return ok_; }
    size_t bytes() const { return bytes_; }
    size_t read_bytes() const { return read_; }

    // Reads the next `n` bytes into dst: a positive whole number of pages,
    // within the entry. False with `why` on a short file or a page whose
    // checksum does not match.
    bool read(void* dst, size_t n, std::string* why) {
      if (!ok_) throw std::logic_error("NvmeFolderStore::Reader: the entry is absent");
      if (n == 0 || n % store_->o_.page_bytes != 0 || read_ + n > bytes_)
        throw std::invalid_argument("NvmeFolderStore::Reader: a read must be whole pages within the entry");
      const size_t page = store_->o_.page_bytes;
      auto* p = static_cast<uint8_t*>(dst);
      while (n > 0) {
        if (fd_ < 0 && !open_part(why)) return false;
        const size_t room = std::min(n, file_size() - in_file_);
        if (!read_all(fd_, p, room, why, path_)) return false;
        for (size_t off = 0; off < room; off += page) {
          const size_t idx = (in_file_ + off) / page;
          if (crc32c(p + off, page) != want_[idx]) {
            *why = "checksum mismatch on page " + std::to_string(idx) + " of " + path_;
            return false;
          }
        }
        p += room;
        n -= room;
        in_file_ += room;
        read_ += room;
        if (in_file_ == file_size()) {
          ::close(fd_);
          fd_ = -1;
          in_file_ = 0;
          ++file_index_;
        }
      }
      return true;
    }

   private:
    size_t file_size() const {
      return std::min(store_->o_.file_bytes, bytes_ - file_index_ * store_->o_.file_bytes);
    }
    // Opens the current file and loads its CRC sidecar.
    bool open_part(std::string* why) {
      path_ = dir_ + "/" + file_name(file_index_, ".bin");
      fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd_ < 0) {
        *why = "cannot open " + path_ + ": " + std::strerror(errno);
        return false;
      }
      const std::string crc_path = dir_ + "/" + file_name(file_index_, ".crc");
      want_.assign(file_size() / store_->o_.page_bytes, 0);
      const int cfd = ::open(crc_path.c_str(), O_RDONLY | O_CLOEXEC);
      if (cfd < 0) {
        *why = "cannot open " + crc_path + ": " + std::strerror(errno);
        return false;
      }
      const bool got = read_all(cfd, want_.data(), want_.size() * sizeof(uint32_t), why, crc_path);
      ::close(cfd);
      return got;
    }

    const NvmeFolderStore* store_;
    std::string key_, dir_, path_;
    size_t bytes_ = 0;
    size_t read_ = 0;
    size_t in_file_ = 0;
    size_t file_index_ = 0;
    int fd_ = -1;
    std::vector<uint32_t> want_;
    bool ok_ = false;
  };

  // Whether an entry is committed under `key`.
  bool contains(const std::string& key) const {
    check_key(key);
    std::error_code ec;
    return std::filesystem::is_directory(entry_dir(key), ec);
  }

  // Bytes held by committed entries.
  int64_t bytes_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_;
  }
  int64_t bytes_free() const { return o_.capacity_bytes - bytes_used(); }

  // Stores `bytes` of data under `key`, replacing any previous entry. False
  // with `why` when it would not fit or a file write fails; the previous
  // entry is left in place on failure.
  bool put(const std::string& key, const void* data, size_t bytes, std::string* why) {
    check_key(key);
    check_bytes(bytes);
    if (!fits(key, bytes, why)) return false;
    const std::string tmp = tmp_dir(key);
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
    return commit_tmp(key, tmp, static_cast<int64_t>(bytes), why);
  }

  // Reads the entry under `key` into `dst` (`bytes` long, the size it was put
  // with). False with `why` when the entry is absent, a file is missing or
  // short, or a page's checksum does not match.
  bool get(const std::string& key, void* dst, size_t bytes, std::string* why) const {
    check_key(key);
    int64_t stored = -1;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = sizes_.find(key);
      if (it != sizes_.end()) stored = it->second;
    }
    if (stored < 0) {
      *why = "no entry " + key;
      return false;
    }
    if (static_cast<int64_t>(bytes) != stored) {
      *why = "entry " + key + " holds " + std::to_string(stored) + " bytes, not " + std::to_string(bytes);
      return false;
    }
    const size_t files = file_count(bytes);
    std::vector<int> ok(files, 0);
    std::vector<std::string> errors(files);
    run(files, [&](size_t f) {
      const size_t off = f * o_.file_bytes;
      const size_t n = std::min(o_.file_bytes, bytes - off);
      ok[f] = read_file(entry_dir(key), f, static_cast<uint8_t*>(dst) + off, n, &errors[f]) ? 1 : 0;
    });
    for (size_t f = 0; f < files; ++f)
      if (!ok[f]) {
        *why = errors[f];
        return false;
      }
    return true;
  }

  // Removes an entry's folder and its bytes from the accounting, and its
  // session's folder when no other entry is left in it.
  void remove(const std::string& key) {
    check_key(key);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sizes_.find(key);
    if (it == sizes_.end()) return;
    drop_dir(key);
    used_ -= it->second;
    sizes_.erase(it);
  }

  // The folder an entry lives in: `<root>/<session>/<entry>`.
  std::string entry_dir(const std::string& key) const { return o_.root + "/" + key; }

  // The number of files an entry of `bytes` takes.
  size_t file_count(size_t bytes) const { return (bytes + o_.file_bytes - 1) / o_.file_bytes; }

 private:
  // A key is `<session>/<entry>`: two non-empty names, neither hidden (a
  // leading dot marks the store's temp folders) nor holding a separator.
  static void check_key(const std::string& key) {
    const size_t slash = key.find('/');
    const auto valid = [](const std::string& part) {
      return !part.empty() && part[0] != '.' && part.find_first_of("/\\") == std::string::npos;
    };
    if (slash == std::string::npos || key.find('/', slash + 1) != std::string::npos ||
        !valid(key.substr(0, slash)) || !valid(key.substr(slash + 1)))
      throw std::invalid_argument("NvmeFolderStore: a key is <session>/<entry>, not '" + key + "'");
  }

  std::string session_dir(const std::string& key) const { return o_.root + "/" + key.substr(0, key.find('/')); }

  std::string tmp_dir(const std::string& key) const {
    std::string flat = key;
    std::replace(flat.begin(), flat.end(), '/', '.');
    return o_.root + "/.tmp-" + flat;
  }

  void check_bytes(size_t bytes) const {
    if (bytes == 0 || bytes % o_.page_bytes != 0)
      throw std::invalid_argument("NvmeFolderStore: an entry must be a positive whole number of pages");
  }

  // Whether `bytes` under `key` fits beside the other entries (the entry it
  // replaces is released first).
  bool fits(const std::string& key, size_t bytes, std::string* why) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t old = sizes_.count(key) ? sizes_.at(key) : 0;
    const int64_t total = used_ - old + static_cast<int64_t>(bytes);
    if (total > o_.capacity_bytes) {
      *why = "the cache would hold " + std::to_string(total) + " bytes, above its capacity of " +
             std::to_string(o_.capacity_bytes);
      return false;
    }
    return true;
  }

  // The commit: the capacity is checked again, then the previous folder goes
  // and the temp folder takes the entry's name.
  bool commit_tmp(const std::string& key, const std::string& tmp, int64_t bytes, std::string* why) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t old = sizes_.count(key) ? sizes_[key] : 0;
    if (used_ - old + bytes > o_.capacity_bytes) {
      std::error_code ec;
      std::filesystem::remove_all(tmp, ec);
      *why = "the cache would hold " + std::to_string(used_ - old + bytes) + " bytes, above its capacity of " +
             std::to_string(o_.capacity_bytes);
      return false;
    }
    drop_dir(key);
    std::error_code ec;
    std::filesystem::create_directories(session_dir(key), ec);
    if (!ec) std::filesystem::rename(tmp, entry_dir(key), ec);
    if (ec) {
      std::filesystem::remove_all(tmp, ec);
      *why = "cannot commit " + key + ": " + ec.message();
      sizes_.erase(key);
      used_ -= old;
      return false;
    }
    used_ += bytes - old;
    sizes_[key] = bytes;
    return true;
  }

  // Removes an entry's folder, then its session's folder if it is empty. The
  // caller holds the mutex.
  void drop_dir(const std::string& key) {
    std::error_code ec;
    std::filesystem::remove_all(entry_dir(key), ec);
    std::filesystem::remove(session_dir(key), ec);
  }

  // Entries already on disk count toward the capacity; leftover temp folders
  // from an interrupted write are removed, as are sessions left empty.
  void rescan() {
    std::error_code ec;
    for (const auto& session : std::filesystem::directory_iterator(o_.root, ec)) {
      const std::string name = session.path().filename().string();
      if (!session.is_directory(ec)) continue;
      if (name.rfind(".tmp-", 0) == 0) {
        std::filesystem::remove_all(session.path(), ec);
        continue;
      }
      for (const auto& entry : std::filesystem::directory_iterator(session.path(), ec)) {
        if (!entry.is_directory(ec)) continue;
        const std::string key = name + "/" + entry.path().filename().string();
        int64_t bytes = 0;
        for (const auto& f : std::filesystem::directory_iterator(entry.path(), ec))
          if (f.is_regular_file(ec) && f.path().extension() == ".bin") bytes += static_cast<int64_t>(f.file_size(ec));
        sizes_[key] = bytes;
        used_ += bytes;
      }
      std::filesystem::remove(session.path(), ec);  // only if it held nothing
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
  std::map<std::string, int64_t> sizes_;  // committed entries (by key) and their bytes
  int64_t used_ = 0;
};

}  // namespace dgpp
