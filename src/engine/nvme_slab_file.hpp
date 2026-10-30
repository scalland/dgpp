#pragma once
// The NVMe cold tier's page file: one descriptor on a preallocated file of
// fixed-size records. This class owns the file (direct I/O with a buffered
// fallback, the advisory lock that makes one world per slab, the capacity
// claim) and moves page records at their offsets. What a page means — its
// CRC-32C, the index, adoption across restarts — stays with NvmeTier, so a
// second backend can replace this file without touching those rules.
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/log.hpp"

namespace dgpp {

class NvmeSlabFile {
 public:
  NvmeSlabFile() = default;
  ~NvmeSlabFile() { close(); }
  NvmeSlabFile(const NvmeSlabFile&) = delete;
  NvmeSlabFile& operator=(const NvmeSlabFile&) = delete;

  // Opens the file with direct I/O first (the page cache would grow into the
  // memory the model plans on); a filesystem that refuses it gets buffered
  // I/O only when allow_buffered says so. The advisory lock makes this the
  // world's only process on the file. Unless keep_contents, the file is
  // truncated after the lock is taken and its whole `bytes` claimed now
  // (unwritten extents): a full disk is a startup error, never a torn write.
  void open(const std::string& path, bool require_existing, bool allow_buffered, bool keep_contents, int rank,
            size_t bytes) {
    close();
    path_ = path;
    rank_ = rank;
    const int flags = O_RDWR | O_CLOEXEC | (require_existing ? 0 : O_CREAT);
    fd_ = ::open(path.c_str(), flags | O_DIRECT, 0600);
    direct_ = fd_ >= 0;
    if (fd_ < 0 && errno == EINVAL) fd_ = ::open(path.c_str(), flags, 0600);
    if (fd_ < 0) throw std::runtime_error("NvmeTier: cannot create " + path + ": " + std::strerror(errno));
    // One world per slab: a second process on the same binding (a duplicate
    // deployment) would truncate and write under the first.
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      const int err = errno;
      close();
      throw std::runtime_error("NvmeTier: " + path + " is held by another process (" + std::strerror(err) +
                               "): one world per slab — stop the other deployment or give this one another "
                               "nvme_cache.path");
    }
    if (!direct_ && !allow_buffered) {
      close();
      throw std::runtime_error("NvmeTier: " + path +
                               " refuses direct I/O (the filesystem buffers it, and the page cache would grow into "
                               "the memory the model plans on): put nvme_cache.path on a local filesystem, or set "
                               "nvme_cache.allow_buffered");
    }
    if (keep_contents) return;
    // Opening with O_TRUNC would destroy another world's slab before the
    // advisory lock can refuse us. Clear it only after acquiring ownership.
    if (::ftruncate(fd_, 0) != 0) {
      const int err = errno;
      close();
      throw std::runtime_error("NvmeTier: cannot truncate " + path + ": " + std::strerror(err));
    }
    claim(bytes);
  }

  // Clears the contents of the file already open (the lock stays), then
  // claims `bytes` again.
  void recreate(size_t bytes) {
    if (::ftruncate(fd_, 0) != 0)
      throw std::runtime_error("NvmeTier: cannot truncate " + path_ + ": " + std::strerror(errno));
    claim(bytes);
  }

  void close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }

  bool is_open() const { return fd_ >= 0; }
  bool direct() const { return direct_; }

  // One page record at its offset. The caller's buffer is aligned for direct
  // I/O. False with `why` on a failed or short transfer; EINTR is retried.
  bool write_at(int64_t page, size_t page_bytes, const void* buf, std::string* why) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t left = page_bytes;
    off_t off = static_cast<off_t>(static_cast<size_t>(page) * page_bytes);
    while (left > 0) {
      const ssize_t n = ::pwrite(fd_, p, left, off);
      if (n < 0) {
        if (errno == EINTR) continue;
        *why = "write of page " + std::to_string(page) + ": " + std::strerror(errno);
        return false;
      }
      if (n == 0) {
        *why = "short write of page " + std::to_string(page);
        return false;
      }
      p += n;
      off += n;
      left -= static_cast<size_t>(n);
    }
    return true;
  }

  bool read_at(int64_t page, size_t page_bytes, void* buf, std::string* why) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t left = page_bytes;
    off_t off = static_cast<off_t>(static_cast<size_t>(page) * page_bytes);
    while (left > 0) {
      const ssize_t n = ::pread(fd_, p, left, off);
      if (n < 0) {
        if (errno == EINTR) continue;
        *why = "read of page " + std::to_string(page) + ": " + std::strerror(errno);
        return false;
      }
      if (n == 0) {
        *why = "short read of page " + std::to_string(page);
        return false;
      }
      p += n;
      off += n;
      left -= static_cast<size_t>(n);
    }
    return true;
  }

 private:
  // The whole capacity claimed now (unwritten extents); a filesystem that
  // cannot preallocate gets a sparse file, said out loud.
  void claim(size_t bytes) {
    const off_t total = static_cast<off_t>(bytes);
    if (::posix_fallocate(fd_, 0, total) != 0) {
      const int err = errno;
      if (::ftruncate(fd_, total) != 0) {
        close();
        throw std::runtime_error("NvmeTier: cannot size " + path_ + " to " + std::to_string(total) +
                                 " bytes: " + std::strerror(err));
      }
      DGPP_LOG_WARN("rank {}: NVMe cache: the filesystem cannot preallocate {} ({}); the slab is sparse and a "
                    "full disk would fail a spill instead",
                    rank_, path_, std::strerror(err));
    }
  }

  int fd_ = -1;
  bool direct_ = false;
  std::string path_;
  int rank_ = 0;
};

}  // namespace dgpp
