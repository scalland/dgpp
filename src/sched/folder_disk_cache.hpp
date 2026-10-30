#pragma once
// The cold tier's index for the folder backend (disk-cache step 1): which
// entries sit on disk, in which folder, under LRU retention within a byte
// capacity. Unlike DiskCache (the slab index), an entry owns every byte it
// needs: its snapshot blob, its token ids and its cache blocks are one
// folder of files, and no block is shared with another entry. Two sessions
// that hold the same document therefore keep two full copies, which is what
// a per-tenant store has to guarantee.
//
// Folder names are direct: `<session>/<entry>`, where `session` is the hash
// of prompt_cache_key (or of the token prefix when the request carries no
// key) and `entry` is the hash of the entry's own token prefix. A session
// can hold several entries at different cut points; each one is its own
// folder, so no two entries ever share files.
//
// Every mutation is a scheduler decision (a spill or restore started by the
// policy, a commit rank 0 journaled), like DiskCache. This index holds no
// bytes and no file handles: it names folders and says which to write,
// read or delete. The caller removes the folders of every entry a spill
// evicts.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "sched/prefix_cache.hpp"

namespace dgpp::sched {

class FolderDiskCache {
 public:
  struct Config {
    int64_t capacity_bytes = 0;  // the bytes every resident entry's files may hold (0: off)
    // prompt_cache_key pins protect their entries while the pinned entries
    // hold at most this share of the capacity; past it a pinned entry is as
    // evictable as any other. 100: pins always protect.
    int pin_max_pct = 50;
  };
  enum class State : int { kSpilling = 0, kResident = 1, kRestoring = 2 };
  struct Entry {
    std::vector<int64_t> ids;
    PrefixCache::Images images;
    int64_t position = 0;
    int64_t next_token = -1;   // the MTP lookahead constraint (-1: none)
    int64_t mtp_position = 0;  // the draft block's row counter at the cut
    uint64_t hash = 0;         // of the ids (with the images) at the cut
    std::string session;       // the session folder's name
    std::string name;          // the entry folder's name, inside the session
    int64_t bytes = 0;         // the files' total bytes
    State state = State::kResident;
    uint64_t op = 0;           // the pending spill/restore op
    int memory = -1;           // the PrefixCache entry holding it, or -1
    uint64_t last_use = 0;
    bool live = false;
    std::string key;           // prompt_cache_key: pinned while it lasts
  };
  struct Stats {
    int64_t spills = 0;         // spills begun
    int64_t spilled = 0;        // committed
    int64_t spill_failed = 0;
    int64_t spill_skipped = 0;  // no room after evicting everything evictable
    int64_t restores = 0;
    int64_t restored = 0;
    int64_t restore_failed = 0;
    int64_t evictions = 0;      // entries evicted for room
    int64_t bytes_written = 0;  // committed spills' bytes
    int64_t bytes_read = 0;     // restores' bytes
    int64_t tokens_restored = 0;  // prompt tokens the committed restores brought back
  };
  // A spill's plan: the entry's folder, and the entries evicted for room
  // (their folders are the caller's to delete).
  struct Evicted {
    int entry = -1;
    std::string session, name;
    int64_t position = 0;
    int memory = -1;  // the memory entry that held it too, or -1
  };
  struct SpillPlan {
    bool ok = false;
    bool duplicate = false;  // an identical entry is already on disk
    int entry = -1;
    std::string session, name;
    std::vector<Evicted> evicted;
  };
  struct RestorePlan {
    std::string session, name;
    int64_t bytes = 0;
  };

  FolderDiskCache() = default;
  explicit FolderDiskCache(const Config& cfg);

  bool enabled() const { return cfg_.capacity_bytes > 0; }
  const Config& config() const { return cfg_; }
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }
  int live_entries() const;
  int pinned_entries() const;  // live entries with a key
  int64_t bytes_used() const { return used_bytes_; }
  int64_t capacity_bytes() const { return cfg_.capacity_bytes; }
  const Entry& entry(int index) const { return entries_.at(static_cast<size_t>(index)); }
  std::vector<int> live() const;  // every live entry's index, ascending
  // Whether the accounting matches the live entries (the tests): used bytes
  // are their sum, and every live entry is in the hash index.
  bool consistent(std::string* why = nullptr) const;

  // The session and entry folder names for a prefix. The session comes from
  // the key when there is one, else from the prefix (so keyless entries of
  // one prefix share a session folder and differ only in their entry name).
  static std::string session_name(const std::string& key, uint64_t prefix_hash);
  static std::string entry_name(uint64_t hash, int64_t position);

  // The deepest entry that is on disk (resident or restoring) at one of the
  // prompt's cuts, with the prompt's first `position` ids and a matching
  // lookahead token; -1 when none. Spilling entries are not offered.
  int lookup(const std::vector<int64_t>& prompt, const std::vector<int64_t>& cuts,
             const std::vector<uint64_t>& cut_hashes, const PrefixCache::Images& images = {}) const;
  int find_exact(const int64_t* ids, int64_t n, uint64_t hash, const PrefixCache::Images& images,
                 int64_t next_token) const;

  // ---- spills ------------------------------------------------------------
  // Plans an entry of `bytes` over ids[0..position). Evicts least recently
  // used resident entries until the bytes fit. The entry starts kSpilling;
  // commit or abort it by the op's outcome. Not ok (nothing changed) when
  // the entry cannot fit even alone, or an identical entry is on disk.
  SpillPlan begin_spill(const int64_t* ids, int64_t position, int64_t next_token,
                        const PrefixCache::Images& images, int64_t mtp_position, int64_t bytes,
                        uint64_t op, uint64_t now, int memory_entry, const std::string& key = "");
  void commit_spill(int entry);
  void abort_spill(int entry);

  // ---- restores ----------------------------------------------------------
  // The entry's folder to read; the entry holds kRestoring (never evicted)
  // until the op commits, cancels or aborts.
  RestorePlan begin_restore(int entry, uint64_t op);
  void commit_restore(int entry, int memory_entry, uint64_t now);
  void cancel_restore(int entry);
  void abort_restore(int entry);

  // ---- room and bookkeeping ----------------------------------------------
  void on_memory_evicted(int entry);
  void touch(int entry, uint64_t now);
  // Frees the least recently used resident entry; its index, or -1.
  int evict_lru(int* memory = nullptr);

 private:
  int lru_victim() const;
  bool pins_protect() const;
  void kill_entry(int entry);

  Config cfg_;
  std::vector<Entry> entries_;
  std::unordered_multimap<uint64_t, int> by_hash_;
  int64_t used_bytes_ = 0;
  Stats stats_;
};

}  // namespace dgpp::sched
