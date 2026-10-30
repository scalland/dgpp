// The folder backend's index (sched/folder_disk_cache.hpp, disk-cache step 1):
// each entry owns its bytes and its folder, LRU retention within a byte
// capacity, the spill and restore state machine with its aborts, pins up to
// their cap, and direct folder names for sessions and entries.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "sched/folder_disk_cache.hpp"

namespace {

using dgpp::sched::FolderDiskCache;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

FolderDiskCache::Config config(int64_t capacity, int pin_max_pct = 50) {
  FolderDiskCache::Config c;
  c.capacity_bytes = capacity;
  c.pin_max_pct = pin_max_pct;
  return c;
}

std::vector<int64_t> counted(int64_t n, int64_t base = 0) {
  std::vector<int64_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = base + i;
  return v;
}

constexpr int64_t kEntry = 40960;  // 10 pages

DGPP_TEST(folder_cache_spill_commit_lookup_and_restore) {
  FolderDiskCache d(config(1 << 20));
  require(d.enabled() && d.bytes_used() == 0, "empty at start");
  const auto prompt = counted(20);
  const auto plan = d.begin_spill(prompt.data(), 12, -1, {}, 0, kEntry, /*op=*/1, /*now=*/1, /*memory=*/0);
  require(plan.ok && plan.entry == 0, "the spill is planned");
  require(!plan.session.empty() && plan.name.find('-') != std::string::npos, "the folder names are set");
  require(d.bytes_used() == kEntry, "its bytes are accounted from the start");
  require(d.entry(0).state == FolderDiskCache::State::kSpilling, "spilling until the commit");
  std::vector<int64_t> cuts = {12};
  std::vector<uint64_t> hashes = {dgpp::sched::PrefixCache::hash_prefix(prompt.data(), 12)};
  require(d.lookup(prompt, cuts, hashes) < 0, "a spilling entry is not offered");
  d.commit_spill(0);
  require(d.lookup(prompt, cuts, hashes) == 0, "the committed entry is offered at its cut");
  const auto restore = d.begin_restore(0, /*op=*/2);
  require(restore.session == plan.session && restore.name == plan.name, "the restore reads the same folder");
  require(restore.bytes == kEntry, "the restore reads the entry's bytes");
  require(d.entry(0).state == FolderDiskCache::State::kRestoring, "restoring holds the entry");
  d.commit_restore(0, /*memory=*/3, /*now=*/5);
  require(d.entry(0).state == FolderDiskCache::State::kResident && d.entry(0).memory == 3,
          "the restore lands and links to its memory entry");
  require(d.consistent(), "the accounting is consistent");
}

DGPP_TEST(folder_cache_lru_eviction_returns_the_evicted_folders) {
  FolderDiskCache d(config(2 * kEntry + 4096));  // room for two entries
  const auto a = counted(8, 100), b = counted(8, 200), c = counted(8, 300);
  const auto pa = d.begin_spill(a.data(), 8, -1, {}, 0, kEntry, 1, /*now=*/1, -1);
  d.commit_spill(pa.entry);
  const auto pb = d.begin_spill(b.data(), 8, -1, {}, 0, kEntry, 2, /*now=*/2, -1);
  d.commit_spill(pb.entry);
  const auto pc = d.begin_spill(c.data(), 8, -1, {}, 0, kEntry, 3, /*now=*/3, -1);
  require(pc.ok, "the third entry fits by evicting the least recently used one");
  require(pc.evicted.size() == 1 && pc.evicted[0].entry == pa.entry, "the oldest entry is the victim");
  require(pc.evicted[0].session == pa.session && pc.evicted[0].name == pa.name,
          "the victim's folder names are returned for the caller to delete");
  require(d.bytes_used() == 2 * kEntry, "two entries' bytes remain");
  require(d.consistent(), "consistent after eviction");
}

DGPP_TEST(folder_cache_duplicates_and_oversize_are_refused) {
  FolderDiskCache d(config(kEntry));
  const auto p = counted(8);
  const auto first = d.begin_spill(p.data(), 8, -1, {}, 0, kEntry, 1, 1, -1);
  require(first.ok, "the first entry fits");
  d.commit_spill(first.entry);
  const auto again = d.begin_spill(p.data(), 8, -1, {}, 0, kEntry, 2, 2, -1);
  require(again.duplicate && !again.ok, "an identical entry is already on disk");
  const auto q = counted(8, 50);
  const auto big = d.begin_spill(q.data(), 8, -1, {}, 0, kEntry + 4096, 3, 3, -1);
  require(!big.ok && !big.duplicate, "an entry larger than the capacity is refused");
  require(d.stats().spill_skipped == 1, "the refusal is counted");
  require(d.live_entries() == 1, "nothing was evicted for an entry that could not fit");
}

DGPP_TEST(folder_cache_abort_releases_bytes_and_folder) {
  FolderDiskCache d(config(1 << 20));
  const auto p = counted(8);
  const auto plan = d.begin_spill(p.data(), 8, -1, {}, 0, kEntry, 1, 1, -1);
  d.abort_spill(plan.entry);
  require(d.bytes_used() == 0 && d.live_entries() == 0, "an aborted spill frees its bytes");
  require(d.consistent(), "consistent after the abort");

  const auto q = counted(8, 9);
  const auto plan2 = d.begin_spill(q.data(), 8, -1, {}, 0, kEntry, 2, 2, -1);
  d.commit_spill(plan2.entry);
  d.begin_restore(plan2.entry, 3);
  d.abort_restore(plan2.entry);
  require(d.bytes_used() == 0, "a failed restore drops the entry and its bytes");
  require(d.stats().restore_failed == 1, "the failed restore is counted");
}

DGPP_TEST(folder_cache_pins_survive_unpinned_entries_within_the_cap) {
  FolderDiskCache d(config(2 * kEntry + 4096, /*pin_max_pct=*/50));
  const auto pinned = counted(8, 10), plain = counted(8, 20), third = counted(8, 30);
  const auto pp = d.begin_spill(pinned.data(), 8, -1, {}, 0, kEntry, 1, /*now=*/1, -1, "tenant-a");
  d.commit_spill(pp.entry);
  const auto pl = d.begin_spill(plain.data(), 8, -1, {}, 0, kEntry, 2, /*now=*/2, -1);
  d.commit_spill(pl.entry);
  const auto pt = d.begin_spill(third.data(), 8, -1, {}, 0, kEntry, 3, /*now=*/3, -1);
  require(pt.ok && pt.evicted.size() == 1 && pt.evicted[0].entry == pl.entry,
          "the unpinned entry is evicted before the older pinned one");
  require(d.pinned_entries() == 1, "the pinned entry survives");
}

DGPP_TEST(folder_cache_sessions_group_by_key_and_entries_stay_separate) {
  // Two cut points of one keyed session share a session folder; each entry
  // has its own folder name, so no two entries ever share files.
  const std::string s1 = FolderDiskCache::session_name("agent-7", 111);
  const std::string s2 = FolderDiskCache::session_name("agent-7", 222);
  require(s1 == s2, "one key, one session folder");
  require(FolderDiskCache::session_name("agent-8", 111) != s1, "another key, another session");
  require(FolderDiskCache::session_name("", 111) == FolderDiskCache::session_name("", 111),
          "keyless sessions derive from the prefix");
  require(FolderDiskCache::entry_name(0xabcULL, 8) != FolderDiskCache::entry_name(0xabcULL, 16),
          "entries at different positions get different folders");

  FolderDiskCache d(config(1 << 20));
  const auto p8 = counted(8), p16 = counted(16);
  const auto a = d.begin_spill(p8.data(), 8, -1, {}, 0, kEntry, 1, 1, -1, "agent-7");
  const auto b = d.begin_spill(p16.data(), 16, -1, {}, 0, kEntry, 2, 2, -1, "agent-7");
  require(a.session == b.session && a.name != b.name, "same session folder, different entry folders");
}

}  // namespace
