#include "sched/folder_disk_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace dgpp::sched {

namespace {
bool same_image(const ImageInput& a, const ImageInput& b) {
  return a.offset == b.offset && a.tokens == b.tokens && a.width == b.width &&
         a.height == b.height && a.grid == b.grid && a.rgb == b.rgb;
}
bool same_image_key(const PrefixCache::ImageKey& a, const PrefixCache::ImageKey& b) {
  return a.input == b.input || (a.hash == b.hash && same_image(*a.input, *b.input));
}
size_t image_count(const PrefixCache::Images& images, int64_t position) {
  size_t n = 0;
  while (n < images.size() && images[n].input->offset <= position) ++n;
  return n;
}
std::string hex16(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}
}  // namespace

FolderDiskCache::FolderDiskCache(const Config& cfg) : cfg_(cfg) {
  if (cfg_.capacity_bytes < 0) throw std::invalid_argument("FolderDiskCache: a negative capacity");
  if (cfg_.pin_max_pct < 0 || cfg_.pin_max_pct > 100)
    throw std::invalid_argument("FolderDiskCache: pin_max_pct must lie in [0, 100]");
}

int FolderDiskCache::live_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live ? 1 : 0;
  return n;
}

int FolderDiskCache::pinned_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live && !e.key.empty() ? 1 : 0;
  return n;
}

std::vector<int> FolderDiskCache::live() const {
  std::vector<int> out;
  for (size_t i = 0; i < entries_.size(); ++i)
    if (entries_[i].live) out.push_back(static_cast<int>(i));
  return out;
}

bool FolderDiskCache::consistent(std::string* why) const {
  const auto fail = [&](const std::string& msg) {
    if (why != nullptr) *why = msg;
    return false;
  };
  int64_t sum = 0;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live) continue;
    sum += e.bytes;
    bool found = false;
    const auto range = by_hash_.equal_range(e.hash);
    for (auto it = range.first; it != range.second; ++it)
      if (it->second == static_cast<int>(i)) found = true;
    if (!found) return fail("live entry " + std::to_string(i) + " is missing from the hash index");
  }
  if (sum != used_bytes_)
    return fail("used bytes " + std::to_string(used_bytes_) + " differ from the entries' sum " + std::to_string(sum));
  if (used_bytes_ > cfg_.capacity_bytes) return fail("used bytes exceed the capacity");
  return true;
}

std::string FolderDiskCache::session_name(const std::string& key, uint64_t prefix_hash) {
  if (key.empty()) return hex16(prefix_hash);
  // FNV-1a over the key's bytes: a stable name for a key of any length.
  uint64_t h = 0xcbf29ce484222325ull;
  for (const unsigned char c : key) h = (h ^ c) * 0x100000001b3ull;
  return hex16(h);
}

std::string FolderDiskCache::entry_name(uint64_t hash, int64_t position) {
  return hex16(hash) + "-" + std::to_string(position);
}

int FolderDiskCache::find_exact(const int64_t* ids, int64_t n, uint64_t hash, const PrefixCache::Images& images,
                                int64_t next_token) const {
  const auto range = by_hash_.equal_range(hash);
  int best = -1;
  for (auto it = range.first; it != range.second; ++it) {
    const Entry& e = entries_[static_cast<size_t>(it->second)];
    if (!e.live || e.position != n) continue;
    if (e.next_token >= 0 && e.next_token != next_token) continue;
    if (!std::equal(e.ids.begin(), e.ids.end(), ids)) continue;
    const size_t count = image_count(images, n);
    if (e.images.size() != count ||
        !std::equal(e.images.begin(), e.images.end(), images.begin(), same_image_key))
      continue;
    if (best < 0 || it->second < best) best = it->second;
  }
  return best;
}

int FolderDiskCache::lookup(const std::vector<int64_t>& prompt, const std::vector<int64_t>& cuts,
                            const std::vector<uint64_t>& cut_hashes, const PrefixCache::Images& images) const {
  if (cuts.size() != cut_hashes.size())
    throw std::invalid_argument("FolderDiskCache::lookup: cuts and hashes differ");
  for (size_t i = cuts.size(); i-- > 0;) {
    const int64_t c = cuts[i];
    if (c <= 0 || c >= static_cast<int64_t>(prompt.size())) continue;
    const int e = find_exact(prompt.data(), c, cut_hashes[i], images, prompt[static_cast<size_t>(c)]);
    if (e < 0) continue;
    if (entries_[static_cast<size_t>(e)].state == State::kSpilling) continue;
    return e;
  }
  return -1;
}

FolderDiskCache::SpillPlan FolderDiskCache::begin_spill(const int64_t* ids, int64_t position, int64_t next_token,
                                                        const PrefixCache::Images& images, int64_t mtp_position,
                                                        int64_t bytes, uint64_t op, uint64_t now, int memory_entry,
                                                        const std::string& key) {
  SpillPlan plan;
  if (!enabled()) return plan;
  if (position <= 0) throw std::invalid_argument("FolderDiskCache: empty entry");
  if (bytes <= 0) throw std::invalid_argument("FolderDiskCache: an entry holds no bytes");
  const uint64_t h = PrefixCache::with_images(PrefixCache::hash_prefix(ids, position), position, images);
  if (find_exact(ids, position, h, images, next_token) >= 0) {  // already on disk
    plan.duplicate = true;
    return plan;
  }
  if (bytes > cfg_.capacity_bytes) {  // cannot fit even alone
    ++stats_.spill_skipped;
    return plan;
  }
  // Room: evict least recently used resident entries until the bytes fit.
  while (used_bytes_ + bytes > cfg_.capacity_bytes) {
    const int candidate = lru_victim();
    if (candidate < 0) {
      ++stats_.spill_skipped;
      return plan;
    }
    Evicted victim;
    victim.entry = candidate;
    victim.session = entries_[static_cast<size_t>(candidate)].session;
    victim.name = entries_[static_cast<size_t>(candidate)].name;
    int memory = -1;
    if (evict_lru(&memory) != candidate) throw std::logic_error("FolderDiskCache: the eviction chose another victim");
    victim.memory = memory;
    plan.evicted.push_back(victim);
  }
  Entry e;
  e.ids.assign(ids, ids + position);
  e.images.assign(images.begin(), images.begin() + static_cast<std::ptrdiff_t>(image_count(images, position)));
  e.position = position;
  e.next_token = next_token;
  e.mtp_position = mtp_position;
  e.hash = h;
  e.session = session_name(key, PrefixCache::hash_prefix(ids, position));
  e.name = entry_name(h, position);
  e.bytes = bytes;
  e.state = State::kSpilling;
  e.op = op;
  e.memory = memory_entry;
  e.last_use = now;
  e.live = true;
  e.key = key;
  const int index = static_cast<int>(entries_.size());
  entries_.push_back(std::move(e));
  by_hash_.emplace(h, index);
  used_bytes_ += bytes;
  ++stats_.spills;
  plan.ok = true;
  plan.entry = index;
  plan.session = entries_[static_cast<size_t>(index)].session;
  plan.name = entries_[static_cast<size_t>(index)].name;
  return plan;
}

void FolderDiskCache::commit_spill(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kSpilling) throw std::logic_error("FolderDiskCache: committing an entry not spilling");
  e.state = State::kResident;
  e.op = 0;
  stats_.bytes_written += e.bytes;
  ++stats_.spilled;
}

void FolderDiskCache::abort_spill(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kSpilling) throw std::logic_error("FolderDiskCache: aborting an entry not spilling");
  ++stats_.spill_failed;
  kill_entry(entry);
}

FolderDiskCache::RestorePlan FolderDiskCache::begin_restore(int entry, uint64_t op) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kResident) throw std::logic_error("FolderDiskCache: restoring an entry not resident");
  e.state = State::kRestoring;
  e.op = op;
  ++stats_.restores;
  stats_.bytes_read += e.bytes;
  return RestorePlan{e.session, e.name, e.bytes};
}

void FolderDiskCache::commit_restore(int entry, int memory_entry, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kRestoring) throw std::logic_error("FolderDiskCache: committing an entry not restoring");
  e.state = State::kResident;
  e.op = 0;
  e.last_use = now;
  e.memory = memory_entry;
  ++stats_.restored;
}

void FolderDiskCache::cancel_restore(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kRestoring) throw std::logic_error("FolderDiskCache: cancelling an entry not restoring");
  e.state = State::kResident;
  e.op = 0;
  --stats_.restores;
  stats_.bytes_read -= e.bytes;
}

void FolderDiskCache::abort_restore(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live || e.state != State::kRestoring) throw std::logic_error("FolderDiskCache: aborting an entry not restoring");
  ++stats_.restore_failed;
  kill_entry(entry);
}

void FolderDiskCache::on_memory_evicted(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (e.live) e.memory = -1;
}

void FolderDiskCache::touch(int entry, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (e.live) e.last_use = now;
}

bool FolderDiskCache::pins_protect() const {
  if (cfg_.pin_max_pct >= 100) return true;
  if (cfg_.pin_max_pct <= 0) return false;
  int64_t pinned = 0;
  for (const Entry& e : entries_)
    if (e.live && !e.key.empty()) pinned += e.bytes;
  return pinned * 100 <= static_cast<int64_t>(cfg_.pin_max_pct) * cfg_.capacity_bytes;
}

int FolderDiskCache::lru_victim() const {
  // The least recently used resident unpinned entry; a pinned one only when
  // every resident entry is pinned, or when the pins hold more than the cap.
  const bool protect = pins_protect();
  int victim = -1;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.state != State::kResident) continue;
    if (victim < 0) {
      victim = static_cast<int>(i);
      continue;
    }
    const Entry& v = entries_[static_cast<size_t>(victim)];
    const bool pinned = protect && !e.key.empty(), victim_pinned = protect && !v.key.empty();
    if ((victim_pinned && !pinned) || (pinned == victim_pinned && e.last_use < v.last_use))
      victim = static_cast<int>(i);
  }
  return victim;
}

int FolderDiskCache::evict_lru(int* memory) {
  const int victim = lru_victim();
  if (memory != nullptr) *memory = victim < 0 ? -1 : entries_[static_cast<size_t>(victim)].memory;
  if (victim < 0) return -1;
  ++stats_.evictions;
  kill_entry(victim);
  return victim;
}

void FolderDiskCache::kill_entry(int entry) {
  Entry& e = entries_.at(static_cast<size_t>(entry));
  if (!e.live) throw std::logic_error("FolderDiskCache: killing a dead entry");
  used_bytes_ -= e.bytes;
  const auto range = by_hash_.equal_range(e.hash);
  for (auto it = range.first; it != range.second; ++it)
    if (it->second == entry) {
      by_hash_.erase(it);
      break;
    }
  e.live = false;
  e.memory = -1;
  e.key.clear();
  e.bytes = 0;
  std::vector<int64_t>().swap(e.ids);
  PrefixCache::Images().swap(e.images);
}

}  // namespace dgpp::sched
