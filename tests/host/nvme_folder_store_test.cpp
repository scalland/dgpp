// Unit tests for the folder backend's store (engine/nvme_folder_store.hpp):
// byte-exact round trips across files, streamed writes and reads in page runs
// that cross file boundaries, capacity accounting and refusal, checksum
// detection, replacement, key validation, restart rescan and size checks.
// Host only; no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/nvme_folder_store.hpp"

namespace fs = std::filesystem;

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

std::vector<uint8_t> pattern(size_t bytes, uint8_t seed) {
  std::vector<uint8_t> v(bytes);
  for (size_t i = 0; i < bytes; ++i) v[i] = static_cast<uint8_t>(seed + i * 131u + (i >> 12));
  return v;
}

dgpp::NvmeFolderStore::Options options(const std::string& root, int64_t capacity) {
  dgpp::NvmeFolderStore::Options o;
  o.root = root;
  o.capacity_bytes = capacity;
  o.page_bytes = 4096;
  o.file_bytes = 16384;  // four pages a file, so a 40 KiB entry spans three files
  o.io_threads = 4;
  return o;
}

void test_round_trip_across_files(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(40960, 7);
  std::string why;
  CHECK(store.put("sess-a/e0", data.data(), data.size(), &why));
  CHECK(store.file_count(data.size()) == 3);
  CHECK(store.contains("sess-a/e0"));
  CHECK(store.bytes_used() == 40960);
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(store.get("sess-a/e0", back.data(), back.size(), &why));
  CHECK(back == data);
  CHECK(fs::exists(store.entry_dir("sess-a/e0") + "/part-2.bin"));
  CHECK(fs::exists(store.entry_dir("sess-a/e0") + "/part-2.crc"));
}

void test_capacity_refusal_keeps_previous(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 50000));
  const auto small = pattern(20480, 1);
  const auto big = pattern(40960, 2);
  std::string why;
  CHECK(store.put("sess-cap/e0", small.data(), small.size(), &why));
  // The replacement fits: the old 20 KiB is released first, so 40 KiB <= 50000.
  CHECK(store.put("sess-cap/e0", big.data(), big.size(), &why));
  CHECK(!store.put("sess-other/e0", big.data(), big.size(), &why));
  CHECK(!why.empty());
  CHECK(!store.contains("sess-other/e0"));
  CHECK(store.bytes_used() == 40960);
  std::vector<uint8_t> back(big.size(), 0);
  CHECK(store.get("sess-cap/e0", back.data(), back.size(), &why));
  CHECK(back == big);
}

void test_remove_releases_bytes_and_empty_session(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(8192, 3);
  std::string why;
  CHECK(store.put("sess-rm/e0", data.data(), data.size(), &why));
  store.remove("sess-rm/e0");
  CHECK(!store.contains("sess-rm/e0"));
  CHECK(store.bytes_used() == 0);
  CHECK(!fs::exists(root + "/sess-rm"));
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(!store.get("sess-rm/e0", back.data(), back.size(), &why));
}

void test_entries_of_one_session_stay_separate(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto a = pattern(8192, 12), b = pattern(12288, 13);
  std::string why;
  CHECK(store.put("sess-two/a", a.data(), a.size(), &why));
  CHECK(store.put("sess-two/b", b.data(), b.size(), &why));
  store.remove("sess-two/a");
  CHECK(!store.contains("sess-two/a"));
  CHECK(fs::exists(root + "/sess-two"));  // b still lives in it
  std::vector<uint8_t> back(b.size(), 0);
  CHECK(store.get("sess-two/b", back.data(), back.size(), &why));
  CHECK(back == b);
}

void test_checksum_detects_corruption(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(40960, 9);
  std::string why;
  CHECK(store.put("sess-bad/e0", data.data(), data.size(), &why));
  // Flip one byte in the middle file's payload.
  const std::string path = store.entry_dir("sess-bad/e0") + "/part-1.bin";
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    char c = 0;
    f.seekg(100);
    f.get(c);
    f.clear();
    f.seekp(100);
    f.put(static_cast<char>(c ^ 0x5a));
  }
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(!store.get("sess-bad/e0", back.data(), back.size(), &why));
  CHECK(why.find("checksum mismatch") != std::string::npos);
}

void test_replacement_updates_accounting(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto first = pattern(8192, 4);
  const auto second = pattern(16384, 5);
  std::string why;
  CHECK(store.put("sess-rep/e0", first.data(), first.size(), &why));
  CHECK(store.put("sess-rep/e0", second.data(), second.size(), &why));
  CHECK(store.bytes_used() == 16384);
  std::vector<uint8_t> back(second.size(), 0);
  CHECK(store.get("sess-rep/e0", back.data(), back.size(), &why));
  CHECK(back == second);
  // The old entry's files are gone: the 16 KiB entry is one file, part-0.
  CHECK(fs::exists(store.entry_dir("sess-rep/e0") + "/part-0.bin"));
  CHECK(!fs::exists(store.entry_dir("sess-rep/e0") + "/part-1.bin"));
}

// Streamed writes in page runs that do not line up with the file boundary
// (12 KiB, then 28 KiB across the 16 KiB file edge), read back in other runs.
void test_streamed_entry_crosses_files(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(40960, 11);
  std::string why;
  {
    dgpp::NvmeFolderStore::Writer w(store, "sess-s/e0", data.size(), &why);
    CHECK(w.ok());
    CHECK(w.append(data.data(), 12288, &why));
    CHECK(w.append(data.data() + 12288, 28672, &why));
    CHECK(w.commit(&why));
  }
  CHECK(store.contains("sess-s/e0"));
  CHECK(store.bytes_used() == 40960);
  CHECK(fs::exists(store.entry_dir("sess-s/e0") + "/part-2.crc"));
  // The same bytes as a whole-buffer get.
  std::vector<uint8_t> whole(data.size(), 0);
  CHECK(store.get("sess-s/e0", whole.data(), whole.size(), &why));
  CHECK(whole == data);
  std::vector<uint8_t> back(data.size(), 0);
  dgpp::NvmeFolderStore::Reader r(store, "sess-s/e0", &why);
  CHECK(r.ok() && r.bytes() == data.size());
  CHECK(r.read(back.data(), 8192, &why));
  CHECK(r.read(back.data() + 8192, 32768, &why));
  CHECK(r.read_bytes() == data.size());
  CHECK(back == data);
}

void test_streamed_abort_leaves_nothing(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(8192, 14);
  std::string why;
  {
    dgpp::NvmeFolderStore::Writer w(store, "sess-x/e0", data.size(), &why);
    CHECK(w.ok());
    CHECK(w.append(data.data(), 4096, &why));
    // Destroyed before the commit: the temp folder goes.
  }
  CHECK(!store.contains("sess-x/e0"));
  CHECK(store.bytes_used() == 0);
  bool temp_left = false;
  for (const auto& e : fs::directory_iterator(root))
    if (e.path().filename().string().rfind(".tmp-", 0) == 0) temp_left = true;
  CHECK(!temp_left);
}

void test_streamed_read_detects_corruption(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(16384, 15);
  std::string why;
  CHECK(store.put("sess-sb/e0", data.data(), data.size(), &why));
  {
    std::fstream f(store.entry_dir("sess-sb/e0") + "/part-0.bin", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(5000);
    f.put('\x7f');
  }
  dgpp::NvmeFolderStore::Reader r(store, "sess-sb/e0", &why);
  CHECK(r.ok());
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(!r.read(back.data(), back.size(), &why));
  CHECK(why.find("checksum mismatch") != std::string::npos);
}

void test_key_validation(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(4096, 16);
  std::string why;
  const char* bad[] = {"flat", "a/b/c", "/entry", "session/", ".hidden/e0", "sess/..", "a\\b/c"};
  for (const char* key : bad) {
    bool threw = false;
    try {
      store.put(key, data.data(), data.size(), &why);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  CHECK(store.bytes_used() == 0);
}

void test_rescan_after_restart(const std::string& root) {
  std::string why;
  const auto data = pattern(24576, 6);
  {
    dgpp::NvmeFolderStore store(options(root, 1 << 20));
    CHECK(store.put("sess-keep/e0", data.data(), data.size(), &why));
  }
  // An interrupted write leaves a temp folder; the next start removes it.
  fs::create_directories(root + "/.tmp-sess-torn.e0");
  dgpp::NvmeFolderStore again(options(root, 1 << 20));
  CHECK(again.contains("sess-keep/e0"));
  CHECK(!fs::exists(root + "/.tmp-sess-torn.e0"));
  CHECK(again.bytes_used() == 24576);
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(again.get("sess-keep/e0", back.data(), back.size(), &why));
  CHECK(back == data);
}

void test_wrong_size_is_refused(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(8192, 8);
  std::string why;
  CHECK(store.put("sess-size/e0", data.data(), data.size(), &why));
  std::vector<uint8_t> back(4096, 0);
  CHECK(!store.get("sess-size/e0", back.data(), back.size(), &why));
  CHECK(why.find("holds") != std::string::npos);
}

}  // namespace

int main() {
  // Each test gets its own root: the store's accounting is per root, so a
  // leftover entry from one test must not count against another.
  const fs::path base = fs::temp_directory_path() / "dgpp-folder-store-test";
  const std::vector<std::pair<std::string, void (*)(const std::string&)>> tests = {
      {"round-trip", test_round_trip_across_files},
      {"capacity", test_capacity_refusal_keeps_previous},
      {"remove", test_remove_releases_bytes_and_empty_session},
      {"sessions", test_entries_of_one_session_stay_separate},
      {"checksum", test_checksum_detects_corruption},
      {"replace", test_replacement_updates_accounting},
      {"stream", test_streamed_entry_crosses_files},
      {"stream-abort", test_streamed_abort_leaves_nothing},
      {"stream-checksum", test_streamed_read_detects_corruption},
      {"keys", test_key_validation},
      {"rescan", test_rescan_after_restart},
      {"size", test_wrong_size_is_refused},
  };
  fs::remove_all(base);
  for (const auto& [name, run] : tests) {
    const std::string root = (base / name).string();
    fs::remove_all(root);
    run(root);
    fs::remove_all(root);
  }
  fs::remove_all(base);
  if (failures) {
    std::fprintf(stderr, "nvme_folder_store_test: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("nvme_folder_store_test: all checks passed\n");
  return 0;
}
