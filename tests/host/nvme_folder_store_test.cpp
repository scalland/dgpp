// Unit tests for the folder backend's store (engine/nvme_folder_store.hpp):
// byte-exact round trips across files, capacity accounting and refusal,
// checksum detection, replacement, restart rescan and size checks. Host only;
// no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
  CHECK(store.put("sess-a", data.data(), data.size(), &why));
  CHECK(store.file_count(data.size()) == 3);
  CHECK(store.contains("sess-a"));
  CHECK(store.bytes_used() == 40960);
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(store.get("sess-a", back.data(), back.size(), &why));
  CHECK(back == data);
  CHECK(fs::exists(store.session_dir("sess-a") + "/part-2.bin"));
  CHECK(fs::exists(store.session_dir("sess-a") + "/part-2.crc"));
}

void test_capacity_refusal_keeps_previous(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 50000));
  const auto small = pattern(20480, 1);
  const auto big = pattern(40960, 2);
  std::string why;
  CHECK(store.put("sess-cap", small.data(), small.size(), &why));
  // The replacement fits: the old 20 KiB is released first, so 40 KiB <= 50000.
  CHECK(store.put("sess-cap", big.data(), big.size(), &why));
  CHECK(!store.put("sess-other", big.data(), big.size(), &why));
  CHECK(!why.empty());
  CHECK(!store.contains("sess-other"));
  CHECK(store.bytes_used() == 40960);
  std::vector<uint8_t> back(big.size(), 0);
  CHECK(store.get("sess-cap", back.data(), back.size(), &why));
  CHECK(back == big);
}

void test_remove_releases_bytes(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(8192, 3);
  std::string why;
  CHECK(store.put("sess-rm", data.data(), data.size(), &why));
  store.remove("sess-rm");
  CHECK(!store.contains("sess-rm"));
  CHECK(store.bytes_used() == 0);
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(!store.get("sess-rm", back.data(), back.size(), &why));
}

void test_checksum_detects_corruption(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(40960, 9);
  std::string why;
  CHECK(store.put("sess-bad", data.data(), data.size(), &why));
  // Flip one byte in the middle file's payload.
  const std::string path = store.session_dir("sess-bad") + "/part-1.bin";
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
  CHECK(!store.get("sess-bad", back.data(), back.size(), &why));
  CHECK(why.find("checksum mismatch") != std::string::npos);
}

void test_replacement_updates_accounting(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto first = pattern(8192, 4);
  const auto second = pattern(16384, 5);
  std::string why;
  CHECK(store.put("sess-rep", first.data(), first.size(), &why));
  CHECK(store.put("sess-rep", second.data(), second.size(), &why));
  CHECK(store.bytes_used() == 16384);
  std::vector<uint8_t> back(second.size(), 0);
  CHECK(store.get("sess-rep", back.data(), back.size(), &why));
  CHECK(back == second);
  // The old entry's files are gone: the 16 KiB entry is one file, part-0.
  CHECK(fs::exists(store.session_dir("sess-rep") + "/part-0.bin"));
  CHECK(!fs::exists(store.session_dir("sess-rep") + "/part-1.bin"));
}

void test_rescan_after_restart(const std::string& root) {
  std::string why;
  const auto data = pattern(24576, 6);
  {
    dgpp::NvmeFolderStore store(options(root, 1 << 20));
    CHECK(store.put("sess-keep", data.data(), data.size(), &why));
  }
  // An interrupted write leaves a temp folder; the next start removes it.
  fs::create_directories(root + "/.tmp-sess-torn");
  dgpp::NvmeFolderStore again(options(root, 1 << 20));
  CHECK(again.contains("sess-keep"));
  CHECK(!fs::exists(root + "/.tmp-sess-torn"));
  CHECK(again.bytes_used() == 24576);
  std::vector<uint8_t> back(data.size(), 0);
  CHECK(again.get("sess-keep", back.data(), back.size(), &why));
  CHECK(back == data);
}

void test_wrong_size_is_refused(const std::string& root) {
  dgpp::NvmeFolderStore store(options(root, 1 << 20));
  const auto data = pattern(8192, 8);
  std::string why;
  CHECK(store.put("sess-size", data.data(), data.size(), &why));
  std::vector<uint8_t> back(4096, 0);
  CHECK(!store.get("sess-size", back.data(), back.size(), &why));
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
      {"remove", test_remove_releases_bytes},
      {"checksum", test_checksum_detects_corruption},
      {"replace", test_replacement_updates_accounting},
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
