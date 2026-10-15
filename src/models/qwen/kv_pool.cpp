#include "models/qwen/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {
void validate(const QwenKvPoolShape& s) {
  if (s.layers < 0 || s.kv_heads <= 0 || s.dim <= 0 || s.idx_dim <= 0 || s.kpool <= 0 ||
      s.block_tokens <= 0 || s.max_requests <= 0 || s.token_slots <= 0)
    throw std::invalid_argument("QwenKvPool: every shape field must be positive");
  if (s.block_tokens % s.kpool != 0)
    throw std::invalid_argument("QwenKvPool: block_tokens must be a multiple of kpool");
  if (s.token_slots % s.block_tokens != 0)
    throw std::invalid_argument("QwenKvPool: token_slots must be a multiple of block_tokens");
}
}  // namespace

QwenKvPool::~QwenKvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
  cudaFree(k_scale_base_);
  cudaFree(v_scale_base_);
  cudaFree(idx_base_);
  cudaFree(ring_base_);
}

size_t QwenKvPool::cache_bytes(const QwenKvPoolShape& s) {
  validate(s);
  if (s.format != LatentFormat::kBf16 && s.format != LatentFormat::kFp8 &&
      s.format != LatentFormat::kFp4)
    throw std::invalid_argument("QwenKvPool: the K/V cache's format must be bf16, fp8 or fp4");
  const size_t L = static_cast<size_t>(s.layers);
  const size_t elems = static_cast<size_t>(s.token_slots) * s.kv_heads * s.dim;
  const size_t kv = s.format == LatentFormat::kFp8   ? elems
                    : s.format == LatentFormat::kFp4 ? elems / 2 + elems / 16
                                                     : elems * 2;
  const size_t scales = s.format == LatentFormat::kBf16
                            ? 0
                            : static_cast<size_t>(s.token_slots) * s.kv_heads * 4;
  const size_t idx = static_cast<size_t>(s.token_slots / s.kpool) * s.idx_dim * 2;
  const size_t ring = static_cast<size_t>(s.max_requests) * s.kpool * s.idx_dim * 2;
  const size_t table = PagedBlockTable::table_bytes(s.max_requests, s.token_slots / s.block_tokens);
  return L * (2 * kv + 2 * scales + idx + ring) + table;
}

void QwenKvPool::init(const QwenKvPoolShape& shape) {
  if (initialized_) throw std::logic_error("QwenKvPool: init twice");
  validate(shape);
  shape_ = shape;
  const size_t L = static_cast<size_t>(shape_.layers);
  auto alloc = [](void** p, size_t bytes) { DGPP_CUDA_OK(cudaMalloc(p, std::max<size_t>(bytes, 1))); };
  alloc(reinterpret_cast<void**>(&k_base_), L * layer_kv_bytes());
  alloc(reinterpret_cast<void**>(&v_base_), L * layer_kv_bytes());
  alloc(reinterpret_cast<void**>(&k_scale_base_), L * layer_scale_bytes());
  alloc(reinterpret_cast<void**>(&v_scale_base_), L * layer_scale_bytes());
  alloc(reinterpret_cast<void**>(&idx_base_), L * layer_idx_elems() * 2);
  alloc(reinterpret_cast<void**>(&ring_base_), L * layer_ring_elems() * 2);
  table_.init(shape_.max_requests, shape_.block_tokens, shape_.token_slots / shape_.block_tokens);
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

void QwenKvPool::check_req(int req, const char* what) const {
  if (!initialized_) throw std::logic_error(std::string("QwenKvPool: ") + what + " before init");
  if (req < 0 || req >= shape_.max_requests)
    throw std::out_of_range(std::string("QwenKvPool: ") + what + " request " + std::to_string(req));
}

QwenQsaCache QwenKvPool::view(int layer) const {
  if (!initialized_) throw std::logic_error("QwenKvPool: view before init");
  if (layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("QwenKvPool: layer " + std::to_string(layer));
  QwenQsaCache c;
  const size_t l = static_cast<size_t>(layer);
  c.k_cache = k_base_ + l * (layer_kv_bytes() / 2);  // the plane's base either way
  c.v_cache = v_base_ + l * (layer_kv_bytes() / 2);
  // The scale plane is [layers][token_slots][kv_heads] floats: the per-layer
  // stride in ELEMENTS (layer_scale_bytes() is already bytes — dividing the
  // product by 4 here walked every layer >= 1 to a quarter stride, into the
  // wrong layer's scales and off the end at the top layers: the fp8/fp4
  // engine's illegal access / NaN, 2026-10-14).
  const size_t scale_stride = layer_scale_bytes() / 4;
  c.k_scales = shape_.format == LatentFormat::kBf16 ? nullptr : k_scale_base_ + l * scale_stride;
  c.v_scales = shape_.format == LatentFormat::kBf16 ? nullptr : v_scale_base_ + l * scale_stride;
  c.row_bytes = static_cast<int64_t>(kv_row_bytes());
  c.format = shape_.format;
  c.index_cache = idx_base_ + l * layer_idx_elems();
  c.ring = ring_base_ + l * layer_ring_elems();
  c.block_tables = const_cast<int32_t*>(table_.device_tables());
  c.block_tokens = shape_.block_tokens;
  c.blocks_per_request = static_cast<int>(table_.total_blocks());
  c.max_requests = shape_.max_requests;
  return c;
}

uint16_t* QwenKvPool::ring(int layer, int req) const {
  check_req(req, "ring");
  if (layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("QwenKvPool: layer " + std::to_string(layer));
  return ring_base_ + static_cast<size_t>(layer) * layer_ring_elems() +
         static_cast<size_t>(req) * shape_.kpool * shape_.idx_dim;
}

size_t QwenKvPool::ring_bytes_per_request() const {
  return static_cast<size_t>(shape_.kpool) * shape_.idx_dim * 2;
}

void QwenKvPool::reset_request(int req, cudaStream_t stream) {
  check_req(req, "reset_request");
  for (int l = 0; l < shape_.layers; ++l)
    DGPP_CUDA_OK(cudaMemsetAsync(ring(l, req), 0, ring_bytes_per_request(), stream));
  release_request_blocks(req, stream);
}

void QwenKvPool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("QwenKvPool: reset_all before init");
  const size_t L = static_cast<size_t>(shape_.layers);
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, L * layer_kv_bytes(), stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, L * layer_kv_bytes(), stream));
  if (shape_.format != LatentFormat::kBf16) {
    DGPP_CUDA_OK(cudaMemsetAsync(k_scale_base_, 0, L * layer_scale_bytes(), stream));
    DGPP_CUDA_OK(cudaMemsetAsync(v_scale_base_, 0, L * layer_scale_bytes(), stream));
  }
  DGPP_CUDA_OK(cudaMemsetAsync(idx_base_, 0, L * layer_idx_elems() * 2, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(ring_base_, 0, L * layer_ring_elems() * 2, stream));
  table_.reset_all(stream);
}

void QwenKvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  const size_t kv_blk = static_cast<size_t>(shape_.block_tokens) * kv_row_elems();
  const size_t kv_blk_bytes = static_cast<size_t>(shape_.block_tokens) * kv_row_bytes();
  const size_t sc_blk = shape_.format == LatentFormat::kBf16
                            ? 0
                            : static_cast<size_t>(shape_.block_tokens) * shape_.kv_heads;
  const size_t idx_blk = static_cast<size_t>(pools_per_block()) * shape_.idx_dim;
  for (int l = 0; l < shape_.layers; ++l) {
    const QwenQsaCache c = view(l);
    DGPP_CUDA_OK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(c.k_cache) + static_cast<size_t>(dst) * kv_blk_bytes,
                                 reinterpret_cast<uint8_t*>(c.k_cache) + static_cast<size_t>(src) * kv_blk_bytes,
                                 kv_blk_bytes, cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(c.v_cache) + static_cast<size_t>(dst) * kv_blk_bytes,
                                 reinterpret_cast<uint8_t*>(c.v_cache) + static_cast<size_t>(src) * kv_blk_bytes,
                                 kv_blk_bytes, cudaMemcpyDeviceToDevice, stream));
    if (sc_blk) {
      DGPP_CUDA_OK(cudaMemcpyAsync(c.k_scales + static_cast<size_t>(dst) * sc_blk,
                                   c.k_scales + static_cast<size_t>(src) * sc_blk, sc_blk * 4,
                                   cudaMemcpyDeviceToDevice, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(c.v_scales + static_cast<size_t>(dst) * sc_blk,
                                   c.v_scales + static_cast<size_t>(src) * sc_blk, sc_blk * 4,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(c.index_cache + static_cast<size_t>(dst) * idx_blk,
                                 c.index_cache + static_cast<size_t>(src) * idx_blk, idx_blk * 2,
                                 cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace dgpp
