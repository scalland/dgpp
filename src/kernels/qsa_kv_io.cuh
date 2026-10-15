#pragma once
// QSA cache row I/O: the quantized K/V formats for the Qwen3.8-Flash-Next
// paged caches (engine.kv_dtype on the qwen4_exp family, 2026-10-13).
// One token's row holds every kv head's row back to back (kv_heads * dim
// elements). Stored widths per row of E elements:
//   bf16       2E                     the historical cache, bitwise unchanged
//   fp8        E e4m3 codes, plus a plane of fp32 row scales (one per token
//              and kv head): scale = absmax/448, code = e4m3(x * 448/absmax)
//   fp4        E/2 e2m1 nibbles with one e4m3 scale per 16 elements after
//              the codes and the fp32 row-scale plane (the NVFP4 recipe,
//              kernels/latent_format.hpp)
// The attention kernels keep their bf16 shared-memory tiles exactly as
// they are: a quantized gather decodes into the tile on the way in (the
// DSA latent tiles' pattern, kernels/dsa.cu), so the arithmetic above the
// cache is bitwise the bf16 kernel over the dequantized rows. The codecs
// are the shared latent_format.hpp device functions the host reference and
// the pool agree on bit for bit.
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace qsa_io {

// The stored bytes of one token's row (E = kv_heads * dim elements).
constexpr int64_t row_bytes(LatentFormat f, int64_t e) {
  switch (f) {
    case LatentFormat::kFp8: return e;
    case LatentFormat::kFp4: return e / 2 + e / 16;
    default: return e * 2;
  }
}

// Decode one 8-element vector of one row into a bf16 smem row.
//   row     the row's first stored byte (plane base + phys * row_bytes)
//   elem    the vector's first element within the row (kvh * dim + c8 * 8)
//   dim     the head dim — head = elem / dim selects the row scale
//   codes_bytes the nibble region's size (E/2) — where the fp4 block scales
//           start within the row (0 for bf16/fp8)
//   scales  the token's row-scale row (k_row_scales + phys * kv_heads);
//           unused (may be null) for bf16 and fp4_block
template <LatentFormat F>
__device__ __forceinline__ void decode_vec(uint16_t* dst, const uint8_t* row, int64_t elem,
                                           int dim, int64_t codes_bytes, const float* scales) {
  if constexpr (F == LatentFormat::kBf16) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(row + elem * 2);
  } else if constexpr (F == LatentFormat::kFp8) {
    const uint2 c = *reinterpret_cast<const uint2*>(row + elem);
    float s = __ldg(scales + elem / dim);
    if (!(s == s)) s = 0.0f;  // a never-written or corrupt scale slot decodes to
                              // zeros, never to NaN (a NaN K/V poisons the
                              // router's logits and the MoE's expert ids)
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&c);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float f = fp8_e4m3_bits_to_float(b[i]);
      dst[i] = float_to_bf16_bits(f == f ? f * s : 0.0f);  // 0x7F (NaN) -> 0
    }
  } else {  // kFp4: the NVFP4 row — nibbles, e4m3 scales per 16, times the row scale
    const uint32_t c = *reinterpret_cast<const uint32_t*>(row + elem / 2);
    float block_scale = fp8_e4m3_bits_to_float(row[codes_bytes + elem / kLatentFp4Block]);
    if (!(block_scale == block_scale)) block_scale = 0.0f;  // a 0x7F scale byte -> 0
    const float s = block_scale * __ldg(scales + elem / dim);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[2 * i] = latent_fp4_decode_bf16(static_cast<uint8_t>(b[i] & 0x0Fu), s);
      dst[2 * i + 1] = latent_fp4_decode_bf16(static_cast<uint8_t>(b[i] >> 4), s);
    }
  }
}


// The attention tile gather (2026-10-13, engine.kv_dtype on the Qwen K/V
// caches): n rows of the kv head `kvh` of both planes decode into the
// kernels' own bf16 smem tiles (stride D+8 u16 — the tile row the bf16
// kernel's cp.async path fills, so the arithmetic above the tile is
// bitwise the bf16 kernel's over the dequantized rows). A kv head's stored
// slice of one token's row: bf16 2D bytes, fp8 D code bytes (plus one fp32
// row scale per token & head in the plane), fp4 D/2 nibble bytes plus D/16
// e4m3 block scales after them (plus the row-scale plane). rb is one
// token's whole stored row; `phys` the tile's physical token slots. The
// bf16 kernels never call this (their cp.async path is untouched).
__device__ __forceinline__ int64_t head_bytes(LatentFormat f, int64_t D) {
  return f == LatentFormat::kFp4   ? D / 2 + D / 16
         : f == LatentFormat::kFp8 ? D
                                   : D * 2;
}

template <int D>
__device__ __forceinline__ void gather_tile(
    const uint8_t* __restrict__ kplane, const uint8_t* __restrict__ vplane,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales, LatentFormat f,
    int64_t rb, int kv_heads, const int64_t* __restrict__ phys, int n, int kvh, uint16_t* kt,
    uint16_t* vt) {
  constexpr int kVec = D / 8;
  const int64_t codes_bytes = static_cast<int64_t>(kv_heads) * D / 2;  // fp4's scale area
  const int total = n * kVec * 2;
  for (int idx = static_cast<int>(threadIdx.x); idx < total; idx += static_cast<int>(blockDim.x)) {
    const int which = idx / (n * kVec);
    const int rem = idx - which * (n * kVec);
    const int tt = rem / kVec;
    const int c8 = rem - tt * kVec;
    if (tt >= n) continue;
    // Row-wide element index: decode_vec turns it into the format's byte
    // offset, picks the row scale (elem / D = the kv head) and the fp4
    // block byte (codes_bytes + elem / 16) — the append kernel's layout.
    const int64_t elem = static_cast<int64_t>(kvh) * D + static_cast<int64_t>(c8) * 8;
    const uint8_t* row = (which == 0 ? kplane : vplane) + phys[tt] * rb;
    uint16_t* d = (which == 0 ? kt : vt) + static_cast<int64_t>(tt) * (D + 8) +
                  static_cast<int64_t>(c8) * 8;
    const float* base = which == 0 ? k_scales : v_scales;
    const float* sc = base ? base + phys[tt] * kv_heads : nullptr;
    if (f == LatentFormat::kFp8)
      decode_vec<LatentFormat::kFp8>(d, row, elem, D, 0, sc);
    else if (f == LatentFormat::kFp4)
      decode_vec<LatentFormat::kFp4>(d, row, elem, D, codes_bytes, sc);
    else
      decode_vec<LatentFormat::kBf16>(d, row, elem, D, 0, nullptr);
  }
}


// The runtime-format form for kernels that keep one code path (the warp
// prefill's stage): `row` is the token row's first stored byte (the whole
// row, every head), `elem` the row-wide element index, `head_dim` the head
// dim (elem / head_dim selects the row scale), `codes_bytes` the nibble
// region's width (fp4: the row's element count / 2, where its e4m3 block
// scales start), `scales` the token's row-scale row (null with bf16).
__device__ __forceinline__ void decode_head(uint16_t* dst, const uint8_t* row, int64_t elem,
                                            int head_dim, int64_t codes_bytes, const float* scales,
                                            LatentFormat f) {
  if (f == LatentFormat::kFp8)
    decode_vec<LatentFormat::kFp8>(dst, row, elem, head_dim, 0, scales);
  else if (f == LatentFormat::kFp4)
    decode_vec<LatentFormat::kFp4>(dst, row, elem, head_dim, codes_bytes, scales);
  else
    decode_vec<LatentFormat::kBf16>(dst, row, elem, head_dim, 0, nullptr);
}

}  // namespace qsa_io
}  // namespace dgpp
