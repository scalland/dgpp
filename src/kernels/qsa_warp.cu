// QSA sparse attention prefill, one warp per (query, KV group) -- the shape of
// SGLang's sparse GQA prefill kernel (BLOCK_N 16, one warp per program) on
// GB10's tensor cores. Default for long prefill walks; DGPP_QSA_WARP=0 falls
// back to the partial kernels (models/qwen/layers.cpp).
//
// The whole online softmax stays in one warp's registers: Q as m16
// A-fragments (the group's <= 16 query heads are the M rows), K/V tiles of 16
// selected tokens gathered into shared memory with cp.async (L1-allocating),
// S = Q K^T and O += P V on mma.sync m16n8k16, P reused from S's accumulator
// layout as the A-fragment (FlashAttention-2), no block barrier. It writes the
// normalized fp32 output straight into the caller's c_out -- no m/l/c
// partials, no combine pass. Lanes 0-15 resolve a tile's 16 physical rows
// (token -> block table) in one round; a serial chain starved the copies.
// One stage by default: ncu showed the two-stage form at 2 blocks/SM
// (shared-memory bound); one stage gives 5 warps/SM and residency hides the
// gather (DGPP_QSA_WARP_STAGES=2 at build time keeps the double buffer).
//
// Numerics: probabilities rounded to bf16 for P V, the denominator summing
// the unrounded values (the partial kernels' rule); the dots' summation order
// is the tensor cores'. Tolerance-equal to qsa_attn_prefill_partial +
// dsa_attn_combine (qsa_test), not bitwise -- the same class as SGLang's
// kernel, which also rounds P to bf16.
#include "kernels/qsa.hpp"
#include "kernels/qsa_kv_io.cuh"

#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kD = 256;
constexpr int kTileTok = 16;
constexpr int kRow = kD + 8;  // padded bf16 row: conflict-free ldmatrix
constexpr int kStageElems = 2 * kTileTok * kRow;  // K then V
#ifndef DGPP_QSA_WARP_STAGES
#define DGPP_QSA_WARP_STAGES 1
#endif
constexpr int kStages = DGPP_QSA_WARP_STAGES;
constexpr size_t kSmem = size_t(kStages) * kStageElems * 2;  // bytes
// fp8 codes staged raw (D4): 16 tokens x (K, V) x 256 code bytes, after the bf16 stages.
constexpr int kRawBytes = 2 * kTileTok * kD;
constexpr size_t kSmemFp8 = kSmem + size_t(kRawBytes);

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}

__global__ __launch_bounds__(32) void qsa_attn_prefill_warp_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ req_ids, const int32_t* __restrict__ topk,
    int topk_stride, const int32_t* __restrict__ counts, int local_heads, int kv_heads, int block_tokens,
    const int32_t* __restrict__ block_tables, int blocks_per_request, float scale, float* __restrict__ out,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales, LatentFormat format,
    int64_t row_bytes) {
  extern __shared__ __align__(16) uint16_t sm[];
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int h0 = kvh * group;
  const int lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int width = kv_heads * kD;
  const int cnt = counts[r];
  const int32_t* toks = topk + r * topk_stride;
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const float sl2 = scale * 1.4426950408889634f;

  // Q A-fragments, 16 k-steps x 4 registers; rows past the group are zero.
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + r * q_row_stride + static_cast<int64_t>(h0) * kD;
    const bool v0 = g < group, v1 = g + 8 < group;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c + 8) : 0u;
    }
  }

  const uint8_t* __restrict__ kplane = reinterpret_cast<const uint8_t*>(k_cache);
  const uint8_t* __restrict__ vplane = reinterpret_cast<const uint8_t*>(v_cache);
  // Stage a tile: 16 tokens x (K, V) x 32 16-byte chunks = 1024 chunks, 32 per lane.
  auto stage = [&](int tile, int buf) {
    uint16_t* kt = sm + buf * kStageElems;
    uint16_t* vt = kt + kTileTok * kRow;
    const int base = tile * kTileTok;
    if (format == LatentFormat::kFp8) {
      // fp8 codes (D4): the tile's raw code bytes cp.async into the raw region
      // (lanes 0-15 copy each row's K chunk, lanes 16-31 its V chunk, one
      // 16-byte chunk per lane per token), then the warp dequantizes them into
      // the bf16 stage with the same arithmetic as qsa_io::decode_vec (fp8).
      uint8_t* raw = reinterpret_cast<uint8_t*>(sm + kStages * kStageElems);
      const int plane = lane >> 4, chunk = lane & 15;
      int64_t my_off = 0, my_sidx = 0;  // lanes 0..15: the row's code byte / row scale index
      if (lane < kTileTok && base + lane < cnt) {
        const int tok = toks[base + lane];
        const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
        my_off = phys * row_bytes + static_cast<int64_t>(kvh) * kD;
        my_sidx = phys * kv_heads + kvh;
      }
      const uint8_t* src_plane = plane ? vplane : kplane;
      const float* sc_plane = plane ? v_scales : k_scales;
#pragma unroll
      for (int i = 0; i < kTileTok; ++i) {
        const int64_t off = __shfl_sync(0xffffffffu, my_off, i);
        if (base + i < cnt)
          cp_async16(raw + plane * (kTileTok * kD) + i * kD + chunk * 16, src_plane + off + chunk * 16, true);
      }
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group 0;\n" ::);
      __syncwarp();  // the raw codes are complete before any lane reads them
#pragma unroll
      for (int i = 0; i < kTileTok; ++i) {
        const int64_t sidx = __shfl_sync(0xffffffffu, my_sidx, i);
        uint16_t* dst = sm + buf * kStageElems + plane * (kTileTok * kRow) + i * kRow + chunk * 16;
        if (base + i < cnt) {
          float s = __ldg(sc_plane + sidx);
          if (!(s == s)) s = 0.0f;  // a corrupt scale slot decodes to zeros, never NaN
          uint4 c4 = *reinterpret_cast<const uint4*>(raw + plane * (kTileTok * kD) + i * kD + chunk * 16);
          const uint8_t* b = reinterpret_cast<const uint8_t*>(&c4);
          float v[16];
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const float f = fp8_e4m3_bits_to_float(b[j]);
            v[j] = f == f ? f * s : 0.0f;  // 0x7F (NaN) -> 0
          }
          *reinterpret_cast<uint4*>(dst) =
              make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
          *reinterpret_cast<uint4*>(dst + 8) =
              make_uint4(pack_bf16(v[8], v[9]), pack_bf16(v[10], v[11]), pack_bf16(v[12], v[13]), pack_bf16(v[14], v[15]));
        } else {
          // Past the list: zeros, as the bf16 path's zero-fill writes.
          *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
          *reinterpret_cast<uint4*>(dst + 8) = make_uint4(0u, 0u, 0u, 0u);
        }
      }
      __syncwarp();
      return;
    }
    // Lanes 0..15 resolve the tile's 16 physical rows in one round of loads
    // (a serial token -> block-table chain per issue step starved the copies).
    if (format != LatentFormat::kBf16) {
      // The quantized cache: the warp decodes the tile's rows into the same
      // bf16 stages (qsa_kv_io.cuh); no cp.async group, the syncwarp below
      // the wait stands in for the copy's completion.
      int64_t my_off = 0, my_soff = 0;
      if (lane < kTileTok && base + lane < cnt) {
        const int tok = toks[base + lane];
        const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
        my_off = phys * row_bytes;                       // the row's first stored byte
        my_soff = phys * kv_heads;  // decode_head adds elem/kD (this head) itself
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int64_t off = __shfl_sync(0xffffffffu, my_off, i);
        const int64_t soff = __shfl_sync(0xffffffffu, my_soff, i);
        const int64_t elem = static_cast<int64_t>(kvh) * kD + lane * 8;  // the row-wide index
        if (base + i < cnt) {
          qsa_io::decode_head(kt + i * kRow + lane * 8, kplane + off, elem, kD,
                              static_cast<int64_t>(width) / 2,
                              k_scales ? k_scales + soff : nullptr, format);
          qsa_io::decode_head(vt + i * kRow + lane * 8, vplane + off, elem, kD,
                              static_cast<int64_t>(width) / 2,
                              v_scales ? v_scales + soff : nullptr, format);
        } else {
          // Past the list: zeros, as the bf16 path's zero-fill cp.async
          // writes. The masked P is 0, but 0 x a stale NaN in the V stage
          // is NaN (the fp8/fp4 engine's non-finite logits, 2026-10-14).
          *reinterpret_cast<uint4*>(kt + i * kRow + lane * 8) = make_uint4(0u, 0u, 0u, 0u);
          *reinterpret_cast<uint4*>(vt + i * kRow + lane * 8) = make_uint4(0u, 0u, 0u, 0u);
        }
      }
      __syncwarp();
      return;
    }
    int64_t my_off = 0;
    if (lane < kTileTok && base + lane < cnt) {
      const int tok = toks[base + lane];
      const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
      my_off = phys * width + static_cast<int64_t>(kvh) * kD;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {            // token i, lane = its 16-byte chunk
      const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
      const bool valid = base + i < cnt;
      cp_async16(kt + i * kRow + lane * 8, k_cache + off, valid);
      cp_async16(vt + i * kRow + lane * 8, v_cache + off, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows g, g + 8 (lane-partial l)

  const int tiles = (cnt + kTileTok - 1) / kTileTok;
  if (kStages == 2 && tiles > 0) stage(0, 0);
  for (int it = 0; it < tiles; ++it) {
    const int buf = kStages == 2 ? (it & 1) : 0;
    if (kStages == 1) {
      // One stage: residency (more warps per SM) hides the gather instead.
      stage(it, 0);
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if (it + 1 < tiles) {
      stage(it + 1, buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncwarp();
    const uint16_t* kt = sm + buf * kStageElems;
    const uint16_t* vt = kt + kTileTok * kRow;
    // S = Q K^T: two n8 tiles (tokens 0-7, 8-15).
    float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      ldsm_x4(b, kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8);
      mma_bf16(sc[0], qa[ks], b[0], b[1]);
      mma_bf16(sc[1], qa[ks], b[2], b[3]);
    }
    // Mask tokens past the list, scale into the exp2 domain, row maxima over the quad.
    const int n = min(kTileTok, cnt - it * kTileTok);
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = j * 8 + 2 * t + e;
        sc[j][e] = col < n ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = col < n ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = exp2f(m0 - n0), a1 = exp2f(m1 - n1);  // m = -inf on the first tile: 0
    m0 = n0;
    m1 = n1;
    float p[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      p[j][0] = exp2f(sc[j][0] - n0);
      p[j][1] = exp2f(sc[j][1] - n0);
      p[j][2] = exp2f(sc[j][2] - n1);
      p[j][3] = exp2f(sc[j][3] - n1);
    }
    l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
    l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
    // P as the A-fragment over k = the tile's 16 tokens.
    const uint32_t pa[4] = {pack_bf16(p[0][0], p[0][1]), pack_bf16(p[0][2], p[0][3]),
                            pack_bf16(p[1][0], p[1][1]), pack_bf16(p[1][2], p[1][3])};
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      acc[d][0] *= a0;
      acc[d][1] *= a0;
      acc[d][2] *= a1;
      acc[d][3] *= a1;
    }
    // O += P V: 16 x4.trans loads, two n8 dim tiles each.
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      ldsm_x4_t(b, vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8);
      mma_bf16(acc[dp * 2], pa, b[0], b[1]);
      mma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
    }
    __syncwarp();  // this buffer's reads done before the next stage() overwrites it
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* o0 = out + (r * local_heads + h0 + g) * kD;
  float* o1 = out + (r * local_heads + h0 + g + 8) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (g < group) *reinterpret_cast<float2*>(o0 + col) = make_float2(acc[d][0] * i0, acc[d][1] * i0);
    if (g + 8 < group) *reinterpret_cast<float2*>(o1 + col) = make_float2(acc[d][2] * i1, acc[d][3] * i1);
  }
}

}  // namespace

bool qsa_warp_supported(int dim, int local_heads, int kv_heads) {
  return dim == kD && kv_heads > 0 && local_heads % kv_heads == 0 && local_heads / kv_heads <= 16;
}

void qsa_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                           const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk, int topk_stride,
                           const int32_t* counts, int rows, int local_heads, int kv_heads, int block_tokens,
                           const int32_t* block_tables, int blocks_per_request, float scale, float* out,
                           cudaStream_t stream, const float* k_scales, const float* v_scales,
                           LatentFormat format, int64_t row_bytes) {
  if (rows <= 0) return;
  if (!qsa_warp_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("qsa_attn_prefill_warp: dim 256, <= 16 query heads per KV head");
  static const bool opted = [] {  // once per process, thread-safe (TP ranks may share one)
    DGPP_CUDA_OK(cudaFuncSetAttribute(qsa_attn_prefill_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmemFp8)));
    // All of L1 as shared memory: residency here is bounded by the 33.8 KB stages.
    DGPP_CUDA_OK(cudaFuncSetAttribute(qsa_attn_prefill_warp_kernel,
                                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return true;
  }();
  (void)opted;
  const int64_t rb = row_bytes > 0
                         ? row_bytes
                         : qsa_io::row_bytes(format, static_cast<int64_t>(kv_heads) * kD);
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads));
  // fp8 adds the raw code region; the other formats keep their existing footprint.
  const size_t smem = format == LatentFormat::kFp8 ? kSmemFp8 : kSmem;
  qsa_attn_prefill_warp_kernel<<<grid, 32, smem, stream>>>(q, q_row_stride, k_cache, v_cache, req_ids, topk,
                                                              topk_stride, counts, local_heads, kv_heads,
                                                              block_tokens, block_tables, blocks_per_request,
                                                              scale, out, k_scales, v_scales, format, rb);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
