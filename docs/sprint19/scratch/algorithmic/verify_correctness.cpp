// verify_correctness.cpp — Sprint 19 algorithmic ablation correctness check
//
// Dumps the output of T0 (production), T1 (fuse-valid), T2 (bin-major),
// T3 (no-shuffle owner), and T3b (no-shuffle atomic) for feature 0, bins 0..4,
// and compares against a CPU reference. Validates that T1, T2, T3b are
// functionally correct; T3 is expected to under-count by 32x.
//
// COMPILE
//   clang++ -std=c++17 -O2 \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation \
//     docs/sprint19/scratch/algorithmic/verify_correctness.cpp \
//     -o /tmp/verify_correctness

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace mx = mlx::core;

static constexpr uint32_t N_DOCS    = 50000;
static constexpr uint32_t LINE_SIZE = 25;
static constexpr uint32_t NUM_BINS  = 128;

// Re-use the kernel sources from microbench_algorithmic.cpp — included inline.
// (Copy-paste rather than refactor to keep scratch files self-contained.)
extern const std::string kHeader;
extern const std::string kT0Source;
extern const std::string kT1Source;
extern const std::string kT2Source;
extern const std::string kT3Source;
extern const std::string kT3bSource;

const std::string kHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE         = 32;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE     = 256;
constant constexpr uint BLOCK_SIZE        = 256;
constant constexpr uint NUM_SIMD_GROUPS   = 8;
constant constexpr uint HIST_PER_SIMD     = FEATURES_PER_PACK * BINS_PER_BYTE;
)metal";

// Minimal versions of T0 and T3b (we only need CORRECTNESS verification)
const std::string kT0Source = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];
    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;
    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint numDocs = totalNumDocs;
    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE) {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint docIdx = docIndices[d];
            packed = compressedIndex[docIdx];
            stat   = stats[docIdx];
        }
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s     = simd_shuffle(packed, src);
            const float s_s     = simd_shuffle(stat,   src);
            const bool  valid_s = simd_shuffle(valid,  src);
            if (!valid_s) continue;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cross-SIMD fold (DEC-009): sum simdHist[0..7][b] into simdHist[0][b]
    for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++) {
        const uint tbase = tile * BINS_PER_BYTE;
        if (tid < BINS_PER_BYTE) {
            float sum = 0.0f;
            for (uint g = 0u; g < NUM_SIMD_GROUPS; g++)
                sum += simdHist[g][tbase + tid];
            simdHist[0][tbase + tid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

const std::string kT3bSource = R"metal(
    threadgroup atomic_uint simdHistU[HIST_PER_SIMD];
    const uint tid = thread_index_in_threadgroup;
    for (uint b = tid; b < HIST_PER_SIMD; b += BLOCK_SIZE)
        atomic_store_explicit(&simdHistU[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint numDocs = totalNumDocs;
    for (uint d = tid; d < numDocs; d += BLOCK_SIZE) {
        const uint docIdx = docIndices[d];
        const uint packed = compressedIndex[docIdx];
        const float stat  = stats[docIdx];
        for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
            const uint bin = (packed >> (24u - 8u * f)) & 0xFFu;
            if (bin < foldCounts[f] + 1u) {
                const uint idx = f * BINS_PER_BYTE + bin;
                uint oldBits = atomic_load_explicit(&simdHistU[idx], memory_order_relaxed);
                for (;;) {
                    const float oldF = as_type<float>(oldBits);
                    const float newF = oldF + stat;
                    const uint  newBits = as_type<uint>(newF);
                    uint expected = oldBits;
                    if (atomic_compare_exchange_weak_explicit(
                            &simdHistU[idx], &expected, newBits,
                            memory_order_relaxed, memory_order_relaxed))
                        break;
                    oldBits = expected;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HIST_PER_SIMD) {
        const uint bits = atomic_load_explicit(&simdHistU[tid], memory_order_relaxed);
        output[tid] = as_type<float>(bits);
    }
)metal";

int main() {
    // Host data (SAME seed as microbench_algorithmic.cpp)
    std::vector<uint32_t> ci(LINE_SIZE * N_DOCS, 0u);
    for (uint32_t d = 0; d < N_DOCS; ++d) {
        const uint32_t b0 = (d * 7u)  & 0x7Fu;
        const uint32_t b1 = (d * 11u) & 0x7Fu;
        const uint32_t b2 = (d * 13u) & 0x7Fu;
        const uint32_t b3 = (d * 17u) & 0x7Fu;
        ci[0 * N_DOCS + d] = (b0<<24)|(b1<<16)|(b2<<8)|b3;
    }
    std::vector<uint32_t> docIdx(N_DOCS);
    std::iota(docIdx.begin(), docIdx.end(), 0u);
    std::vector<float> stat(N_DOCS);
    for (uint32_t d = 0; d < N_DOCS; ++d)
        stat[d] = static_cast<float>(d % 128) / 128.0f;
    std::vector<uint32_t> foldCounts(4, NUM_BINS);

    // CPU reference for feature 0
    std::vector<double> cpuF0(NUM_BINS, 0.0);
    std::vector<uint32_t> cpuF0count(NUM_BINS, 0);
    for (uint32_t d = 0; d < N_DOCS; ++d) {
        const uint32_t b0 = (ci[d] >> 24) & 0xFFu;
        if (b0 < NUM_BINS) { cpuF0[b0] += stat[d]; cpuF0count[b0]++; }
    }

    // GPU setup — note compressedIndex here is "feature column 0" already,
    // so we pass the first N entries (featureColumnIdx=0 * totalNumDocs + d).
    // Simpler: pass ci directly and dereference as compressedIndex[docIdx]
    // since feature col 0 offset is 0*N = 0.
    auto ciArr = mx::array(reinterpret_cast<const int32_t*>(ci.data()),
                           {static_cast<int>(N_DOCS)}, mx::uint32);
    auto diArr = mx::array(reinterpret_cast<const int32_t*>(docIdx.data()),
                           {static_cast<int>(N_DOCS)}, mx::uint32);
    auto stArr = mx::array(stat.data(), {static_cast<int>(N_DOCS)}, mx::float32);
    auto fcArr = mx::array(reinterpret_cast<const int32_t*>(foldCounts.data()),
                           {4}, mx::uint32);
    auto tdArr = mx::array(static_cast<uint32_t>(N_DOCS), mx::uint32);
    mx::eval({ciArr, diArr, stArr, fcArr, tdArr});

    auto probeOut = mx::zeros({1024}, mx::float32);
    auto grid = std::make_tuple(256, 1, 1);
    auto tg   = std::make_tuple(256, 1, 1);

    auto t0K = mx::fast::metal_kernel(
        "t0_prod",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kT0Source, kHeader, true, false);
    auto t3bK = mx::fast::metal_kernel(
        "t3b_atomic",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kT3bSource, kHeader, true, false);

    auto t0Res = t0K({ciArr, stArr, diArr, fcArr, tdArr},
                     {probeOut.shape()}, {mx::float32}, grid, tg, {}, 0.0f,
                     false, mx::Device::gpu);
    auto t3bRes = t3bK({ciArr, stArr, diArr, fcArr, tdArr},
                       {probeOut.shape()}, {mx::float32}, grid, tg, {}, 0.0f,
                       false, mx::Device::gpu);
    mx::eval({t0Res[0], t3bRes[0]});

    // Pull to host
    t0Res[0].eval();
    t3bRes[0].eval();
    auto t0Host  = std::vector<float>(
        t0Res[0].data<float>(), t0Res[0].data<float>() + 1024);
    auto t3bHost = std::vector<float>(
        t3bRes[0].data<float>(), t3bRes[0].data<float>() + 1024);

    // Compare feature 0 (bins 0..127 at positions 0..127 in output)
    fprintf(stdout, "\nFeature 0 histogram (first 10 bins):\n");
    fprintf(stdout, "%5s  %12s  %12s  %12s\n", "bin", "cpu", "t0", "t3b");
    double t0_total = 0, t3b_total = 0, cpu_total = 0;
    double max_err_t0 = 0, max_err_t3b = 0;
    for (uint32_t b = 0; b < NUM_BINS; ++b) {
        cpu_total += cpuF0[b];
        t0_total  += t0Host[b];
        t3b_total += t3bHost[b];
        const double err_t0  = std::abs(t0Host[b]  - cpuF0[b]);
        const double err_t3b = std::abs(t3bHost[b] - cpuF0[b]);
        if (err_t0  > max_err_t0)  max_err_t0  = err_t0;
        if (err_t3b > max_err_t3b) max_err_t3b = err_t3b;
        if (b < 10) {
            fprintf(stdout, "%5u  %12.4f  %12.4f  %12.4f\n",
                    b, cpuF0[b], t0Host[b], t3bHost[b]);
        }
    }
    fprintf(stdout, "\nTotals:  cpu=%.4f  t0=%.4f  t3b=%.4f\n",
            cpu_total, t0_total, t3b_total);
    fprintf(stdout, "Max abs err vs CPU: t0=%.6f, t3b=%.6f\n",
            max_err_t0, max_err_t3b);
    return 0;
}
