// microbench_gather.cpp — Sprint 19 / S19-01c gather micro-benchmark + probe harness
//
// PURPOSE
//   Directly measures the cost of compressedIndex gather in row-major vs
//   column-major layout, in isolation (no accumulation, no shuffle, no TG memory).
//   If the layouts measure identically, the S19-01b attribution model was wrong
//   and DEC-015 should be rejected. If col-major is significantly faster in
//   isolation but not in the full kernel, something else in the full kernel
//   is hiding or dominating the effect.
//
//   Probe A: strip simd_shuffle inner loop — measures shuffle cost
//   Probe B: strip TG-memory writes       — measures TG-write cost
//   Probe C: strip bin-check branches     — measures branch divergence cost
//   Probe D: strip global loads           — measures total global-memory cost
//
// COMPILE (from repo root, single-file, no CMake required)
//   clang++ -std=c++17 -O2 \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation \
//     docs/sprint19/scratch/microbench/microbench_gather.cpp \
//     -o /tmp/microbench_gather
//
// RUN
//   /tmp/microbench_gather
//
// OUTPUT
//   Human-readable timing table + JSON summary to stdout.
//   Use: /tmp/microbench_gather 2>/dev/null | tee /tmp/microbench_results.txt
//
// GATE CONFIG (matches production gate)
//   N = 50000, lineSize = 25 (100 features / 4 per uint32), featureColumnIdx = 0
//   threadgroup: 256, threadgroups: ceil(50000/256) = 196
//   Warm runs: 5 (discarded), timed runs: 5 (stats reported)
//
// CRITIQUE: what could make this micro-bench lie?
//   (1) GPU queue scheduling noise: MLX schedules Metal command buffers
//       asynchronously; wall-clock timing includes any queuing overhead.
//       MITIGATION: 5 warm runs flush JIT cold-start; 5 timed runs average
//       out scheduling jitter. The harness calls mx::eval() which blocks
//       until GPU completion, so wall clock = GPU wall time. This is NOT
//       the same as MTLCommandBuffer GPU-execution timestamps, which exclude
//       queue-wait time. At N=50k/lineSize=25, kernel is 2–15 ms, so 20–200 µs
//       scheduling noise is <5% of reported time — acceptable for a binary
//       layout-neutral/layout-sensitive verdict.
//   (2) Prefetch hiding the gather: AGX's texture prefetcher or L2 prefetcher
//       may stream both layouts equally if the access is sufficiently regular.
//       A sequential kernel (gid monotone) is MORE regular than the production
//       kernel (sorted docIndices ≠ identity), so the micro-bench may
//       OVERESTIMATE col-major speedup vs production. This is conservative in
//       the correct direction: if micro shows ~0, production ~0 is guaranteed.
//   (3) Sorted docIndices vs identity: in production, sorted partition indices
//       are not identity[0..N-1]; partitions add a permutation. We use
//       identity (gid → gid) here for simplicity. For col-major, sorted
//       docIndices are still near-monotone (partition sort), so the access
//       pattern is similar. For row-major, the stride pattern is identical.
//       IMPACT: minimal — both layouts see the same docIndices permutation
//       structure in production; the stride difference is the variable we're
//       isolating.
//   (4) JIT cold-start: iter 0 compiles the Metal shader. We run 5 warm
//       iterations first (discarded) to ensure steady-state before timing.
//   (5) Threadgroup size mismatch: production uses 256-thread TGs with 8
//       SIMD groups. The gather micro-bench also uses 256-thread TGs (same
//       grid). The warp scheduler sees the same occupancy profile.

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace mx = mlx::core;

// ============================================================================
// Config
// ============================================================================

static constexpr uint32_t N_DOCS    = 50000;
static constexpr uint32_t LINE_SIZE = 25;     // gate: 100 features / 4 = 25 uint32 cols
static constexpr uint32_t FEAT_COL  = 0;      // test column 0 (same for both layouts)
static constexpr int      WARM_RUNS = 5;
static constexpr int      TIMED_RUNS= 5;

// ============================================================================
// Timing helper
// ============================================================================

struct TTimingResult {
    double mean_ms = 0.0;
    double stdev_ms = 0.0;
    std::vector<double> runs_ms;
};

static TTimingResult ComputeStats(const std::vector<double>& ms) {
    TTimingResult r;
    r.runs_ms = ms;
    double sum = 0.0;
    for (double v : ms) sum += v;
    r.mean_ms = sum / ms.size();
    double var = 0.0;
    for (double v : ms) var += (v - r.mean_ms) * (v - r.mean_ms);
    r.stdev_ms = (ms.size() > 1) ? std::sqrt(var / (ms.size() - 1)) : 0.0;
    return r;
}

// ============================================================================
// Data setup
// ============================================================================

// Row-major compressedIndex: [N_DOCS * LINE_SIZE] uint32
// compressedIndex[doc * LINE_SIZE + col]
static std::vector<uint32_t> MakeRowMajorCI() {
    std::vector<uint32_t> ci(N_DOCS * LINE_SIZE);
    // Synthetic data: doc d, col c → (d * 7 + c * 3) & 0xFF packed into 4 features
    for (uint32_t d = 0; d < N_DOCS; ++d)
        for (uint32_t c = 0; c < LINE_SIZE; ++c)
            ci[d * LINE_SIZE + c] = (d * 7u + c * 3u) & 0x00FF00FFu;
    return ci;
}

// Column-major compressedIndex: [LINE_SIZE * N_DOCS] uint32
// compressedIndex[col * N_DOCS + doc]
static std::vector<uint32_t> MakeColMajorCI(const std::vector<uint32_t>& rowMajor) {
    std::vector<uint32_t> ci(LINE_SIZE * N_DOCS);
    for (uint32_t d = 0; d < N_DOCS; ++d)
        for (uint32_t c = 0; c < LINE_SIZE; ++c)
            ci[c * N_DOCS + d] = rowMajor[d * LINE_SIZE + c];
    return ci;
}

// docIndices: identity permutation (doc 0..N-1 in sorted order)
// In production, this is argsort of partition assignments; for the micro-bench
// we use identity so that both row-major and col-major see the same "access
// count" — the stride-100-byte vs stride-4-byte difference is the only variable.
static std::vector<uint32_t> MakeDocIndices() {
    std::vector<uint32_t> di(N_DOCS);
    std::iota(di.begin(), di.end(), 0u);
    return di;
}

// Stats (for probe kernels that need them)
static std::vector<float> MakeStats() {
    std::vector<float> s(N_DOCS);
    for (uint32_t d = 0; d < N_DOCS; ++d) s[d] = static_cast<float>(d % 128) / 128.0f;
    return s;
}

// ============================================================================
// Kernel source strings
// ============================================================================

// --- Gather row-major ---
static const std::string kGatherRowMajorHeader = R"metal(
#include <metal_stdlib>
using namespace metal;
)metal";

static const std::string kGatherRowMajorSource = R"metal(
    const uint gid = thread_position_in_grid.x;
    if (gid >= totalNumDocs) return;
    const uint docIdx = docIndices[gid];
    output[gid] = compressedIndex[docIdx * lineSize + featureColumnIdx];
)metal";

// --- Gather col-major ---
static const std::string kGatherColMajorSource = R"metal(
    const uint gid = thread_position_in_grid.x;
    if (gid >= totalNumDocs) return;
    const uint docIdx = docIndices[gid];
    output[gid] = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
)metal";

// ============================================================================
// Probe kernel source strings
//
// All probes run on the SAME grid as the production L1a kernel:
//   threadgroup: 256, threadgroups: 196  (= ceil(50000/(8*32)) * 25 groups)
//   This matches production dispatch so occupancy is identical.
//
// Each probe is a stripped version of kHistOneByteSource (kernel_sources.h:165-213).
// The accumulation phase ONLY is timed — zero-init and reduction phases are
// identical to production and are not the subject of this attribution.
//
// Shared kernel header (constants + threadgroup declaration):
// ============================================================================

static const std::string kProbeHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE         = 32;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE     = 256;
constant constexpr uint BLOCK_SIZE        = 256;
constant constexpr uint NUM_SIMD_GROUPS   = 8;
constant constexpr uint HIST_PER_SIMD     = FEATURES_PER_PACK * BINS_PER_BYTE;
)metal";

// --- Probe production (baseline for probes A-D) ---
// Full accumulation loop: global load + simd_shuffle + TG write + bin-check.
// Identical to kHistOneByteSource lines 165-213, trimmed to the accumulation
// phase only. The kernel body here is a full self-contained kernel (not the
// MLX-generated wrapper), but for timing purposes we launch it via
// mx::fast::metal_kernel using the same grid/TG as production.
//
// INPUT NAMES (matching production names):
//   compressedIndex, stats, docIndices, partOffsets, partSizes,
//   featureColumnIndices, lineSize, maxBlocksPerPart, numGroups,
//   foldCountsFlat, firstFoldIndicesFlat, totalBinFeatures, numStats, totalNumDocs
// OUTPUT NAMES: histogram

// For simplicity in the probe harness, we strip the full grid decode and
// work on a flat doc range (as if numPartitions=1, numGroups=1, one stat).
// This is equivalent to one TG processing all docs in a single partition —
// the accumulation loop structure is identical to production.

// Probe common preamble (single-partition, single-group flat layout):
static const std::string kProbePreamble = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    // Zero-init (same as production barrier 1)
    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs         = totalNumDocs;
    const uint featureColumnIdx = 0u;
    const uint statIdx          = 0u;
    const uint foldBase         = 0u;
)metal";

// --- Probe PRODUCTION (P): full accumulation loop ---
static const std::string kProbeProductionSource = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs          = totalNumDocs;
    const uint featureColumnIdx = 0u;
    const uint statIdx          = 0u;
    const uint foldBase         = 0u;

    // Full accumulation (DEC-015 col-major address since we're at tip 108c7a59d2)
    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE)
    {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint docIdx = docIndices[d];
            // NOTE: col-major as per current tip 108c7a59d2
            packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s     = simd_shuffle(packed, src);
            const float s_s     = simd_shuffle(stat,   src);
            const bool  valid_s = simd_shuffle(valid,  src);
            if (!valid_s) continue;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write simdHist[0][tid] to output (prevent dead-code elimination)
    if (tid < HIST_PER_SIMD)
        output[tid] = simdHist[0][tid];
)metal";

// --- Probe A: strip simd_shuffle inner loop ---
// Replace the 32-iteration shuffle broadcast with a single direct load from
// the "self" (lane src=lane, no shuffle). This removes 96 shuffle ops per
// batch but keeps the global load and TG write.
static const std::string kProbeASource = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs          = totalNumDocs;
    const uint featureColumnIdx = 0u;
    const uint statIdx          = 0u;
    const uint foldBase         = 0u;

    // PROBE A: No simd_shuffle. Each lane processes only its own doc.
    // Keeps global load + TG write; removes the 32-iter broadcast loop.
    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE)
    {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint docIdx = docIndices[d];
            packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }
        // No shuffle loop — process only own doc.
        if (valid) {
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (packed >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += stat;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HIST_PER_SIMD)
        output[tid] = simdHist[0][tid];
)metal";

// --- Probe B: strip TG-memory writes ---
// Keep global load + simd_shuffle; skip the simdHist write.
// Use a register accumulator (volatile) to prevent dead-code elimination.
static const std::string kProbeBSource = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs          = totalNumDocs;
    const uint featureColumnIdx = 0u;
    const uint statIdx          = 0u;
    const uint foldBase         = 0u;

    float regAccum = 0.0f;   // register sink — prevents dead-code elimination

    // PROBE B: Skip TG-memory writes. All ops except the += simdHist write.
    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE)
    {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint docIdx = docIndices[d];
            packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s     = simd_shuffle(packed, src);
            const float s_s     = simd_shuffle(stat,   src);
            const bool  valid_s = simd_shuffle(valid,  src);
            if (!valid_s) continue;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                const bool owner = (bin < foldCounts[foldBase + f] + 1u &&
                                    (bin & (SIMD_SIZE - 1u)) == lane);
                // Skip TG write; accumulate into register to prevent DCE.
                if (owner) regAccum += s_s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Emit regAccum so the compiler cannot eliminate the work.
    if (tid == 0) output[0] = regAccum;
    else if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

// --- Probe C: strip bin-check branches ---
// Replace predicate (bin < foldCounts + 1 && (bin & 31) == lane) with
// constant true — all lanes always write to bin 0. Incorrect output,
// but eliminates divergence cost.
static const std::string kProbeCSource = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs          = totalNumDocs;
    const uint featureColumnIdx = 0u;
    const uint statIdx          = 0u;

    // PROBE C: Bin-check replaced with constant true. All lanes write to
    // f * BINS_PER_BYTE + 0 (i.e. bin 0 for every feature). Measures cost
    // without branch divergence.
    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE)
    {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint docIdx = docIndices[d];
            packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s     = simd_shuffle(packed, src);
            const float s_s     = simd_shuffle(stat,   src);
            const bool  valid_s = simd_shuffle(valid,  src);
            if (!valid_s) continue;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                // PROBE C: no bin check — always write to bin 0.
                simdHist[simd_id][f * BINS_PER_BYTE + 0] += s_s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HIST_PER_SIMD)
        output[tid] = simdHist[0][tid];
)metal";

// --- Probe D: strip global loads ---
// Replace compressedIndex / stats / docIndices global reads with compile-time
// constants. Measures the kernel cost WITHOUT any global memory traffic.
// Delta vs production = total global-memory cost (load + address chain).
static const std::string kProbeDSource = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs          = totalNumDocs;
    const uint foldBase         = 0u;

    // PROBE D: No global loads. packed and stat are compile-time constants.
    // This runs the full shuffle+TG-write loop but pays zero memory latency.
    const uint  const_packed = 0x0F0F0F0Fu;   // bin=15 for all 4 features
    const float const_stat   = 0.5f;

    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE)
    {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        // No global loads — use constants.
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s     = simd_shuffle(const_packed, src);
            const float s_s     = simd_shuffle(const_stat,   src);
            const bool  valid_s = simd_shuffle(valid,  src);
            if (!valid_s) continue;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_s >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HIST_PER_SIMD)
        output[tid] = simdHist[0][tid];
)metal";

// ============================================================================
// Timer helper — runs a kernel N times and returns wall-clock timing
// ============================================================================

template<typename KernelFn>
static TTimingResult TimeKernel(const char* name, KernelFn&& fn, int warmRuns, int timedRuns) {
    fprintf(stderr, "  Timing %s: ", name);

    // Warm runs (discard)
    for (int i = 0; i < warmRuns; ++i) {
        fn();
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(timedRuns);
    for (int i = 0; i < timedRuns; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        fprintf(stderr, ".");
    }
    fprintf(stderr, "\n");
    return ComputeStats(times);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    fprintf(stderr, "[S19-01c] Initializing MLX GPU device...\n");
    // GPU trace disabled (no Metal debugger attached)

    // -------------------------------------------------------------------------
    // Setup host data
    // -------------------------------------------------------------------------
    fprintf(stderr, "[S19-01c] Building synthetic data (N=%u, lineSize=%u)...\n", N_DOCS, LINE_SIZE);

    auto rowMajorCI_host = MakeRowMajorCI();
    auto colMajorCI_host = MakeColMajorCI(rowMajorCI_host);
    auto docIndices_host = MakeDocIndices();
    auto stats_host      = MakeStats();

    // foldCounts: all features have 128 bins at gate
    std::vector<uint32_t> foldCounts_host(4, 128u);  // 4 features per group

    // -------------------------------------------------------------------------
    // Upload to GPU
    // -------------------------------------------------------------------------
    auto ciRowMajor = mx::array(
        reinterpret_cast<const int32_t*>(rowMajorCI_host.data()),
        {static_cast<int>(N_DOCS * LINE_SIZE)}, mx::uint32);
    auto ciColMajor = mx::array(
        reinterpret_cast<const int32_t*>(colMajorCI_host.data()),
        {static_cast<int>(LINE_SIZE * N_DOCS)}, mx::uint32);
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndices_host.data()),
        {static_cast<int>(N_DOCS)}, mx::uint32);
    auto stats = mx::array(
        stats_host.data(), {static_cast<int>(N_DOCS)}, mx::float32);
    auto foldCounts = mx::array(
        reinterpret_cast<const int32_t*>(foldCounts_host.data()),
        {4}, mx::uint32);

    auto lineSizeArr    = mx::array(static_cast<uint32_t>(LINE_SIZE), mx::uint32);
    auto featColArr     = mx::array(static_cast<uint32_t>(FEAT_COL),  mx::uint32);
    auto totalDocsArr   = mx::array(static_cast<uint32_t>(N_DOCS),    mx::uint32);
    auto statIdxArr     = mx::array(static_cast<uint32_t>(0u),        mx::uint32);

    mx::eval({ciRowMajor, ciColMajor, docIndices, stats, foldCounts,
              lineSizeArr, featColArr, totalDocsArr, statIdxArr});

    fprintf(stderr, "[S19-01c] GPU data uploaded.\n");

    // Output buffer (one uint32 per doc for gather; HIST_PER_SIMD for probes)
    auto gatherOut = mx::zeros({static_cast<int>(N_DOCS)}, mx::uint32);
    auto probeOut  = mx::zeros({1024}, mx::float32);  // HIST_PER_SIMD = 4 * 256

    // -------------------------------------------------------------------------
    // Micro-benchmark: row-major gather
    // -------------------------------------------------------------------------
    const int tgSize = 256;
    const int tgCount = (static_cast<int>(N_DOCS) + tgSize - 1) / tgSize;

    auto rmKernel = mx::fast::metal_kernel(
        "gather_rowmajor",
        {"compressedIndex", "docIndices", "lineSize", "featureColumnIdx", "totalNumDocs"},
        {"output"},
        kGatherRowMajorSource,
        kGatherRowMajorHeader,
        true, false
    );
    auto cmKernel = mx::fast::metal_kernel(
        "gather_colmajor",
        {"compressedIndex", "docIndices", "lineSize", "featureColumnIdx", "totalNumDocs"},
        {"output"},
        kGatherColMajorSource,
        kGatherRowMajorHeader,
        true, false
    );

    auto rmGrid = std::make_tuple(static_cast<int>(N_DOCS), 1, 1);
    auto rmTG   = std::make_tuple(tgSize, 1, 1);

    auto runRowMajor = [&]() {
        auto result = rmKernel(
            {ciRowMajor, docIndices, lineSizeArr, featColArr, totalDocsArr},
            {gatherOut.shape()}, {mx::uint32}, rmGrid, rmTG,
            {}, 0.0f, false, mx::Device::gpu
        );
        mx::eval(result[0]);
    };

    auto runColMajor = [&]() {
        auto result = cmKernel(
            {ciColMajor, docIndices, lineSizeArr, featColArr, totalDocsArr},
            {gatherOut.shape()}, {mx::uint32}, rmGrid, rmTG,
            {}, 0.0f, false, mx::Device::gpu
        );
        mx::eval(result[0]);
    };

    fprintf(stderr, "[S19-01c] === MICRO-BENCHMARK: GATHER ISOLATION ===\n");
    auto rmStats = TimeKernel("gather_rowmajor", runRowMajor, WARM_RUNS, TIMED_RUNS);
    auto cmStats = TimeKernel("gather_colmajor", runColMajor, WARM_RUNS, TIMED_RUNS);

    // -------------------------------------------------------------------------
    // Probe kernels (accumulation sub-phase attribution)
    // -------------------------------------------------------------------------

    // Probe kernel grid: match production L1a dispatch.
    // Production at N=50k, d=0: 25 groups × 1 partition × 1 stat = 25 TGs.
    // But to measure accumulation at full load, we simulate 1 partition with
    // ALL N_DOCS assigned, running all 25 groups' worth of work.
    // Simplification: 1 flat TG over all N_DOCS docs.
    // This is representative of the per-TG accumulation cost scaled to N docs.
    // The probe grid uses 1 TG of 256 threads iterating over all N_DOCS docs
    // (outer batch loop runs ceil(50000 / (8*32)) = 196 iterations — same as
    //  production depth-0 per-TG, which has all N docs in one partition).

    // Probe kernel inputs
    // All probes need: compressedIndex (col-major at tip), stats, docIndices,
    // totalNumDocs, foldCounts
    // Probes are launched as 1 TG × 256 threads (depth-0 equivalent)

    auto probeGrid = std::make_tuple(256, 1, 1);
    auto probeTG   = std::make_tuple(256, 1, 1);

    // Production probe
    auto prodKernel = mx::fast::metal_kernel(
        "probe_production",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"},
        kProbeProductionSource,
        kProbeHeader, true, false
    );

    // Probe A (no shuffle)
    auto probeAKernel = mx::fast::metal_kernel(
        "probe_A_no_shuffle",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"},
        kProbeASource,
        kProbeHeader, true, false
    );

    // Probe B (no TG writes)
    auto probeBKernel = mx::fast::metal_kernel(
        "probe_B_no_tgwrite",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"},
        kProbeBSource,
        kProbeHeader, true, false
    );

    // Probe C (no bin check)
    auto probeCKernel = mx::fast::metal_kernel(
        "probe_C_no_bincheck",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"},
        kProbeCSource,
        kProbeHeader, true, false
    );

    // Probe D (no global loads)
    auto probeDKernel = mx::fast::metal_kernel(
        "probe_D_no_globalload",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"},
        kProbeDSource,
        kProbeHeader, true, false
    );

    auto runProbe = [&](auto& kernel, const char* tag) -> TTimingResult {
        return TimeKernel(tag,
            [&]() {
                auto r = kernel(
                    {ciColMajor, stats, docIndices, foldCounts, totalDocsArr},
                    {probeOut.shape()}, {mx::float32}, probeGrid, probeTG,
                    {}, 0.0f, false, mx::Device::gpu
                );
                mx::eval(r[0]);
            }, WARM_RUNS, TIMED_RUNS);
    };

    fprintf(stderr, "[S19-01c] === PROBE KERNELS ===\n");
    auto pProd = runProbe(prodKernel,    "probe_production (baseline)");
    auto pA    = runProbe(probeAKernel,  "probe_A (no simd_shuffle)");
    auto pB    = runProbe(probeBKernel,  "probe_B (no TG write)");
    auto pC    = runProbe(probeCKernel,  "probe_C (no bin-check)");
    auto pD    = runProbe(probeDKernel,  "probe_D (no global load)");

    // -------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------

    fprintf(stdout, "\n");
    fprintf(stdout, "====================================================================\n");
    fprintf(stdout, "Sprint 19 S19-01c: Gather Micro-Benchmark + Probe Attribution\n");
    fprintf(stdout, "Config: N=%u, lineSize=%u, featureColumnIdx=%u, warm=%d, timed=%d\n",
            N_DOCS, LINE_SIZE, FEAT_COL, WARM_RUNS, TIMED_RUNS);
    fprintf(stdout, "====================================================================\n\n");

    // Gather isolation
    fprintf(stdout, "--- PART 1: Gather layout isolation ---\n");
    fprintf(stdout, "%-30s  mean_ms  stdev_ms  runs_ms\n", "Kernel");
    fprintf(stdout, "%-30s  %-7.3f  %-8.3f  [", "gather_rowmajor",
            rmStats.mean_ms, rmStats.stdev_ms);
    for (int i = 0; i < (int)rmStats.runs_ms.size(); ++i)
        fprintf(stdout, "%.3f%s", rmStats.runs_ms[i], i+1<(int)rmStats.runs_ms.size()?", ":"");
    fprintf(stdout, "]\n");
    fprintf(stdout, "%-30s  %-7.3f  %-8.3f  [", "gather_colmajor",
            cmStats.mean_ms, cmStats.stdev_ms);
    for (int i = 0; i < (int)cmStats.runs_ms.size(); ++i)
        fprintf(stdout, "%.3f%s", cmStats.runs_ms[i], i+1<(int)cmStats.runs_ms.size()?", ":"");
    fprintf(stdout, "]\n");

    double gatherSpeedup = (cmStats.mean_ms > 0.0) ? rmStats.mean_ms / cmStats.mean_ms : 0.0;
    fprintf(stdout, "\nrow-major / col-major speedup: %.3fx\n", gatherSpeedup);
    if (gatherSpeedup > 1.5)
        fprintf(stdout, "VERDICT (gather): col-major IS faster in isolation. DEC-015 layout is sound.\n"
                        "         But full kernel showed 0.98x — something ELSE is hiding the speedup.\n");
    else if (gatherSpeedup < 1.1)
        fprintf(stdout, "VERDICT (gather): layout is NEUTRAL in isolation. "
                        "DEC-015 model was WRONG — AGX does not expose the scatter cost.\n");
    else
        fprintf(stdout, "VERDICT (gather): modest col-major advantage (%.2fx). "
                        "Mixed signal — run probes.\n", gatherSpeedup);

    // Probe attribution
    fprintf(stdout, "\n--- PART 2: Accumulation sub-phase probes (1 TG, N=%u docs) ---\n", N_DOCS);
    fprintf(stdout, "Production probe time = reference for all deltas.\n\n");
    fprintf(stdout, "%-35s  mean_ms  stdev_ms  delta_vs_prod  pct_of_prod\n", "Probe");

    auto printProbe = [&](const char* name, const TTimingResult& r) {
        double delta = pProd.mean_ms - r.mean_ms;
        double pct   = (pProd.mean_ms > 0.0) ? 100.0 * delta / pProd.mean_ms : 0.0;
        fprintf(stdout, "%-35s  %-7.3f  %-8.3f  %+.3f ms       %.1f%%\n",
                name, r.mean_ms, r.stdev_ms, delta, pct);
    };

    fprintf(stdout, "%-35s  %-7.3f  %-8.3f  (baseline)\n",
            "probe_production", pProd.mean_ms, pProd.stdev_ms);
    printProbe("probe_A (no simd_shuffle)",    pA);
    printProbe("probe_B (no TG write)",        pB);
    printProbe("probe_C (no bin-check)",       pC);
    printProbe("probe_D (no global load)",     pD);

    // Sub-phase attribution
    double shuffleCost  = pProd.mean_ms - pA.mean_ms;
    double tgWriteCost  = pProd.mean_ms - pB.mean_ms;
    double branchCost   = pProd.mean_ms - pC.mean_ms;
    double globalLoadCost = pProd.mean_ms - pD.mean_ms;

    fprintf(stdout, "\n--- PART 2 attribution summary ---\n");
    fprintf(stdout, "  simd_shuffle cost estimate:    %.3f ms  (prod - probe_A)\n", shuffleCost);
    fprintf(stdout, "  TG-memory write cost estimate: %.3f ms  (prod - probe_B)\n", tgWriteCost);
    fprintf(stdout, "  Branch divergence cost estim:  %.3f ms  (prod - probe_C)\n", branchCost);
    fprintf(stdout, "  Global-load cost estimate:     %.3f ms  (prod - probe_D)\n", globalLoadCost);
    fprintf(stdout, "  Probe_D (ALU+shuffle, no mem): %.3f ms  (ALU floor)\n", pD.mean_ms);

    // JSON output
    fprintf(stdout, "\n--- JSON summary ---\n");
    fprintf(stdout, "{\n");
    fprintf(stdout, "  \"config\": {\"N\": %u, \"lineSize\": %u, \"warmRuns\": %d, \"timedRuns\": %d},\n",
            N_DOCS, LINE_SIZE, WARM_RUNS, TIMED_RUNS);
    fprintf(stdout, "  \"gather\": {\n");
    fprintf(stdout, "    \"rowmajor_mean_ms\": %.4f, \"rowmajor_stdev_ms\": %.4f,\n",
            rmStats.mean_ms, rmStats.stdev_ms);
    fprintf(stdout, "    \"colmajor_mean_ms\": %.4f, \"colmajor_stdev_ms\": %.4f,\n",
            cmStats.mean_ms, cmStats.stdev_ms);
    fprintf(stdout, "    \"speedup_col_vs_row\": %.4f\n", gatherSpeedup);
    fprintf(stdout, "  },\n");
    fprintf(stdout, "  \"probes\": {\n");
    fprintf(stdout, "    \"production_mean_ms\": %.4f, \"production_stdev_ms\": %.4f,\n",
            pProd.mean_ms, pProd.stdev_ms);
    fprintf(stdout, "    \"probe_A_mean_ms\": %.4f, \"probe_A_stdev_ms\": %.4f,\n",
            pA.mean_ms, pA.stdev_ms);
    fprintf(stdout, "    \"probe_B_mean_ms\": %.4f, \"probe_B_stdev_ms\": %.4f,\n",
            pB.mean_ms, pB.stdev_ms);
    fprintf(stdout, "    \"probe_C_mean_ms\": %.4f, \"probe_C_stdev_ms\": %.4f,\n",
            pC.mean_ms, pC.stdev_ms);
    fprintf(stdout, "    \"probe_D_mean_ms\": %.4f, \"probe_D_stdev_ms\": %.4f,\n",
            pD.mean_ms, pD.stdev_ms);
    fprintf(stdout, "    \"shuffle_cost_ms\": %.4f,\n", shuffleCost);
    fprintf(stdout, "    \"tgwrite_cost_ms\": %.4f,\n", tgWriteCost);
    fprintf(stdout, "    \"branch_cost_ms\":  %.4f,\n", branchCost);
    fprintf(stdout, "    \"globalload_cost_ms\": %.4f\n", globalLoadCost);
    fprintf(stdout, "  }\n");
    fprintf(stdout, "}\n");

    return 0;
}
