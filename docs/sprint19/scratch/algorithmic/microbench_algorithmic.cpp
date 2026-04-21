// microbench_algorithmic.cpp — Sprint 19 algorithmic ablation harness
//
// PURPOSE
//   Direct measurement of algorithmic alternatives to the L1a histogram
//   accumulator's 32-iter simd_shuffle inner loop. Methodology matches
//   docs/sprint19/scratch/microbench/microbench_gather.cpp exactly:
//   1 TG × 256 threads processing all N=50k docs in a single partition,
//   timed via wall-clock + mx::eval() blocking.
//
//   The production probe at branch tip (DEC-015 col-major address) measures
//   2.357 ms ± 0.035 ms (docs/sprint19/reattribution.md §3). All variants
//   here are compared against that number using the SAME grid (1 TG × 256
//   threads) and the SAME data (identity docIndices, synthetic packed data).
//
//   Variants measured:
//     T0: production baseline (reproduction of probe_production)
//     T1: fuse-valid — pack valid bit into MSB of packed, 2 shuffles/src
//     T2: bin-major accumulation (sort-by-bin pre-built buckets, accum only)
//     T3: no-shuffle per-lane — each lane owns its own doc, no broadcast,
//         bin-check evaluated per lane; structurally the same work but
//         shuffle chain eliminated (correctness: each doc written once by
//         its owner thread if any; this is an INCORRECT accumulator for
//         comparison purposes — it measures the lower bound of a
//         shuffle-free formulation)
//     T3b: no-shuffle + atomic writes to common simdHist[0][bin]
//         (correct accumulator — each doc writes to the single shared
//         simdHist[0][bin] via atomic_fetch_add; eliminates shuffle AND
//         the per-SIMD-group partial layout; measures the cost of
//         threadgroup-scope atomics as the shuffle replacement)
//
// COMPILE
//   clang++ -std=c++17 -O2 \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation \
//     docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp \
//     -o /tmp/microbench_algorithmic && /tmp/microbench_algorithmic
//
// OUTPUT
//   Human-readable timing table + JSON summary to stdout.

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
// Config — matches docs/sprint19/scratch/microbench/microbench_gather.cpp
// ============================================================================

static constexpr uint32_t N_DOCS     = 50000;
static constexpr uint32_t LINE_SIZE  = 25;
static constexpr uint32_t FEAT_COL   = 0;
static constexpr int      WARM_RUNS  = 5;
static constexpr int      TIMED_RUNS = 5;
static constexpr uint32_t NUM_BINS   = 128;  // gate config

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

template<typename KernelFn>
static TTimingResult TimeKernel(const char* name, KernelFn&& fn,
                                int warmRuns, int timedRuns) {
    fprintf(stderr, "  Timing %s: ", name);
    for (int i = 0; i < warmRuns; ++i) { fn(); }
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
// Data setup
// ============================================================================

// Col-major compressedIndex: [LINE_SIZE * N_DOCS] uint32, matching tip
// 108c7a59d2 layout (DEC-015 col-major). Bin values: (doc * 7) & 0x7F
// packed into 4 features per uint32 (bins 0..127).
static std::vector<uint32_t> MakeColMajorCI() {
    std::vector<uint32_t> ci(LINE_SIZE * N_DOCS, 0u);
    for (uint32_t d = 0; d < N_DOCS; ++d) {
        // Feature 0: bin0, Feature 1: bin1, ... packed high byte first
        const uint32_t b0 = (d * 7u)  & 0x7Fu;
        const uint32_t b1 = (d * 11u) & 0x7Fu;
        const uint32_t b2 = (d * 13u) & 0x7Fu;
        const uint32_t b3 = (d * 17u) & 0x7Fu;
        const uint32_t packed =
            (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
        ci[0 * N_DOCS + d] = packed;
    }
    return ci;
}

static std::vector<uint32_t> MakeDocIndices() {
    std::vector<uint32_t> di(N_DOCS);
    std::iota(di.begin(), di.end(), 0u);
    return di;
}

static std::vector<float> MakeStats() {
    std::vector<float> s(N_DOCS);
    for (uint32_t d = 0; d < N_DOCS; ++d)
        s[d] = static_cast<float>(d % 128) / 128.0f;
    return s;
}

// ============================================================================
// Pre-pass data for T2 (bin-major): per-feature bucket index.
// For each feature f, bucketOffsets[f * (NUM_BINS+1) + b] = first doc index
// in bucket b of feature f. bucketDocs[f * N + i] = doc index (0..N-1) for
// the i-th doc in feature f's bucket list (sorted by bin).
//
// This pre-pass cost is NOT counted in the accumulation benchmark — we are
// measuring the lower bound (best case) for sort-by-bin. The REAL cost of
// the pre-pass is discussed in the research note.
// ============================================================================

struct TBinMajorLayout {
    std::vector<uint32_t> bucketOffsets;  // [FEATURES_PER_PACK * (NUM_BINS+1)]
    std::vector<uint32_t> bucketDocs;     // [FEATURES_PER_PACK * N]
};

static TBinMajorLayout MakeBinMajor(const std::vector<uint32_t>& ciColMajor) {
    TBinMajorLayout lo;
    lo.bucketOffsets.assign(4u * (NUM_BINS + 1u), 0u);
    lo.bucketDocs.assign(4u * N_DOCS, 0u);

    for (uint32_t f = 0; f < 4; ++f) {
        // Histogram of bins for feature f
        std::vector<uint32_t> counts(NUM_BINS, 0u);
        for (uint32_t d = 0; d < N_DOCS; ++d) {
            const uint32_t packed = ciColMajor[0 * N_DOCS + d];
            const uint32_t bin = (packed >> (24u - 8u * f)) & 0xFFu;
            if (bin < NUM_BINS) counts[bin]++;
        }
        // Prefix sum → bucketOffsets
        uint32_t acc = 0;
        for (uint32_t b = 0; b < NUM_BINS; ++b) {
            lo.bucketOffsets[f * (NUM_BINS + 1) + b] = acc;
            acc += counts[b];
        }
        lo.bucketOffsets[f * (NUM_BINS + 1) + NUM_BINS] = acc;
        // Fill buckets
        std::vector<uint32_t> cursor(NUM_BINS, 0u);
        for (uint32_t d = 0; d < N_DOCS; ++d) {
            const uint32_t packed = ciColMajor[0 * N_DOCS + d];
            const uint32_t bin = (packed >> (24u - 8u * f)) & 0xFFu;
            if (bin < NUM_BINS) {
                const uint32_t off = lo.bucketOffsets[f * (NUM_BINS + 1) + bin]
                                   + cursor[bin]++;
                lo.bucketDocs[f * N_DOCS + off] = d;
            }
        }
    }
    return lo;
}

// ============================================================================
// Kernel bodies — all match the probe style of microbench_gather.cpp
// ============================================================================

static const std::string kHeader = R"metal(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE         = 32;
constant constexpr uint FEATURES_PER_PACK = 4;
constant constexpr uint BINS_PER_BYTE     = 256;
constant constexpr uint BLOCK_SIZE        = 256;
constant constexpr uint NUM_SIMD_GROUPS   = 8;
constant constexpr uint HIST_PER_SIMD     = FEATURES_PER_PACK * BINS_PER_BYTE;
)metal";

// ---------------------------------------------------------------------------
// T0: production baseline (reproduction of probe_production)
// Bit-identical to reattribution.md probe_production. Reference for delta.
// ---------------------------------------------------------------------------
static const std::string kT0Source = R"metal(
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

    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE) {
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
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

// ---------------------------------------------------------------------------
// T1: fuse-valid — pack the valid bit into MSB of packed.
// Invalid lanes have packed=0; the bin-check `bin < foldCounts[f]+1` still
// fires for bin=0, but since valid lanes write bin with correct values and
// we dedicate the MSB to the valid flag (never used by 128-bin features),
// we can test `p_s >> 31` instead of `simd_shuffle(valid, src)`.
//
// Shuffle count per src: 2 (packed, stat) vs 3 for T0. One shuffle removed.
// Expected saving: ~1/3 of shuffle cost per reattribution.md §5.3.
// ---------------------------------------------------------------------------
static const std::string kT1Source = R"metal(
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
    const uint VALID_BIT        = 0x80000000u;

    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE) {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        uint  packed = 0u;
        float stat   = 0.0f;
        if (valid) {
            const uint docIdx = docIndices[d];
            packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
            // 128-bin features use bits 6..0 of each byte — the top nibble of
            // each 8-bit slot has room for the valid sentinel in the MSB of
            // the full uint32 (bit 31) without colliding with any bin byte.
            // At gate config, feature 0 occupies bits 24..30, so bit 31 is free.
            packed |= VALID_BIT;
            stat   = stats[statIdx * totalNumDocs + docIdx];
        }
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s = simd_shuffle(packed, src);
            const float s_s = simd_shuffle(stat,   src);
            if ((p_s & VALID_BIT) == 0u) continue;  // fused valid check
            // Mask off the valid bit before extracting feature-0 bin.
            const uint p_clean = p_s & 0x7FFFFFFFu;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_clean >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

// ---------------------------------------------------------------------------
// T2: bin-major accumulation (sort-by-bin, pre-built buckets).
//
// Assumes a pre-pass has built per-feature doc buckets:
//   bucketOffsets[f*(NUM_BINS+1) + b] = first doc index in bucket b
//   bucketDocs[f*N + i]               = doc index at position i
//
// Accumulation: each SIMD group owns a slice of bins, lane l of group g
// owns bins {g*32 + l, g*32 + l + 256, ...} (stride NUM_SIMD_GROUPS*SIMD_SIZE
// across 256 bins). Wait — better: assign bin b to (thread_id = b & 255). Then
// thread t loops over all 4 features, reads its bin-t bucket for each, and
// sums stats directly. No shuffle, no cross-lane communication.
//
// Each thread's work: for each feature f in 0..3, iterate from
//   bucketOffsets[f*(NUM_BINS+1) + t] to bucketOffsets[f*(NUM_BINS+1) + t + 1]
// and sum stats[bucketDocs[f*N + i]]. Write result to simdHist[0][f*256 + t].
//
// CRITICAL: We bypass simdHist[8][1024] — only simdHist[0][1024] is used
// (single-SIMD-group-equivalent shared layout). No cross-SIMD fold needed.
//
// This is the BEST CASE: we assume the pre-pass is free. In reality the pre-
// pass is a non-trivial bin-count + prefix-scan + scatter over all N docs for
// each of 25 feature groups × 4 features × every partition at every depth.
// ---------------------------------------------------------------------------
static const std::string kT2Source = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs  = totalNumDocs;
    const uint numBins  = 128u;
    const uint binStride = 129u;  // NUM_BINS + 1

    // Each of 256 threads owns 1 bin per feature (128 bins, 2 sweeps per feature
    // with tid mod 128 ownership; for simplicity here, threads 0..127 each
    // handle 1 bin, threads 128..255 handle bins 0..127 for features 2..3).
    // CLEANER: tid in [0, 128) handles bin tid for features 0,1; tid in [128,256)
    // handles bin tid-128 for features 2,3.
    //
    // Even cleaner (used here): each thread handles 2 features at 1 bin.
    //   bin_t    = tid & 127
    //   feat_lo  = (tid >> 7) * 2           // 0 for tid<128, 2 for tid>=128
    // Then each thread sums stats over bucket for (feat_lo, bin_t) and
    // (feat_lo+1, bin_t).

    const uint bin_t   = tid & 127u;
    const uint feat_lo = (tid >> 7) * 2u;

    for (uint fi = 0u; fi < 2u; ++fi) {
        const uint f = feat_lo + fi;
        const uint start = bucketOffsets[f * binStride + bin_t];
        const uint end   = bucketOffsets[f * binStride + bin_t + 1u];
        float sum = 0.0f;
        for (uint i = start; i < end; ++i) {
            const uint docIdx = bucketDocs[f * numDocs + i];
            sum += stats[0u * totalNumDocs + docIdx];
        }
        simdHist[0][f * BINS_PER_BYTE + bin_t] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

// ---------------------------------------------------------------------------
// T3: no-shuffle per-lane — each thread processes its OWN doc (no simd_shuffle
// broadcast). Owner-lane predicate (bin & 31 == lane) still applies. Since each
// thread processes one doc per outer-batch iteration and only owner lanes write,
// 31/32 of docs are DROPPED. This under-counts by 32×.
//
// WHAT THIS MEASURES: Lower bound on a shuffle-free single-owner formulation
// (structurally invalid — proves that removing shuffles alone is not enough).
// Compare against probe_A in reattribution.md (0.325 ms) which had the same
// under-counting bug.
// ---------------------------------------------------------------------------
static const std::string kT3Source = R"metal(
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

    for (uint batch_start = simd_id * SIMD_SIZE;
         batch_start < numDocs;
         batch_start += NUM_SIMD_GROUPS * SIMD_SIZE) {
        const uint d     = batch_start + lane;
        const bool valid = (d < numDocs);
        if (!valid) continue;
        const uint docIdx = docIndices[d];
        const uint packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
        const float stat  = stats[statIdx * totalNumDocs + docIdx];
        for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
            const uint bin = (packed >> (24u - 8u * f)) & 0xFFu;
            if (bin < foldCounts[foldBase + f] + 1u &&
                (bin & (SIMD_SIZE - 1u)) == lane) {
                simdHist[simd_id][f * BINS_PER_BYTE + bin] += stat;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

// ---------------------------------------------------------------------------
// T3b: no-shuffle + threadgroup-atomic writes to a single shared histogram.
// Each thread processes its OWN doc, atomically adds to simdHist[0][f*256+bin].
// This is the CORRECT no-shuffle alternative — all docs contribute once.
//
// Replaces: 32-iter shuffle chain with 1 atomic_fetch_add per feature per doc.
// Cost comparison: 256 threads × 196 batches × 4 features = 200,704 atomics
// per SIMD group; compared to 196 × 96 = 18,816 shuffles. 10× more operations,
// but each atomic has higher latency than a shuffle.
//
// NOTE: MSL atomic_float on threadgroup memory requires iOS 16/macOS 13 and
// the metal::atomic<float> type; pre-14 systems use uint bit-cast CAS.
// We use the CAS loop pattern to stay portable.
// ---------------------------------------------------------------------------
static const std::string kT3bSource = R"metal(
    // Use uint-backed TG memory for CAS-float pattern
    threadgroup atomic_uint simdHistU[HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    // Zero-init
    for (uint b = tid; b < HIST_PER_SIMD; b += BLOCK_SIZE)
        atomic_store_explicit(&simdHistU[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs          = totalNumDocs;
    const uint featureColumnIdx = 0u;
    const uint statIdx          = 0u;
    const uint foldBase         = 0u;

    // Each thread processes its own doc at stride BLOCK_SIZE (no shuffle).
    for (uint d = tid; d < numDocs; d += BLOCK_SIZE) {
        const uint docIdx = docIndices[d];
        const uint packed = compressedIndex[featureColumnIdx * totalNumDocs + docIdx];
        const float stat  = stats[statIdx * totalNumDocs + docIdx];
        for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
            const uint bin = (packed >> (24u - 8u * f)) & 0xFFu;
            if (bin < foldCounts[foldBase + f] + 1u) {
                const uint idx = f * BINS_PER_BYTE + bin;
                // CAS-float add
                uint oldBits = atomic_load_explicit(&simdHistU[idx], memory_order_relaxed);
                for (;;) {
                    const float oldF = as_type<float>(oldBits);
                    const float newF = oldF + stat;
                    const uint  newBits = as_type<uint>(newF);
                    uint expected = oldBits;
                    if (atomic_compare_exchange_weak_explicit(
                            &simdHistU[idx], &expected, newBits,
                            memory_order_relaxed, memory_order_relaxed)) {
                        break;
                    }
                    oldBits = expected;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Writeback to output (same shape as T0)
    if (tid < HIST_PER_SIMD) {
        const uint bits = atomic_load_explicit(&simdHistU[tid], memory_order_relaxed);
        output[tid] = as_type<float>(bits);
    }
)metal";

// ---------------------------------------------------------------------------
// A1: T1 fuse-valid + BATCH_DOCS=64 wider-batch.
//
// Each lane holds 2 docs in registers (packed_lo/hi, stat_lo/hi). Outer batch
// stride doubles to NUM_SIMD_GROUPS * 64 = 512 docs, halving the outer-loop
// iteration count (98 iters at N=50000 vs 196 at BATCH_DOCS=32). The inner
// shuffle work doubles to 64 shuffle pairs per outer iter, but the second-slab
// loads can issue concurrently with first-slab shuffle work — load-shuffle
// pipelining hides AGX memory latency (projected ~0.3–1 ms toy-kernel saving
// per `docs/sprint19/ablation_accumulation.md` §3.1).
//
// Parity: identical to T1 — MSB sentinel, reduction order unchanged within
// each SIMD group (doc-order preserved: lo batch then hi batch, src 0..31 in
// each). γ_7 unchanged.
// ---------------------------------------------------------------------------
static const std::string kA1Source = R"metal(
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
    const uint VALID_BIT        = 0x80000000u;
    const uint BATCH_DOCS       = 64u;
    const uint OUTER_STRIDE     = NUM_SIMD_GROUPS * BATCH_DOCS;

    for (uint batch_start = simd_id * BATCH_DOCS;
         batch_start < numDocs;
         batch_start += OUTER_STRIDE) {
        const uint d_lo     = batch_start + lane;
        const uint d_hi     = batch_start + SIMD_SIZE + lane;
        const bool valid_lo = (d_lo < numDocs);
        const bool valid_hi = (d_hi < numDocs);

        uint  packed_lo = 0u, packed_hi = 0u;
        float stat_lo = 0.0f, stat_hi = 0.0f;
        if (valid_lo) {
            const uint docIdx = docIndices[d_lo];
            packed_lo = compressedIndex[featureColumnIdx * totalNumDocs + docIdx] | VALID_BIT;
            stat_lo   = stats[statIdx * totalNumDocs + docIdx];
        }
        if (valid_hi) {
            const uint docIdx = docIndices[d_hi];
            packed_hi = compressedIndex[featureColumnIdx * totalNumDocs + docIdx] | VALID_BIT;
            stat_hi   = stats[statIdx * totalNumDocs + docIdx];
        }

        // First slab: shuffle through lane's packed_lo/stat_lo.
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s = simd_shuffle(packed_lo, src);
            const float s_s = simd_shuffle(stat_lo,   src);
            if ((p_s & VALID_BIT) == 0u) continue;
            const uint p_clean = p_s & 0x7FFFFFFFu;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_clean >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
        // Second slab: shuffle through lane's packed_hi/stat_hi.
        for (uint src = 0u; src < SIMD_SIZE; ++src) {
            const uint  p_s = simd_shuffle(packed_hi, src);
            const float s_s = simd_shuffle(stat_hi,   src);
            if ((p_s & VALID_BIT) == 0u) continue;
            const uint p_clean = p_s & 0x7FFFFFFFu;
            for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
                const uint bin = (p_clean >> (24u - 8u * f)) & 0xFFu;
                if (bin < foldCounts[foldBase + f] + 1u &&
                    (bin & (SIMD_SIZE - 1u)) == lane) {
                    simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HIST_PER_SIMD) output[tid] = simdHist[0][tid];
)metal";

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    fprintf(stderr, "[S19-algorithmic] Initializing MLX GPU device...\n");

    auto ciColMajor_host = MakeColMajorCI();
    auto docIndices_host = MakeDocIndices();
    auto stats_host      = MakeStats();
    auto binMajor        = MakeBinMajor(ciColMajor_host);

    // foldCounts: all 4 features have NUM_BINS (128)
    std::vector<uint32_t> foldCounts_host(4, NUM_BINS);

    auto ciColMajor = mx::array(
        reinterpret_cast<const int32_t*>(ciColMajor_host.data()),
        {static_cast<int>(LINE_SIZE * N_DOCS)}, mx::uint32);
    auto docIndices = mx::array(
        reinterpret_cast<const int32_t*>(docIndices_host.data()),
        {static_cast<int>(N_DOCS)}, mx::uint32);
    auto stats = mx::array(
        stats_host.data(), {static_cast<int>(N_DOCS)}, mx::float32);
    auto foldCounts = mx::array(
        reinterpret_cast<const int32_t*>(foldCounts_host.data()),
        {4}, mx::uint32);
    auto bucketOffsets = mx::array(
        reinterpret_cast<const int32_t*>(binMajor.bucketOffsets.data()),
        {static_cast<int>(binMajor.bucketOffsets.size())}, mx::uint32);
    auto bucketDocs = mx::array(
        reinterpret_cast<const int32_t*>(binMajor.bucketDocs.data()),
        {static_cast<int>(binMajor.bucketDocs.size())}, mx::uint32);

    auto totalDocsArr = mx::array(static_cast<uint32_t>(N_DOCS), mx::uint32);

    mx::eval({ciColMajor, docIndices, stats, foldCounts,
              bucketOffsets, bucketDocs, totalDocsArr});

    fprintf(stderr, "[S19-algorithmic] GPU data uploaded. "
                    "Pre-pass bucket layout built on host.\n");

    auto probeOut = mx::zeros({1024}, mx::float32);
    auto probeGrid = std::make_tuple(256, 1, 1);
    auto probeTG   = std::make_tuple(256, 1, 1);

    // -------------------------------------------------------------------------
    // Kernel registration
    // -------------------------------------------------------------------------
    auto t0Kernel = mx::fast::metal_kernel(
        "t0_production",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kT0Source, kHeader, true, false);

    auto t1Kernel = mx::fast::metal_kernel(
        "t1_fuse_valid",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kT1Source, kHeader, true, false);

    auto t2Kernel = mx::fast::metal_kernel(
        "t2_bin_major",
        {"bucketOffsets", "bucketDocs", "stats", "totalNumDocs"},
        {"output"}, kT2Source, kHeader, true, false);

    auto t3Kernel = mx::fast::metal_kernel(
        "t3_no_shuffle_owner",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kT3Source, kHeader, true, false);

    auto t3bKernel = mx::fast::metal_kernel(
        "t3b_no_shuffle_atomic",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kT3bSource, kHeader, true, false);

    auto a1Kernel = mx::fast::metal_kernel(
        "a1_t1_batch64",
        {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
        {"output"}, kA1Source, kHeader, true, false);

    // -------------------------------------------------------------------------
    // Time each variant
    // -------------------------------------------------------------------------
    auto run5 = [&](auto& kernel, const std::vector<mx::array>& inputs) {
        auto r = kernel(inputs, {probeOut.shape()}, {mx::float32},
                        probeGrid, probeTG, {}, 0.0f, false, mx::Device::gpu);
        mx::eval(r[0]);
    };

    fprintf(stderr, "[S19-algorithmic] === ALGORITHMIC PROBES ===\n");

    auto t0Stats = TimeKernel("T0 production baseline",
        [&]() { run5(t0Kernel,
            {ciColMajor, stats, docIndices, foldCounts, totalDocsArr}); },
        WARM_RUNS, TIMED_RUNS);

    auto t1Stats = TimeKernel("T1 fuse-valid (2 shuffles/src)",
        [&]() { run5(t1Kernel,
            {ciColMajor, stats, docIndices, foldCounts, totalDocsArr}); },
        WARM_RUNS, TIMED_RUNS);

    auto t2Stats = TimeKernel("T2 bin-major (accum only, pre-pass free)",
        [&]() { run5(t2Kernel,
            {bucketOffsets, bucketDocs, stats, totalDocsArr}); },
        WARM_RUNS, TIMED_RUNS);

    auto t3Stats = TimeKernel("T3 no-shuffle owner (drops 31/32 docs)",
        [&]() { run5(t3Kernel,
            {ciColMajor, stats, docIndices, foldCounts, totalDocsArr}); },
        WARM_RUNS, TIMED_RUNS);

    auto t3bStats = TimeKernel("T3b no-shuffle + TG atomic-CAS",
        [&]() { run5(t3bKernel,
            {ciColMajor, stats, docIndices, foldCounts, totalDocsArr}); },
        WARM_RUNS, TIMED_RUNS);

    auto a1Stats = TimeKernel("A1 T1+BATCH_DOCS=64 wider-batch",
        [&]() { run5(a1Kernel,
            {ciColMajor, stats, docIndices, foldCounts, totalDocsArr}); },
        WARM_RUNS, TIMED_RUNS);

    // -------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------
    fprintf(stdout, "\n");
    fprintf(stdout, "====================================================================\n");
    fprintf(stdout, "Sprint 19 Algorithmic Ablation — Accumulation Variants\n");
    fprintf(stdout, "Config: N=%u, NUM_BINS=%u, 1 TG x 256 threads, warm=%d, timed=%d\n",
            N_DOCS, NUM_BINS, WARM_RUNS, TIMED_RUNS);
    fprintf(stdout, "====================================================================\n\n");

    fprintf(stdout, "%-40s  mean_ms  stdev_ms  delta_vs_T0  pct_of_T0\n", "Variant");
    auto print = [&](const char* name, const TTimingResult& r, const TTimingResult& ref) {
        double delta = r.mean_ms - ref.mean_ms;
        double pct   = (ref.mean_ms > 0.0) ? 100.0 * (r.mean_ms / ref.mean_ms) : 0.0;
        fprintf(stdout, "%-40s  %-7.3f  %-8.3f  %+7.3f ms    %6.1f%%\n",
                name, r.mean_ms, r.stdev_ms, delta, pct);
    };
    print("T0 production baseline",                 t0Stats,  t0Stats);
    print("T1 fuse-valid (2 shuffles/src)",         t1Stats,  t0Stats);
    print("T2 bin-major (accum-only, free pre-pass)", t2Stats, t0Stats);
    print("T3 no-shuffle owner (under-counts 32x)", t3Stats,  t0Stats);
    print("T3b no-shuffle + TG atomic-CAS",         t3bStats, t0Stats);
    print("A1 T1+BATCH_DOCS=64 wider-batch",        a1Stats,  t0Stats);
    fprintf(stdout, "\n  (A1 vs T1 stack delta: %+7.3f ms  %6.1f%%)\n",
            a1Stats.mean_ms - t1Stats.mean_ms,
            (t1Stats.mean_ms > 0.0) ? 100.0 * (a1Stats.mean_ms / t1Stats.mean_ms) : 0.0);

    // Run-detail dump
    auto runs = [&](const char* tag, const TTimingResult& r) {
        fprintf(stdout, "  %s runs_ms: [", tag);
        for (size_t i = 0; i < r.runs_ms.size(); ++i)
            fprintf(stdout, "%.3f%s", r.runs_ms[i],
                    i+1 < r.runs_ms.size() ? ", " : "");
        fprintf(stdout, "]\n");
    };
    fprintf(stdout, "\n--- Per-run timings ---\n");
    runs("T0",  t0Stats);
    runs("T1",  t1Stats);
    runs("T2",  t2Stats);
    runs("T3",  t3Stats);
    runs("T3b", t3bStats);
    runs("A1",  a1Stats);

    // JSON
    fprintf(stdout, "\n--- JSON summary ---\n{\n");
    fprintf(stdout, "  \"config\": {\"N\": %u, \"NUM_BINS\": %u, \"warm\": %d, \"timed\": %d},\n",
            N_DOCS, NUM_BINS, WARM_RUNS, TIMED_RUNS);
    auto js = [&](const char* k, const TTimingResult& r, bool last) {
        fprintf(stdout, "    \"%s\": {\"mean_ms\": %.4f, \"stdev_ms\": %.4f}%s\n",
                k, r.mean_ms, r.stdev_ms, last ? "" : ",");
    };
    fprintf(stdout, "  \"probes\": {\n");
    js("T0_production", t0Stats, false);
    js("T1_fuse_valid", t1Stats, false);
    js("T2_bin_major_accum_only", t2Stats, false);
    js("T3_no_shuffle_owner_undercount", t3Stats, false);
    js("T3b_no_shuffle_atomic",  t3bStats, false);
    js("A1_t1_batch64_wider",    a1Stats,  true);
    fprintf(stdout, "  }\n}\n");

    return 0;
}
