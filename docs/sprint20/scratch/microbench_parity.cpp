// microbench_parity.cpp — Sprint 20 D1: T3b vs T0 full DEC-008 parity sweep
//
// PURPOSE
//   Sweep the 18-config DEC-008 envelope (N × loss × bins) and verify that
//   T3b (threadgroup atomic-CAS no-shuffle accumulator) produces histogram
//   output within the DEC-008 ulp bounds vs T0 (production shuffle baseline).
//   Also runs a 100-run determinism check on the gate config (50k/RMSE/128b).
//
//   CRITICAL DESIGN NOTE (from CRITIQUE phase):
//     T0 here uses the FOLDING variant from verify_correctness.cpp — it includes
//     the DEC-009 cross-SIMD 8-term linear fold that merges all 8 SIMD-group
//     partial histograms into a single per-bin sum before output.
//     The T0 in microbench_algorithmic.cpp does NOT fold — it outputs only
//     simdHist[0] (1/8 of the sum). Using the non-folding T0 would falsely fail
//     all configs (T3b produces full sum; non-folding T0 produces 1/8 of it).
//     The T3b source is taken verbatim from microbench_algorithmic.cpp.
//
//   DEC-008 envelope:
//     RMSE     ulp = 0   (bit-exact)
//     Logloss  ulp <= 4
//     MultiClass (approxDim=3): max ulp across all 3 stat dims <= 8
//
//   Folds semantics (S19-13 alignment):
//     foldCounts[f] = NumBins - 1  for all features (no-NaN ordinal)
//     Effective valid bin range: [0, NumBins-2] (i.e. NumBins-1 bins)
//     kernel bin-check: bin < foldCounts[f] + 1u  =>  bin < NumBins-1 + 1 = NumBins
//     Wait — this means all bins 0..NumBins-1 are valid (NumBins total valid bins).
//     Reread: foldCounts[f] = NumBins - 1, kernel checks bin < foldCounts[f] + 1
//     = bin < NumBins. So bins 0..NumBins-1 are valid. Correct.
//
// COMPILE
//   clang++ -std=c++17 -O2 \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation \
//     docs/sprint20/scratch/microbench_parity.cpp \
//     -o /tmp/microbench_parity
//
// RUN
//   DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/mlx/0.31.1/lib /tmp/microbench_parity
//
// OUTPUT
//   Per-config JSON to .cache/profiling/sprint20/d1_parity/<config>.json
//   Summary table + determinism verdict to stdout
//   d1_parity.md report content to stdout (redirect to docs/sprint20/d1_parity.md)

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <sys/stat.h>
#endif

namespace mx = mlx::core;

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t FEATURES_PER_PACK = 4u;
static constexpr uint32_t BINS_PER_BYTE     = 256u;
static constexpr uint32_t HIST_PER_SIMD     = FEATURES_PER_PACK * BINS_PER_BYTE; // 1024
static constexpr uint32_t LINE_SIZE         = 25u;  // matches microbench_algorithmic.cpp

// ============================================================================
// Config grid
// ============================================================================

enum class LossType { RMSE, Logloss, MultiClass };

struct Config {
    uint32_t rows;
    LossType loss;
    uint32_t bins;
    // derived
    uint32_t approxDim() const { return (loss == LossType::MultiClass) ? 3u : 1u; }
    const char* lossName() const {
        switch (loss) {
            case LossType::RMSE:       return "rmse";
            case LossType::Logloss:    return "logloss";
            case LossType::MultiClass: return "multiclass";
        }
        return "unknown";
    }
    std::string key() const {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%u_%s_d6_%ubins",
                      rows, lossName(), bins);
        return buf;
    }
    // DEC-008 ulp threshold
    uint32_t ulpThreshold() const {
        switch (loss) {
            case LossType::RMSE:       return 0u;   // bit-exact
            case LossType::Logloss:    return 4u;
            case LossType::MultiClass: return 8u;
        }
        return 0u;
    }
};

static std::vector<Config> MakeConfigGrid() {
    std::vector<Config> grid;
    for (uint32_t rows : {1000u, 10000u, 50000u}) {
        for (LossType loss : {LossType::RMSE, LossType::Logloss, LossType::MultiClass}) {
            for (uint32_t bins : {32u, 128u}) {
                grid.push_back({rows, loss, bins});
            }
        }
    }
    assert(grid.size() == 18u);
    return grid;
}

// ============================================================================
// Synthetic data generation
// ============================================================================

struct SynthData {
    std::vector<uint32_t> ciColMajor;  // [LINE_SIZE * N_DOCS] uint32
    std::vector<uint32_t> docIndices;  // [N_DOCS]
    // stats[dim][N_DOCS] — approxDim independent stat arrays
    std::vector<std::vector<float>> stats;
    std::vector<uint32_t> foldCounts;  // [4] — all set to bins-1 (S19-13 semantics)
    uint32_t numDocs;
};

static SynthData MakeSynthData(const Config& cfg) {
    const uint32_t N = cfg.rows;
    const uint32_t numBins = cfg.bins;
    SynthData d;
    d.numDocs = N;

    // compressedIndex — col-major: [LINE_SIZE * N] entries, feature col 0 is used
    d.ciColMajor.resize(LINE_SIZE * N, 0u);
    for (uint32_t doc = 0u; doc < N; ++doc) {
        // Pack 4 features per uint32 (same pattern as microbench_algorithmic.cpp)
        // Bin values use multipliers 7, 11, 13, 17 with masking to 0xFF
        const uint32_t b0 = (doc * 7u)  & 0xFFu;
        const uint32_t b1 = (doc * 11u) & 0xFFu;
        const uint32_t b2 = (doc * 13u) & 0xFFu;
        const uint32_t b3 = (doc * 17u) & 0xFFu;
        d.ciColMajor[0 * N + doc] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }

    // docIndices — identity permutation
    d.docIndices.resize(N);
    std::iota(d.docIndices.begin(), d.docIndices.end(), 0u);

    // foldCounts — S19-13 semantics: folds = NumBins - 1
    d.foldCounts.assign(4u, numBins - 1u);

    // stat arrays — one per approxDim
    const uint32_t aDim = cfg.approxDim();
    d.stats.resize(aDim);
    for (uint32_t dim = 0u; dim < aDim; ++dim) {
        d.stats[dim].resize(N);
        // Each dim gets a distinct stat pattern to stress MultiClass non-determinism
        switch (dim) {
            case 0:
                for (uint32_t doc = 0u; doc < N; ++doc)
                    d.stats[dim][doc] = static_cast<float>(doc % 128u) / 128.0f;
                break;
            case 1:
                for (uint32_t doc = 0u; doc < N; ++doc)
                    d.stats[dim][doc] = static_cast<float>((doc + 7u) % 64u) / 64.0f;
                break;
            case 2:
                for (uint32_t doc = 0u; doc < N; ++doc)
                    d.stats[dim][doc] = static_cast<float>((doc + 13u) % 32u) / 32.0f;
                break;
            default:
                for (uint32_t doc = 0u; doc < N; ++doc)
                    d.stats[dim][doc] = 0.5f;
                break;
        }
    }

    return d;
}

// ============================================================================
// Metal kernel sources
// ============================================================================

// Shared header — same as microbench_algorithmic.cpp
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
// T0: production baseline WITH full DEC-009 cross-SIMD linear fold.
//
// IMPORTANT: this is the FOLDING variant from verify_correctness.cpp, NOT the
// non-folding variant from microbench_algorithmic.cpp. The folding version
// produces the correct full per-bin sum across all 8 SIMD groups. Without the
// fold, output[tid] = simdHist[0][tid] which holds only the SIMD-0 partial —
// 1/8 of the true sum. Comparing T3b (full sum) against the non-folding T0
// (1/8 sum) would produce a guaranteed FAIL on every config.
// ---------------------------------------------------------------------------
static const std::string kT0Source = R"metal(
    threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD];

    const uint tid     = thread_index_in_threadgroup;
    const uint lane    = tid & (SIMD_SIZE - 1u);
    const uint simd_id = tid >> 5u;

    // Zero-init per SIMD group
    for (uint b = lane; b < HIST_PER_SIMD; b += SIMD_SIZE)
        simdHist[simd_id][b] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint numDocs = totalNumDocs;

    // Accumulation: cooperative shuffle-broadcast loop (T0 production pattern)
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

    // DEC-009 cross-SIMD linear fold: merge 8 SIMD-group partials into simdHist[0].
    // Without this fold, output[tid] = simdHist[0][tid] = 1/8 of true sum — WRONG.
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

// ---------------------------------------------------------------------------
// T3b: no-shuffle + threadgroup atomic-CAS accumulator.
// Verbatim from microbench_algorithmic.cpp::kT3bSource.
// Each thread processes its own doc at stride BLOCK_SIZE, adds via CAS-float
// to a single shared simdHistU[HIST_PER_SIMD]. Cross-SIMD fold not needed —
// simdHistU[b] already holds the full per-bin sum across all threads.
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

    // Writeback to output (same shape as T0: 1024 floats)
    if (tid < HIST_PER_SIMD) {
        const uint bits = atomic_load_explicit(&simdHistU[tid], memory_order_relaxed);
        output[tid] = as_type<float>(bits);
    }
)metal";

// NOTE: T3b uses `compressedIndex[featureColumnIdx * totalNumDocs + docIdx]`
// but T0 uses `compressedIndex[docIdx]`. In the harness we pass the GPU array
// that starts at feature column 0 (first N entries of ciColMajor), so both
// indexing patterns refer to the same physical data:
//   - T0 gets ciArr directly (shape [N]) → compressedIndex[docIdx] = ci[0*N + docIdx]
//   - T3b gets ciArr directly (shape [N]) → compressedIndex[0 * totalNumDocs + docIdx]
//     = compressedIndex[docIdx] (since totalNumDocs = N)
// Both are equivalent for a single feature column. No mismatch.

// ============================================================================
// ULP difference (32-bit IEEE 754)
// ============================================================================

static uint32_t UlpDiff(float a, float b) {
    if (a == b) return 0u;
    // Handle NaN/Inf gracefully — treat as max distance
    if (!std::isfinite(a) || !std::isfinite(b)) return 0xFFFFFFFFu;
    uint32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    // Same sign: bit-distance = |ia - ib|
    // Different sign: distance wraps through 0/-0, handle with signed arithmetic
    if ((ia >> 31u) == (ib >> 31u)) {
        return (ia > ib) ? (ia - ib) : (ib - ia);
    }
    // Different signs: convert to sign-magnitude, then distance
    // Map negative floats: flip all bits; positive floats: flip sign bit
    auto toUlpSpace = [](uint32_t bits) -> int64_t {
        if (bits & 0x80000000u)
            return -static_cast<int64_t>(bits & 0x7FFFFFFFu);
        return static_cast<int64_t>(bits);
    };
    int64_t da = toUlpSpace(ia), db = toUlpSpace(ib);
    int64_t diff = da - db;
    return static_cast<uint32_t>(diff < 0 ? -diff : diff);
}

// FNV-1a 64-bit hash over raw float bytes
static uint64_t HashHistogram(const std::vector<float>& h) {
    uint64_t hash = 14695981039346656037ULL;
    for (float v : h) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        for (int b = 0; b < 4; ++b) {
            hash ^= static_cast<uint64_t>((bits >> (b * 8)) & 0xFFu);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

// ============================================================================
// Per-dim parity record
// ============================================================================

struct DimRecord {
    uint32_t dim;
    uint32_t max_ulp;
    float    max_abs_diff;
    bool     bit_exact;
    bool     pass;           // pass = max_ulp <= threshold
    uint32_t threshold;
};

struct ConfigRecord {
    Config cfg;
    std::vector<DimRecord> dims;
    uint64_t t0_hash;
    uint64_t t3b_hash;
    uint32_t max_ulp_overall;   // max across all dims
    float    max_abs_diff_overall;
    bool     bin_exact;         // bit-exact across all dims
    bool     parity_pass;       // pass = all dims pass
};

// ============================================================================
// GPU kernel registration
// ============================================================================

using KernelFn = mx::fast::CustomKernelFunction;

struct Kernels {
    KernelFn t0;
    KernelFn t3b;
};

static Kernels BuildKernels() {
    // Input names for T0: compressedIndex, stats, docIndices, foldCounts, totalNumDocs
    // Input names for T3b: compressedIndex, stats, docIndices, foldCounts, totalNumDocs
    return {
        mx::fast::metal_kernel(
            "t0_parity_folding",
            {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
            {"output"},
            kT0Source, kHeader, true, false),
        mx::fast::metal_kernel(
            "t3b_parity_atomic",
            {"compressedIndex", "stats", "docIndices", "foldCounts", "totalNumDocs"},
            {"output"},
            kT3bSource, kHeader, true, false),
    };
}

// ============================================================================
// Run one (kernel, dim) pair
// ============================================================================

static std::vector<float> RunKernel(
        KernelFn& kernel,
        const mx::array& ciArr,
        const mx::array& statArr,   // shape [N] — one dim at a time
        const mx::array& diArr,
        const mx::array& fcArr,
        const mx::array& tdArr) {

    auto probeOut = mx::zeros({static_cast<int>(HIST_PER_SIMD)}, mx::float32);
    auto grid = std::make_tuple(256, 1, 1);
    auto tg   = std::make_tuple(256, 1, 1);

    auto res = kernel({ciArr, statArr, diArr, fcArr, tdArr},
                      {probeOut.shape()}, {mx::float32},
                      grid, tg, {}, 0.0f, false, mx::Device::gpu);
    mx::eval(res[0]);

    return std::vector<float>(res[0].data<float>(),
                              res[0].data<float>() + HIST_PER_SIMD);
}

// ============================================================================
// Sweep one config
// ============================================================================

static ConfigRecord SweepConfig(const Config& cfg, Kernels& kernels) {
    const SynthData synth = MakeSynthData(cfg);
    const uint32_t N = synth.numDocs;

    // Upload CI (feature col 0: first N uint32 entries of ciColMajor)
    mx::array ciArr = mx::array(
        reinterpret_cast<const int32_t*>(synth.ciColMajor.data()),
        {static_cast<int>(N)}, mx::uint32);
    mx::array diArr = mx::array(
        reinterpret_cast<const int32_t*>(synth.docIndices.data()),
        {static_cast<int>(N)}, mx::uint32);
    mx::array fcArr = mx::array(
        reinterpret_cast<const int32_t*>(synth.foldCounts.data()),
        {4}, mx::uint32);
    mx::array tdArr = mx::array(static_cast<uint32_t>(N), mx::uint32);
    mx::eval({ciArr, diArr, fcArr, tdArr});

    ConfigRecord rec;
    rec.cfg = cfg;
    rec.max_ulp_overall = 0u;
    rec.max_abs_diff_overall = 0.0f;
    rec.bin_exact = true;
    rec.parity_pass = true;

    // Combine hashes across all dims to produce a single per-config hash
    uint64_t t0HashCombined = 0x0, t3bHashCombined = 0x0;

    for (uint32_t dim = 0u; dim < cfg.approxDim(); ++dim) {
        mx::array statArr = mx::array(
            synth.stats[dim].data(),
            {static_cast<int>(N)}, mx::float32);
        mx::eval(statArr);

        auto t0Out  = RunKernel(kernels.t0,  ciArr, statArr, diArr, fcArr, tdArr);
        auto t3bOut = RunKernel(kernels.t3b, ciArr, statArr, diArr, fcArr, tdArr);

        // Bin-by-bin comparison
        uint32_t dimMaxUlp = 0u;
        float    dimMaxAbs = 0.0f;
        bool     dimBitExact = true;

        for (uint32_t b = 0u; b < HIST_PER_SIMD; ++b) {
            uint32_t ulp = UlpDiff(t0Out[b], t3bOut[b]);
            float    abs = std::fabs(t0Out[b] - t3bOut[b]);
            if (ulp > dimMaxUlp) dimMaxUlp = ulp;
            if (abs > dimMaxAbs) dimMaxAbs = abs;
            if (ulp != 0u) dimBitExact = false;
        }

        const uint32_t thr = cfg.ulpThreshold();
        const bool dimPass = (dimMaxUlp <= thr);

        DimRecord dr;
        dr.dim = dim;
        dr.max_ulp = dimMaxUlp;
        dr.max_abs_diff = dimMaxAbs;
        dr.bit_exact = dimBitExact;
        dr.pass = dimPass;
        dr.threshold = thr;
        rec.dims.push_back(dr);

        if (dimMaxUlp > rec.max_ulp_overall) rec.max_ulp_overall = dimMaxUlp;
        if (dimMaxAbs > rec.max_abs_diff_overall) rec.max_abs_diff_overall = dimMaxAbs;
        if (!dimBitExact) rec.bin_exact = false;
        if (!dimPass) rec.parity_pass = false;

        // Fold hashes — XOR-combine per dim
        t0HashCombined  ^= (HashHistogram(t0Out)  + 0x9e3779b97f4a7c15ULL + dim);
        t3bHashCombined ^= (HashHistogram(t3bOut) + 0x9e3779b97f4a7c15ULL + dim);
    }

    rec.t0_hash  = t0HashCombined;
    rec.t3b_hash = t3bHashCombined;

    return rec;
}

// ============================================================================
// Determinism check: 100 runs on 50k/RMSE/128b
// ============================================================================

struct DetRecord {
    int unique_hashes;
    bool pass;
    std::vector<uint64_t> hashes;  // all 100
};

static DetRecord DeterminismCheck(Kernels& kernels) {
    Config gateCfg = {50000u, LossType::RMSE, 128u};
    const SynthData synth = MakeSynthData(gateCfg);
    const uint32_t N = synth.numDocs;

    mx::array ciArr = mx::array(
        reinterpret_cast<const int32_t*>(synth.ciColMajor.data()),
        {static_cast<int>(N)}, mx::uint32);
    mx::array statArr = mx::array(
        synth.stats[0].data(),
        {static_cast<int>(N)}, mx::float32);
    mx::array diArr = mx::array(
        reinterpret_cast<const int32_t*>(synth.docIndices.data()),
        {static_cast<int>(N)}, mx::uint32);
    mx::array fcArr = mx::array(
        reinterpret_cast<const int32_t*>(synth.foldCounts.data()),
        {4}, mx::uint32);
    mx::array tdArr = mx::array(static_cast<uint32_t>(N), mx::uint32);
    mx::eval({ciArr, statArr, diArr, fcArr, tdArr});

    fprintf(stderr, "  [determinism] 100 × T3b on 50k/RMSE/128b: ");

    DetRecord det;
    det.hashes.reserve(100);

    for (int run = 0; run < 100; ++run) {
        auto out = RunKernel(kernels.t3b, ciArr, statArr, diArr, fcArr, tdArr);
        det.hashes.push_back(HashHistogram(out));
        if ((run + 1) % 20 == 0) fprintf(stderr, "%d ", run + 1);
    }
    fprintf(stderr, "\n");

    // Count unique hashes
    auto sorted = det.hashes;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    det.unique_hashes = static_cast<int>(sorted.size());
    det.pass = (det.unique_hashes == 1);

    return det;
}

// ============================================================================
// Filesystem helpers
// ============================================================================

static void MkdirP(const std::string& path) {
    // Create directory and parents — naive slash split
    std::string partial;
    for (char c : path) {
        partial += c;
        if (c == '/') {
            ::mkdir(partial.c_str(), 0755);
        }
    }
    ::mkdir(path.c_str(), 0755);
}

// ============================================================================
// JSON emit
// ============================================================================

static std::string EscapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

static void WriteConfigJSON(const std::string& dir, const ConfigRecord& rec) {
    const std::string path = dir + "/" + rec.cfg.key() + ".json";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"rows\": %u,\n", rec.cfg.rows);
    fprintf(f, "    \"loss\": \"%s\",\n", EscapeJson(rec.cfg.lossName()).c_str());
    fprintf(f, "    \"bins\": %u,\n", rec.cfg.bins);
    fprintf(f, "    \"approx_dim\": %u,\n", rec.cfg.approxDim());
    fprintf(f, "    \"depth\": 6,\n");
    fprintf(f, "    \"seed\": 42\n");
    fprintf(f, "  },\n");
    fprintf(f, "  \"t0_histogram_hash\": \"0x%016llx\",\n",
            static_cast<unsigned long long>(rec.t0_hash));
    fprintf(f, "  \"t3b_histogram_hash\": \"0x%016llx\",\n",
            static_cast<unsigned long long>(rec.t3b_hash));
    fprintf(f, "  \"bin_exact\": %s,\n", rec.bin_exact ? "true" : "false");
    fprintf(f, "  \"max_abs_diff\": %e,\n", rec.max_abs_diff_overall);
    fprintf(f, "  \"max_ulp_diff\": %u,\n", rec.max_ulp_overall);
    fprintf(f, "  \"ulp_threshold\": %u,\n", rec.cfg.ulpThreshold());
    fprintf(f, "  \"parity_verdict\": \"%s\",\n", rec.parity_pass ? "PASS" : "FAIL");
    fprintf(f, "  \"per_approx_dim\": [\n");
    for (size_t i = 0; i < rec.dims.size(); ++i) {
        const auto& dr = rec.dims[i];
        fprintf(f, "    {\"dim\": %u, \"max_ulp\": %u, \"max_abs_diff\": %e, "
                   "\"bit_exact\": %s, \"verdict\": \"%s\"}%s\n",
                dr.dim, dr.max_ulp, dr.max_abs_diff,
                dr.bit_exact ? "true" : "false",
                dr.pass ? "PASS" : "FAIL",
                (i + 1 < rec.dims.size()) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

static void WriteDetJSON(const std::string& dir, const DetRecord& det) {
    const std::string path = dir + "/determinism_50k_rmse_128b.json";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"config\": {\"rows\": 50000, \"loss\": \"rmse\", \"bins\": 128, \"runs\": 100},\n");
    fprintf(f, "  \"unique_hashes\": %d,\n", det.unique_hashes);
    fprintf(f, "  \"determinism_verdict\": \"%s\",\n", det.pass ? "PASS" : "FAIL");
    fprintf(f, "  \"hashes\": [\n");
    for (size_t i = 0; i < det.hashes.size(); ++i) {
        fprintf(f, "    \"0x%016llx\"%s\n",
                static_cast<unsigned long long>(det.hashes[i]),
                (i + 1 < det.hashes.size()) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

// ============================================================================
// Report generation
// ============================================================================

static void PrintSummary(const std::vector<ConfigRecord>& records,
                         const DetRecord& det,
                         const std::string& outputDir) {
    int pass_count = 0;
    for (const auto& r : records) if (r.parity_pass) pass_count++;

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "S20-D1 Parity Sweep — T3b vs T0 — DEC-008 Envelope\n");
    fprintf(stdout, "================================================================================\n\n");

    fprintf(stdout, "%-30s  %-5s  %-6s  %-9s  %-8s  %-6s\n",
            "config", "dims", "thr", "max_ulp", "max_abs", "verdict");
    fprintf(stdout, "%-30s  %-5s  %-6s  %-9s  %-8s  %-6s\n",
            "------", "----", "---", "-------", "-------", "-------");

    for (const auto& r : records) {
        fprintf(stdout, "%-30s  %5u  %6u  %9u  %8.2e  %-6s\n",
                r.cfg.key().c_str(),
                r.cfg.approxDim(),
                r.cfg.ulpThreshold(),
                r.max_ulp_overall,
                static_cast<double>(r.max_abs_diff_overall),
                r.parity_pass ? "PASS" : "FAIL");
    }

    fprintf(stdout, "\n  PARITY TOTAL:       %d / 18 PASS\n", pass_count);
    fprintf(stdout, "  DETERMINISM:        %d unique hashes / 100 runs  →  %s\n",
            det.unique_hashes, det.pass ? "PASS" : "FAIL");
    fprintf(stdout, "\n  JSON output:        %s/\n", outputDir.c_str());

    // Honest bottom-line
    const bool allPass = (pass_count == 18 && det.pass);
    fprintf(stdout, "\n  D1 VERDICT:  %s\n", allPass ? "PASS" : "FAIL");
    if (!allPass) {
        if (pass_count < 18) {
            fprintf(stdout, "  --> Parity failures detected. Review FAILed configs above.\n");
            fprintf(stdout, "      Per DEC-017: if parity fails, apply Kahan compensation and re-sweep.\n");
            // Identify failure classes
            bool rmsFail = false, logFail = false, mcFail = false;
            for (const auto& r : records) {
                if (!r.parity_pass) {
                    if (r.cfg.loss == LossType::RMSE)       rmsFail = true;
                    if (r.cfg.loss == LossType::Logloss)    logFail = true;
                    if (r.cfg.loss == LossType::MultiClass) mcFail  = true;
                }
            }
            if (mcFail && !rmsFail && !logFail)
                fprintf(stdout, "      NOTE: only MultiClass fails — conditional Kahan (approxDim>1 only) may suffice.\n");
        }
        if (!det.pass)
            fprintf(stdout, "  --> Determinism FAIL: T3b is non-deterministic on AGX. This is a blocking finding.\n");
    }
    fprintf(stdout, "\n");
}

// ============================================================================
// Write d1_parity.md report to stdout (redirect to file)
// ============================================================================

static void PrintMarkdownReport(const std::vector<ConfigRecord>& records,
                                const DetRecord& det) {
    int pass_count = 0;
    for (const auto& r : records) if (r.parity_pass) pass_count++;
    const bool allPass = (pass_count == 18 && det.pass);

    fprintf(stdout, "\n\n===== BEGIN d1_parity.md =====\n\n");
    fprintf(stdout, "# S20-D1 Parity Report — T3b vs T0 DEC-008 Envelope\n\n");
    fprintf(stdout, "**Branch**: `mlx/sprint-20-hist-atomic-cas`  \n");
    fprintf(stdout, "**Date**: 2026-04-19  \n");
    fprintf(stdout, "**Gate**: 18/18 configs within DEC-008 ulp bounds + 100/100 determinism runs\n\n");
    fprintf(stdout, "---\n\n");
    fprintf(stdout, "## Verdict\n\n");
    fprintf(stdout, "**D1 %s** — %d/18 configs pass parity, %d/100 determinism runs identical\n\n",
            allPass ? "PASS" : "FAIL", pass_count,
            det.pass ? 100 : (100 - (det.unique_hashes > 1 ? det.unique_hashes : 0)));
    fprintf(stdout, "---\n\n");
    fprintf(stdout, "## Per-config parity table\n\n");
    fprintf(stdout, "| Config | ApproxDim | ULP threshold | Max ULP | Max abs diff | Verdict |\n");
    fprintf(stdout, "|---|---|---|---|---|---|\n");
    for (const auto& r : records) {
        fprintf(stdout, "| %s | %u | %u | %u | %.2e | **%s** |\n",
                r.cfg.key().c_str(),
                r.cfg.approxDim(),
                r.cfg.ulpThreshold(),
                r.max_ulp_overall,
                static_cast<double>(r.max_abs_diff_overall),
                r.parity_pass ? "PASS" : "FAIL");
    }
    fprintf(stdout, "\n");
    // Per-dim detail for MultiClass
    fprintf(stdout, "### MultiClass per-approxDim detail\n\n");
    fprintf(stdout, "| Config | Dim | Max ULP | Max abs diff | Verdict |\n");
    fprintf(stdout, "|---|---|---|---|---|\n");
    for (const auto& r : records) {
        if (r.cfg.loss == LossType::MultiClass) {
            for (const auto& dr : r.dims) {
                fprintf(stdout, "| %s | %u | %u | %.2e | **%s** |\n",
                        r.cfg.key().c_str(), dr.dim,
                        dr.max_ulp, static_cast<double>(dr.max_abs_diff),
                        dr.pass ? "PASS" : "FAIL");
            }
        }
    }
    fprintf(stdout, "\n---\n\n");
    fprintf(stdout, "## Determinism\n\n");
    fprintf(stdout, "100 × T3b runs on 50k/RMSE/d6/128b/seed42. Unique histogram hashes: **%d**.  \n",
            det.unique_hashes);
    fprintf(stdout, "Verdict: **%s**\n\n", det.pass ? "PASS — bit-exact run-to-run" : "FAIL — non-deterministic");
    fprintf(stdout, "---\n\n");
    fprintf(stdout, "## Honest bottom line\n\n");
    if (allPass) {
        fprintf(stdout, "D1 PASS. All 18 configs within DEC-008 envelope. T3b is bit-exact run-to-run.\n");
        fprintf(stdout, "Proceed to D2 (kernel integration) per sprint plan.\n");
    } else {
        if (pass_count < 18) {
            fprintf(stdout, "D1 FAIL — parity. ");
            bool mcOnly = true;
            for (const auto& r : records) {
                if (!r.parity_pass && r.cfg.loss != LossType::MultiClass)
                    mcOnly = false;
            }
            if (mcOnly)
                fprintf(stdout, "Only MultiClass configs fail. Kahan mitigation scoped to approxDim>1. Re-sweep required.\n");
            else
                fprintf(stdout, "RMSE and/or Logloss configs fail. Full Kahan mitigation required. Re-sweep before D2.\n");
        }
        if (!det.pass)
            fprintf(stdout, "D1 FAIL — non-determinism. T3b produces different histogram hashes across runs. "
                            "This is a fundamental property of FP32 atomic contention order on AGX. "
                            "Kahan will not fix non-determinism (it is a precision fix, not an ordering fix). "
                            "Escalate to Ramos before proceeding.\n");
    }
    fprintf(stdout, "\n---\n\n");
    fprintf(stdout, "## Reproduce commands\n\n");
    fprintf(stdout, "```bash\n");
    fprintf(stdout, "# Compile\n");
    fprintf(stdout, "clang++ -std=c++17 -O2 \\\n");
    fprintf(stdout, "  -I/opt/homebrew/Cellar/mlx/0.31.1/include \\\n");
    fprintf(stdout, "  -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \\\n");
    fprintf(stdout, "  -framework Metal -framework Foundation \\\n");
    fprintf(stdout, "  docs/sprint20/scratch/microbench_parity.cpp \\\n");
    fprintf(stdout, "  -o /tmp/microbench_parity\n\n");
    fprintf(stdout, "# Run\n");
    fprintf(stdout, "DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/mlx/0.31.1/lib /tmp/microbench_parity\n");
    fprintf(stdout, "```\n\n");
    fprintf(stdout, "JSON output: `.cache/profiling/sprint20/d1_parity/`\n");
    fprintf(stdout, "\n===== END d1_parity.md =====\n");
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    fprintf(stderr, "[S20-D1] Initializing MLX GPU device...\n");

    const std::string ROOT =
        "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs"
        "/Programming/Frameworks/catboost-mlx";
    const std::string outDir = ROOT + "/.cache/profiling/sprint20/d1_parity";
    MkdirP(outDir);

    // Build kernels (MLX compiles Metal on first dispatch)
    Kernels kernels = BuildKernels();

    // Warm-up: dispatch once on small data to trigger Metal compilation
    fprintf(stderr, "[S20-D1] Warming up Metal kernels...\n");
    {
        Config warmCfg = {1000u, LossType::RMSE, 32u};
        SynthData warmData = MakeSynthData(warmCfg);
        const uint32_t Nw = warmData.numDocs;
        auto ci = mx::array(reinterpret_cast<const int32_t*>(warmData.ciColMajor.data()),
                            {static_cast<int>(Nw)}, mx::uint32);
        auto st = mx::array(warmData.stats[0].data(), {static_cast<int>(Nw)}, mx::float32);
        auto di = mx::array(reinterpret_cast<const int32_t*>(warmData.docIndices.data()),
                            {static_cast<int>(Nw)}, mx::uint32);
        auto fc = mx::array(reinterpret_cast<const int32_t*>(warmData.foldCounts.data()),
                            {4}, mx::uint32);
        auto td = mx::array(static_cast<uint32_t>(Nw), mx::uint32);
        mx::eval({ci, st, di, fc, td});
        RunKernel(kernels.t0,  ci, st, di, fc, td);
        RunKernel(kernels.t3b, ci, st, di, fc, td);
    }
    fprintf(stderr, "[S20-D1] Metal kernels compiled.\n\n");

    // 18-config parity sweep
    auto grid = MakeConfigGrid();
    std::vector<ConfigRecord> records;
    records.reserve(18);

    for (const auto& cfg : grid) {
        fprintf(stderr, "[S20-D1] Sweeping %s... ", cfg.key().c_str());
        ConfigRecord rec = SweepConfig(cfg, kernels);
        fprintf(stderr, "%s (max_ulp=%u)\n", rec.parity_pass ? "PASS" : "FAIL", rec.max_ulp_overall);
        WriteConfigJSON(outDir, rec);
        records.push_back(rec);
    }

    // 100-run determinism check
    fprintf(stderr, "\n[S20-D1] Running determinism check (100 runs × 50k/RMSE/128b)...\n");
    DetRecord det = DeterminismCheck(kernels);
    fprintf(stderr, "  unique hashes: %d → %s\n", det.unique_hashes, det.pass ? "PASS" : "FAIL");
    WriteDetJSON(outDir, det);

    // Print summary table and markdown report
    PrintSummary(records, det, outDir);
    PrintMarkdownReport(records, det);

    return 0;
}
