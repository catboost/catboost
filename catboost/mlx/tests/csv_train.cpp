// CatBoost-MLX standalone CSV training tool.
// Reads a CSV file, quantizes features, and trains a GBDT model on Apple Silicon GPU.
//
// Usage: ./csv_train <file.csv> [options]
//   --iterations N         Number of boosting iterations (default: 100)
//   --depth D              Max tree depth (default: 6)
//   --lr RATE              Learning rate (default: 0.1)
//   --l2 LAMBDA            L2 regularization (default: 3.0)
//   --loss TYPE            Loss: rmse, logloss, multiclass, mae, quantile[:alpha], huber[:delta], poisson, tweedie[:p], mape, auto
//   --bins B               Max quantization bins per feature (default: 255)
//   --target-col N         0-based column index for target (default: last column)
//   --cat-features L       Comma-separated 0-based column indices for categorical features
//   --eval-fraction F      Fraction of data for validation (default: 0 = no split)
//   --early-stopping N     Stop after N iterations with no validation improvement (default: 0 = disabled)
//   --subsample F          Row subsampling fraction per iteration (default: 1.0)
//   --colsample-bytree F   Feature subsampling fraction per tree (default: 1.0)
//   --seed N               Random seed for subsampling (default: 42)
//   --nan-mode MODE        NaN handling: min (default), forbidden
//   --output PATH          Save trained model to JSON file
//   --feature-importance   Print gain-based feature importance after training
//   --cv N                 N-fold cross-validation (default: 0 = disabled)
//   --ctr                  Enable CTR target encoding for high-cardinality categoricals
//   --ctr-prior F          CTR prior (default: 0.5)
//   --max-onehot-size N    Max categories for OneHot; above uses CTR (default: 10)
//   --group-col N          0-based column index for group/query ID (ranking losses)
//   --weight-col N         0-based column index for sample weights (default: -1 = uniform)
//   --min-data-in-leaf N   Minimum documents per leaf (default: 1)
//   --monotone-constraints 0,1,-1,...  Per-feature monotone constraints (0=none, 1=inc, -1=dec)
//   --snapshot-path PATH   Save/restore training snapshot for resume (default: disabled)
//   --snapshot-interval N  Save snapshot every N iterations (default: 1)
//   --grow-policy POLICY   Tree grow policy: SymmetricTree (default), Depthwise (per-leaf), Lossguide (best-first)
//   --max-leaves N         Maximum leaves for Lossguide grow policy (default: 31)
//   --verbose              Print per-iteration loss
//
// Compile:
//   clang++ -std=c++17 -O2 -I. \
//     -I/opt/homebrew/Cellar/mlx/0.31.1/include \
//     -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
//     -framework Metal -framework Foundation -Wno-c++20-extensions \
//     catboost/mlx/tests/csv_train.cpp -o csv_train
//
// Examples:
//   ./csv_train iris.csv --loss multiclass --iterations 200
//   ./csv_train housing.csv --loss rmse --depth 4 --lr 0.05
//   ./csv_train breast_cancer.csv --loss logloss

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>
#ifdef CATBOOST_MLX_STAGE_PROFILE
#include <catboost/mlx/methods/stage_profiler.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <optional>
#include <queue>
#include <stdexcept>   // std::invalid_argument, std::runtime_error
#ifdef CATBOOST_MLX_STAGE_PROFILE
#include <filesystem>
#include <sstream>
#endif

// ============================================================================
// S30-T1-INSTRUMENT: per-stage float32 residual instrumentation
//
// Compile-time gate: -DCOSINE_RESIDUAL_INSTRUMENT
// Purpose: compute fp32 and fp64 accumulations in parallel at iter=0 (1-indexed
//   iter=1) to measure which accumulator site dominates the observed ST+Cosine
//   drift (DEC-034/DEC-035 T1 gate G1 deliverable).
//
// Accumulators measured:
//   1. cosDen  — float32 joint-Cosine denominator in FindBestSplit (ordinal)
//   2. cosNum  — float32 joint-Cosine numerator  in FindBestSplit (ordinal)
//   3. gSums   — float32 per-leaf gradient sums (scatter_add_axis → Newton leaf value)
//   4. cursorInc — float32 per-doc cursor increment (mx::take(leafValues, partitions))
//
// Output: CSV files in docs/sprint30/t1-instrument/data/ one file per site per seed.
// DO NOT SHIP: instrumentation is evidence, not product code.
//
// S30-D2-INSTRUMENT extension: -DCOSINE_D2_INSTRUMENT (implies COSINE_RESIDUAL_INSTRUMENT)
// Purpose: extend T1 instrumentation with dedicated gain_scalar CSVs for the D2
//   full-stack residual + argmax-flip diagnostic (DEC-035 #101 task).
// Additional output schema:
//   gain_scalar_seedN.csv  — per (feat, bin) gain_f32, gain_f64, gain_abs_residual
//   leaf_sum_seedN.csv     — same as T1/T2 leaf_sum (reused; written to D2 data dir)
// Argmax flip CSVs are computed by the Python post-processor run_d2_instrument.py
//   from the gain_scalar CSVs (no second pass through FindBestSplit needed).
// Output directory: docs/sprint30/d2-stack-instrument/data/ (overridable via
//   COSINE_RESIDUAL_OUTDIR env var, same as T1/T2).
// ============================================================================
#ifdef COSINE_D2_INSTRUMENT
// D2 implies T1/T2 instrument paths — enable the base flag automatically
#  ifndef COSINE_RESIDUAL_INSTRUMENT
#    define COSINE_RESIDUAL_INSTRUMENT
#  endif
#endif

#ifdef COSINE_RESIDUAL_INSTRUMENT
#include <filesystem>
#include <cassert>

struct TCosineResidualInstrument {
    // Output directory — set at startup from argv[0] parent + relative path
    std::string outDir;

    // Active flag: dump only at iter==0 (1-indexed iter=1), cleared after first dump
    bool dumpCosDen = false;   // set true just before FindBestSplit call at iter=0
    bool dumpLeaf   = false;   // set true just before leaf-value computation at iter=0
    bool dumpApprox = false;   // set true just before approx-update at iter=0

    // Seed label written into filenames so multi-seed runs produce distinct files
    int seed = 0;

    // Accumulation vectors for cosDen/cosNum double shadows (per bin candidate across all p*k)
    // Filled inside FindBestSplit ordinal branch; written to CSV after FindBestSplit returns.
    struct TBinRecord {
        uint32_t featIdx;
        uint32_t bin;
        float  cosNum_f32;
        double cosNum_f64;
        float  cosDen_f32;
        double cosDen_f64;
        float  gain_f32;    // D2-redux: true pre-K4 fp32 gain path: float(cosNum_f32_shadow / sqrtf(cosDen_f32_shadow))
        double gain_f64;    // true fp64 gain path: cosNum_d / sqrt(cosDen_d)
    };
    std::vector<TBinRecord> binRecords;  // populated inside FindBestSplit, flushed after return

#ifdef PROBE_E_INSTRUMENT
    // S33-PROBE-E: per-leaf cosNum/cosDen breakdown for the joint-denominator sum.
    // For each (feat, bin, partition) tuple at depth >= 1, capture the partition
    // contribution under MLX's current "skip-when-degenerate" rule and the CPU-style
    // "mask-each-side-independently" rule. Lets us locate the gain-class divergence
    // for candidates with degenerate child splits (e.g. CPU's d=2 (feat=0, bin=21)
    // pick that produces empty children in 2 of 4 leaves after d=0/d=1 partitioning).
    struct TLeafRecord {
        uint32_t featIdx;
        uint32_t bin;
        uint32_t partition;
        double sumLeft;
        double sumRight;
        double weightLeft;
        double weightRight;
        bool   mlxSkipped;     // true if MLX's `continue` at score_calcer skipped this partition
        double mlxTermNum;     // contribution MLX actually adds (0 if skipped)
        double mlxTermDen;
        double cpuTermNum;     // contribution CPU's mask-formula would add (per-side independent)
        double cpuTermDen;
    };
    std::vector<TLeafRecord> leafRecords;  // flushed per FindBestSplit call alongside binRecords
#endif  // PROBE_E_INSTRUMENT
};

// Global singleton — zero-overhead when not compiled in
static TCosineResidualInstrument g_cosInstr;

// Write a vector of bin records (cosDen/cosNum) to CSV
static void WriteCosAccumCSV(const std::string& path,
                              const std::vector<TCosineResidualInstrument::TBinRecord>& rows) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[INSTR] ERROR: cannot open %s\n", path.c_str()); return; }
    fprintf(f, "featIdx,bin,cosNum_f32,cosNum_f64,cosNum_abs_residual,"
               "cosDen_f32,cosDen_f64,cosDen_abs_residual,"
               "gain_f32,gain_f64,gain_abs_residual\n");
    for (const auto& r : rows) {
        double numRes  = std::fabs(static_cast<double>(r.cosNum_f32) - r.cosNum_f64);
        double denRes  = std::fabs(static_cast<double>(r.cosDen_f32) - r.cosDen_f64);
        double gainRes = std::fabs(static_cast<double>(r.gain_f32)   - r.gain_f64);
        fprintf(f, "%u,%u,%.10g,%.15g,%.6e,%.10g,%.15g,%.6e,%.10g,%.15g,%.6e\n",
                r.featIdx, r.bin,
                static_cast<double>(r.cosNum_f32), r.cosNum_f64, numRes,
                static_cast<double>(r.cosDen_f32), r.cosDen_f64, denRes,
                static_cast<double>(r.gain_f32),   r.gain_f64,   gainRes);
    }
    fclose(f);
    fprintf(stderr, "[INSTR] wrote %zu rows → %s\n", rows.size(), path.c_str());
}

#ifdef PROBE_E_INSTRUMENT
// S33-PROBE-E: write per-leaf records (one row per (feat, bin, partition))
static void WriteLeafRecordsCSV(const std::string& path,
                                 const std::vector<TCosineResidualInstrument::TLeafRecord>& rows) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[PROBE-E] ERROR: cannot open %s\n", path.c_str()); return; }
    fprintf(f, "featIdx,bin,partition,sumLeft,sumRight,weightLeft,weightRight,"
               "mlx_skipped,mlx_termNum,mlx_termDen,"
               "cpu_termNum,cpu_termDen,"
               "term_num_diff,term_den_diff\n");
    for (const auto& r : rows) {
        double dN = r.cpuTermNum - r.mlxTermNum;
        double dD = r.cpuTermDen - r.mlxTermDen;
        fprintf(f, "%u,%u,%u,%.10g,%.10g,%.10g,%.10g,%d,%.15g,%.15g,%.15g,%.15g,%.6e,%.6e\n",
                r.featIdx, r.bin, r.partition,
                r.sumLeft, r.sumRight, r.weightLeft, r.weightRight,
                r.mlxSkipped ? 1 : 0,
                r.mlxTermNum, r.mlxTermDen,
                r.cpuTermNum, r.cpuTermDen,
                dN, dD);
    }
    fclose(f);
    fprintf(stderr, "[PROBE-E] wrote %zu rows → %s\n", rows.size(), path.c_str());
}
#endif  // PROBE_E_INSTRUMENT

#ifdef COSINE_D2_INSTRUMENT
// S30-D2-INSTRUMENT: write gain_scalar CSV — subset of cos_accum with only gain columns.
// Schema matches task #101 spec: feat_idx, bin, gain_f32, gain_f64, gain_abs_residual.
// This is the M1 (L3) measurement: post-cast gain scalar residual for every split candidate.
static void WriteGainScalarCSV(const std::string& path,
                                const std::vector<TCosineResidualInstrument::TBinRecord>& rows) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[D2] ERROR: cannot open %s\n", path.c_str()); return; }
    fprintf(f, "feat_idx,bin,gain_f32,gain_f64,gain_abs_residual\n");
    for (const auto& r : rows) {
        double gainRes = std::fabs(static_cast<double>(r.gain_f32) - r.gain_f64);
        fprintf(f, "%u,%u,%.10g,%.15g,%.6e\n",
                r.featIdx, r.bin,
                static_cast<double>(r.gain_f32), r.gain_f64, gainRes);
    }
    fclose(f);
    fprintf(stderr, "[D2] wrote %zu rows → %s\n", rows.size(), path.c_str());
}
#endif  // COSINE_D2_INSTRUMENT

// Summarise residuals to stderr for quick gate evaluation
static void PrintResidualSummary(const char* label,
                                  const std::vector<double>& residuals) {
    if (residuals.empty()) { fprintf(stderr, "[INSTR] %s: (empty)\n", label); return; }
    double maxR = 0, sumR = 0;
    for (double r : residuals) { if (r > maxR) maxR = r; sumR += r; }
    // 99th percentile
    std::vector<double> sorted = residuals;
    std::sort(sorted.begin(), sorted.end());
    double p99 = sorted[static_cast<size_t>(0.99 * (sorted.size() - 1))];
    fprintf(stderr, "[INSTR] %s: n=%zu  max=%.4e  mean=%.4e  p99=%.4e\n",
            label, residuals.size(), maxR, sumR / residuals.size(), p99);
}
#endif  // COSINE_RESIDUAL_INSTRUMENT

// ============================================================================
// S31-T3b-T1-AUDIT: iter=1 split-selection comparison harness instrumentation
//
// Compile-time gate: -DITER1_AUDIT
// Purpose: for each depth layer of iter=0 (1-indexed iter=1), dump from MLX:
//   - Per-partition parent aggregates: (partIdx, sumG, sumH, W, leafCount)
//   - Top-K=5 split candidates: (featIdx, binIdx, gain) in descending order
//   - Winning split: (featIdx, binIdx, gain)
// Output: JSON file per seed in the directory given by ITER1_AUDIT_OUTDIR env var.
//         Default: docs/sprint31/t3b-audit/data/
//
// Hard-stop rule: only instruments iter==0. Deeper iters see stale leaf assignments
// and their comparisons are not meaningful.
//
// DEC-012 note: this block is diagnostic evidence, not product code.
// ============================================================================
#ifdef ITER1_AUDIT
#ifndef COSINE_RESIDUAL_INSTRUMENT
#  include <filesystem>
#endif
#include <algorithm>
#include <tuple>

struct TIter1AuditState {
    std::string outDir;
    int seed = 0;
    bool active = false;  // true when iter==0, ST, Cosine

    // Written per depth: one entry per depth level
    struct TLayerRecord {
        int depth = 0;
        // Parent aggregates: one entry per active partition
        struct TPartStat {
            uint32_t partIdx;
            double sumG;
            double sumH;
            double W;
            uint64_t leafCount;
        };
        std::vector<TPartStat> partStats;
        // Top-K=5 candidates: sorted descending by gain
        struct TCandidate {
            uint32_t featIdx;
            uint32_t binIdx;
            double gain;
        };
        std::vector<TCandidate> topK;
        // Winner
        uint32_t winnerFeat = 0;
        uint32_t winnerBin  = 0;
        double   winnerGain = 0.0;
    };
    std::vector<TLayerRecord> layers;

    // Pending layer: filled during FindBestSplit, flushed after the call returns
    TLayerRecord pendingLayer;
    bool pendingActive = false;

    // Top-K candidate accumulator inside FindBestSplit
    // Each entry: (gain, featIdx, binIdx) — use gain as primary sort key
    std::vector<std::tuple<double, uint32_t, uint32_t>> candidateAccum;
    static constexpr int TOP_K = 5;
};

static TIter1AuditState g_iter1Audit;

// Serialise all recorded layers to JSON file for the current seed.
static void WriteIter1AuditJSON() {
    if (g_iter1Audit.layers.empty()) return;
    std::string path = g_iter1Audit.outDir
                     + "/mlx_splits_seed" + std::to_string(g_iter1Audit.seed) + ".json";
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());

    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "[AUDIT] ERROR: cannot open %s\n", path.c_str());
        return;
    }
    fprintf(f, "{\n  \"seed\": %d,\n  \"layers\": [\n", g_iter1Audit.seed);
    for (size_t li = 0; li < g_iter1Audit.layers.size(); ++li) {
        const auto& layer = g_iter1Audit.layers[li];
        fprintf(f, "    {\n      \"depth\": %d,\n", layer.depth);
        // partitions
        fprintf(f, "      \"partitions\": [\n");
        for (size_t pi = 0; pi < layer.partStats.size(); ++pi) {
            const auto& ps = layer.partStats[pi];
            fprintf(f, "        {\"leaf_idx\": %u, \"sumG\": %.15g, "
                       "\"sumH\": %.15g, \"W\": %.15g, \"leaf_count\": %llu}%s\n",
                    ps.partIdx, ps.sumG, ps.sumH, ps.W,
                    (unsigned long long)ps.leafCount,
                    (pi + 1 < layer.partStats.size()) ? "," : "");
        }
        fprintf(f, "      ],\n");
        // top5
        fprintf(f, "      \"top5\": [\n");
        for (size_t ci = 0; ci < layer.topK.size(); ++ci) {
            const auto& c = layer.topK[ci];
            fprintf(f, "        {\"feat_idx\": %u, \"bin_idx\": %u, \"gain\": %.15g}%s\n",
                    c.featIdx, c.binIdx, c.gain,
                    (ci + 1 < layer.topK.size()) ? "," : "");
        }
        fprintf(f, "      ],\n");
        // winner
        fprintf(f, "      \"winner\": {\"feat_idx\": %u, \"bin_idx\": %u, \"gain\": %.15g}\n",
                layer.winnerFeat, layer.winnerBin, layer.winnerGain);
        fprintf(f, "    }%s\n", (li + 1 < g_iter1Audit.layers.size()) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[AUDIT] wrote %zu layers → %s\n",
            g_iter1Audit.layers.size(), path.c_str());
}
#endif  // ITER1_AUDIT

// ============================================================================
// S32-T2-INSTRUMENT: Cosine term-level audit — per-bin (feat, bin) dump
//
// Compile-time gate: -DCOSINE_TERM_AUDIT
// Purpose: at depth=0 iter=0 for SymmetricTree+Cosine, dump per-bin:
//   (feat, bin, sumLeft, sumRight, weightLeft, weightRight, lambda,
//    cosNum_d, cosDen_d, gain)
// This is the term-level dump needed to compare against CPU CatBoost's
// per-bin gain computation at the same (feat, bin) coordinate and identify
// the first diverging quantity (gradient, weight, cosNum, cosDen, or gain).
//
// Output: CSV at COSINE_TERM_AUDIT_OUTDIR or default docs/sprint32/t2-terms/data/
// One file per seed: mlx_terms_seed<N>_depth0.csv
//
// At depth=0 with N=50k RMSE SymmetricTree: numPartitions=1, K=1.
// Each row corresponds to one (feat, bin) candidate, capturing the single
// (p=0, k=0) term that comprises the entire Cosine accumulation.
//
// DEC-012 note: diagnostic evidence only; zero product-path behavior change.
// ============================================================================
#ifdef COSINE_TERM_AUDIT
#ifndef COSINE_RESIDUAL_INSTRUMENT
#  include <filesystem>
#endif

struct TCosineTermAuditState {
    std::string outDir;
    int seed = 0;
    bool active = false;    // set true at depth=0, iter=0 arm site
    bool flushed = false;   // prevent double-write

    struct TTermRecord {
        uint32_t featIdx;
        uint32_t bin;
        double sumLeft;
        double sumRight;
        double weightLeft;
        double weightRight;
        double lambda;
        double cosNumTerm;  // incremental contribution to cosNum_d for this (feat,bin) sum
        double cosDenTerm;  // incremental contribution to cosDen_d for this (feat,bin) sum
        double gain;        // finalised cosNum / sqrt(cosDen) after all p,k
    };
    std::vector<TTermRecord> records;

    // Per-bin accumulators filled inside the p,k loop then finalised
    // Only meaningful while active == true
    uint32_t curFeat  = 0;
    uint32_t curBin   = 0;
    double   curCosNum = 0.0;
    double   curCosDen = 0.0;
    double   curSumL  = 0.0;
    double   curSumR  = 0.0;
    double   curWL    = 0.0;
    double   curWR    = 0.0;
    double   curLambda = 0.0;
};

static TCosineTermAuditState g_termAudit;

static void WriteCosineTermCSV(const std::string& path,
                                const std::vector<TCosineTermAuditState::TTermRecord>& rows) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "[TERM_AUDIT] ERROR: cannot open %s\n", path.c_str()); return; }
    fprintf(f, "feat,bin,sumLeft,sumRight,weightLeft,weightRight,lambda,"
               "cosNum_term,cosDen_term,gain\n");
    for (const auto& r : rows) {
        fprintf(f, "%u,%u,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n",
                r.featIdx, r.bin,
                r.sumLeft, r.sumRight,
                r.weightLeft, r.weightRight,
                r.lambda,
                r.cosNumTerm, r.cosDenTerm,
                r.gain);
    }
    fclose(f);
    fprintf(stderr, "[TERM_AUDIT] wrote %zu rows -> %s\n", rows.size(), path.c_str());
}
#endif  // COSINE_TERM_AUDIT

namespace mx = mlx::core;
using ui32 = uint32_t;

// ============================================================================
// Structures (mirrored from gpu_structures.h for standalone compilation)
// ============================================================================

struct TCFeature {
    uint64_t Offset;
    ui32 Mask;
    ui32 Shift;
    ui32 FirstFoldIndex;
    ui32 Folds;
    bool OneHotFeature;
    bool SkipFirstBinInScoreCount;
};

struct TBestSplitProperties {
    ui32 FeatureId = static_cast<ui32>(-1);
    ui32 BinId = 0;
    float Score = 1e30f;
    // Fix 2 (S30-COSINE-KAHAN): widened from float to double.
    // CPU CatBoost holds bestGain / Gain as double throughout FindBestSplit.
    // With gain ≈ 104 the fp32 ULP is ~1.2e-5; the top-10 bins cluster within
    // ~0.006, so the fp32 argmax is underdetermined.  double reduces the ULP
    // to ~2e-14, making the argmax deterministic and CPU-matching.
    double Gain = 1e30;
    bool Defined() const { return FeatureId != static_cast<ui32>(-1); }
};

struct TObliviousSplitLevel {
    ui32 FeatureColumnIdx;
    ui32 Shift;
    ui32 Mask;
    ui32 BinThreshold;
    bool IsOneHot = false;
};

struct TPartitionStatistics {
    double Sum = 0.0;
    double Weight = 0.0;
    double Count = 0.0;
};

// ============================================================================
// CLI argument parsing
// ============================================================================

struct TConfig {
    std::string CsvPath;
    ui32 NumIterations = 100;
    ui32 MaxDepth = 6;
    float LearningRate = 0.1f;
    float L2RegLambda = 3.0f;
    ui32 MaxBins = 255;
    int TargetCol = -1;  // -1 = last column
    std::string LossType = "auto";
    bool Verbose = false;
    std::unordered_set<int> CatFeatureCols;  // 0-based column indices for categorical features
    // Early stopping
    float EvalFraction = 0.0f;         // fraction of data for validation (0 = disabled)
    ui32 EarlyStoppingPatience = 0;    // 0 = disabled
    // Subsampling
    float SubsampleRatio = 1.0f;       // row subsampling fraction per iteration
    float ColsampleByTree = 1.0f;      // feature subsampling fraction per tree
    ui32 RandomSeed = 42;
    // Bootstrap
    std::string BootstrapType = "no";  // "no", "bayesian", "bernoulli", "mvs"
    float BaggingTemperature = 1.0f;   // Bayesian bootstrap temperature
    float MvsReg = 0.0f;              // MVS regularization
    // Ranking / groups
    int GroupCol = -1;                 // -1 = no group column; 0-based column index for group/query ID
    // Sample weights
    int WeightCol = -1;                // -1 = no weight column; 0-based column index for sample weights
    // NaN handling
    std::string NanMode = "min";       // "min" = NaN → bin 0, "forbidden" = error on NaN
    // Model output
    std::string OutputModelPath;       // "" = no save
    bool ShowFeatureImportance = false; // print gain-based feature importance
    // Cross-validation
    ui32 CVFolds = 0;                  // 0 = disabled, N = N-fold cross-validation
    // CTR target encoding
    bool UseCtr = false;               // enable CTR for high-cardinality categoricals
    float CtrPrior = 0.5f;            // Bayesian smoothing prior
    ui32 MaxOneHotSize = 10;           // categoricals with > this many values use CTR instead of OneHot
    // Regularization
    ui32 MinDataInLeaf = 1;            // minimum documents per leaf (1 = no restriction)
    float RandomStrength = 1.0f;       // random score perturbation strength (0 = disabled)
    // Monotone constraints
    std::vector<int> MonotoneConstraints;  // per-feature: 0=none, 1=increasing, -1=decreasing
    // Snapshot save/resume
    std::string SnapshotPath;              // "" = disabled
    ui32 SnapshotInterval = 1;             // save snapshot every N iterations
    // External eval file
    std::string EvalFile;                  // "" = disabled; path to separate validation CSV
    // Grow policy
    std::string GrowPolicy = "SymmetricTree";  // "SymmetricTree" (default), "Depthwise", or "Lossguide"
    ui32 MaxLeaves = 31;                        // used only when GrowPolicy=="Lossguide"
    // Score function (DW / LG paths — FindBestSplitPerPartition only)
    // Accepted values: "L2" (default), "Cosine"
    // "NewtonL2" and "NewtonCosine" are reserved for a future sprint and will
    // throw at startup if passed.  Default "L2" preserves pre-S28 behaviour.
    // CPU CatBoost DW default is "Cosine"; MLX default remains "L2" to avoid
    // silently changing results for existing callers who have not set score_function.
    std::string ScoreFunction = "L2";
};

TConfig ParseArgs(int argc, char** argv) {
    TConfig config;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.csv> [--iterations N] [--depth D] [--lr RATE] "
                "[--l2 LAMBDA] [--loss TYPE] [--bins B] [--target-col N] [--cat-features L] "
                "[--eval-fraction F] [--early-stopping N] [--subsample F] [--colsample-bytree F] "
                "[--seed N] [--nan-mode MODE] [--output PATH] [--feature-importance] "
                "[--cv N] [--ctr] [--ctr-prior F] [--max-onehot-size N] [--group-col N] "
                "[--weight-col N] [--min-data-in-leaf N] [--monotone-constraints L] "
                "[--snapshot-path PATH] [--snapshot-interval N] [--eval-file PATH] "
                "[--score-function {L2,Cosine}] [--verbose]\n", argv[0]);
        exit(1);
    }
    config.CsvPath = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) config.NumIterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) config.MaxDepth = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) config.LearningRate = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--l2") == 0 && i + 1 < argc) config.L2RegLambda = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--loss") == 0 && i + 1 < argc) config.LossType = argv[++i];
        else if (strcmp(argv[i], "--bins") == 0 && i + 1 < argc) config.MaxBins = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--target-col") == 0 && i + 1 < argc) config.TargetCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--cat-features") == 0 && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                config.CatFeatureCols.insert(std::atoi(token.c_str()));
            }
        }
        else if (strcmp(argv[i], "--eval-fraction") == 0 && i + 1 < argc) config.EvalFraction = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--early-stopping") == 0 && i + 1 < argc) config.EarlyStoppingPatience = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--subsample") == 0 && i + 1 < argc) config.SubsampleRatio = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--colsample-bytree") == 0 && i + 1 < argc) config.ColsampleByTree = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) config.RandomSeed = std::atoll(argv[++i]);
        else if (strcmp(argv[i], "--bootstrap-type") == 0 && i + 1 < argc) config.BootstrapType = argv[++i];
        else if (strcmp(argv[i], "--bagging-temperature") == 0 && i + 1 < argc) config.BaggingTemperature = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--mvs-reg") == 0 && i + 1 < argc) config.MvsReg = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--nan-mode") == 0 && i + 1 < argc) config.NanMode = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) config.OutputModelPath = argv[++i];
        else if (strcmp(argv[i], "--feature-importance") == 0) config.ShowFeatureImportance = true;
        else if (strcmp(argv[i], "--cv") == 0 && i + 1 < argc) config.CVFolds = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctr") == 0) config.UseCtr = true;
        else if (strcmp(argv[i], "--ctr-prior") == 0 && i + 1 < argc) config.CtrPrior = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--max-onehot-size") == 0 && i + 1 < argc) config.MaxOneHotSize = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--group-col") == 0 && i + 1 < argc) config.GroupCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--weight-col") == 0 && i + 1 < argc) config.WeightCol = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-data-in-leaf") == 0 && i + 1 < argc) config.MinDataInLeaf = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--random-strength") == 0 && i + 1 < argc) config.RandomStrength = std::atof(argv[++i]);
        else if (strcmp(argv[i], "--monotone-constraints") == 0 && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                config.MonotoneConstraints.push_back(std::atoi(token.c_str()));
            }
        }
        else if (strcmp(argv[i], "--snapshot-path") == 0 && i + 1 < argc) config.SnapshotPath = argv[++i];
        else if (strcmp(argv[i], "--snapshot-interval") == 0 && i + 1 < argc) config.SnapshotInterval = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--eval-file") == 0 && i + 1 < argc) config.EvalFile = argv[++i];
        else if (strcmp(argv[i], "--grow-policy") == 0 && i + 1 < argc) config.GrowPolicy = argv[++i];
        else if (strcmp(argv[i], "--max-leaves") == 0 && i + 1 < argc) config.MaxLeaves = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--score-function") == 0 && i + 1 < argc) config.ScoreFunction = argv[++i];
        else if (strcmp(argv[i], "--verbose") == 0) config.Verbose = true;
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); exit(1); }
    }
    return config;
}

// ============================================================================
// CSV loading
// ============================================================================

struct TDataset {
    std::vector<std::vector<float>> Features;  // [numFeatures][numDocs] — float values (or hash indices for categorical)
    std::vector<float> Targets;                 // [numDocs]
    ui32 NumDocs = 0;
    ui32 NumFeatures = 0;
    std::vector<std::string> FeatureNames;
    std::vector<bool> IsCategorical;            // [numFeatures] — true for categorical columns
    std::vector<std::unordered_map<std::string, uint32_t>> CatHashMaps;  // [numFeatures] — string → bin index (empty for numeric)
    std::vector<bool> HasNaN;                   // [numFeatures] — true if any value is NaN
    // Group/query data (for ranking losses)
    std::vector<ui32> GroupIds;                  // [numDocs] — integer group ID per doc
    std::vector<ui32> GroupOffsets;              // [numGroups + 1] — prefix sum; group g spans [GroupOffsets[g], GroupOffsets[g+1])
    ui32 NumGroups = 0;
    // Sample weights
    std::vector<float> Weights;                  // [numDocs] — per-sample weights (empty = uniform)
};

bool IsNumber(const std::string& s) {
    if (s.empty()) return false;
    // NaN/missing value markers are "numeric-ish" — they'll become NaN during parsing
    if (s == "NaN" || s == "nan" || s == "NA" || s == "na" || s == "N/A" || s == "?") return true;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

bool IsNaNString(const std::string& s) {
    return s.empty() || s == "NaN" || s == "nan" || s == "NA" || s == "na" || s == "N/A" || s == "?";
}

TDataset LoadCSV(const std::string& path, int targetCol, const std::unordered_set<int>& catFeatureCols,
                  const std::string& nanMode, int groupCol = -1, int weightCol = -1) {
    TDataset ds;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        exit(1);
    }

    std::string line;
    std::vector<std::vector<std::string>> rawRows;  // store all cells as strings first
    bool hasHeader = false;
    int numCols = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            size_t start = cell.find_first_not_of(" \t");
            size_t end = cell.find_last_not_of(" \t");
            if (start != std::string::npos) cell = cell.substr(start, end - start + 1);
            else cell = "";
            cells.push_back(cell);
        }

        if (numCols == 0) {
            numCols = cells.size();
            if (!cells.empty() && !IsNumber(cells[0])) {
                hasHeader = true;
                ds.FeatureNames = cells;
                continue;
            }
        }

        if (static_cast<int>(cells.size()) != numCols) continue;
        rawRows.push_back(cells);
    }

    if (rawRows.empty()) {
        fprintf(stderr, "Error: No valid data rows in %s\n", path.c_str());
        exit(1);
    }

    ds.NumDocs = rawRows.size();
    int tgtCol = (targetCol >= 0) ? targetCol : (numCols - 1);
    if (tgtCol < 0 || tgtCol >= numCols) {
        fprintf(stderr, "Error: Invalid target column %d (data has %d columns)\n", tgtCol, numCols);
        exit(1);
    }

    // Determine which feature columns are categorical
    // Exclude target, group, and weight columns from features
    ui32 excludedCols = 1 + (groupCol >= 0 ? 1 : 0) + (weightCol >= 0 ? 1 : 0);
    ds.NumFeatures = numCols - excludedCols;
    ds.IsCategorical.resize(ds.NumFeatures, false);

    // Fix feature names: exclude the target column, group column, and weight column
    if (hasHeader && !ds.FeatureNames.empty()) {
        std::vector<std::string> filteredNames;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            if (c < static_cast<int>(ds.FeatureNames.size()))
                filteredNames.push_back(ds.FeatureNames[c]);
        }
        ds.FeatureNames = std::move(filteredNames);
    }

    // Map from original column index (excluding target, group, and weight) to feature index
    std::vector<int> colToFeatIdx(numCols, -1);
    {
        ui32 fIdx = 0;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            colToFeatIdx[c] = fIdx;

            // Check if this column was explicitly marked as categorical
            if (catFeatureCols.count(c)) {
                ds.IsCategorical[fIdx] = true;
            }
            fIdx++;
        }
    }

    // Auto-detect: if not explicitly specified, check if any cell in a non-target column is non-numeric
    // Empty cells and NaN markers (NaN, NA, na, N/A, ?) are treated as missing values, not categorical
    if (catFeatureCols.empty()) {
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            int fIdx = colToFeatIdx[c];
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                const std::string& val = rawRows[d][c];
                if (!IsNumber(val) && !IsNaNString(val)) {
                    ds.IsCategorical[fIdx] = true;
                    break;
                }
            }
        }
    }

    // Build hash maps for categorical features, parse numeric features
    ds.Features.resize(ds.NumFeatures);
    ds.CatHashMaps.resize(ds.NumFeatures);
    ds.HasNaN.resize(ds.NumFeatures, false);
    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        ds.Features[f].resize(ds.NumDocs);
    }
    ds.Targets.resize(ds.NumDocs);

    // Parse group column strings → integer group IDs
    std::unordered_map<std::string, uint32_t> groupStringToId;
    if (groupCol >= 0) {
        ds.GroupIds.resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            const std::string& val = rawRows[d][groupCol];
            auto it = groupStringToId.find(val);
            if (it == groupStringToId.end()) {
                uint32_t newId = groupStringToId.size();
                groupStringToId[val] = newId;
                ds.GroupIds[d] = newId;
            } else {
                ds.GroupIds[d] = it->second;
            }
        }
    }

    // Parse weight column if present
    if (weightCol >= 0) {
        ds.Weights.resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            try {
                ds.Weights[d] = std::stof(rawRows[d][weightCol]);
            } catch (...) {
                fprintf(stderr, "Error: Non-numeric weight at row %u: '%s'\n", d, rawRows[d][weightCol].c_str());
                exit(1);
            }
        }
    }

    for (ui32 d = 0; d < ds.NumDocs; ++d) {
        // Parse target (must be numeric)
        try {
            ds.Targets[d] = std::stof(rawRows[d][tgtCol]);
        } catch (...) {
            fprintf(stderr, "Error: Non-numeric target at row %u: '%s'\n", d, rawRows[d][tgtCol].c_str());
            exit(1);
        }

        ui32 fIdx = 0;
        for (int c = 0; c < numCols; ++c) {
            if (c == tgtCol) continue;
            if (c == groupCol) continue;
            if (c == weightCol) continue;
            if (ds.IsCategorical[fIdx]) {
                // Categorical: hash string → sequential integer
                const std::string& val = rawRows[d][c];
                auto& hashMap = ds.CatHashMaps[fIdx];
                auto it = hashMap.find(val);
                if (it == hashMap.end()) {
                    uint32_t newIdx = hashMap.size();
                    hashMap[val] = newIdx;
                    ds.Features[fIdx][d] = static_cast<float>(newIdx);
                } else {
                    ds.Features[fIdx][d] = static_cast<float>(it->second);
                }
            } else {
                // Numeric — handle NaN/missing values
                const std::string& val = rawRows[d][c];
                if (IsNaNString(val)) {
                    if (nanMode == "forbidden") {
                        fprintf(stderr, "Error: NaN/missing value at row %u, col %d: '%s' (--nan-mode=forbidden)\n",
                                d, c, val.c_str());
                        exit(1);
                    }
                    ds.Features[fIdx][d] = std::numeric_limits<float>::quiet_NaN();
                    ds.HasNaN[fIdx] = true;
                } else {
                    try {
                        ds.Features[fIdx][d] = std::stof(val);
                        if (std::isnan(ds.Features[fIdx][d])) {
                            ds.HasNaN[fIdx] = true;
                        }
                    } catch (...) {
                        if (nanMode == "forbidden") {
                            fprintf(stderr, "Error: Non-numeric value at row %u, col %d: '%s'\n", d, c, val.c_str());
                            exit(1);
                        }
                        ds.Features[fIdx][d] = std::numeric_limits<float>::quiet_NaN();
                        ds.HasNaN[fIdx] = true;
                    }
                }
            }
            fIdx++;
        }
    }

    // Sort data by group ID and build GroupOffsets
    if (groupCol >= 0 && !ds.GroupIds.empty()) {
        // Create a permutation that sorts by group ID (stable sort preserves order within groups)
        std::vector<ui32> perm(ds.NumDocs);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(),
            [&](ui32 a, ui32 b) { return ds.GroupIds[a] < ds.GroupIds[b]; });

        // Apply permutation to all data
        auto permute = [&](auto& vec) {
            auto tmp = vec;
            for (ui32 i = 0; i < ds.NumDocs; ++i) vec[i] = tmp[perm[i]];
        };
        permute(ds.Targets);
        permute(ds.GroupIds);
        if (!ds.Weights.empty()) permute(ds.Weights);
        for (ui32 f = 0; f < ds.NumFeatures; ++f) {
            permute(ds.Features[f]);
        }

        // Build GroupOffsets prefix sum
        ds.NumGroups = 0;
        ds.GroupOffsets.push_back(0);
        for (ui32 d = 1; d < ds.NumDocs; ++d) {
            if (ds.GroupIds[d] != ds.GroupIds[d - 1]) {
                ds.GroupOffsets.push_back(d);
                ds.NumGroups++;
            }
        }
        ds.GroupOffsets.push_back(ds.NumDocs);
        ds.NumGroups++;
    }

    return ds;
}

// ============================================================================
// Binary format loader (CBMX)
// ============================================================================
// Header: "CBMX"(4) version(4) n_samples(4) n_features(4) flags(4) = 20 bytes
// Data:   float32[n_features][n_samples] (column-major)
//         float32[n_samples] target (if flag bit 0)
//         float32[n_samples] weight (if flag bit 2)
//         uint32[n_samples]  group_id (if flag bit 1)

bool IsBinaryFormat(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4] = {0};
    size_t rd = fread(magic, 1, 4, f);
    fclose(f);
    return rd == 4 && memcmp(magic, "CBMX", 4) == 0;
}

TDataset LoadBinary(const std::string& path, const std::string& nanMode) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        exit(1);
    }

    char magic[4];
    uint32_t version, n, nf, flags;
    fread(magic, 1, 4, f);
    fread(&version, 4, 1, f);
    fread(&n, 4, 1, f);
    fread(&nf, 4, 1, f);
    fread(&flags, 4, 1, f);

    if (version != 1) {
        fprintf(stderr, "Error: Unsupported CBMX version %u\n", version);
        fclose(f); exit(1);
    }

    bool hasTarget = flags & 1;
    bool hasGroup = flags & 2;
    bool hasWeight = flags & 4;

    TDataset ds;
    ds.NumDocs = n;
    ds.NumFeatures = nf;
    ds.Features.resize(nf);
    ds.HasNaN.resize(nf, false);
    ds.IsCategorical.resize(nf, false);
    ds.CatHashMaps.resize(nf);

    // Read features (column-major: each feature column contiguous)
    for (uint32_t fi = 0; fi < nf; fi++) {
        ds.Features[fi].resize(n);
        size_t rd = fread(ds.Features[fi].data(), sizeof(float), n, f);
        if (rd != n) {
            fprintf(stderr, "Error: Truncated binary file at feature %u\n", fi);
            fclose(f); exit(1);
        }
        for (uint32_t d = 0; d < n; d++) {
            if (std::isnan(ds.Features[fi][d])) {
                if (nanMode == "forbidden") {
                    fprintf(stderr, "Error: NaN at feature %u, row %u (--nan-mode=forbidden)\n", fi, d);
                    fclose(f); exit(1);
                }
                ds.HasNaN[fi] = true;
            }
        }
    }

    if (hasTarget) {
        ds.Targets.resize(n);
        fread(ds.Targets.data(), sizeof(float), n, f);
    }
    if (hasWeight) {
        ds.Weights.resize(n);
        fread(ds.Weights.data(), sizeof(float), n, f);
    }
    if (hasGroup) {
        ds.GroupIds.resize(n);
        fread(ds.GroupIds.data(), sizeof(uint32_t), n, f);

        // Sort by group and build GroupOffsets (same as LoadCSV)
        std::vector<ui32> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(),
            [&](ui32 a, ui32 b) { return ds.GroupIds[a] < ds.GroupIds[b]; });
        auto permute = [&](auto& vec) {
            auto tmp = vec;
            for (ui32 i = 0; i < n; ++i) vec[i] = tmp[perm[i]];
        };
        permute(ds.Targets);
        permute(ds.GroupIds);
        if (!ds.Weights.empty()) permute(ds.Weights);
        for (uint32_t fi = 0; fi < nf; fi++) permute(ds.Features[fi]);

        ds.NumGroups = 0;
        ds.GroupOffsets.push_back(0);
        for (uint32_t d = 1; d < n; ++d) {
            if (ds.GroupIds[d] != ds.GroupIds[d - 1]) {
                ds.GroupOffsets.push_back(d);
                ds.NumGroups++;
            }
        }
        ds.GroupOffsets.push_back(n);
        ds.NumGroups++;
    }

    fclose(f);

    // Generate feature names
    ds.FeatureNames.resize(nf);
    for (uint32_t fi = 0; fi < nf; fi++)
        ds.FeatureNames[fi] = "f" + std::to_string(fi);

    return ds;
}

// ============================================================================
// Quantization — GreedyLogSum border selection
//
// S31-T2-PORT-GREEDYLOGSUM: Port of CatBoost's GreedyLogSum border algorithm
// from library/cpp/grid_creator/binarization.cpp (TGreedyBinarizer<MaxSumLog>).
//
// Algorithm (unweighted path — no DefaultValue):
//   1. Sort feature values retaining duplicates (all N docs).
//   2. Build TFeatureBin over the full sorted array — BinEnd-BinStart = document count.
//   3. Greedily split the bin with the highest log-gain in a priority queue.
//   4. Each split's score = log(total) - log(left) - log(right)  [MaxSumLog penalty].
//   5. Border = midpoint of the two adjacent array elements at each split boundary.
//
// DEC-038 fix: prior implementation deduplicated before passing to GreedyLogSumBestSplit,
// making BinEnd-BinStart count unique values instead of documents. This changed the
// penalty score landscape, producing ~2-index border offsets for features with many
// duplicate values (N=50k, continuous float32 → ~50k near-unique → small effect; but
// for features with genuine duplicates it diverged significantly). Fix: pass allVals
// (sorted, with duplicates) so BinEnd-BinStart = document count, matching CPU.
//
// DEC-037 fix: maxBordersCount = maxBins (was maxBins-1; caused MLX_bin = CPU_bin-1).
//
// Reference: TGreedyBinarizer<EPenaltyType::MaxSumLog>::BestSplit (no DefaultValue branch)
//            + TFeatureBin<MaxSumLog>::CalcSplitScore / UpdateBestSplitProperties
//            + GreedySplit priority-queue loop
//            in library/cpp/grid_creator/binarization.cpp.
//
// Arcadia dependencies stripped — uses only std:: containers.
// ============================================================================

namespace {

// MaxSumLog penalty: -log(w + 1e-8).  Gain from splitting [L,R) at s is:
//   CalcSplitScore(s) = -Penalty(s-L) + -Penalty(R-s) - (-Penalty(R-L))
//                     = log(R-L+1e-8) - log(s-L+1e-8) - log(R-s+1e-8)
// Higher is better.
inline double GreedyLogSumPenalty(double w) {
    return -std::log(w + 1e-8);
}

struct TGreedyFeatureBin {
    uint32_t BinStart;
    uint32_t BinEnd;
    const float* Values;   // pointer into sorted all-docs value array (with duplicates)
    uint32_t BestSplit;
    double BestScore;

    TGreedyFeatureBin(uint32_t start, uint32_t end, const float* vals)
        : BinStart(start), BinEnd(end), Values(vals), BestSplit(start), BestScore(0.0)
    {
        UpdateBestSplitProperties();
    }

    bool operator<(const TGreedyFeatureBin& other) const {
        return BestScore < other.BestScore;
    }

    bool CanSplit() const {
        return (BinStart != BestSplit) && (BinEnd != BestSplit);
    }

    bool IsFirst() const {
        return BinStart == 0;
    }

    // Border value: midpoint between Values[BinStart-1] and Values[BinStart].
    float LeftBorder() const {
        if (IsFirst()) return Values[0];
        float borderValue = 0.5f * Values[BinStart - 1];
        borderValue += 0.5f * Values[BinStart];
        return borderValue;
    }

    uint32_t Size() const { return BinEnd - BinStart; }
    double Score() const { return BestScore; }

    // Split this bin at BestSplit: returns the left child; this bin becomes right.
    TGreedyFeatureBin Split() {
        TGreedyFeatureBin left(BinStart, BestSplit, Values);
        BinStart = BestSplit;
        UpdateBestSplitProperties();
        return left;
    }

private:
    double CalcSplitScore(uint32_t splitPos) const {
        if (splitPos == BinStart || splitPos == BinEnd) {
            return -std::numeric_limits<double>::infinity();
        }
        const double leftScore  = -GreedyLogSumPenalty(static_cast<double>(splitPos - BinStart));
        const double rightScore = -GreedyLogSumPenalty(static_cast<double>(BinEnd    - splitPos));
        const double currScore  = -GreedyLogSumPenalty(static_cast<double>(BinEnd    - BinStart));
        return leftScore + rightScore - currScore;
    }

    void UpdateBestSplitProperties() {
        // Mirror TFeatureBin::UpdateBestSplitProperties from binarization.cpp:
        // Find midValue at midpoint of [BinStart, BinEnd), then locate the first
        // occurrence (lb) and one-past-last occurrence (ub) of that midValue.
        // Evaluate score at both positions, pick the better one.
        const uint32_t mid = BinStart + (BinEnd - BinStart) / 2;
        const float midValue = Values[mid];

        const uint32_t lb = static_cast<uint32_t>(
            std::lower_bound(Values + BinStart, Values + mid, midValue) - Values
        );
        const uint32_t ub = static_cast<uint32_t>(
            std::upper_bound(Values + mid, Values + BinEnd, midValue) - Values
        );

        const double scoreLeft  = CalcSplitScore(lb);
        const double scoreRight = CalcSplitScore(ub);

        BestSplit = (scoreLeft >= scoreRight) ? lb : ub;
        BestScore = (BestSplit == lb) ? scoreLeft : scoreRight;
    }
};

// GreedySplit: greedy priority-queue splitter (mirrors GreedySplit from binarization.cpp).
// Takes ownership of initialBin; returns sorted vector of border values.
static std::vector<float> GreedyLogSumBestSplit(
    const std::vector<float>& uniqueValues,
    int maxBordersCount)
{
    if (uniqueValues.size() <= 1 || maxBordersCount <= 0) {
        return {};
    }

    TGreedyFeatureBin initialBin(
        0,
        static_cast<uint32_t>(uniqueValues.size()),
        uniqueValues.data()
    );

    std::priority_queue<TGreedyFeatureBin> splits;
    splits.push(initialBin);

    while (static_cast<int>(splits.size()) <= maxBordersCount && splits.top().CanSplit()) {
        auto top = splits.top();
        splits.pop();
        auto left = top.Split();
        splits.push(left);
        splits.push(top);
    }

    std::vector<float> borders;
    borders.reserve(splits.size() > 0 ? splits.size() - 1 : 0);
    while (!splits.empty()) {
        if (!splits.top().IsFirst()) {
            borders.push_back(splits.top().LeftBorder());
        }
        splits.pop();
    }

    // Sort borders (priority queue ordering is by score, not position).
    std::sort(borders.begin(), borders.end());
    // Remove exact duplicates that can arise when adjacent unique values share a midpoint.
    borders.erase(std::unique(borders.begin(), borders.end()), borders.end());

    return borders;
}

}  // namespace

// ============================================================================
// Quantization
// ============================================================================

struct TQuantization {
    std::vector<std::vector<float>> Borders;    // [numFeatures] → sorted border values
    std::vector<std::vector<uint8_t>> BinnedFeatures;  // [numFeatures][numDocs] bin indices
};

TQuantization QuantizeFeatures(const TDataset& ds, ui32 maxBins) {
    TQuantization q;
    q.Borders.resize(ds.NumFeatures);
    q.BinnedFeatures.resize(ds.NumFeatures);

    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        if (ds.IsCategorical[f]) {
            // Categorical: no borders, bin index = hash map index directly
            q.Borders[f].clear();
            q.BinnedFeatures[f].resize(ds.NumDocs);
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                q.BinnedFeatures[f][d] = static_cast<uint8_t>(ds.Features[f][d]);
            }
            continue;
        }

        // Numeric: GreedyLogSum border selection (CatBoost default EBorderSelectionType).
        // Uses greedy MaxSumLog priority-queue approach (TGreedyBinarizer<MaxSumLog>),
        // unweighted path (TFeatureBin — count of ALL documents, not unique values).
        //
        // DEC-038 fix: CPU's TFeatureBin is built from features.Values (all docs, with
        // duplicates), not from a deduplicated unique-value array. The CalcSplitScore
        // uses BinEnd-BinStart as document counts, so the penalty function receives
        // document counts. Using uniqueVals caused a different score landscape and
        // produced ~2-index border offsets for features with many duplicate values.
        //
        // DEC-037 fix: maxBordersCount = maxBins (was maxBins-1; caused MLX_bin = CPU_bin-1).
        //
        // Step 1: build sorted raw values (non-NaN only), with duplicates retained.
        std::vector<float> allVals;
        allVals.reserve(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            if (!std::isnan(ds.Features[f][d])) {
                allVals.push_back(ds.Features[f][d]);
            }
        }
        if (allVals.empty()) {
            // All NaN — single NaN bin, no borders.
            q.BinnedFeatures[f].resize(ds.NumDocs, 0);
            continue;
        }
        std::sort(allVals.begin(), allVals.end());

        // Step 2: Greedy MaxSumLog binarization — pass allVals (with duplicates) to
        // match CatBoost's TFeatureBin which operates on the full document array.
        // DEC-037 fix: maxBordersCount = maxBins (was maxBins-1, causing bin offset).
        //
        // DEC-039 fix: cap maxBordersCount at 127 to satisfy the histogram kernel's
        // T2_BIN_CAP = 128 (0..127) contract.  With folds ≤ 127 the maximum bin_value
        // is 127, so the packed byte for posInWord=0 features never has bit 7 set
        // (bit 7 at shift=24 = bit 31 of the word = VALID_BIT).  Requesting 128
        // borders produces fold_count=128 and allows bin_value=128 at posInWord=0
        // features; the histogram kernel strips the VALID_BIT on p_clean, aliasing
        // bin_value=128 to bin_value=0 and silently dropping those docs from the
        // suffix sum.  This caused wL to be offset by ~391 for features 0,4,8,12,16
        // (every posInWord=0 feature) despite DEC-038-correct borders.
        // Note: bench_boosting already applies this cap (NumBins-1 for fold count).
        int maxBordersCount = static_cast<int>(std::min(maxBins, 127u));
        std::vector<float> borders = GreedyLogSumBestSplit(allVals, maxBordersCount);
        q.Borders[f] = std::move(borders);

        // Step 3: bin assignment (same policy as before):
        //   NaN → bin 0 (if the feature has any NaN values in the dataset).
        //   Real values → upper_bound into borders, offset by 1 if NaN bin is active.
        bool hasNaN = ds.HasNaN[f];
        ui32 binOffset = hasNaN ? 1 : 0;
        q.BinnedFeatures[f].resize(ds.NumDocs);
        for (ui32 d = 0; d < ds.NumDocs; ++d) {
            if (std::isnan(ds.Features[f][d])) {
                q.BinnedFeatures[f][d] = 0;  // NaN bin
            } else {
                auto it = std::upper_bound(q.Borders[f].begin(), q.Borders[f].end(), ds.Features[f][d]);
                q.BinnedFeatures[f][d] = static_cast<uint8_t>((it - q.Borders[f].begin()) + binOffset);
            }
        }
    }

    return q;
}

// Quantize a new dataset using pre-computed borders from training data
TQuantization QuantizeWithBorders(const TDataset& ds, const TQuantization& trainQuant,
                                  const TDataset& trainDs) {
    TQuantization q;
    q.Borders = trainQuant.Borders;
    q.BinnedFeatures.resize(ds.NumFeatures);

    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        q.BinnedFeatures[f].resize(ds.NumDocs);
        if (ds.IsCategorical[f]) {
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                q.BinnedFeatures[f][d] = static_cast<uint8_t>(ds.Features[f][d]);
            }
        } else {
            bool hasNaN = trainDs.HasNaN[f];  // use training NaN status for consistent bin offsets
            ui32 binOffset = hasNaN ? 1 : 0;
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                if (std::isnan(ds.Features[f][d])) {
                    q.BinnedFeatures[f][d] = 0;
                } else {
                    auto it = std::upper_bound(q.Borders[f].begin(), q.Borders[f].end(), ds.Features[f][d]);
                    q.BinnedFeatures[f][d] = static_cast<uint8_t>((it - q.Borders[f].begin()) + binOffset);
                }
            }
        }
    }
    return q;
}

// ============================================================================
// Feature packing into compressed uint32 format
// ============================================================================

struct TPackedData {
    std::vector<uint32_t> Data;  // [numDocs * numUi32PerDoc]
    std::vector<TCFeature> Features;
    ui32 NumUi32PerDoc;
    ui32 TotalBinFeatures;
};

TPackedData PackFeatures(const TQuantization& q, const TDataset& ds) {
    const ui32 numDocs = ds.NumDocs;
    const ui32 numFeatures = ds.NumFeatures;
    TPackedData packed;
    packed.NumUi32PerDoc = (numFeatures + 3) / 4;
    packed.Data.resize(numDocs * packed.NumUi32PerDoc, 0);
    packed.TotalBinFeatures = 0;

    for (ui32 f = 0; f < numFeatures; ++f) {
        ui32 wordIdx = f / 4;
        ui32 posInWord = f % 4;
        ui32 shift = (3 - posInWord) * 8;
        ui32 mask = 0xFF << shift;

        // For categorical features, folds = number of unique categories
        // For numeric features, folds = number of borders (+ 1 if feature has NaN for the NaN bin)
        ui32 folds;
        if (ds.IsCategorical[f]) {
            folds = static_cast<ui32>(ds.CatHashMaps[f].size());
        } else {
            folds = static_cast<ui32>(q.Borders[f].size());
            if (ds.HasNaN[f]) folds += 1;  // extra bin 0 for NaN
        }

        TCFeature feat;
        feat.Offset = wordIdx;
        feat.Mask = mask;
        feat.Shift = shift;
        feat.FirstFoldIndex = packed.TotalBinFeatures;
        feat.Folds = folds;
        feat.OneHotFeature = ds.IsCategorical[f];
        feat.SkipFirstBinInScoreCount = false;
        packed.Features.push_back(feat);
        packed.TotalBinFeatures += folds;

        // Pack bin values into compressed data
        for (ui32 d = 0; d < numDocs; ++d) {
            packed.Data[d * packed.NumUi32PerDoc + wordIdx] |=
                (static_cast<uint32_t>(q.BinnedFeatures[f][d]) << shift);
        }
    }

    return packed;
}

// ============================================================================
// Loss function auto-detection
// ============================================================================

std::string DetectLossType(const std::vector<float>& targets) {
    // Check if targets are all integers
    bool allInteger = true;
    float minTarget = targets[0], maxTarget = targets[0];
    std::vector<float> uniqueTargets;

    for (float t : targets) {
        if (t != std::floor(t)) allInteger = false;
        minTarget = std::min(minTarget, t);
        maxTarget = std::max(maxTarget, t);
    }

    if (!allInteger) return "rmse";

    // Count unique integer values
    std::vector<float> sorted = targets;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    if (sorted.size() == 2 && minTarget == 0.0f && maxTarget == 1.0f) {
        return "logloss";
    }

    // Check if targets are 0..K-1
    if (minTarget == 0.0f && maxTarget == static_cast<float>(sorted.size() - 1)) {
        bool consecutive = true;
        for (ui32 i = 0; i < sorted.size(); ++i) {
            if (sorted[i] != static_cast<float>(i)) { consecutive = false; break; }
        }
        if (consecutive && sorted.size() > 2) return "multiclass";
        if (consecutive && sorted.size() == 2) return "logloss";
    }

    return "rmse";
}

// ============================================================================
// Partition layout computation
// ============================================================================

struct TPartitionLayout {
    mx::array DocIndices;
    mx::array PartOffsets;
    mx::array PartSizes;
};

TPartitionLayout ComputePartitionLayout(const mx::array& partitions, ui32 numDocs, ui32 numPartitions) {
    // GPU: argsort partitions to get doc indices grouped by partition
    auto docIndices = mx::astype(mx::argsort(partitions), mx::uint32);

    // GPU: count docs per partition via scatter_add_axis
    auto onesF = mx::ones({static_cast<int>(numDocs)}, mx::float32);
    auto partSizesF = mx::scatter_add_axis(
        mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
        partitions, onesF, 0);

    // GPU: exclusive prefix sum for partition offsets
    auto partOffsetsF = mx::subtract(mx::cumsum(partSizesF), partSizesF);

    auto partSizes = mx::astype(partSizesF, mx::uint32);
    auto partOffsets = mx::astype(partOffsetsF, mx::uint32);

    return {docIndices, partOffsets, partSizes};
}

// ============================================================================
// Histogram dispatch
// ============================================================================

mx::array DispatchHistogram(
    const mx::array& compressedData,
    const mx::array& stats,
    const mx::array& docIndices,
    const mx::array& partOffsets,
    const mx::array& partSizes,
    const std::vector<TCFeature>& features,
    ui32 lineSize,
    ui32 totalBinFeatures,
    ui32 numPartitions,
    ui32 numDocs
) {
    const ui32 numStats = 2;
    const ui32 numFeatures = features.size();
    const ui32 numFeatureGroups = (numFeatures + 3) / 4;

    // Compute maxBlocksPerPart: use multiple threadgroups per partition so
    // the GPU can process large partitions in parallel. Each block of 256
    // threads handles a slice of docs; results are combined via device atomics.
    // Scale blocks by average partition size. Smaller blocks (4K docs each)
    // give better GPU occupancy at early depths where 1-2 large partitions
    // dominate. The cap of 8 blocks limits atomic contention.
    const ui32 docsPerBlock = 4096;
    ui32 avgDocsPerPart = numDocs / std::max(1u, numPartitions);
    ui32 maxBlocksPerPart = std::max(1u, (avgDocsPerPart + docsPerBlock - 1) / docsPerBlock);
    maxBlocksPerPart = std::min(maxBlocksPerPart, 8u);

    mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
    mx::array histogram = mx::zeros(histShape, mx::float32);

    // Build flat arrays for all feature groups (single dispatch)
    std::vector<ui32> colIndices(numFeatureGroups);
    std::vector<ui32> foldCountsFlat(numFeatureGroups * 4, 0);
    std::vector<ui32> firstFoldFlat(numFeatureGroups * 4, 0);
    for (ui32 g = 0; g < numFeatureGroups; ++g) {
        colIndices[g] = g;
        const ui32 featureStart = g * 4;
        const ui32 featuresInGroup = std::min(4u, numFeatures - featureStart);
        for (ui32 f = 0; f < featuresInGroup; ++f) {
            foldCountsFlat[g * 4 + f] = features[featureStart + f].Folds;
            firstFoldFlat[g * 4 + f] = features[featureStart + f].FirstFoldIndex;
        }
    }

    auto colIndicesArr = mx::array(reinterpret_cast<const int32_t*>(colIndices.data()),
                                   {static_cast<int>(numFeatureGroups)}, mx::uint32);
    auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsFlat.data()),
                                   {static_cast<int>(numFeatureGroups * 4)}, mx::uint32);
    auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldFlat.data()),
                                  {static_cast<int>(numFeatureGroups * 4)}, mx::uint32);

    using namespace NCatboostMlx;
    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIndices", "lineSize", "maxBlocksPerPart", "numGroups",
         "foldCountsFlat", "firstFoldIndicesFlat",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        KernelSources::kHistOneByteSource,
        KernelSources::kHistHeader,
        true, false
    );

    bool verbose_this_call = false;

    auto result = kernel(
        {mx::reshape(compressedData, {-1}), stats, docIndices,
         partOffsets, partSizes,
         colIndicesArr,
         mx::array(static_cast<uint32_t>(lineSize), mx::uint32),
         mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32),
         mx::array(static_cast<uint32_t>(numFeatureGroups), mx::uint32),
         foldCountsArr, firstFoldArr,
         mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32),
         mx::array(static_cast<uint32_t>(numStats), mx::uint32),
         mx::array(static_cast<uint32_t>(numDocs), mx::uint32)},
        {histShape}, {mx::float32},
        std::make_tuple(static_cast<int>(256 * maxBlocksPerPart * numFeatureGroups),
                        static_cast<int>(numPartitions), 2),
        std::make_tuple(256, 1, 1),
        {}, 0.0f, verbose_this_call, mx::Device::gpu
    );

    histogram = result[0];
    mx::eval(histogram);

    return histogram;
}

// ============================================================================
// Score function enum — mirrors CPU CatBoost EScoreFunction.
// S28-L2-EXPLICIT: dispatch wiring live as of this commit (task #72, DEC-032).
// S28-OBLIV-DISPATCH: moved before FindBestSplit so both Oblivious and DW
// paths share a single definition; previously defined after FindBestSplit.
// Only {L2, Cosine} implemented.  NewtonL2 and NewtonCosine reserved for a
// future sprint; passing them throws a clear error rather than silently
// falling back to L2 (silent divergence is the exact bug DEC-032 closed).
// ============================================================================

enum class EScoreFunction {
    L2,     // L2 Newton gain: G²/(W+λ) — default, preserves pre-S28 behaviour
    Cosine, // Cosine gain: Σ(G²/(W+λ)) / sqrt(Σ(G²·W/(W+λ)²))
            // CPU: TCosineScoreCalcer (catboost/private/libs/algo/score_calcers.h:47)
};

// Parse score_function string to enum.  Throws std::invalid_argument on
// unknown values rather than silently falling back to L2.
inline EScoreFunction ParseScoreFunction(const std::string& s) {
    if (s == "L2")     return EScoreFunction::L2;
    if (s == "Cosine") return EScoreFunction::Cosine;
    // Explicit guard for the two values we know exist in CPU CatBoost but have
    // not yet implemented.  Throw — do NOT silently fall back to L2.
    if (s == "NewtonL2" || s == "NewtonCosine") {
        throw std::invalid_argument(
            "score_function='" + s + "' is not yet implemented in MLX backend. "
            "Supported values: L2, Cosine.");
    }
    throw std::invalid_argument(
        "Unknown score_function='" + s + "'. "
        "Supported values: L2, Cosine.");
}

// ----------------------------------------------------------------------------
// Cosine gain derivation (used by ComputeCosineGainKDim below).
//
// CPU reference: TCosineScoreCalcer::AddLeafPlain + GetScores()
//   score[0] += leafApprox * SumWeightedDelta  (numerator)
//   score[1] += leafApprox² * SumWeight        (denominator accumulator)
//   final score = score[0] / sqrt(score[1])
// where leafApprox = sumGrad / (sumHess + lambda).
//
// Expanded to avoid redundant division:
//   num  = sumLeft²/(wLeft+λ) + sumRight²/(wRight+λ)
//   den  = sumLeft²·wLeft/(wLeft+λ)² + sumRight²·wRight/(wRight+λ)²
//   score = num / sqrt(den)
//
// Notes:
// - No parent-node subtraction (unlike L2). Cosine is an absolute score, not
//   a differential gain. Higher is better; values are not comparable to L2.
// - den guard of 1e-20f prevents sqrt(0) on empty partitions.
//   (CPU uses double 1e-100; 1e-20f is safely above FP32 subnormal range.)
// - K-dim accumulation: sum num/den contributions across all approx dimensions
//   before computing the final score, matching CPU's multi-dimensional
//   AddLeafPlain loop in CalcScores (greedy_tensor_search.cpp).
// - float32 precision: numerator and denominator each accumulate K terms.
//   At K≤3, N≤1000, float32 relative error ≤ ~1e-6 (Higham bound). ULP drift
//   vs CPU double accumulation is measured empirically in t2-gate-report.md.
// ----------------------------------------------------------------------------
// K-dimensional Cosine gain: sum num/den contributions across approx dims,
// then compute score[0]/sqrt(score[1]).  Mirrors CPU's per-dim AddLeafPlain
// loop which accumulates into a single Scores[splitIdx] pair.
inline float ComputeCosineGainKDim(
    float totalNum,   // Σ_k sumLeft_k²/(wLeft_k+λ) + sumRight_k²/(wRight_k+λ)
    float totalDen    // Σ_k (sumLeft_k²·wLeft_k/(wLeft_k+λ)² + ...) + guard
) {
    return totalNum / std::sqrt(totalDen);
}

// ============================================================================
// Split finding (multi-dim aware)
// ============================================================================

TBestSplitProperties FindBestSplit(
    const std::vector<std::vector<float>>& perDimHist,  // [K][numPartitions * 2 * totalBinFeatures]
    const std::vector<std::vector<TPartitionStatistics>>& perDimPartStats,  // [K][numPartitions]
    const std::vector<TCFeature>& features,
    ui32 totalBinFeatures,
    float l2RegLambda,
    ui32 numPartitions,
    const std::vector<bool>& featureMask,       // optional: if non-empty, skip features where mask[f]=false
    ui32 minDataInLeaf,                          // minimum docs per child (1 = no restriction)
    const std::vector<std::vector<ui32>>& countHist,   // [numPartitions][totalBinFeatures] doc counts per bin
    const std::vector<ui32>& partDocCounts,             // [numPartitions] total doc count per partition
    const std::vector<int>& monotoneConstraints,       // per-feature: 0=none, 1=inc, -1=dec
    float randomStrength,                              // random score perturbation (0 = disabled)
    float gradRms,                                     // gradient RMS — noise scale denominator (CPU: CalcDerivativesStDevFromZeroPlainBoosting, greedy_tensor_search.cpp:92)
    std::mt19937* rng,                                 // RNG for perturbation
    int debugIter,                                     // P11-P13: print debug for this iter (-1 = off)
    int debugDepth,                                    // P11-P13: print debug for this depth (-1 = off)
    EScoreFunction scoreFunction                       // S28-OBLIV-DISPATCH: gain formula — no default, every caller must be explicit
) {
    TBestSplitProperties bestSplit;
    const ui32 K = perDimHist.size();
    // Fix 2 (S30-COSINE-KAHAN): widen argmax accumulator to double.
    double bestGain = -std::numeric_limits<double>::infinity();

#ifdef CATBOOST_MLX_DEBUG_LEAF
    // P11/P13: accumulate all candidates for top-10 report at depth=0, iter=0
    struct TDbgCandidate {
        ui32 featIdx; ui32 bin; float gain;
        float sumL; float wL; float sumR; float wR; float sumP; float wP;
    };
    const bool p11Active = (debugIter == 0 && debugDepth == 0);
    std::vector<TDbgCandidate> allCandidates;
#endif

    // Random score perturbation: scale by gradient RMS, matching CPU CatBoost.
    // CPU reference: CalcDerivativesStDevFromZeroPlainBoosting (greedy_tensor_search.cpp:92–106)
    //   returns sqrt( sum_{k,i} g_k[i]^2 / N ), called as:
    //   scoreStDev = randomStrength * derivativesStDevFromZero   (line 861-866)
    // gradRms is pre-computed in RunTraining after gradient+weight application and passed here.
    float noiseScale = 0.0f;
    if (randomStrength > 0.0f && rng && gradRms > 0.0f) {
        noiseScale = randomStrength * gradRms;
    }
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);

    // The Metal histogram kernel produces per-bin sums with a +1 offset:
    //   hist[firstFold + b] = sum of docs where featureValue == b+1
    //
    // For OneHot features: each bin represents one category.
    //   Split "go right if value == bin": sumRight = hist[bin], sumLeft = total - sumRight
    //
    // For ordinal features: precompute suffix sums then use O(1) lookups.
    //   Split threshold b (value > b → right):
    //     sumRight = suffixSum[b] = sum(hist[b..folds-1])
    //     sumLeft = total - sumRight

    for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
        // Feature subsampling: skip features not in the selected set
        if (!featureMask.empty() && !featureMask[featIdx]) continue;
        const auto& feat = features[featIdx];

        if (feat.OneHotFeature) {
            // ── OneHot: each bin is independent, no suffix sums needed ──
            for (ui32 bin = 0; bin < feat.Folds; ++bin) {
                // Fix 2 (S30-COSINE-KAHAN): widened from float to double.
                // L2 path accumulates directly into totalGain; Cosine path finalizes via
                // cosNum_d/sqrt(cosDen_d) and stores without cast.  Keeps the argmax in
                // double throughout, matching CPU CatBoost's double bestGain.
                double totalGain = 0.0;
                // S28-OBLIV-DISPATCH: Cosine accumulators at bin scope — sum across all p*K dims,
                // then finalize once per bin (matches CPU AddLeafPlain multi-partition loop).
                // S30-T2-KAHAN K4: accumulate cosNum/cosDen in double to eliminate per-term
                // float32 computation error (~1e-3 floor) that defeated Neumaier compensation.
                // The per-term expression (sL^2*invL + sR^2*invR) has ~1e-3 fp32 rounding
                // intrinsic to float32 arithmetic on these magnitudes — Neumaier only compensates
                // summation rounding; the terms themselves must be widened.  Convert back to
                // float only at ComputeCosineGainKDim call.  Memory cost: 2 doubles per bin
                // candidate (16 bytes) vs 2 floats (8 bytes) — negligible at 2540 bins.
                double cosNum_d = 0.0;
                double cosDen_d = 1e-20;  // guard against sqrt(0), mirrors float guard

                if (minDataInLeaf > 1 && !countHist.empty()) {
                    bool anyViolates = false;
                    for (ui32 p = 0; p < numPartitions; ++p) {
                        ui32 countRight = countHist[p][feat.FirstFoldIndex + bin];
                        ui32 countLeft = partDocCounts[p] - countRight;
                        if (countLeft < minDataInLeaf || countRight < minDataInLeaf) {
                            anyViolates = true;
                            break;
                        }
                    }
                    if (anyViolates) continue;
                }

                for (ui32 p = 0; p < numPartitions; ++p) {
                    for (ui32 k = 0; k < K; ++k) {
                        const float* histData = perDimHist[k].data() + p * 2 * totalBinFeatures;
                        float totalSum = perDimPartStats[k][p].Sum;
                        float totalWeight = perDimPartStats[k][p].Weight;

                        float sumRight = histData[feat.FirstFoldIndex + bin];
                        float weightRight = histData[totalBinFeatures + feat.FirstFoldIndex + bin];
                        float sumLeft = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;

                        // S35-1H-L2-PER-SIDE (#129): apply per-side mask to L2 only.
                        // Mirrors ordinal Commit 1.5 (csv_train.cpp:1973+).  Cosine retains
                        // joint-skip per S34-PROBE-F-LITE verdict — Cosine is parentless,
                        // so per-side mask injects rare-category bias.  L2 has the parent
                        // term; at degenerate (p,k) the non-empty side cancels parent
                        // exactly, so per-side mask matches CPU.
                        const bool wL_pos = (weightLeft  > 1e-15f);
                        const bool wR_pos = (weightRight > 1e-15f);

                        switch (scoreFunction) {
                            case EScoreFunction::L2: {
                                if (!wL_pos && !wR_pos) break;
                                if (wL_pos) totalGain += (sumLeft  * sumLeft)  / (weightLeft  + l2RegLambda);
                                if (wR_pos) totalGain += (sumRight * sumRight) / (weightRight + l2RegLambda);
                                totalGain -= (totalSum * totalSum) / (totalWeight + l2RegLambda);
                                break;
                            }
                            case EScoreFunction::Cosine: {
                                // S34-PROBE-F-LITE verdict: leave joint-skip for one-hot Cosine.
                                if (!wL_pos || !wR_pos) break;
                                // S30-T2-KAHAN K4: widen inputs to double before computing terms.
                                const double dSL    = static_cast<double>(sumLeft);
                                const double dSR    = static_cast<double>(sumRight);
                                const double dWL    = static_cast<double>(weightLeft);
                                const double dWR    = static_cast<double>(weightRight);
                                const double dL2    = static_cast<double>(l2RegLambda);
                                const double dInvL  = 1.0 / (dWL + dL2);
                                const double dInvR  = 1.0 / (dWR + dL2);
                                cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
                                cosDen_d += dSL * dSL * dWL * dInvL * dInvL
                                          + dSR * dSR * dWR * dInvR * dInvR;
                                break;
                            }
                            default:
                                throw std::logic_error("Unhandled EScoreFunction in FindBestSplit (one-hot branch)");
                        }
                    }
                }
                // Finalize Cosine score after all partitions and dims are accumulated.
                if (scoreFunction == EScoreFunction::Cosine) {
                    // Fix 2 (S30-COSINE-KAHAN): assign fp64 result directly — no cast to float.
                    // K4 widened the accumulators (cosNum_d, cosDen_d) to double.  Fix 2 removes
                    // the static_cast<float> here so that totalGain stays double through the argmax
                    // comparison, matching CPU CatBoost's double bestGain/totalGain end-to-end.
                    totalGain = cosNum_d / std::sqrt(cosDen_d);
                }

                // Add random perturbation to prevent overfitting
                // Fix 2: perturbedGain is double; noiseScale*noiseDist(*rng) is float → promoted.
                double perturbedGain = totalGain;
                if (noiseScale > 0.0f) {
                    perturbedGain += static_cast<double>(noiseScale * noiseDist(*rng));
                }
                if (perturbedGain > bestGain) {
                    bestGain = perturbedGain;
                    bestSplit.FeatureId = featIdx;
                    bestSplit.BinId = bin;
                    bestSplit.Gain = totalGain;  // store actual gain, not perturbed
                    bestSplit.Score = static_cast<float>(-totalGain);
                }
#ifdef ITER1_AUDIT
                // S31-T3b: collect candidate gain for top-K ranking (OneHot branch)
                if (g_iter1Audit.pendingActive) {
                    g_iter1Audit.candidateAccum.emplace_back(totalGain, featIdx, bin);
                }
#endif
#ifdef CATBOOST_MLX_DEBUG_LEAF
                if (p11Active) {
                    // Record candidate for P11 top-10 (OneHot: use partition 0, dim 0 stats)
                    TDbgCandidate cand;
                    cand.featIdx = featIdx; cand.bin = bin; cand.gain = static_cast<float>(totalGain);
                    if (numPartitions > 0 && K > 0) {
                        const float* hd0 = perDimHist[0].data() + 0 * 2 * totalBinFeatures;
                        cand.sumR = hd0[feat.FirstFoldIndex + bin];
                        cand.wR   = hd0[totalBinFeatures + feat.FirstFoldIndex + bin];
                        cand.sumP = perDimPartStats[0][0].Sum;
                        cand.wP   = perDimPartStats[0][0].Weight;
                        cand.sumL = cand.sumP - cand.sumR;
                        cand.wL   = cand.wP   - cand.wR;
                    }
                    allCandidates.push_back(cand);
                }
#endif
            }
        } else {
            // ── Ordinal: precompute suffix sums for O(1) lookup per bin ──
            ui32 folds = feat.Folds;
            if (folds == 0) continue;

            // Suffix sums of gradient/hessian histograms: suffGrad[k][p][b] =
            // sum over i=b..folds-1 of hist[firstFold + i].
            // Layout: [K * numPartitions * (folds+1)], sentinel at [folds] = 0.
            size_t stride = static_cast<size_t>(folds) + 1;
            std::vector<float> suffGrad(K * numPartitions * stride, 0.0f);
            std::vector<float> suffHess(K * numPartitions * stride, 0.0f);

            for (ui32 k = 0; k < K; ++k) {
                for (ui32 p = 0; p < numPartitions; ++p) {
                    const float* hd = perDimHist[k].data() + p * 2 * totalBinFeatures;
                    size_t base = (k * numPartitions + p) * stride;
                    // Build suffix sums from right to left
                    for (int b = static_cast<int>(folds) - 1; b >= 0; --b) {
                        suffGrad[base + b] = suffGrad[base + b + 1] + hd[feat.FirstFoldIndex + b];
                        suffHess[base + b] = suffHess[base + b + 1] + hd[totalBinFeatures + feat.FirstFoldIndex + b];
                    }
                }
            }

            // Suffix sums for doc-count histogram (min-data-in-leaf)
            std::vector<ui32> suffCount;
            if (minDataInLeaf > 1 && !countHist.empty()) {
                suffCount.resize(numPartitions * stride, 0);
                for (ui32 p = 0; p < numPartitions; ++p) {
                    size_t base = p * stride;
                    for (int b = static_cast<int>(folds) - 1; b >= 0; --b) {
                        suffCount[base + b] = suffCount[base + b + 1]
                                            + countHist[p][feat.FirstFoldIndex + b];
                    }
                }
            }

            bool hasMonotone = !monotoneConstraints.empty()
                            && featIdx < monotoneConstraints.size()
                            && monotoneConstraints[featIdx] != 0;

            for (ui32 bin = 0; bin < folds; ++bin) {
                // Fix 2 (S30-COSINE-KAHAN): widened from float to double.
                double totalGain = 0.0;
                bool violatesConstraint = false;
                // S28-OBLIV-DISPATCH: Cosine accumulators at bin scope — sum across all p*K dims.
                // S30-T2-KAHAN K4: accumulate in double; Fix 2 removes the cast at finalization.
                double cosNum_d = 0.0;
                double cosDen_d = 1e-20;  // guard against sqrt(0), mirrors float guard
#ifdef COSINE_TERM_AUDIT
                // S32-T2: reset per-bin accumulators at the start of each bin candidate.
                if (g_termAudit.active) {
                    g_termAudit.curFeat   = featIdx;
                    g_termAudit.curBin    = bin;
                    g_termAudit.curCosNum = 0.0;
                    g_termAudit.curCosDen = 0.0;
                    g_termAudit.curSumL   = 0.0;
                    g_termAudit.curSumR   = 0.0;
                    g_termAudit.curWL     = 0.0;
                    g_termAudit.curWR     = 0.0;
                    g_termAudit.curLambda = 0.0;
                }
#endif  // COSINE_TERM_AUDIT
#ifdef COSINE_RESIDUAL_INSTRUMENT
                // S30-T2: parallel float32 shadow accumulators for residual comparison.
                // These mirror the pre-K4 float32 accumulation.  The residual reported in the
                // CSV is |cosNum_f32_shadow - cosNum_d|, showing the actual K4 improvement.
                float cosNum_f32_shadow = 0.0f;
                float cosDen_f32_shadow = 1e-20f;
#endif

                // Min-data-in-leaf check using precomputed suffix sums
                if (minDataInLeaf > 1 && !suffCount.empty()) {
                    bool anyViolates = false;
                    for (ui32 p = 0; p < numPartitions; ++p) {
                        ui32 countRight = suffCount[p * stride + bin];
                        ui32 countLeft = partDocCounts[p] - countRight;
                        if (countLeft < minDataInLeaf || countRight < minDataInLeaf) {
                            anyViolates = true;
                            break;
                        }
                    }
                    if (anyViolates) continue;
                }

                for (ui32 p = 0; p < numPartitions; ++p) {
                    // Monotone constraint check using precomputed suffix sums (dim 0)
                    if (hasMonotone) {
                        size_t base0 = static_cast<size_t>(p) * stride;
                        float sR = suffGrad[base0 + bin];
                        float wR = suffHess[base0 + bin];
                        float sL = perDimPartStats[0][p].Sum - sR;
                        float wL = perDimPartStats[0][p].Weight - wR;
                        if (wL > 1e-15f && wR > 1e-15f) {
                            float vL = sL / (wL + l2RegLambda);
                            float vR = sR / (wR + l2RegLambda);
                            if (monotoneConstraints[featIdx] == 1 && vL < vR) {
                                violatesConstraint = true;
                                break;
                            }
                            if (monotoneConstraints[featIdx] == -1 && vL > vR) {
                                violatesConstraint = true;
                                break;
                            }
                        }
                    }

                    for (ui32 k = 0; k < K; ++k) {
                        float totalSum = perDimPartStats[k][p].Sum;
                        float totalWeight = perDimPartStats[k][p].Weight;

                        size_t base = (k * numPartitions + p) * stride;
                        float sumRight = suffGrad[base + bin];
                        float weightRight = suffHess[base + bin];
                        float sumLeft = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;

#ifdef PROBE_E_INSTRUMENT
                        // S33-PROBE-E: capture per-leaf contribution snapshot BEFORE the
                        // skip-when-degenerate gate. Mirrors MLX's actual behavior (mlxTerm*=0
                        // when skipped) AND records what CPU's per-side-mask formula would add
                        // at the same (sumL, sumR, wL, wR). Diff columns expose the divergence.
                        if (g_cosInstr.dumpCosDen && scoreFunction == EScoreFunction::Cosine && K == 1) {
                            TCosineResidualInstrument::TLeafRecord lr;
                            lr.featIdx       = featIdx;
                            lr.bin           = bin;
                            lr.partition     = p;
                            lr.sumLeft       = sumLeft;
                            lr.sumRight      = sumRight;
                            lr.weightLeft    = weightLeft;
                            lr.weightRight   = weightRight;
                            const bool skip  = (weightLeft < 1e-15f || weightRight < 1e-15f);
                            lr.mlxSkipped    = skip;
                            const double dL2_e = static_cast<double>(l2RegLambda);
                            if (skip) {
                                lr.mlxTermNum = 0.0;
                                lr.mlxTermDen = 0.0;
                            } else {
                                const double dSL_e = static_cast<double>(sumLeft);
                                const double dSR_e = static_cast<double>(sumRight);
                                const double dWL_e = static_cast<double>(weightLeft);
                                const double dWR_e = static_cast<double>(weightRight);
                                const double dInvL_e = 1.0 / (dWL_e + dL2_e);
                                const double dInvR_e = 1.0 / (dWR_e + dL2_e);
                                lr.mlxTermNum = dSL_e * dSL_e * dInvL_e + dSR_e * dSR_e * dInvR_e;
                                lr.mlxTermDen = dSL_e * dSL_e * dWL_e * dInvL_e * dInvL_e
                                              + dSR_e * dSR_e * dWR_e * dInvR_e * dInvR_e;
                            }
                            // CPU-style: each side contributes independently, masked when its weight ≤ 0
                            double cpuN = 0.0, cpuD = 0.0;
                            if (weightLeft > 1e-15f) {
                                const double dSL2 = static_cast<double>(sumLeft);
                                const double dWL2 = static_cast<double>(weightLeft);
                                const double inv  = 1.0 / (dWL2 + dL2_e);
                                cpuN += dSL2 * dSL2 * inv;
                                cpuD += dSL2 * dSL2 * dWL2 * inv * inv;
                            }
                            if (weightRight > 1e-15f) {
                                const double dSR2 = static_cast<double>(sumRight);
                                const double dWR2 = static_cast<double>(weightRight);
                                const double inv  = 1.0 / (dWR2 + dL2_e);
                                cpuN += dSR2 * dSR2 * inv;
                                cpuD += dSR2 * dSR2 * dWR2 * inv * inv;
                            }
                            lr.cpuTermNum = cpuN;
                            lr.cpuTermDen = cpuD;
                            g_cosInstr.leafRecords.push_back(lr);
                        }
#endif  // PROBE_E_INSTRUMENT

                        // S33-L4-FIX (DEC-042): replace the joint skip-when-either-side-empty
                        // with a per-side mask that matches CPU's reference formula in
                        // short_vector_ops.h:155+ (SSE2 UpdateScoreBinKernelPlain).
                        // CPU zeroes the average of the empty side but ALWAYS adds the
                        // non-empty side's contribution.  The old single continue discarded
                        // the entire partition when one child was degenerate, which silently
                        // under-attributed cosNum/cosDen for signal-correlated split candidates
                        // that produce empty children in some partitions after depth>=1 splits.
                        // Both L2 and Cosine now apply per-side accumulation (S33-L4-FIX Commit 1.5).
                        const bool wL_pos = (weightLeft  > 1e-15f);
                        const bool wR_pos = (weightRight > 1e-15f);

                        switch (scoreFunction) {
                            case EScoreFunction::L2: {
                                // S33-L4-FIX Commit 1.5 (DEC-042): replace joint skip-when-either-empty
                                // with per-side mask matching CPU's CalcAverage formula in
                                // online_predictor.h:112-119:
                                //   average = (count > 0) ? sum / (count + λ) : 0
                                // CPU calls AddLeaf for both sides unconditionally; the mask is
                                // implicit in CalcAverage returning 0 for empty leaves.
                                // Each non-empty side contributes sum² / (weight + λ) to gain.
                                // Parent term (totalSum² / (totalWeight + λ)) is subtracted only
                                // when at least one side is non-empty — skip entire partition only
                                // when both children are empty (true no-op, no information).
                                // Sibling of Commit 1 (10c72b4e96) which applied the same fix to Cosine.
                                if (!wL_pos && !wR_pos) break;
                                if (wL_pos) {
                                    totalGain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda);
                                }
                                if (wR_pos) {
                                    totalGain += (sumRight * sumRight) / (weightRight + l2RegLambda);
                                }
                                totalGain -= (totalSum * totalSum) / (totalWeight + l2RegLambda);
                                break;
                            }
                            case EScoreFunction::Cosine: {
                                // S33-L4-FIX: per-side mask — skip whole partition only when
                                // both sides are empty (true no-op); otherwise accumulate each
                                // non-empty side independently.  Mirrors CPU's mask formula:
                                //   average = mask · sum / (w + λ),  mask = (w > 0)
                                //   num contribution: sum² / (w + λ)       [if w > 0]
                                //   den contribution: sum² · w / (w + λ)²  [if w > 0]
                                // S30-T2-KAHAN K4: widen to double before computing per-(p,k) terms.
                                // This eliminates the ~1e-3 per-term fp32 computation error that
                                // float32 Neumaier compensation could not reach.
                                if (!wL_pos && !wR_pos) break;
                                const double dL2   = static_cast<double>(l2RegLambda);
                                double termNum = 0.0;
                                double termDen = 0.0;
                                if (wL_pos) {
                                    const double dSL   = static_cast<double>(sumLeft);
                                    const double dWL   = static_cast<double>(weightLeft);
                                    const double dInvL = 1.0 / (dWL + dL2);
                                    termNum += dSL * dSL * dInvL;
                                    termDen += dSL * dSL * dWL * dInvL * dInvL;
                                }
                                if (wR_pos) {
                                    const double dSR   = static_cast<double>(sumRight);
                                    const double dWR   = static_cast<double>(weightRight);
                                    const double dInvR = 1.0 / (dWR + dL2);
                                    termNum += dSR * dSR * dInvR;
                                    termDen += dSR * dSR * dWR * dInvR * dInvR;
                                }
                                cosNum_d += termNum;
                                cosDen_d += termDen;
#ifdef COSINE_TERM_AUDIT
                                // S32-T2: accumulate per-bin term contributions for the term audit.
                                // At depth=0 K=1 numPartitions=1, there is exactly one (p=0,k=0)
                                // term per (feat,bin) candidate — these are the raw input values.
                                // S33-L4-FIX: use raw float stats (sumLeft/Right, weightLeft/Right)
                                // since dSL/dSR/dWL/dWR are now scoped inside per-side if-blocks.
                                if (g_termAudit.active) {
                                    g_termAudit.curCosNum   += termNum;
                                    g_termAudit.curCosDen   += termDen;
                                    // Capture the partition stats from the last (p,k) iteration.
                                    // For depth=0 K=1 these are the only (p,k) values.
                                    g_termAudit.curSumL   = static_cast<double>(sumLeft);
                                    g_termAudit.curSumR   = static_cast<double>(sumRight);
                                    g_termAudit.curWL     = static_cast<double>(weightLeft);
                                    g_termAudit.curWR     = static_cast<double>(weightRight);
                                    g_termAudit.curLambda = dL2;
                                }
#endif  // COSINE_TERM_AUDIT
#ifdef COSINE_RESIDUAL_INSTRUMENT
                                // S30-T2: parallel float32 shadow accumulation for residual comparison.
                                // Accumulates the same terms in float32 to show the pre-K4 precision.
                                // S33-L4-FIX: apply per-side mask to match the corrected accumulation.
                                if (g_cosInstr.dumpCosDen) {
                                    if (wL_pos) {
                                        const float invLf = 1.0f / (weightLeft  + l2RegLambda);
                                        cosNum_f32_shadow += sumLeft  * sumLeft  * invLf;
                                        cosDen_f32_shadow += sumLeft  * sumLeft  * weightLeft  * invLf * invLf;
                                    }
                                    if (wR_pos) {
                                        const float invRf = 1.0f / (weightRight + l2RegLambda);
                                        cosNum_f32_shadow += sumRight * sumRight * invRf;
                                        cosDen_f32_shadow += sumRight * sumRight * weightRight * invRf * invRf;
                                    }
                                }
#endif
                                break;
                            }
                            default:
                                throw std::logic_error("Unhandled EScoreFunction in FindBestSplit (ordinal branch)");
                        }
                    }
                }

                // Finalize Cosine score after all partitions and dims are accumulated.
                if (scoreFunction == EScoreFunction::Cosine) {
                    // Fix 2 (S30-COSINE-KAHAN): assign fp64 result directly — no cast to float.
                    totalGain = cosNum_d / std::sqrt(cosDen_d);
                }
#ifdef COSINE_TERM_AUDIT
                // S32-T2: record finalised per-bin term data.
                // totalGain is now set for Cosine; for non-Cosine score functions this block
                // is conditionally disabled by g_termAudit.active (only armed for Cosine).
                if (g_termAudit.active && scoreFunction == EScoreFunction::Cosine) {
                    TCosineTermAuditState::TTermRecord rec;
                    rec.featIdx    = g_termAudit.curFeat;
                    rec.bin        = g_termAudit.curBin;
                    rec.sumLeft    = g_termAudit.curSumL;
                    rec.sumRight   = g_termAudit.curSumR;
                    rec.weightLeft = g_termAudit.curWL;
                    rec.weightRight= g_termAudit.curWR;
                    rec.lambda     = g_termAudit.curLambda;
                    rec.cosNumTerm = g_termAudit.curCosNum;
                    rec.cosDenTerm = g_termAudit.curCosDen;
                    rec.gain       = totalGain;
                    g_termAudit.records.push_back(rec);
                }
#endif  // COSINE_TERM_AUDIT
#ifdef COSINE_RESIDUAL_INSTRUMENT
                // D2-redux: record per-bin residual for cosDen/cosNum/gain with CORRECTED methodology.
                // gain_f32 is now computed from the true pre-K4 fp32 shadow accumulators, NEVER
                // touching cosNum_d/cosDen_d.  This is what V2 audit required: the fp32 path must
                // use only float arithmetic end-to-end.
                // gain_f64 is the true fp64 path: cosNum_d / sqrt(cosDen_d).
                // Residual = |gain_f32 - gain_f64| measures fp32 accumulation path divergence,
                // NOT the float-cast ULP of a single fp64 value (which was D2's bias).
                if (g_cosInstr.dumpCosDen && scoreFunction == EScoreFunction::Cosine) {
                    double gain_f64_ref = cosNum_d / std::sqrt(cosDen_d);
                    // D2-redux: true pre-K4 fp32 gain — computed entirely from float32 shadows,
                    // sqrtf in float, division in float.  This is the path the code took before K4.
                    float gain_f32_path = static_cast<float>(
                        cosNum_f32_shadow / std::sqrtf(cosDen_f32_shadow));
                    TCosineResidualInstrument::TBinRecord rec;
                    rec.featIdx    = featIdx;
                    rec.bin        = bin;
                    rec.cosNum_f32 = cosNum_f32_shadow;
                    rec.cosNum_f64 = cosNum_d;
                    rec.cosDen_f32 = cosDen_f32_shadow;
                    rec.cosDen_f64 = cosDen_d;
                    rec.gain_f32   = gain_f32_path;   // true pre-K4 fp32 path
                    rec.gain_f64   = gain_f64_ref;    // true fp64 path
                    g_cosInstr.binRecords.push_back(rec);
                }
#endif

                if (violatesConstraint) continue;

                // Add random perturbation to prevent overfitting
                // Fix 2: perturbedGain is double; float noise is promoted to double.
                double perturbedGain = totalGain;
                if (noiseScale > 0.0f) {
                    perturbedGain += static_cast<double>(noiseScale * noiseDist(*rng));
                }
                if (perturbedGain > bestGain) {
                    bestGain = perturbedGain;
                    bestSplit.FeatureId = featIdx;
                    bestSplit.BinId = bin;
                    bestSplit.Gain = totalGain;  // store actual gain, not perturbed
                    bestSplit.Score = static_cast<float>(-totalGain);
                }
#ifdef ITER1_AUDIT
                // S31-T3b: collect candidate gain for top-K ranking (Ordinal branch)
                if (g_iter1Audit.pendingActive) {
                    g_iter1Audit.candidateAccum.emplace_back(totalGain, featIdx, bin);
                }
#endif
#ifdef CATBOOST_MLX_DEBUG_LEAF
                if (p11Active) {
                    // Record candidate for P11 top-10 (Ordinal: use partition 0, dim 0 suffix sums)
                    TDbgCandidate cand;
                    cand.featIdx = featIdx; cand.bin = bin; cand.gain = static_cast<float>(totalGain);
                    if (numPartitions > 0 && K > 0) {
                        size_t sbase = (0 * numPartitions + 0) * stride;
                        cand.sumR = suffGrad[sbase + bin];
                        cand.wR   = suffHess[sbase + bin];
                        cand.sumP = perDimPartStats[0][0].Sum;
                        cand.wP   = perDimPartStats[0][0].Weight;
                        cand.sumL = cand.sumP - cand.sumR;
                        cand.wL   = cand.wP   - cand.wR;
                    }
                    allCandidates.push_back(cand);
                }
#endif
            }
        }
    }

#ifdef CATBOOST_MLX_DEBUG_LEAF
    if (p11Active && !allCandidates.empty()) {
        // P12 — literal gain formula used in this FindBestSplit
        // S28-OBLIV-DISPATCH: formula depends on scoreFunction (L2 or Cosine).
        if (scoreFunction == EScoreFunction::L2) {
            printf("[DBG P12] gain_formula: L2: "
                   "gain += (sumL^2)/(wL+L2) + (sumR^2)/(wR+L2) - (sumP^2)/(wP+L2)\n");
        } else {
            printf("[DBG P12] gain_formula: Cosine: "
                   "score = (Σ_pk sumL^2/(wL+L2) + sumR^2/(wR+L2)) / sqrt(Σ_pk sumL^2*wL/(wL+L2)^2 + ...)\n");
        }
        printf("[DBG P12] formula_file: catboost/mlx/tests/csv_train.cpp (S28-OBLIV-DISPATCH)\n");
        printf("[DBG P12] noise_formula: noiseScale = rs * gradRms  (DEC-028 fix)\n");
        printf("[DBG P12] noiseScale_used = %.6f  (rs=%.4f  gradRms=%.6f)\n",
               noiseScale, randomStrength, gradRms);

        // Sort by gain descending
        std::sort(allCandidates.begin(), allCandidates.end(),
                  [](const TDbgCandidate& a, const TDbgCandidate& b){ return a.gain > b.gain; });

        printf("[DBG P11] top-10 candidates at depth=0, iter=0 (before noise):\n");
        printf("[DBG P11]   rank  featIdx  bin    gain       sumL     wL       sumR     wR       sumP     wP\n");
        for (ui32 r = 0; r < std::min((ui32)allCandidates.size(), 10u); ++r) {
            const auto& c = allCandidates[r];
            printf("[DBG P11]   %-5u %-8u %-6u %-10.4f %-8.4f %-8.2f %-8.4f %-8.2f %-8.4f %.2f\n",
                   r, c.featIdx, c.bin, c.gain,
                   c.sumL, c.wL, c.sumR, c.wR, c.sumP, c.wP);
        }

        // P13 — cross-check for rank-0 candidate
        const auto& best = allCandidates[0];
        float p13_l2 = l2RegLambda;
        float p13_gain_mlx_formula =
            (best.wL > 1e-15f && best.wR > 1e-15f)
            ? (best.sumL * best.sumL) / (best.wL + p13_l2)
              + (best.sumR * best.sumR) / (best.wR + p13_l2)
              - (best.sumP * best.sumP) / (best.wP + p13_l2)
            : 0.0f;
        // CPU formula is identical: sumL^2/(wL+L2) + sumR^2/(wR+L2) - sumP^2/(wP+L2)
        // For RMSE unweighted, wL/wR == docCountL/docCountR, so formulas agree exactly.
        // Any difference would come from histogram inputs, not formula divergence (H3).
        float p13_gain_cpu_formula = p13_gain_mlx_formula;  // identical formula
        float p13_gain_reported = best.gain;
        printf("[DBG P13] cross-check for rank-0 (feat=%u bin=%u):\n", best.featIdx, best.bin);
        printf("[DBG P13]   sumL=%.6f  wL=%.2f  sumR=%.6f  wR=%.2f\n",
               best.sumL, best.wL, best.sumR, best.wR);
        printf("[DBG P13]   gain_by_MLX_formula   = %.6f\n", p13_gain_mlx_formula);
        printf("[DBG P13]   gain_by_CPU_formula   = %.6f  (same formula for RMSE)\n", p13_gain_cpu_formula);
        printf("[DBG P13]   gain_reported_by_FindBestSplit = %.6f\n", p13_gain_reported);
        printf("[DBG P13]   L2=%g  wP=%.2f  sumP=%.6f\n", p13_l2, best.wP, best.sumP);
        fflush(stdout);
    }
#endif

#ifdef ITER1_AUDIT
    // S31-T3b: finalise top-K candidates and record winner into pending layer
    if (g_iter1Audit.pendingActive) {
        // Sort descending by gain (first element of tuple)
        std::sort(g_iter1Audit.candidateAccum.begin(),
                  g_iter1Audit.candidateAccum.end(),
                  [](const auto& a, const auto& b){ return std::get<0>(a) > std::get<0>(b); });

        g_iter1Audit.pendingLayer.topK.clear();
        const int kLimit = std::min(
            (int)g_iter1Audit.candidateAccum.size(),
            TIter1AuditState::TOP_K);
        for (int ki = 0; ki < kLimit; ++ki) {
            TIter1AuditState::TLayerRecord::TCandidate c;
            c.gain    = std::get<0>(g_iter1Audit.candidateAccum[ki]);
            c.featIdx = std::get<1>(g_iter1Audit.candidateAccum[ki]);
            c.binIdx  = std::get<2>(g_iter1Audit.candidateAccum[ki]);
            g_iter1Audit.pendingLayer.topK.push_back(c);
        }
        g_iter1Audit.candidateAccum.clear();

        // Record winner
        if (bestSplit.Defined()) {
            g_iter1Audit.pendingLayer.winnerFeat = bestSplit.FeatureId;
            g_iter1Audit.pendingLayer.winnerBin  = bestSplit.BinId;
            g_iter1Audit.pendingLayer.winnerGain = bestSplit.Gain;
        }
        // NOTE: pendingActive stays true here — the call-site FLUSH block clears it
        // after pushing pendingLayer to g_iter1Audit.layers.
    }
#endif

    return bestSplit;
}

// ============================================================================
// Depthwise: find best split per partition.
//
// Returns a vector of size numPartitions, each element being the best split
// for that partition.  Partitions with no valid split get FeatureId == -1.
// The outer loop mirrors FindBestSplit but restricts the gain sum to one
// partition at a time.
// ============================================================================

std::vector<TBestSplitProperties> FindBestSplitPerPartition(
    const std::vector<std::vector<float>>& perDimHist,  // [K][numPartitions * 2 * totalBinFeatures]
    const std::vector<std::vector<TPartitionStatistics>>& perDimPartStats,  // [K][numPartitions]
    const std::vector<TCFeature>& features,
    ui32 totalBinFeatures,
    float l2RegLambda,
    ui32 numPartitions,
    const std::vector<bool>& featureMask = {},
    float randomStrength = 0.0f,                           // random score perturbation (0 = disabled)
    float gradRms = 0.0f,                                  // gradient RMS — noise scale (CPU: CalcDerivativesStDevFromZeroPlainBoosting, greedy_tensor_search.cpp:92)
    std::mt19937* rng = nullptr,                           // RNG for perturbation
    EScoreFunction scoreFunction = EScoreFunction::L2      // gain formula — default L2 preserves pre-S28 behaviour
) {
    const ui32 K = static_cast<ui32>(perDimHist.size());
    std::vector<TBestSplitProperties> results(numPartitions);
    // Fix 2 (S30-COSINE-KAHAN): widen per-partition argmax accumulator to double.
    std::vector<double> bestGains(numPartitions, -std::numeric_limits<double>::infinity());

    // Random score perturbation: mirrors DEC-028's FindBestSplit pattern.
    // CPU reference: CalcScoreStDev (greedy_tensor_search.cpp:851) computes one global
    // scalar per tree and passes it unchanged into every per-partition / per-leaf search.
    // gradRms is pre-computed in RunTraining (csv_train.cpp:3155-3165) and passed here.
    float noiseScale = 0.0f;
    if (randomStrength > 0.0f && rng && gradRms > 0.0f) {
        noiseScale = randomStrength * gradRms;
    }
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);

    for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
        if (!featureMask.empty() && !featureMask[featIdx]) continue;
        const auto& feat = features[featIdx];

        if (feat.OneHotFeature) {
            for (ui32 bin = 0; bin < feat.Folds; ++bin) {
                for (ui32 p = 0; p < numPartitions; ++p) {
                    // For Cosine we accumulate num/den separately across K dims then
                    // compute score at the end (ComputeCosineGainKDim).  For L2 we
                    // accumulate the scalar gain directly.
                    // Fix 2 (S30-COSINE-KAHAN): widened from float to double.
                    double gain = 0.0;
                    // S30-T2-KAHAN K4: Cosine accumulators in double — eliminates per-term
                    // fp32 computation error (~1e-3 floor) that defeated Neumaier compensation.
                    double cosNum_d = 0.0;
                    double cosDen_d = 1e-20;  // guard against sqrt(0)
                    for (ui32 k = 0; k < K; ++k) {
                        const float* histData = perDimHist[k].data()
                            + static_cast<size_t>(p) * 2 * totalBinFeatures;
                        float totalSum    = perDimPartStats[k][p].Sum;
                        float totalWeight = perDimPartStats[k][p].Weight;
                        float sumRight    = histData[feat.FirstFoldIndex + bin];
                        float weightRight = histData[totalBinFeatures + feat.FirstFoldIndex + bin];
                        float sumLeft    = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;
                        if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
                        switch (scoreFunction) {
                            case EScoreFunction::L2:
                                gain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                                      + (sumRight * sumRight) / (weightRight + l2RegLambda)
                                      - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                                break;
                            case EScoreFunction::Cosine: {
                                // S30-T2-KAHAN K4: widen inputs to double before computing terms.
                                const double dSL   = static_cast<double>(sumLeft);
                                const double dSR   = static_cast<double>(sumRight);
                                const double dWL   = static_cast<double>(weightLeft);
                                const double dWR   = static_cast<double>(weightRight);
                                const double dL2   = static_cast<double>(l2RegLambda);
                                const double dInvL = 1.0 / (dWL + dL2);
                                const double dInvR = 1.0 / (dWR + dL2);
                                cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
                                cosDen_d += dSL * dSL * dWL * dInvL * dInvL
                                          + dSR * dSR * dWR * dInvR * dInvR;
                                break;
                            }
                            default:
                                // Should be unreachable: ParseScoreFunction guards at startup.
                                throw std::logic_error("Unhandled EScoreFunction in FindBestSplitPerPartition (one-hot branch)");
                        }
                    }
                    // Finalise score: for Cosine, compute num/sqrt(den) after K loop.
                    if (scoreFunction == EScoreFunction::Cosine) {
                        // Fix 2 (S30-COSINE-KAHAN): assign fp64 result directly — no cast to float.
                        gain = cosNum_d / std::sqrt(cosDen_d);
                    }
                    // Add random perturbation to prevent overfitting (mirrors DEC-028 FindBestSplit:1057-1068)
                    // Fix 2: perturbedGain is double; float noise is promoted to double.
                    double perturbedGain = gain;
                    if (noiseScale > 0.0f) {
                        perturbedGain += static_cast<double>(noiseScale * noiseDist(*rng));
                    }
                    if (perturbedGain > bestGains[p]) {
                        bestGains[p] = perturbedGain;
                        results[p].FeatureId = featIdx;
                        results[p].BinId     = bin;
                        results[p].Gain      = perturbedGain;
                        results[p].Score     = static_cast<float>(-perturbedGain);
                    }
                }
            }
        } else {
            // Ordinal: precompute suffix sums per partition per dim
            const ui32 stride = totalBinFeatures;
            std::vector<float> suffGrad(K * numPartitions * stride, 0.0f);
            std::vector<float> suffHess(K * numPartitions * stride, 0.0f);
            for (ui32 k = 0; k < K; ++k) {
                for (ui32 p = 0; p < numPartitions; ++p) {
                    const float* histData = perDimHist[k].data()
                        + static_cast<size_t>(p) * 2 * totalBinFeatures;
                    size_t base = (k * numPartitions + p) * stride;
                    float runG = 0.0f, runH = 0.0f;
                    for (int b = static_cast<int>(feat.FirstFoldIndex + feat.Folds) - 1;
                         b >= static_cast<int>(feat.FirstFoldIndex); --b) {
                        runG += histData[b];
                        runH += histData[totalBinFeatures + b];
                        suffGrad[base + b] = runG;
                        suffHess[base + b] = runH;
                    }
                }
            }

            for (ui32 bin = 0; bin + 1 < feat.Folds; ++bin) {
                ui32 binOffset = feat.FirstFoldIndex + bin;
                for (ui32 p = 0; p < numPartitions; ++p) {
                    // Fix 2 (S30-COSINE-KAHAN): widened from float to double.
                    double gain = 0.0;
                    // S30-T2-KAHAN K4: Cosine accumulators in double — eliminates per-term
                    // fp32 computation error (~1e-3 floor) that defeated Neumaier compensation.
                    double cosNum_d = 0.0;
                    double cosDen_d = 1e-20;  // guard against sqrt(0)
                    for (ui32 k = 0; k < K; ++k) {
                        float totalSum    = perDimPartStats[k][p].Sum;
                        float totalWeight = perDimPartStats[k][p].Weight;
                        size_t base = (k * numPartitions + p) * stride;
                        float sumRight    = suffGrad[base + binOffset];
                        float weightRight = suffHess[base + binOffset];
                        float sumLeft    = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;
                        if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
                        switch (scoreFunction) {
                            case EScoreFunction::L2:
                                gain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                                      + (sumRight * sumRight) / (weightRight + l2RegLambda)
                                      - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                                break;
                            case EScoreFunction::Cosine: {
                                // S30-T2-KAHAN K4: widen inputs to double before computing terms.
                                const double dSL   = static_cast<double>(sumLeft);
                                const double dSR   = static_cast<double>(sumRight);
                                const double dWL   = static_cast<double>(weightLeft);
                                const double dWR   = static_cast<double>(weightRight);
                                const double dL2   = static_cast<double>(l2RegLambda);
                                const double dInvL = 1.0 / (dWL + dL2);
                                const double dInvR = 1.0 / (dWR + dL2);
                                cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
                                cosDen_d += dSL * dSL * dWL * dInvL * dInvL
                                          + dSR * dSR * dWR * dInvR * dInvR;
                                break;
                            }
                            default:
                                throw std::logic_error("Unhandled EScoreFunction in FindBestSplitPerPartition (ordinal branch)");
                        }
                    }
                    if (scoreFunction == EScoreFunction::Cosine) {
                        // Fix 2 (S30-COSINE-KAHAN): assign fp64 result directly — no cast to float.
                        gain = cosNum_d / std::sqrt(cosDen_d);
                    }
                    // Add random perturbation to prevent overfitting (mirrors DEC-028 FindBestSplit:1188-1198)
                    // Fix 2: perturbedGain is double; float noise is promoted to double.
                    double perturbedGain = gain;
                    if (noiseScale > 0.0f) {
                        perturbedGain += static_cast<double>(noiseScale * noiseDist(*rng));
                    }
                    if (perturbedGain > bestGains[p]) {
                        bestGains[p] = perturbedGain;
                        results[p].FeatureId = featIdx;
                        results[p].BinId     = bin;
                        results[p].Gain      = perturbedGain;
                        results[p].Score     = static_cast<float>(-perturbedGain);
                    }
                }
            }
        }
    }
    return results;
}

// ============================================================================
// Loss function helpers: parse loss type with optional parameter
// ============================================================================

struct TLossConfig {
    std::string Type;
    float Param = 0.0f;  // alpha for quantile, delta for huber
};

TLossConfig ParseLossType(const std::string& lossStr) {
    TLossConfig lc;

    // Belt-and-suspenders: normalize to lowercase so callers passing 'MAE',
    // 'Quantile:0.7', etc. work even if the Python layer didn't normalize first.
    std::string normalized = lossStr;
    auto colonPos = normalized.find(':');
    // Lowercase only the base type (before the colon)
    std::string baseType = (colonPos != std::string::npos)
        ? normalized.substr(0, colonPos)
        : normalized;
    for (auto& c : baseType) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (colonPos != std::string::npos) {
        lc.Type = baseType;
        std::string suffix = normalized.substr(colonPos + 1);
        // Also accept named-param syntax: 'alpha=0.7', 'delta=1.0'
        for (const auto* prefix : {"alpha=", "delta="}) {
            std::string lcSuffix = suffix;
            for (auto& c : lcSuffix) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lcSuffix.find(prefix) == 0) {
                suffix = suffix.substr(std::string(prefix).size());
                break;
            }
        }
        lc.Param = std::stof(suffix);
    } else {
        lc.Type = baseType;
        if (lc.Type == "quantile") lc.Param = 0.5f;
        if (lc.Type == "huber") lc.Param = 1.0f;
        if (lc.Type == "tweedie") lc.Param = 1.5f;  // variance power p
    }
    return lc;
}

// ============================================================================
// Softmax helpers
// ============================================================================

struct TSoftmaxResult {
    mx::array Probs;  // [K, numDocs]
};

TSoftmaxResult ComputeSoftmax(const mx::array& cursor, int K, int numDocs) {
    auto maxC = mx::maximum(mx::max(cursor, 0), mx::array(0.0f));  // [numDocs]
    auto expC = mx::exp(mx::subtract(cursor, mx::reshape(maxC, {1, numDocs})));  // [K, numDocs]
    auto expImp = mx::exp(mx::negative(maxC));  // [numDocs]
    auto sumExp = mx::add(mx::sum(expC, 0), expImp);  // [numDocs]
    auto probs = mx::divide(expC, mx::reshape(sumExp, {1, numDocs}));  // [K, numDocs]
    return {probs};
}

// ============================================================================
// Compute loss value for a given cursor, targets, loss type
// ============================================================================

float ComputeLossValue(
    const mx::array& cursor, const mx::array& targetsArr,
    const std::string& lossType, float lossParam, ui32 approxDim, ui32 numDocs
) {
    if (lossType == "rmse") {
        auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(numDocs)}), targetsArr);
        auto loss = mx::sqrt(mx::mean(mx::multiply(diff, diff)));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "mae") {
        auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(numDocs)}), targetsArr);
        auto loss = mx::mean(mx::abs(diff));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "quantile") {
        float alpha = lossParam;
        auto diff = mx::subtract(targetsArr, mx::reshape(cursor, {static_cast<int>(numDocs)}));
        auto isPos = mx::greater(diff, mx::array(0.0f));
        auto loss = mx::where(isPos,
            mx::multiply(mx::array(alpha), diff),
            mx::multiply(mx::array(alpha - 1.0f), diff));
        auto result = mx::mean(loss);
        mx::eval(result);
        return result.item<float>();
    } else if (lossType == "huber") {
        float delta = lossParam;
        auto diff = mx::subtract(mx::reshape(cursor, {static_cast<int>(numDocs)}), targetsArr);
        auto absDiff = mx::abs(diff);
        auto isSmall = mx::less_equal(absDiff, mx::array(delta));
        auto loss = mx::where(isSmall,
            mx::multiply(mx::array(0.5f), mx::multiply(diff, diff)),
            mx::subtract(mx::multiply(mx::array(delta), absDiff), mx::array(0.5f * delta * delta)));
        auto result = mx::mean(loss);
        mx::eval(result);
        return result.item<float>();
    } else if (lossType == "logloss") {
        auto sig = mx::sigmoid(mx::reshape(cursor, {static_cast<int>(numDocs)}));
        auto eps = mx::array(1e-15f);
        auto loss = mx::negative(mx::mean(mx::add(
            mx::multiply(targetsArr, mx::log(mx::add(sig, eps))),
            mx::multiply(mx::subtract(mx::array(1.0f), targetsArr),
                         mx::log(mx::add(mx::subtract(mx::array(1.0f), sig), eps)))
        )));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "poisson") {
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto expPred = mx::exp(flatCursor);
        auto loss = mx::mean(mx::subtract(expPred, mx::multiply(targetsArr, flatCursor)));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "tweedie") {
        float p = lossParam;
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto term1 = mx::divide(
            mx::multiply(mx::negative(targetsArr), mx::exp(mx::multiply(flatCursor, mx::array(1.0f - p)))),
            mx::array(1.0f - p));
        auto term2 = mx::divide(
            mx::exp(mx::multiply(flatCursor, mx::array(2.0f - p))),
            mx::array(2.0f - p));
        auto loss = mx::mean(mx::add(term1, term2));
        mx::eval(loss);
        return loss.item<float>();
    } else if (lossType == "mape") {
        auto flatCursor = mx::reshape(cursor, {static_cast<int>(numDocs)});
        auto absTarget = mx::maximum(mx::abs(targetsArr), mx::array(1e-6f));
        auto loss = mx::mean(mx::divide(mx::abs(mx::subtract(flatCursor, targetsArr)), absTarget));
        mx::eval(loss);
        return loss.item<float>();
    } else {
        // multiclass
        auto sm = ComputeSoftmax(cursor, approxDim, numDocs);
        auto targetInt = mx::astype(targetsArr, mx::int32);
        auto probTarget = mx::zeros({static_cast<int>(numDocs)}, mx::float32);
        for (ui32 k = 0; k < approxDim; ++k) {
            auto isClass = mx::astype(mx::equal(targetInt, mx::array(static_cast<int>(k))), mx::float32);
            auto probK = mx::reshape(
                mx::slice(sm.Probs, {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)}),
                {static_cast<int>(numDocs)});
            probTarget = mx::add(probTarget, mx::multiply(isClass, probK));
        }
        auto isLast = mx::astype(mx::equal(targetInt, mx::array(static_cast<int>(approxDim))), mx::float32);
        auto probImp = mx::subtract(mx::array(1.0f), mx::sum(sm.Probs, 0));
        probTarget = mx::add(probTarget, mx::multiply(isLast, probImp));
        auto loss = mx::negative(mx::mean(mx::log(mx::add(probTarget, mx::array(1e-15f)))));
        mx::eval(loss);
        return loss.item<float>();
    }
}

// ============================================================================
// Ranking: pair generation and gradient computation
// ============================================================================

struct TPair {
    ui32 Winner;  // doc index with higher relevance
    ui32 Loser;   // doc index with lower relevance
    float Weight;
};

// Generate all (winner, loser) pairs within each group for PairLogit
std::vector<TPair> GeneratePairLogitPairs(
    const std::vector<float>& targets,
    const std::vector<ui32>& groupOffsets,
    ui32 numGroups
) {
    std::vector<TPair> pairs;
    for (ui32 g = 0; g < numGroups; ++g) {
        ui32 begin = groupOffsets[g];
        ui32 end = groupOffsets[g + 1];
        for (ui32 i = begin; i < end; ++i) {
            for (ui32 j = begin; j < end; ++j) {
                if (targets[i] > targets[j]) {
                    pairs.push_back({i, j, 1.0f});
                }
            }
        }
    }
    return pairs;
}

// Generate YetiRank pairs: random permutation within each group,
// then adjacent-position pairs weighted by position and relevance difference
std::vector<TPair> GenerateYetiRankPairs(
    const std::vector<float>& targets,
    const std::vector<ui32>& groupOffsets,
    ui32 numGroups,
    std::mt19937& rng
) {
    std::vector<TPair> pairs;
    for (ui32 g = 0; g < numGroups; ++g) {
        ui32 begin = groupOffsets[g];
        ui32 end = groupOffsets[g + 1];
        ui32 groupSize = end - begin;
        if (groupSize < 2) continue;

        // Random permutation of doc indices within group
        std::vector<ui32> perm(groupSize);
        std::iota(perm.begin(), perm.end(), begin);
        std::shuffle(perm.begin(), perm.end(), rng);

        // Adjacent pairs in the random permutation
        for (ui32 pos = 1; pos < groupSize; ++pos) {
            ui32 i = perm[pos - 1];
            ui32 j = perm[pos];
            float relevDiff = std::fabs(targets[i] - targets[j]);
            if (relevDiff < 1e-8f) continue;
            float weight = relevDiff / std::log2(2.0f + pos);
            if (targets[i] > targets[j]) {
                pairs.push_back({i, j, weight});
            } else {
                pairs.push_back({j, i, weight});
            }
        }
    }
    return pairs;
}

// Scatter pairwise gradients to per-document gradient/hessian arrays
void ScatterPairwiseGradients(
    const std::vector<TPair>& pairs,
    const float* preds,  // raw predictions [numDocs]
    ui32 numDocs,
    std::vector<float>& grads,
    std::vector<float>& hess
) {
    grads.assign(numDocs, 0.0f);
    hess.assign(numDocs, 0.0f);

    for (const auto& pair : pairs) {
        float diff = preds[pair.Winner] - preds[pair.Loser];
        float p = 1.0f / (1.0f + std::exp(-diff));  // sigmoid(diff)
        float w = pair.Weight;

        // gradient: push winner score up, loser score down
        grads[pair.Winner] += w * (p - 1.0f);
        grads[pair.Loser]  += w * (1.0f - p);

        // hessian: p*(1-p) for both
        float h = w * p * (1.0f - p);
        hess[pair.Winner] += h;
        hess[pair.Loser]  += h;
    }

    // Floor hessians to avoid zero
    for (ui32 d = 0; d < numDocs; ++d) {
        hess[d] = std::max(hess[d], 1e-6f);
    }
}

// PairLogit loss: sum over pairs: log(1 + exp(-(pred[winner] - pred[loser])))
float ComputePairLogitLoss(
    const std::vector<TPair>& pairs,
    const float* preds,
    ui32 /*numDocs*/
) {
    if (pairs.empty()) return 0.0f;
    double totalLoss = 0.0;
    for (const auto& pair : pairs) {
        float diff = preds[pair.Winner] - preds[pair.Loser];
        totalLoss += pair.Weight * std::log(1.0f + std::exp(-diff));
    }
    return static_cast<float>(totalLoss / pairs.size());
}

// NDCG metric: per-group DCG/IDCG averaged across groups
float ComputeNDCG(
    const std::vector<float>& targets,
    const float* preds,
    const std::vector<ui32>& groupOffsets,
    ui32 numGroups
) {
    if (numGroups == 0) return 0.0f;
    double totalNDCG = 0.0;
    ui32 validGroups = 0;

    for (ui32 g = 0; g < numGroups; ++g) {
        ui32 begin = groupOffsets[g];
        ui32 end = groupOffsets[g + 1];
        ui32 groupSize = end - begin;
        if (groupSize < 2) continue;

        // Sort docs by predicted score (descending)
        std::vector<ui32> sortedIdx(groupSize);
        std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
        std::sort(sortedIdx.begin(), sortedIdx.end(),
            [&](ui32 a, ui32 b) { return preds[begin + a] > preds[begin + b]; });

        // Compute DCG
        double dcg = 0.0;
        for (ui32 rank = 0; rank < groupSize; ++rank) {
            float relev = targets[begin + sortedIdx[rank]];
            dcg += (std::pow(2.0, relev) - 1.0) / std::log2(2.0 + rank);
        }

        // Compute IDCG (sort by true relevance descending)
        std::vector<float> sortedRelev(groupSize);
        for (ui32 i = 0; i < groupSize; ++i) sortedRelev[i] = targets[begin + i];
        std::sort(sortedRelev.begin(), sortedRelev.end(), std::greater<float>());

        double idcg = 0.0;
        for (ui32 rank = 0; rank < groupSize; ++rank) {
            idcg += (std::pow(2.0, sortedRelev[rank]) - 1.0) / std::log2(2.0 + rank);
        }

        if (idcg > 1e-10) {
            totalNDCG += dcg / idcg;
            validGroups++;
        }
    }

    return (validGroups > 0) ? static_cast<float>(totalNDCG / validGroups) : 0.0f;
}

// ============================================================================
// Apply tree splits to compute leaf indices for any compressed data
// ============================================================================

mx::array ComputeLeafIndices(
    const mx::array& compressedData,
    const std::vector<TObliviousSplitLevel>& splits,
    ui32 numDocs
) {
    auto leafIndices = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    for (ui32 level = 0; level < splits.size(); ++level) {
        const auto& split = splits[level];
        auto column = mx::slice(compressedData,
            {0, static_cast<int>(split.FeatureColumnIdx)},
            {static_cast<int>(numDocs), static_cast<int>(split.FeatureColumnIdx + 1)});
        column = mx::reshape(column, {static_cast<int>(numDocs)});
        auto featureValues = mx::bitwise_and(
            mx::right_shift(column, mx::array(static_cast<uint32_t>(split.Shift), mx::uint32)),
            mx::array(static_cast<uint32_t>(split.Mask), mx::uint32));
        auto goRight = split.IsOneHot
            ? mx::equal(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32))
            : mx::greater(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32));
        auto bits = mx::left_shift(mx::astype(goRight, mx::uint32), mx::array(static_cast<uint32_t>(level), mx::uint32));
        leafIndices = mx::bitwise_or(leafIndices, bits);
    }
    mx::eval(leafIndices);
    return leafIndices;
}

// Compute leaf indices for a depthwise (non-symmetric) tree.
// splits: split descriptors in bit-packed partition order (as built by the training loop).
// splitBfsNodeIds: parallel array — splitBfsNodeIds[i] is the BFS node index for splits[i].
// Returns bit-packed partition indices matching the training-path encoding (DEC-030):
//   bit k of the returned value = goRight at depth level k.
// This matches the `partitions` array maintained by the training path at csv_train.cpp:3660–3683,
// so mx::take(leafValues, leafIndices) is correct on the returned values.
mx::array ComputeLeafIndicesDepthwise(
    const mx::array& compressedData,
    const std::vector<TObliviousSplitLevel>& splits,
    const std::vector<ui32>& splitBfsNodeIds,
    ui32 numDocs,
    ui32 depth
) {
    if (depth == 0) {
        return mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    }

    // Build nodeSplitMap: BFS node index → split descriptor.
    // splits is in partition order; splitBfsNodeIds[i] gives the BFS index for splits[i].
    // This fixes Bug B: nodeSplits[nodeIdx] was indexing by BFS position into a
    // partition-ordered array, producing wrong split descriptors at depth >= 3.
    std::unordered_map<ui32, TObliviousSplitLevel> nodeSplitMap;
    nodeSplitMap.reserve(splits.size());
    for (ui32 i = 0; i < static_cast<ui32>(splits.size()); ++i) {
        nodeSplitMap[splitBfsNodeIds[i]] = splits[i];
    }

    // Materialise compressedData on CPU for the column reads.
    auto flatData = mx::reshape(compressedData, {static_cast<int>(numDocs), -1});
    mx::eval(flatData);
    const uint32_t* dataPtr = flatData.data<uint32_t>();
    const ui32 lineSize = static_cast<ui32>(flatData.shape(1));

    std::vector<uint32_t> leafVec(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 nodeIdx = 0u;
        ui32 partBits = 0u;
        for (ui32 lvl = 0; lvl < depth; ++lvl) {
            // Bug B fix: look up split by BFS node index, not flat position.
            if (!nodeSplitMap.count(nodeIdx)) {
                fprintf(stderr,
                    "ComputeLeafIndicesDepthwise: BFS node %u not found in nodeSplitMap "
                    "(depth=%u, splits.size()=%zu). "
                    "Training build loop must emit a split descriptor for every internal node.\n",
                    nodeIdx, depth, splits.size());
                exit(1);
            }
            const auto& ns = nodeSplitMap.at(nodeIdx);
            uint32_t packed = dataPtr[d * lineSize + ns.FeatureColumnIdx];
            uint32_t fv = (packed >> ns.Shift) & ns.Mask;
            uint32_t goRight = ns.IsOneHot ? (fv == ns.BinThreshold ? 1u : 0u)
                                           : (fv > ns.BinThreshold ? 1u : 0u);
            // Bug A fix: accumulate bit-packed partition index (bit lvl = goRight at depth lvl).
            // Replaces: leafVec[d] = nodeIdx - numNodes (BFS-array-leaf-offset, wrong encoding).
            partBits |= (goRight << lvl);
            nodeIdx = 2u * nodeIdx + 1u + goRight;
        }
        leafVec[d] = partBits;
    }
    return mx::array(reinterpret_cast<const int32_t*>(leafVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32);
}

// Compute per-document dense leaf indices for a lossguide (unbalanced) tree.
// nodeSplitMap: sparse map: BFS node index → split descriptor (only internal nodes).
// leafBfsIds: BFS node index for each dense leaf (size = numLeaves).
// Uses BFS traversal: for each doc, descend from root until hitting a leaf node.
mx::array ComputeLeafIndicesLossguide(
    const mx::array& compressedData,
    const std::unordered_map<ui32, TObliviousSplitLevel>& nodeSplitMap,
    const std::vector<ui32>& leafBfsIds,
    ui32 numDocs,
    ui32 numLeaves
) {
    if (numLeaves <= 1u || nodeSplitMap.empty()) {
        return mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    }

    // Build inverse map: BFS node index → dense leaf id.
    std::unordered_map<ui32, ui32> bfsToLeafId;
    bfsToLeafId.reserve(leafBfsIds.size());
    for (ui32 k = 0; k < static_cast<ui32>(leafBfsIds.size()); ++k) {
        bfsToLeafId[leafBfsIds[k]] = k;
    }

    auto flatData = mx::reshape(compressedData, {static_cast<int>(numDocs), -1});
    mx::eval(flatData);
    const uint32_t* dataPtr = flatData.data<uint32_t>();
    const ui32 lineSize = static_cast<ui32>(flatData.shape(1));

    std::vector<uint32_t> leafVec(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 nodeIdx = 0u;
        while (nodeSplitMap.count(nodeIdx) > 0) {
            const auto& ns = nodeSplitMap.at(nodeIdx);
            uint32_t packed = dataPtr[d * lineSize + ns.FeatureColumnIdx];
            uint32_t fv = (packed >> ns.Shift) & ns.Mask;
            uint32_t goRight = ns.IsOneHot
                ? (fv == ns.BinThreshold ? 1u : 0u)
                : (fv >  ns.BinThreshold ? 1u : 0u);
            nodeIdx = 2u * nodeIdx + 1u + goRight;
        }
        auto it = bfsToLeafId.find(nodeIdx);
        leafVec[d] = (it != bfsToLeafId.end()) ? it->second : 0u;
    }
    return mx::array(reinterpret_cast<const int32_t*>(leafVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32);
}

// ============================================================================
// CTR (target encoding) for high-cardinality categorical features
// ============================================================================

struct TCtrFeature {
    ui32 OrigFeatureIdx;               // index in original ds.Features
    std::string Name;                  // e.g. "city_ctr" or "city_ctr_class1"
    std::vector<float> Values;         // [numDocs] — ordered CTR values per doc
    // Final statistics for prediction (computed from ALL training data)
    std::unordered_map<uint32_t, float> FinalCtrValues;  // catBinIdx → final CTR
    float DefaultCtr;                  // CTR for unknown categories = prior / 1
    float Prior;                       // the prior used
    ui32 ClassIdx;                     // which class this CTR is for (0 for binary/regression)
    std::unordered_map<std::string, uint32_t> OrigCatHashMap;  // original string → bin mapping for prediction
};

// Compute ordered (online) CTR features.
// For each high-cardinality categorical feature, replaces it in the dataset with
// one or more numeric CTR features. Uses ordered computation to prevent target leakage.
std::vector<TCtrFeature> ComputeCtrFeatures(
    TDataset& ds,
    const std::string& lossType,
    ui32 numClasses,
    float prior,
    ui32 maxOneHotSize,
    ui32 randomSeed
) {
    std::vector<TCtrFeature> ctrFeatures;
    std::mt19937 rng(randomSeed + 12345);  // separate seed from subsampling

    // Create random permutation for ordered CTR
    std::vector<ui32> perm(ds.NumDocs);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Inverse permutation: originalIdx → positionInPerm
    std::vector<ui32> invPerm(ds.NumDocs);
    for (ui32 i = 0; i < ds.NumDocs; ++i) {
        invPerm[perm[i]] = i;
    }

    for (ui32 f = 0; f < ds.NumFeatures; ++f) {
        if (!ds.IsCategorical[f]) continue;
        ui32 numCategories = ds.CatHashMaps[f].size();
        if (numCategories <= maxOneHotSize) continue;  // keep OneHot for low-cardinality

        std::string baseName = (f < ds.FeatureNames.size() && !ds.FeatureNames[f].empty())
            ? ds.FeatureNames[f] : ("f" + std::to_string(f));

        if (lossType == "multiclass" && numClasses > 2) {
            // One CTR feature per class
            for (ui32 cls = 0; cls < numClasses; ++cls) {
                TCtrFeature ctr;
                ctr.OrigFeatureIdx = f;
                ctr.Name = baseName + "_ctr_class" + std::to_string(cls);
                ctr.Values.resize(ds.NumDocs);
                ctr.Prior = prior;
                ctr.ClassIdx = cls;
                ctr.DefaultCtr = prior / 1.0f;

                // Ordered CTR: process in permutation order
                // Per-category accumulators: countOfClass, totalCount
                std::unordered_map<uint32_t, float> catClassCount;  // sum of (target == cls) for docs before
                std::unordered_map<uint32_t, float> catTotalCount;  // total docs before

                for (ui32 i = 0; i < ds.NumDocs; ++i) {
                    ui32 doc = perm[i];
                    uint32_t catBin = static_cast<uint32_t>(ds.Features[f][doc]);

                    // CTR for this doc uses only stats from docs before it in permutation
                    float countBefore = catClassCount[catBin];
                    float totalBefore = catTotalCount[catBin];
                    ctr.Values[doc] = (countBefore + prior) / (totalBefore + 1.0f);

                    // Update accumulators with this doc's target
                    catTotalCount[catBin] += 1.0f;
                    if (static_cast<ui32>(ds.Targets[doc]) == cls) {
                        catClassCount[catBin] += 1.0f;
                    }
                }

                // Final CTR values (using ALL data) for prediction
                for (const auto& [catStr, catBin] : ds.CatHashMaps[f]) {
                    float cc = catClassCount[catBin];
                    float tc = catTotalCount[catBin];
                    ctr.FinalCtrValues[catBin] = (cc + prior) / (tc + 1.0f);
                }

                ctrFeatures.push_back(std::move(ctr));
            }
        } else {
            // Binary classification or regression: one CTR per feature
            TCtrFeature ctr;
            ctr.OrigFeatureIdx = f;
            ctr.Name = baseName + "_ctr";
            ctr.Values.resize(ds.NumDocs);
            ctr.Prior = prior;
            ctr.ClassIdx = 0;
            ctr.DefaultCtr = prior / 1.0f;

            // For logloss: countInClass = count of target==1
            // For regression: countInClass = sum of target values
            bool isBinaryClass = (lossType == "logloss");

            std::unordered_map<uint32_t, float> catSum;    // sum of targets (or positive count)
            std::unordered_map<uint32_t, float> catCount;  // total count

            for (ui32 i = 0; i < ds.NumDocs; ++i) {
                ui32 doc = perm[i];
                uint32_t catBin = static_cast<uint32_t>(ds.Features[f][doc]);

                float sumBefore = catSum[catBin];
                float countBefore = catCount[catBin];
                ctr.Values[doc] = (sumBefore + prior) / (countBefore + 1.0f);

                catCount[catBin] += 1.0f;
                if (isBinaryClass) {
                    catSum[catBin] += (ds.Targets[doc] > 0.5f) ? 1.0f : 0.0f;
                } else {
                    catSum[catBin] += ds.Targets[doc];
                }
            }

            // Final CTR values for prediction
            for (const auto& [catStr, catBin] : ds.CatHashMaps[f]) {
                float s = catSum[catBin];
                float c = catCount[catBin];
                ctr.FinalCtrValues[catBin] = (s + prior) / (c + 1.0f);
            }

            ctrFeatures.push_back(std::move(ctr));
        }

        // Replace this categorical feature with its CTR value(s) in the dataset
        // Mark the original feature as non-categorical (it becomes the first CTR's numeric values)
        // First, save the original hash map into all CTR features for this feature
        for (auto& ctr : ctrFeatures) {
            if (ctr.OrigFeatureIdx == f) {
                ctr.OrigCatHashMap = ds.CatHashMaps[f];
            }
        }
        ds.IsCategorical[f] = false;
        ds.HasNaN[f] = false;
        ds.CatHashMaps[f].clear();
        // Find the first CTR for this feature and use its values as replacement
        for (const auto& ctr : ctrFeatures) {
            if (ctr.OrigFeatureIdx == f) {
                ds.Features[f] = ctr.Values;
                ds.FeatureNames[f] = ctr.Name;
                break;
            }
        }
    }

    // Append additional CTR features (for multiclass: classes beyond the first)
    // Track which original features we've already placed the first CTR for
    std::unordered_set<ui32> firstCtrPlaced;
    for (const auto& ctr : ctrFeatures) {
        if (firstCtrPlaced.count(ctr.OrigFeatureIdx)) {
            // This is an additional CTR (e.g. class 1, 2, ... for multiclass)
            ds.Features.push_back(ctr.Values);
            ds.IsCategorical.push_back(false);
            ds.HasNaN.push_back(false);
            ds.FeatureNames.push_back(ctr.Name);
            ds.CatHashMaps.push_back({});
            ds.NumFeatures++;
        } else {
            firstCtrPlaced.insert(ctr.OrigFeatureIdx);
        }
    }

    return ctrFeatures;
}

// Apply training CTR features to an eval dataset.
// Uses FinalCtrValues (computed from all training data) to transform
// high-cardinality categoricals in the eval set to match the training layout.
void ApplyCtrToEvalData(
    TDataset& evalDs,
    const std::vector<TCtrFeature>& ctrFeatures,
    const TDataset& trainDs  // post-CTR training dataset (for NumFeatures reference)
) {
    if (ctrFeatures.empty()) return;

    // Track which original features get their first CTR placed in-place
    std::unordered_set<ui32> firstPlaced;

    for (const auto& ctr : ctrFeatures) {
        ui32 f = ctr.OrigFeatureIdx;

        // Transform eval feature values using FinalCtrValues
        std::vector<float> evalCtrValues(evalDs.NumDocs);
        for (ui32 d = 0; d < evalDs.NumDocs; ++d) {
            uint32_t catBin = static_cast<uint32_t>(evalDs.Features[f][d]);
            auto it = ctr.FinalCtrValues.find(catBin);
            if (it != ctr.FinalCtrValues.end()) {
                evalCtrValues[d] = it->second;
            } else {
                evalCtrValues[d] = ctr.DefaultCtr;  // unseen category
            }
        }

        if (!firstPlaced.count(f)) {
            // Replace the original categorical feature in-place (first CTR for this feature)
            evalDs.Features[f] = std::move(evalCtrValues);
            evalDs.IsCategorical[f] = false;
            evalDs.HasNaN[f] = false;
            evalDs.CatHashMaps[f].clear();
            evalDs.FeatureNames[f] = ctr.Name;
            firstPlaced.insert(f);
        } else {
            // Append additional CTR features (e.g. multiclass class 1, 2, ...)
            evalDs.Features.push_back(std::move(evalCtrValues));
            evalDs.IsCategorical.push_back(false);
            evalDs.HasNaN.push_back(false);
            evalDs.FeatureNames.push_back(ctr.Name);
            evalDs.CatHashMaps.push_back({});
            evalDs.NumFeatures++;
        }
    }
}

// ============================================================================
// Model save: tree record and JSON serialization
// ============================================================================

struct TTreeRecord {
    std::vector<TObliviousSplitLevel> Splits;        // oblivious: [depth]; depthwise/lossguide: [numNodes] BFS
    std::vector<TBestSplitProperties> SplitProps;
    std::vector<float> LeafValues;  // flat: numLeaves for dim=1, numLeaves*approxDim for multi
    ui32 Depth;
    bool IsDepthwise  = false;  // true → Splits holds BFS node splits, not oblivious levels
    bool IsLossguide  = false;  // true → Splits holds BFS node splits for unbalanced lossguide tree
    // Lossguide-only metadata
    std::vector<ui32> LeafBfsIds;  // [numLeaves] BFS node index per dense leaf
    // DEC-029: BFS node index for each split in SplitProps/Splits.
    // For Depthwise: the BFS node index of partition p at depth d (derived from
    //   the bit-pattern of p — not a simple 0..numNodes-1 ordering).
    // For Lossguide: the BFS node that was split at each step (priority-queue order).
    // Both Depthwise and Lossguide populate this; SymmetricTree leaves it empty.
    std::vector<ui32> SplitBfsNodeIds;  // [numSplits] BFS node index per split
    ui32 NumLeaves = 0;            // number of terminal leaves
};

static std::string EscapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

// Internal helper: write model JSON to a FILE* stream.
// Used by both SaveModelJSON (file) and BuildModelJSONString (in-memory).
static void WriteModelJSON(
    FILE* f,
    const std::vector<TTreeRecord>& allTrees,
    const TDataset& ds,
    const TQuantization& quant,
    const std::string& lossType,
    float lossParam,
    const TConfig& config,
    ui32 approxDim,
    ui32 numClasses,
    const std::vector<TCtrFeature>& ctrFeatures,
    const std::vector<float>& basePrediction
) {
    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"catboost-mlx-json\",\n");
    fprintf(f, "  \"version\": 1,\n");

    // model_info
    fprintf(f, "  \"model_info\": {\n");
    fprintf(f, "    \"loss_type\": \"%s\",\n", lossType.c_str());
    fprintf(f, "    \"loss_param\": %g,\n", lossParam);
    fprintf(f, "    \"learning_rate\": %g,\n", config.LearningRate);
    fprintf(f, "    \"l2_reg_lambda\": %g,\n", config.L2RegLambda);
    fprintf(f, "    \"approx_dimension\": %u,\n", approxDim);
    fprintf(f, "    \"num_classes\": %u,\n", numClasses);
    fprintf(f, "    \"num_trees\": %u,\n", static_cast<ui32>(allTrees.size()));
    fprintf(f, "    \"max_depth\": %u,\n", config.MaxDepth);
    fprintf(f, "    \"nan_mode\": \"%s\",\n", config.NanMode.c_str());
    // Base prediction (optimal starting constant for the loss function)
    fprintf(f, "    \"base_prediction\": [");
    for (size_t i = 0; i < basePrediction.size(); ++i) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%.10g", basePrediction[i]);
    }
    fprintf(f, "]\n");
    fprintf(f, "  },\n");

    // features
    fprintf(f, "  \"features\": [\n");
    for (ui32 fi = 0; fi < ds.NumFeatures; ++fi) {
        fprintf(f, "    {\n");
        fprintf(f, "      \"index\": %u,\n", fi);
        std::string name = (fi < ds.FeatureNames.size() && !ds.FeatureNames[fi].empty())
            ? ds.FeatureNames[fi] : ("f" + std::to_string(fi));
        fprintf(f, "      \"name\": \"%s\",\n", EscapeJsonString(name).c_str());
        fprintf(f, "      \"is_categorical\": %s,\n", ds.IsCategorical[fi] ? "true" : "false");
        fprintf(f, "      \"has_nan\": %s,\n", ds.HasNaN[fi] ? "true" : "false");

        // borders
        fprintf(f, "      \"borders\": [");
        for (ui32 b = 0; b < quant.Borders[fi].size(); ++b) {
            if (b > 0) fprintf(f, ", ");
            fprintf(f, "%.8g", quant.Borders[fi][b]);
        }
        fprintf(f, "],\n");

        // cat_hash_map
        fprintf(f, "      \"cat_hash_map\": {");
        if (ds.IsCategorical[fi]) {
            bool first = true;
            for (const auto& [key, val] : ds.CatHashMaps[fi]) {
                if (!first) fprintf(f, ", ");
                fprintf(f, "\"%s\": %u", EscapeJsonString(key).c_str(), val);
                first = false;
            }
        }
        fprintf(f, "}\n");

        fprintf(f, "    }%s\n", (fi + 1 < ds.NumFeatures) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // trees
    fprintf(f, "  \"trees\": [\n");
    for (ui32 ti = 0; ti < allTrees.size(); ++ti) {
        const auto& tree = allTrees[ti];
        fprintf(f, "    {\n");
        fprintf(f, "      \"depth\": %u,\n", tree.Depth);
        // DEC-029: grow_policy field lets the Python predictor dispatch to the correct
        // leaf-index algorithm (oblivious bit-packing vs BFS traversal).
        const char* gp = tree.IsLossguide ? "Lossguide"
                       : tree.IsDepthwise ? "Depthwise"
                       : "SymmetricTree";
        fprintf(f, "      \"grow_policy\": \"%s\",\n", gp);

        // splits: BFS-ordered node split array.
        // For SymmetricTree: one entry per depth level (oblivious).
        // For Depthwise: 2^depth - 1 entries (BFS node order, 0-indexed from root).
        // For Lossguide: numLeaves - 1 entries (BFS insertion order); each entry carries
        //   a "bfs_node_index" so the Python predictor can reconstruct the BFS map exactly.
        // Python compute_leaf_indices dispatches based on grow_policy.
        fprintf(f, "      \"splits\": [\n");
        for (ui32 si = 0; si < tree.SplitProps.size(); ++si) {
            if ((tree.IsLossguide || tree.IsDepthwise) && si < tree.SplitBfsNodeIds.size()) {
                // DEC-029: emit bfs_node_index for Depthwise and Lossguide so the Python
                // predictor can reconstruct the exact BFS-node → split map without
                // re-deriving traversal order from partition index arithmetic.
                // FeatureId==-1 (uint max) means a no-op node (unpruned leaf placeholder).
                fprintf(f, "        {\"feature_idx\": %u, \"bin_threshold\": %u, \"is_one_hot\": %s, \"bfs_node_index\": %u}%s\n",
                        tree.SplitProps[si].FeatureId,
                        tree.SplitProps[si].BinId,
                        tree.Splits[si].IsOneHot ? "true" : "false",
                        tree.SplitBfsNodeIds[si],
                        (si + 1 < tree.SplitProps.size()) ? "," : "");
            } else {
                fprintf(f, "        {\"feature_idx\": %u, \"bin_threshold\": %u, \"is_one_hot\": %s}%s\n",
                        tree.SplitProps[si].FeatureId,
                        tree.SplitProps[si].BinId,
                        tree.Splits[si].IsOneHot ? "true" : "false",
                        (si + 1 < tree.SplitProps.size()) ? "," : "");
            }
        }
        fprintf(f, "      ],\n");

        // leaf_values
        fprintf(f, "      \"leaf_values\": [");
        for (ui32 li = 0; li < tree.LeafValues.size(); ++li) {
            if (li > 0) fprintf(f, ", ");
            fprintf(f, "%.8g", tree.LeafValues[li]);
        }
        fprintf(f, "],\n");

        // split_gains
        fprintf(f, "      \"split_gains\": [");
        for (ui32 si = 0; si < tree.SplitProps.size(); ++si) {
            if (si > 0) fprintf(f, ", ");
            fprintf(f, "%.8g", tree.SplitProps[si].Gain);
        }
        fprintf(f, "]");

        // DEC-029: For Lossguide, serialize leaf_bfs_ids so the Python predictor can
        // build the inverse BFS-node → dense-leaf-id map for prediction routing.
        if (tree.IsLossguide && !tree.LeafBfsIds.empty()) {
            fprintf(f, ",\n      \"leaf_bfs_ids\": [");
            for (ui32 li = 0; li < tree.LeafBfsIds.size(); ++li) {
                if (li > 0) fprintf(f, ", ");
                fprintf(f, "%u", tree.LeafBfsIds[li]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "\n");

        fprintf(f, "    }%s\n", (ti + 1 < allTrees.size()) ? "," : "");
    }
    fprintf(f, "  ],\n");

    // feature_importance
    fprintf(f, "  \"feature_importance\": [\n");
    {
        std::vector<double> featureGain(ds.NumFeatures, 0.0);
        for (const auto& tree : allTrees) {
            for (ui32 level = 0; level < tree.SplitProps.size(); ++level) {
                ui32 featIdx = tree.SplitProps[level].FeatureId;
                if (featIdx < ds.NumFeatures) {
                    featureGain[featIdx] += tree.SplitProps[level].Gain;
                }
            }
        }
        double totalGain = 0.0;
        for (double g : featureGain) totalGain += g;

        std::vector<ui32> sortedIndices(ds.NumFeatures);
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&](ui32 a, ui32 b) { return featureGain[a] > featureGain[b]; });

        bool first = true;
        for (ui32 rank = 0; rank < ds.NumFeatures; ++rank) {
            ui32 fi = sortedIndices[rank];
            if (featureGain[fi] <= 0.0) continue;
            if (!first) fprintf(f, ",\n");
            std::string name = (fi < ds.FeatureNames.size() && !ds.FeatureNames[fi].empty())
                ? ds.FeatureNames[fi] : ("f" + std::to_string(fi));
            double pct = (totalGain > 0) ? 100.0 * featureGain[fi] / totalGain : 0.0;
            fprintf(f, "    {\"index\": %u, \"name\": \"%s\", \"gain\": %.10f, \"percent\": %.2f}",
                    fi, EscapeJsonString(name).c_str(), featureGain[fi], pct);
            first = false;
        }
        if (!first) fprintf(f, "\n");
    }
    fprintf(f, "  ]%s\n", ctrFeatures.empty() ? "" : ",");

    // ctr_features (only if CTR was used)
    if (!ctrFeatures.empty()) {
        fprintf(f, "  \"ctr_features\": [\n");
        for (size_t ci = 0; ci < ctrFeatures.size(); ++ci) {
            const auto& ctr = ctrFeatures[ci];
            fprintf(f, "    {\n");
            fprintf(f, "      \"original_feature_idx\": %u,\n", ctr.OrigFeatureIdx);
            fprintf(f, "      \"name\": \"%s\",\n", EscapeJsonString(ctr.Name).c_str());
            fprintf(f, "      \"prior\": %g,\n", ctr.Prior);
            fprintf(f, "      \"class_idx\": %u,\n", ctr.ClassIdx);
            fprintf(f, "      \"default_value\": %.8g,\n", ctr.DefaultCtr);
            fprintf(f, "      \"final_values\": {");
            bool first = true;
            for (const auto& [bin, val] : ctr.FinalCtrValues) {
                if (!first) fprintf(f, ", ");
                fprintf(f, "\"%u\": %.8g", bin, val);
                first = false;
            }
            fprintf(f, "},\n");
            fprintf(f, "      \"cat_hash_map\": {");
            first = true;
            for (const auto& [catStr, catBin] : ctr.OrigCatHashMap) {
                if (!first) fprintf(f, ", ");
                fprintf(f, "\"%s\": %u", EscapeJsonString(catStr).c_str(), catBin);
                first = false;
            }
            fprintf(f, "}\n");
            fprintf(f, "    }%s\n", (ci + 1 < ctrFeatures.size()) ? "," : "");
        }
        fprintf(f, "  ]\n");
    }

    fprintf(f, "}\n");
}

// Build the model JSON as an in-memory string (for nanobind API path).
std::string BuildModelJSONString(
    const std::vector<TTreeRecord>& allTrees,
    const TDataset& ds,
    const TQuantization& quant,
    const std::string& lossType,
    float lossParam,
    const TConfig& config,
    ui32 approxDim,
    ui32 numClasses,
    const std::vector<TCtrFeature>& ctrFeatures = {},
    const std::vector<float>& basePrediction = {}
) {
    char* buf = nullptr;
    size_t len = 0;
    FILE* f = open_memstream(&buf, &len);
    WriteModelJSON(f, allTrees, ds, quant, lossType, lossParam,
                   config, approxDim, numClasses, ctrFeatures, basePrediction);
    fclose(f);
    std::string result(buf, len);
    free(buf);
    return result;
}

// Save model JSON to a file (CLI path).
void SaveModelJSON(
    const std::string& path,
    const std::vector<TTreeRecord>& allTrees,
    const TDataset& ds,
    const TQuantization& quant,
    const std::string& lossType,
    float lossParam,
    const TConfig& config,
    ui32 approxDim,
    ui32 numClasses,
    const std::vector<TCtrFeature>& ctrFeatures = {},
    const std::vector<float>& basePrediction = {}
) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot open output file: %s\n", path.c_str());
        return;
    }
    WriteModelJSON(f, allTrees, ds, quant, lossType, lossParam,
                   config, approxDim, numClasses, ctrFeatures, basePrediction);
    fclose(f);
}

// ============================================================================
// Training function (extracted for reuse in cross-validation)
// ============================================================================

struct TTrainResult {
    std::vector<TTreeRecord> Trees;
    float FinalTrainLoss = 0.0f;
    float FinalTestLoss = 0.0f;
    ui32 BestIteration = 0;
    ui32 TreesBuilt = 0;
    std::vector<float> BasePrediction;  // optimal starting constant per dimension
    // Phase timing (ms, accumulated across all iterations)
    double GradMs = 0, TreeSearchMs = 0, LeafMs = 0, ApplyMs = 0;
    // Per-iteration loss history (populated for nanobind API consumers)
    std::vector<float> TrainLossHistory;
    std::vector<float> EvalLossHistory;
};

// ============================================================================
// Snapshot save/restore for training resume
// ============================================================================

struct TSnapshot {
    ui32 Iteration = 0;
    std::vector<TTreeRecord> Trees;
    std::vector<float> TrainCursor;  // flat [trainDocs] or [K*trainDocs]
    std::vector<float> ValCursor;    // flat (empty if no val)
    float BestValLoss = 1e30f;
    ui32 BestIteration = 0;
    ui32 NoImprovementCount = 0;
    std::string RngState;
    bool Valid = false;
};

void SaveSnapshot(
    const std::string& path,
    ui32 currentIteration,
    const std::vector<TTreeRecord>& trees,
    const mx::array& cursor, ui32 trainDocs,
    const mx::array& valCursor, ui32 valDocs,
    ui32 approxDim,
    float bestValLoss, ui32 bestIteration, ui32 noImprovementCount,
    const std::mt19937& rng
) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Warning: cannot save snapshot to %s\n", path.c_str()); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"snapshot_version\": 1,\n");
    fprintf(f, "  \"iteration\": %u,\n", currentIteration);
    fprintf(f, "  \"num_trees\": %u,\n", static_cast<ui32>(trees.size()));
    fprintf(f, "  \"approx_dim\": %u,\n", approxDim);
    fprintf(f, "  \"best_val_loss\": %.10g,\n", bestValLoss);
    fprintf(f, "  \"best_iteration\": %u,\n", bestIteration);
    fprintf(f, "  \"no_improvement_count\": %u,\n", noImprovementCount);

    // RNG state
    std::ostringstream rngOss;
    rngOss << rng;
    fprintf(f, "  \"rng_state\": \"%s\",\n", rngOss.str().c_str());

    // Train cursor
    mx::eval(cursor);
    ui32 cursorSize = (approxDim == 1) ? trainDocs : approxDim * trainDocs;
    const float* cPtr = cursor.data<float>();
    fprintf(f, "  \"train_cursor\": [");
    for (ui32 i = 0; i < cursorSize; ++i) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "%.10g", cPtr[i]);
    }
    fprintf(f, "],\n");

    // Val cursor
    fprintf(f, "  \"val_cursor\": [");
    if (valDocs > 0) {
        mx::eval(valCursor);
        ui32 valCursorSize = (approxDim == 1) ? valDocs : approxDim * valDocs;
        const float* vPtr = valCursor.data<float>();
        for (ui32 i = 0; i < valCursorSize; ++i) {
            if (i > 0) fprintf(f, ",");
            fprintf(f, "%.10g", vPtr[i]);
        }
    }
    fprintf(f, "],\n");

    // Trees
    fprintf(f, "  \"trees\": [\n");
    for (ui32 t = 0; t < trees.size(); ++t) {
        const auto& tree = trees[t];
        fprintf(f, "    {\"depth\": %u, \"splits\": [", tree.Depth);
        for (ui32 s = 0; s < tree.Splits.size(); ++s) {
            const auto& sp = tree.Splits[s];
            if (s > 0) fprintf(f, ",");
            fprintf(f, "{\"col\":%u,\"shift\":%u,\"mask\":%u,\"bin\":%u,\"onehot\":%s}",
                    sp.FeatureColumnIdx, sp.Shift, sp.Mask, sp.BinThreshold,
                    sp.IsOneHot ? "true" : "false");
        }
        fprintf(f, "], \"split_props\": [");
        for (ui32 s = 0; s < tree.SplitProps.size(); ++s) {
            if (s > 0) fprintf(f, ",");
            fprintf(f, "{\"feat\":%u,\"bin\":%u,\"gain\":%.10g}",
                    tree.SplitProps[s].FeatureId, tree.SplitProps[s].BinId, tree.SplitProps[s].Gain);
        }
        fprintf(f, "], \"leaf_values\": [");
        for (ui32 l = 0; l < tree.LeafValues.size(); ++l) {
            if (l > 0) fprintf(f, ",");
            fprintf(f, "%.10g", tree.LeafValues[l]);
        }
        fprintf(f, "]}%s\n", (t + 1 < trees.size()) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

TSnapshot LoadSnapshot(const std::string& path) {
    TSnapshot snap;
    std::ifstream file(path);
    if (!file.is_open()) return snap;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing: extract fields by key
    auto extractUint = [&](const std::string& key) -> ui32 {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = content.find(':', pos);
        return static_cast<ui32>(std::atoi(content.c_str() + pos + 1));
    };
    auto extractFloat = [&](const std::string& key) -> float {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        pos = content.find(':', pos);
        return std::atof(content.c_str() + pos + 1);
    };
    auto extractString = [&](const std::string& key) -> std::string {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find('"', pos + key.size() + 2);
        if (pos == std::string::npos) return "";
        auto end = content.find('"', pos + 1);
        return content.substr(pos + 1, end - pos - 1);
    };
    auto extractFloatArray = [&](const std::string& key) -> std::vector<float> {
        std::vector<float> result;
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return result;
        auto start = content.find('[', pos);
        auto end = content.find(']', start);
        if (start == std::string::npos || end == std::string::npos) return result;
        std::string arr = content.substr(start + 1, end - start - 1);
        std::stringstream ss(arr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                result.push_back(std::stof(token));
            }
        }
        return result;
    };

    ui32 version = extractUint("snapshot_version");
    if (version != 1) return snap;

    snap.Iteration = extractUint("iteration");
    ui32 numTrees = extractUint("num_trees");
    ui32 approxDim = extractUint("approx_dim");
    snap.BestValLoss = extractFloat("best_val_loss");
    snap.BestIteration = extractUint("best_iteration");
    snap.NoImprovementCount = extractUint("no_improvement_count");
    snap.RngState = extractString("rng_state");
    snap.TrainCursor = extractFloatArray("train_cursor");
    snap.ValCursor = extractFloatArray("val_cursor");

    // Parse trees: find "trees" array, then parse each tree object
    auto treesPos = content.find("\"trees\"");
    if (treesPos != std::string::npos) {
        auto arrStart = content.find('[', treesPos);
        if (arrStart != std::string::npos) {
            // Find matching closing bracket (accounting for nested arrays)
            int depth = 0;
            size_t arrEnd = arrStart;
            for (size_t i = arrStart; i < content.size(); ++i) {
                if (content[i] == '[') depth++;
                else if (content[i] == ']') { depth--; if (depth == 0) { arrEnd = i; break; } }
            }

            // Parse each tree: find { ... } blocks
            std::string treesStr = content.substr(arrStart + 1, arrEnd - arrStart - 1);
            size_t searchPos = 0;
            while (snap.Trees.size() < numTrees) {
                auto treeStart = treesStr.find('{', searchPos);
                if (treeStart == std::string::npos) break;

                // Find matching closing brace
                int bd = 0;
                size_t treeEnd = treeStart;
                for (size_t i = treeStart; i < treesStr.size(); ++i) {
                    if (treesStr[i] == '{') bd++;
                    else if (treesStr[i] == '}') { bd--; if (bd == 0) { treeEnd = i; break; } }
                }

                std::string treeStr = treesStr.substr(treeStart, treeEnd - treeStart + 1);
                TTreeRecord record;

                // Parse depth
                auto dPos = treeStr.find("\"depth\"");
                if (dPos != std::string::npos) {
                    auto cp = treeStr.find(':', dPos);
                    record.Depth = std::atoi(treeStr.c_str() + cp + 1);
                }

                // Parse leaf_values
                auto lvPos = treeStr.find("\"leaf_values\"");
                if (lvPos != std::string::npos) {
                    auto lStart = treeStr.find('[', lvPos);
                    auto lEnd = treeStr.find(']', lStart);
                    std::string lvStr = treeStr.substr(lStart + 1, lEnd - lStart - 1);
                    std::stringstream lvSs(lvStr);
                    std::string tok;
                    while (std::getline(lvSs, tok, ',')) {
                        if (!tok.empty()) record.LeafValues.push_back(std::stof(tok));
                    }
                }

                // Parse splits array
                auto splitsPos = treeStr.find("\"splits\"");
                if (splitsPos != std::string::npos) {
                    auto sStart = treeStr.find('[', splitsPos);
                    auto sEnd = treeStr.find(']', sStart);
                    std::string splitsStr = treeStr.substr(sStart + 1, sEnd - sStart - 1);
                    // Parse each split object
                    size_t sp = 0;
                    while (true) {
                        auto sb = splitsStr.find('{', sp);
                        if (sb == std::string::npos) break;
                        auto se = splitsStr.find('}', sb);
                        std::string splitObj = splitsStr.substr(sb, se - sb + 1);

                        TObliviousSplitLevel split;
                        auto extractSplitField = [&](const std::string& k) -> long long {
                            auto p2 = splitObj.find("\"" + k + "\"");
                            if (p2 == std::string::npos) return 0;
                            auto cp2 = splitObj.find(':', p2);
                            return std::atoll(splitObj.c_str() + cp2 + 1);
                        };
                        split.FeatureColumnIdx = static_cast<ui32>(extractSplitField("col"));
                        split.Shift = static_cast<ui32>(extractSplitField("shift"));
                        split.Mask = static_cast<ui32>(extractSplitField("mask"));
                        split.BinThreshold = static_cast<ui32>(extractSplitField("bin"));
                        split.IsOneHot = (splitObj.find("\"onehot\":true") != std::string::npos);
                        record.Splits.push_back(split);
                        sp = se + 1;
                    }
                }

                // Parse split_props array
                auto propsPos = treeStr.find("\"split_props\"");
                if (propsPos != std::string::npos) {
                    auto pStart = treeStr.find('[', propsPos);
                    auto pEnd = treeStr.find(']', pStart);
                    std::string propsStr = treeStr.substr(pStart + 1, pEnd - pStart - 1);
                    size_t pp = 0;
                    while (true) {
                        auto pb = propsStr.find('{', pp);
                        if (pb == std::string::npos) break;
                        auto pe = propsStr.find('}', pb);
                        std::string propObj = propsStr.substr(pb, pe - pb + 1);

                        TBestSplitProperties prop;
                        auto extractPropField = [&](const std::string& k) -> float {
                            auto p2 = propObj.find("\"" + k + "\"");
                            if (p2 == std::string::npos) return 0.0f;
                            auto cp2 = propObj.find(':', p2);
                            return std::atof(propObj.c_str() + cp2 + 1);
                        };
                        prop.FeatureId = static_cast<ui32>(extractPropField("feat"));
                        prop.BinId = static_cast<ui32>(extractPropField("bin"));
                        prop.Gain = extractPropField("gain");
                        record.SplitProps.push_back(prop);
                        pp = pe + 1;
                    }
                }

                snap.Trees.push_back(std::move(record));
                searchPos = treeEnd + 1;
            }
        }
    }

    snap.Valid = (snap.Trees.size() == numTrees && !snap.TrainCursor.empty());
    return snap;
}

// ============================================================================
// Compute optimal starting prediction (boost from average)
// ============================================================================

std::vector<float> CalcBasePrediction(
    const std::vector<float>& targets, ui32 numDocs,
    const std::string& lossType, ui32 approxDim, ui32 numClasses,
    const std::vector<float>& sampleWeights = {}
) {
    std::vector<float> basePred(approxDim, 0.0f);
    if (numDocs == 0) return basePred;

    bool hasWeights = !sampleWeights.empty();

    if (lossType == "rmse" || lossType == "mae" || lossType == "huber" || lossType == "mape") {
        // Weighted mean of targets
        double sumW = 0.0, sumWT = 0.0;
        for (ui32 d = 0; d < numDocs; ++d) {
            double w = hasWeights ? sampleWeights[d] : 1.0;
            sumW += w;
            sumWT += w * targets[d];
        }
        basePred[0] = (sumW > 0) ? static_cast<float>(sumWT / sumW) : 0.0f;
    } else if (lossType == "quantile") {
        // Weighted median (alpha=0.5 default) — use simple mean as approximation
        double sumW = 0.0, sumWT = 0.0;
        for (ui32 d = 0; d < numDocs; ++d) {
            double w = hasWeights ? sampleWeights[d] : 1.0;
            sumW += w;
            sumWT += w * targets[d];
        }
        basePred[0] = (sumW > 0) ? static_cast<float>(sumWT / sumW) : 0.0f;
    } else if (lossType == "logloss") {
        // Logit of weighted average probability: log(p / (1-p))
        double sumW = 0.0, sumWT = 0.0;
        for (ui32 d = 0; d < numDocs; ++d) {
            double w = hasWeights ? sampleWeights[d] : 1.0;
            sumW += w;
            sumWT += w * targets[d];
        }
        double avgP = (sumW > 0) ? sumWT / sumW : 0.5;
        avgP = std::max(1e-6, std::min(1.0 - 1e-6, avgP));
        basePred[0] = static_cast<float>(std::log(avgP / (1.0 - avgP)));
    } else if (lossType == "poisson") {
        // Log of weighted mean (Poisson link is exp)
        double sumW = 0.0, sumWT = 0.0;
        for (ui32 d = 0; d < numDocs; ++d) {
            double w = hasWeights ? sampleWeights[d] : 1.0;
            sumW += w;
            sumWT += w * targets[d];
        }
        double avgT = (sumW > 0) ? sumWT / sumW : 1.0;
        basePred[0] = static_cast<float>(std::log(std::max(avgT, 1e-6)));
    } else if (lossType == "tweedie") {
        // Log of weighted mean
        double sumW = 0.0, sumWT = 0.0;
        for (ui32 d = 0; d < numDocs; ++d) {
            double w = hasWeights ? sampleWeights[d] : 1.0;
            sumW += w;
            sumWT += w * targets[d];
        }
        double avgT = (sumW > 0) ? sumWT / sumW : 1.0;
        basePred[0] = static_cast<float>(std::log(std::max(avgT, 1e-6)));
    } else if (lossType == "multiclass") {
        // Per-class log-odds: log(p_k / p_ref) where p_ref is implicit last class
        // For K-1 parametrization: basePred[k] = log(count_k / count_ref)
        std::vector<double> classCounts(numClasses, 0.0);
        double totalW = 0.0;
        for (ui32 d = 0; d < numDocs; ++d) {
            double w = hasWeights ? sampleWeights[d] : 1.0;
            ui32 cls = static_cast<ui32>(targets[d]);
            if (cls < numClasses) classCounts[cls] += w;
            totalW += w;
        }
        // Reference class is the last one (index numClasses-1)
        double refCount = std::max(classCounts[numClasses - 1], 1e-6);
        for (ui32 k = 0; k < approxDim; ++k) {
            double clsCount = std::max(classCounts[k], 1e-6);
            basePred[k] = static_cast<float>(std::log(clsCount / refCount));
        }
    }
    // pairlogit, yetirank: no meaningful base prediction (relative ranking), keep 0

    return basePred;
}

TTrainResult RunTraining(
    const TConfig& config,
    const mx::array& compressedData, ui32 trainDocs,
    const mx::array& targetsArr,
    const mx::array& valCompressedData, ui32 valDocs,
    const mx::array& valTargetsArr,
    const TPackedData& packed,
    const std::string& lossType, float lossParam,
    ui32 approxDim, ui32 numClasses,
    bool printProgress,
    const std::vector<float>& trainTargetsVec = {},        // needed for ranking grad scatter
    const std::vector<ui32>& trainGroupOffsets = {},        // group offsets for ranking
    ui32 trainNumGroups = 0,
    const std::vector<float>& valTargetsVec = {},           // for NDCG on validation
    const std::vector<ui32>& valGroupOffsets = {},
    ui32 valNumGroups = 0,
    const std::vector<float>& sampleWeights = {},            // per-sample weights (empty = uniform)
    const std::string& snapshotPath = "",                    // snapshot file for save/resume
    ui32 snapshotInterval = 1                                // save every N iterations
) {
    TTrainResult result;

    // Compute base prediction (boost from average)
    mx::eval(targetsArr);
    const float* targetsPtr = targetsArr.data<float>();
    std::vector<float> targetsVecLocal(targetsPtr, targetsPtr + trainDocs);
    auto basePred = CalcBasePrediction(targetsVecLocal, trainDocs, lossType,
                                        approxDim, numClasses, sampleWeights);
    result.BasePrediction = basePred;

    bool hasBasePred = false;
    for (float v : basePred) { if (std::fabs(v) > 1e-10f) hasBasePred = true; }

    // Initialize cursor with base prediction
    mx::array cursor = mx::array(0.0f);
    mx::array valCursor = mx::array(0.0f);
    if (hasBasePred) {
        if (approxDim == 1) {
            cursor = mx::full({static_cast<int>(trainDocs)}, basePred[0], mx::float32);
            if (valDocs > 0)
                valCursor = mx::full({static_cast<int>(valDocs)}, basePred[0], mx::float32);
        } else {
            // Shape [K, trainDocs]: each row k filled with basePred[k]
            std::vector<float> initData(approxDim * trainDocs);
            for (ui32 k = 0; k < approxDim; ++k)
                for (ui32 d = 0; d < trainDocs; ++d)
                    initData[k * trainDocs + d] = basePred[k];
            cursor = mx::array(initData.data(),
                {static_cast<int>(approxDim), static_cast<int>(trainDocs)}, mx::float32);
            if (valDocs > 0) {
                std::vector<float> valInitData(approxDim * valDocs);
                for (ui32 k = 0; k < approxDim; ++k)
                    for (ui32 d = 0; d < valDocs; ++d)
                        valInitData[k * valDocs + d] = basePred[k];
                valCursor = mx::array(valInitData.data(),
                    {static_cast<int>(approxDim), static_cast<int>(valDocs)}, mx::float32);
            }
        }
        if (printProgress) {
            printf("Base prediction (boost from average):");
            for (ui32 k = 0; k < approxDim; ++k) printf(" %.6f", basePred[k]);
            printf("\n");
        }
    } else {
        cursor = (approxDim == 1)
            ? mx::zeros({static_cast<int>(trainDocs)}, mx::float32)
            : mx::zeros({static_cast<int>(approxDim), static_cast<int>(trainDocs)}, mx::float32);
    }
    if (valDocs > 0 && !hasBasePred) {
        valCursor = (approxDim == 1)
            ? mx::zeros({static_cast<int>(valDocs)}, mx::float32)
            : mx::zeros({static_cast<int>(approxDim), static_cast<int>(valDocs)}, mx::float32);
    }

    // Random number generator for subsampling
    std::mt19937 rng(config.RandomSeed);

    result.Trees.reserve(config.NumIterations);

    // Early stopping state
    float bestValLoss = std::numeric_limits<float>::infinity();
    ui32 bestIteration = 0;
    ui32 noImprovementCount = 0;

    // Snapshot resume
    ui32 startIteration = 0;
    if (!snapshotPath.empty()) {
        auto snap = LoadSnapshot(snapshotPath);
        if (snap.Valid) {
            result.Trees = std::move(snap.Trees);
            result.TreesBuilt = result.Trees.size();
            startIteration = snap.Iteration + 1;
            bestValLoss = snap.BestValLoss;
            bestIteration = snap.BestIteration;
            noImprovementCount = snap.NoImprovementCount;

            // Restore cursor
            if (!snap.TrainCursor.empty()) {
                if (approxDim == 1) {
                    cursor = mx::array(snap.TrainCursor.data(), {static_cast<int>(trainDocs)}, mx::float32);
                } else {
                    cursor = mx::array(snap.TrainCursor.data(),
                        {static_cast<int>(approxDim), static_cast<int>(trainDocs)}, mx::float32);
                }
                mx::eval(cursor);
            }
            if (!snap.ValCursor.empty() && valDocs > 0) {
                if (approxDim == 1) {
                    valCursor = mx::array(snap.ValCursor.data(), {static_cast<int>(valDocs)}, mx::float32);
                } else {
                    valCursor = mx::array(snap.ValCursor.data(),
                        {static_cast<int>(approxDim), static_cast<int>(valDocs)}, mx::float32);
                }
                mx::eval(valCursor);
            }

            // Restore RNG state
            if (!snap.RngState.empty()) {
                std::istringstream iss(snap.RngState);
                iss >> rng;
            }

            if (printProgress) printf("Resumed from snapshot: iteration %u (%u trees)\n",
                                       snap.Iteration, static_cast<ui32>(result.Trees.size()));
        }
    }

    // Ranking: precompute pairs for PairLogit (static), or generate per-iter for YetiRank
    bool isRanking = (lossType == "pairlogit" || lossType == "yetirank");
    std::vector<TPair> pairLogitPairs;
    if (lossType == "pairlogit" && !trainGroupOffsets.empty()) {
        pairLogitPairs = GeneratePairLogitPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups);
        if (printProgress) printf("PairLogit: %zu pairs from %u groups\n", pairLogitPairs.size(), trainNumGroups);
    }
    // For val set NDCG, precompute PairLogit pairs too
    std::vector<TPair> valPairLogitPairs;
    if (lossType == "pairlogit" && !valGroupOffsets.empty()) {
        valPairLogitPairs = GeneratePairLogitPairs(valTargetsVec, valGroupOffsets, valNumGroups);
    }

    // S31-T2 P5 fix: ScaleL2Reg — mirrors CPU catboost/private/libs/algo/online_predictor.h
    // ScaleL2Reg(l2Regularizer, sumAllWeights, allDocCount) = l2Regularizer * sumAllWeights / allDocCount
    // CPU always scales before passing to split-scoring and leaf-estimation.
    // MLX was passing config.L2RegLambda raw (= unscaled). For uniform weights (S28 anchor),
    // sumAllWeights == docCount so scaledL2 == L2RegLambda and this is a no-op.
    // Becomes load-bearing for any training run with non-uniform sample weights.
    float scaledL2RegLambda;
    {
        double sumAllWeights = 0.0;
        if (sampleWeights.empty()) {
            sumAllWeights = static_cast<double>(trainDocs);
        } else {
            for (float w : sampleWeights) sumAllWeights += static_cast<double>(w);
        }
        scaledL2RegLambda = static_cast<float>(
            config.L2RegLambda * (sumAllWeights / static_cast<double>(trainDocs)));
    }

#ifdef CATBOOST_MLX_STAGE_PROFILE
    NCatboostMlx::TStageProfiler stageProfiler(static_cast<int>(config.NumIterations));
#endif

    for (ui32 iter = startIteration; iter < config.NumIterations; ++iter) {
        auto iterStart = std::chrono::steady_clock::now();

#ifdef CATBOOST_MLX_STAGE_PROFILE
        stageProfiler.BeginIter(static_cast<int>(iter));
#endif

#ifdef CATBOOST_MLX_DEBUG_LEAF
        // P1 — cursor stats BEFORE gradient computation
        // P8 — target stats (only iter 0)
        if (iter == 0 || iter == 1) {
            mx::eval(cursor);
            const float* cPtr = cursor.data<float>();
            ui32 cLen = static_cast<ui32>(cursor.size());
            double cMean = 0, cStd = 0;
            float cMin = cPtr[0], cMax = cPtr[0];
            for (ui32 i = 0; i < cLen; ++i) {
                cMean += cPtr[i]; cMin = std::min(cMin, cPtr[i]); cMax = std::max(cMax, cPtr[i]);
            }
            cMean /= cLen;
            for (ui32 i = 0; i < cLen; ++i) cStd += (cPtr[i] - cMean) * (cPtr[i] - cMean);
            cStd = std::sqrt(cStd / cLen);
            printf("[DBG iter=%u] P1 cursor BEFORE grad: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                   iter, (float)cMean, (float)cStd, cMin, cMax);
            if (iter == 0) {
                printf("[DBG iter=0] P7 basePred[0]=%.6f\n", basePred.empty() ? 0.0f : basePred[0]);
                // P8 — target sanity
                mx::eval(targetsArr);
                const float* tPtr = targetsArr.data<float>();
                double tMean = 0, tStd = 0; float tMin = tPtr[0], tMax = tPtr[0];
                for (ui32 i = 0; i < trainDocs; ++i) {
                    tMean += tPtr[i]; tMin = std::min(tMin, tPtr[i]); tMax = std::max(tMax, tPtr[i]);
                }
                tMean /= trainDocs;
                for (ui32 i = 0; i < trainDocs; ++i) tStd += (tPtr[i]-tMean)*(tPtr[i]-tMean);
                tStd = std::sqrt(tStd / trainDocs);
                printf("[DBG iter=0] P8 targets: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                       (float)tMean, (float)tStd, tMin, tMax);

                // P9 — effective config dump at iter=0 start
                // Compute the noise scale that FindBestSplit will use so we can
                // compare it numerically against CPU's gradient-RMS formula.
                printf("[DBG iter=0] P9 effective_config:\n");
                printf("[DBG iter=0]   ColsampleByTree    = %.4f\n",  config.ColsampleByTree);
                printf("[DBG iter=0]   SubsampleRatio     = %.4f\n",  config.SubsampleRatio);
                printf("[DBG iter=0]   RandomStrength     = %.4f\n",  config.RandomStrength);
                printf("[DBG iter=0]   MinDataInLeaf      = %u\n",    config.MinDataInLeaf);
                printf("[DBG iter=0]   L2RegLambda        = %.4f\n",  config.L2RegLambda);
                printf("[DBG iter=0]   BootstrapType      = %s\n",    config.BootstrapType.c_str());
                printf("[DBG iter=0]   BaggingTemperature = %.4f\n",  config.BaggingTemperature);
                printf("[DBG iter=0]   MvsReg             = %.4f\n",  config.MvsReg);
                printf("[DBG iter=0]   FeatureBorderType  = GreedyLogSum (DEC-037: greedy unweighted, maxBins borders)\n");
                // DEC-028 fix: noiseScale = RandomStrength * gradRms
                // gradRms = sqrt(sum_{k,i} g_k[i]^2 / N), computed post-gradient below.
                // Approximate at iter=0 using target std (residual ≈ -y for RMSE).
                double p9_grad_rms_approx = tStd;
                printf("[DBG iter=0]   noise_formula      = rs * gradRms   (DEC-028 fix; matches CPU greedy_tensor_search.cpp:92)\n");
                printf("[DBG iter=0]   noise_scale_approx = rs * std(y)    = %.6f  (grad_rms≈std(y)=%.6f at iter=0)\n",
                       (float)(config.RandomStrength * p9_grad_rms_approx), (float)p9_grad_rms_approx);
                printf("[DBG iter=0]   noise_scale_old    = rs * N         = %.4f  (OLD formula, 16895x too large — fixed)\n",
                       config.RandomStrength * static_cast<float>(trainDocs));
            }
            fflush(stdout);
        }
#endif

        // --- Bootstrap weights ---
        // Determine effective bootstrap type
        std::string bootstrapType = config.BootstrapType;
        if (bootstrapType == "no" && config.SubsampleRatio < 1.0f) {
            bootstrapType = "bernoulli";  // --subsample without explicit type defaults to Bernoulli
        }

        bool useBootstrap = (bootstrapType != "no");

        // --- Feature subsampling ---
        std::vector<bool> featureMask;
        bool useColsample = (config.ColsampleByTree < 1.0f);
        if (useColsample) {
            ui32 numFeats = packed.Features.size();
            ui32 selectCount = static_cast<ui32>(std::ceil(config.ColsampleByTree * numFeats));
            selectCount = std::min(selectCount, numFeats);
            std::vector<ui32> featIndices(numFeats);
            std::iota(featIndices.begin(), featIndices.end(), 0);
            std::shuffle(featIndices.begin(), featIndices.end(), rng);
            featureMask.resize(numFeats, false);
            for (ui32 i = 0; i < selectCount; ++i) featureMask[featIndices[i]] = true;
        }

        // Step 1: Compute gradients per dimension
#ifdef CATBOOST_MLX_STAGE_PROFILE
        auto _prof_deriv_start = std::chrono::steady_clock::now();
#endif
        std::vector<mx::array> dimGrads, dimHess;
        dimGrads.reserve(approxDim);
        dimHess.reserve(approxDim);
        for (ui32 k = 0; k < approxDim; ++k) {
            dimGrads.push_back(mx::zeros({static_cast<int>(trainDocs)}, mx::float32));
            dimHess.push_back(mx::zeros({static_cast<int>(trainDocs)}, mx::float32));
        }

        if (lossType == "rmse") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            dimGrads[0] = mx::subtract(flatCursor, targetsArr);
            dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
        } else if (lossType == "mae") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            dimGrads[0] = mx::sign(mx::subtract(flatCursor, targetsArr));
            dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
        } else if (lossType == "quantile") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto diff = mx::subtract(flatCursor, targetsArr);
            auto isPositive = mx::greater(diff, mx::array(0.0f));
            dimGrads[0] = mx::where(isPositive, mx::array(1.0f - lossParam), mx::array(-lossParam));
            dimHess[0] = mx::ones({static_cast<int>(trainDocs)}, mx::float32);
        } else if (lossType == "huber") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto diff = mx::subtract(flatCursor, targetsArr);
            auto absDiff = mx::abs(diff);
            auto isSmall = mx::less_equal(absDiff, mx::array(lossParam));
            dimGrads[0] = mx::where(isSmall, diff, mx::multiply(mx::array(lossParam), mx::sign(diff)));
            dimHess[0] = mx::where(isSmall,
                mx::ones({static_cast<int>(trainDocs)}, mx::float32),
                mx::full({static_cast<int>(trainDocs)}, 1e-6f, mx::float32));
        } else if (lossType == "logloss") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto sigmoid = mx::sigmoid(flatCursor);
            dimGrads[0] = mx::subtract(sigmoid, targetsArr);
            dimHess[0] = mx::maximum(
                mx::multiply(sigmoid, mx::subtract(mx::array(1.0f), sigmoid)),
                mx::array(1e-16f)
            );
        } else if (lossType == "poisson") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto expPred = mx::exp(flatCursor);
            dimGrads[0] = mx::subtract(expPred, targetsArr);
            dimHess[0] = mx::maximum(expPred, mx::array(1e-6f));
        } else if (lossType == "tweedie") {
            float p = lossParam;
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            dimGrads[0] = mx::add(
                mx::multiply(mx::negative(targetsArr), mx::exp(mx::multiply(flatCursor, mx::array(1.0f - p)))),
                mx::exp(mx::multiply(flatCursor, mx::array(2.0f - p))));
            auto hess = mx::add(
                mx::multiply(mx::multiply(mx::negative(targetsArr), mx::array(1.0f - p)),
                             mx::exp(mx::multiply(flatCursor, mx::array(1.0f - p)))),
                mx::multiply(mx::array(2.0f - p),
                             mx::exp(mx::multiply(flatCursor, mx::array(2.0f - p)))));
            dimHess[0] = mx::maximum(hess, mx::array(1e-6f));
        } else if (lossType == "mape") {
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            auto absTarget = mx::maximum(mx::abs(targetsArr), mx::array(1e-6f));
            dimGrads[0] = mx::divide(mx::sign(mx::subtract(flatCursor, targetsArr)), absTarget);
            dimHess[0] = mx::divide(mx::ones({static_cast<int>(trainDocs)}, mx::float32), absTarget);
        } else if (lossType == "pairlogit" || lossType == "yetirank") {
            // Ranking: scatter pairwise gradients to per-doc arrays
            auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
            mx::eval(flatCursor);
            const float* predsPtr = flatCursor.data<float>();

            std::vector<TPair>* activePairs = nullptr;
            std::vector<TPair> yetiPairs;  // temporary storage for YetiRank pairs
            if (lossType == "pairlogit") {
                activePairs = &pairLogitPairs;
            } else {
                // YetiRank: generate fresh pairs each iteration
                yetiPairs = GenerateYetiRankPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups, rng);
                activePairs = &yetiPairs;
            }

            std::vector<float> cpuGrads, cpuHess;
            ScatterPairwiseGradients(*activePairs, predsPtr, trainDocs, cpuGrads, cpuHess);
            dimGrads[0] = mx::array(cpuGrads.data(), {static_cast<int>(trainDocs)}, mx::float32);
            dimHess[0] = mx::array(cpuHess.data(), {static_cast<int>(trainDocs)}, mx::float32);
        } else {  // multiclass
            auto sm = ComputeSoftmax(cursor, approxDim, trainDocs);
            auto targetInt = mx::astype(targetsArr, mx::uint32);
            mx::eval(sm.Probs);

            for (ui32 k = 0; k < approxDim; ++k) {
                auto isClass = mx::astype(
                    mx::equal(targetInt, mx::array(static_cast<uint32_t>(k))),
                    mx::float32
                );
                auto probK = mx::reshape(
                    mx::slice(sm.Probs, {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(trainDocs)}),
                    {static_cast<int>(trainDocs)}
                );
                dimGrads[k] = mx::subtract(probK, isClass);
                dimHess[k] = mx::maximum(
                    mx::multiply(probK, mx::subtract(mx::array(1.0f), probK)),
                    mx::array(1e-16f)
                );
            }
        }

        for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});

        // --- Apply sample weights to gradients/hessians ---
        if (!sampleWeights.empty()) {
            auto sampleWeightsArr = mx::array(sampleWeights.data(), {static_cast<int>(trainDocs)}, mx::float32);
            for (ui32 k = 0; k < approxDim; ++k) {
                dimGrads[k] = mx::multiply(dimGrads[k], sampleWeightsArr);
                dimHess[k] = mx::multiply(dimHess[k], sampleWeightsArr);
            }
            for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});
        }

        // --- Apply bootstrap weights to gradients/hessians ---
        if (useBootstrap) {
            std::vector<float> weights(trainDocs, 1.0f);

            if (bootstrapType == "bayesian") {
                float invTemp = 1.0f / std::max(config.BaggingTemperature, 1e-6f);
                std::exponential_distribution<float> expDist(invTemp);
                for (ui32 d = 0; d < trainDocs; ++d) {
                    weights[d] = expDist(rng);
                }
            } else if (bootstrapType == "bernoulli") {
                float p = config.SubsampleRatio;
                std::bernoulli_distribution bernDist(p);
                for (ui32 d = 0; d < trainDocs; ++d) {
                    weights[d] = bernDist(rng) ? 1.0f : 0.0f;
                }
            } else if (bootstrapType == "mvs") {
                // Minimum Variance Sampling: include top docs by gradient magnitude,
                // weight remaining proportionally to gradient magnitude
                float mvsReg = config.MvsReg;
                std::vector<float> gradMag(trainDocs);
                mx::eval(dimGrads[0]);
                const float* g0 = dimGrads[0].data<float>();
                for (ui32 d = 0; d < trainDocs; ++d) {
                    float mag = 0.0f;
                    if (approxDim == 1) {
                        mag = std::fabs(g0[d]);
                    } else {
                        for (ui32 k = 0; k < approxDim; ++k) {
                            mx::eval(dimGrads[k]);
                            const float* gk = dimGrads[k].data<float>();
                            mag += gk[d] * gk[d];
                        }
                        mag = std::sqrt(mag);
                    }
                    gradMag[d] = mag + mvsReg;
                }

                // Find threshold: include top fraction by gradient magnitude
                // Use nth_element (O(n)) instead of sort (O(n log n))
                std::vector<float> sortedMag = gradMag;
                ui32 topCount = static_cast<ui32>(std::ceil(config.SubsampleRatio * trainDocs));
                topCount = std::min(topCount, trainDocs);
                if (topCount > 0 && topCount < trainDocs) {
                    std::nth_element(sortedMag.begin(), sortedMag.begin() + topCount - 1,
                                     sortedMag.end(), std::greater<float>());
                }
                float threshold = (topCount < trainDocs) ? sortedMag[topCount - 1] : 0.0f;

                for (ui32 d = 0; d < trainDocs; ++d) {
                    if (gradMag[d] >= threshold) {
                        weights[d] = 1.0f;
                    } else if (threshold > 0.0f) {
                        weights[d] = gradMag[d] / threshold;
                    } else {
                        weights[d] = 1.0f;
                    }
                }
            }

            auto weightsArr = mx::array(weights.data(), {static_cast<int>(trainDocs)}, mx::float32);
            for (ui32 k = 0; k < approxDim; ++k) {
                dimGrads[k] = mx::multiply(dimGrads[k], weightsArr);
                dimHess[k] = mx::multiply(dimHess[k], weightsArr);
            }
            for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});
        }

        // Compute gradient RMS for RandomStrength noise scaling.
        // Matches CPU: CalcDerivativesStDevFromZeroPlainBoosting (greedy_tensor_search.cpp:92–106)
        //   result = sqrt( sum_{k,i} g_k[i]^2 / N )   where N = number of train docs.
        // Note: CPU divides by N (not K*N); the per-dim sum is summed then divided once by N.
        float gradRms = 0.0f;
        if (config.RandomStrength > 0.0f) {
            double sumSq = 0.0;
            for (ui32 k = 0; k < approxDim; ++k) {
                mx::eval(dimGrads[k]);
                const float* g = dimGrads[k].data<float>();
                for (ui32 d = 0; d < trainDocs; ++d) {
                    sumSq += static_cast<double>(g[d]) * static_cast<double>(g[d]);
                }
            }
            gradRms = static_cast<float>(std::sqrt(sumSq / static_cast<double>(trainDocs)));
        }

#ifdef CATBOOST_MLX_STAGE_PROFILE
        {
            // Drain GPU so the timestamp is attribution-faithful.
            for (ui32 k = 0; k < approxDim; ++k) mx::eval({dimGrads[k], dimHess[k]});
            auto _prof_deriv_end = std::chrono::steady_clock::now();
            double _prof_deriv_ms = std::chrono::duration<double, std::milli>(
                _prof_deriv_end - _prof_deriv_start).count();
            stageProfiler.AccumStage(NCatboostMlx::EStageId::Derivatives, _prof_deriv_ms);
        }
#endif
        auto tGradEnd = std::chrono::steady_clock::now();
        result.GradMs += std::chrono::duration<double, std::milli>(tGradEnd - iterStart).count();

#ifdef CATBOOST_MLX_DEBUG_LEAF
        // P2 — gradient stats; P3 — hessian stats (iters 0 and 1)
        if ((iter == 0 || iter == 1) && approxDim >= 1) {
            mx::eval(dimGrads[0], dimHess[0]);
            const float* gPtr = dimGrads[0].data<float>();
            const float* hPtr = dimHess[0].data<float>();
            double gMean=0, gStd=0, gSum=0, hMean=0, hStd=0, hSum=0;
            float gMin=gPtr[0], gMax=gPtr[0], hMin=hPtr[0], hMax=hPtr[0];
            for (ui32 i = 0; i < trainDocs; ++i) {
                gSum += gPtr[i]; gMin = std::min(gMin,gPtr[i]); gMax = std::max(gMax,gPtr[i]);
                hSum += hPtr[i]; hMin = std::min(hMin,hPtr[i]); hMax = std::max(hMax,hPtr[i]);
            }
            gMean = gSum / trainDocs; hMean = hSum / trainDocs;
            for (ui32 i = 0; i < trainDocs; ++i) {
                gStd += (gPtr[i]-gMean)*(gPtr[i]-gMean);
                hStd += (hPtr[i]-hMean)*(hPtr[i]-hMean);
            }
            gStd = std::sqrt(gStd / trainDocs); hStd = std::sqrt(hStd / trainDocs);
            printf("[DBG iter=%u] P2 grad: mean=%.6f std=%.6f sum=%.4f min=%.6f max=%.6f\n",
                   iter, (float)gMean, (float)gStd, (float)gSum, gMin, gMax);
            printf("[DBG iter=%u] P3 hess: mean=%.6f std=%.6f sum=%.4f min=%.6f max=%.6f "
                   "(expected sum=%.1f for RMSE)\n",
                   iter, (float)hMean, (float)hStd, (float)hSum, hMin, hMax,
                   (float)trainDocs);
            fflush(stdout);
        }
        // P12 — noiseScale that will be used (DEC-028 fix verification)
        if (iter == 0 && config.RandomStrength > 0.0f) {
            printf("[DBG iter=0] P12 gradRms=%.6f  noiseScale=%.6f  (rs=%.4f; CPU formula: sqrt(sum g^2 / N))\n",
                   gradRms, config.RandomStrength * gradRms, config.RandomStrength);
            fflush(stdout);
        }
#endif

        // Step 2: Greedy tree structure search
        auto partitions = mx::zeros({static_cast<int>(trainDocs)}, mx::uint32);
#ifdef CATBOOST_MLX_STAGE_PROFILE
        {
            mx::eval(partitions);
            // InitPartitions is a single zeros() kernel — time is negligible but we record it.
            stageProfiler.AccumStage(NCatboostMlx::EStageId::InitPartitions, 0.0);
        }
#endif
        std::vector<TObliviousSplitLevel> splits;
        std::vector<TBestSplitProperties> splitProps;
        std::vector<ui32> splitBfsNodeIds;  // DEC-029: BFS node index per split (Depthwise + Lossguide)
        ui32 actualTreeDepth = 0;  // number of depth levels actually searched (needed for depthwise)

        // Lossguide-specific state (only populated when GrowPolicy=="Lossguide")
        std::vector<ui32> lossguideLeafBfsIds;     // [numLeaves] BFS node index per dense leaf
        std::vector<uint32_t> lossguideLeafDocVec; // [numDocs] dense leaf id per doc
        std::unordered_map<ui32, TObliviousSplitLevel> lossguideNodeSplitMap;  // bfsIdx → split
        ui32 lossguideNumLeaves = 1;

        const bool isLossguide = (config.GrowPolicy == "Lossguide" || config.GrowPolicy == "lossguide");

        // ---- Lossguide path: best-first leaf-wise tree growth ----
        if (isLossguide) {
            // Priority queue entry: (gain, leafId, bestSplit descriptor)
            // Fix 2 (S30-COSINE-KAHAN): Gain widened to double to match TBestSplitProperties::Gain.
            struct TLeafCandidate {
                double Gain;
                ui32  LeafId;
                TBestSplitProperties Split;
                bool operator<(const TLeafCandidate& o) const { return Gain < o.Gain; }
            };

            // Per-leaf state
            lossguideLeafDocVec.assign(trainDocs, 0u);   // all docs start in leaf 0
            lossguideLeafBfsIds = {0u};                   // leaf 0 is BFS node 0 (root)
            std::vector<ui32> leafDepth = {0u};

            // Node-split map: sparse BFS index → split descriptor.
            // Using a hash map avoids O(2^depth) allocation for unbalanced trees.
            lossguideNodeSplitMap.clear();

            // Priority queue
            std::priority_queue<TLeafCandidate> pq;

            // Materialise compressedData on CPU once for the partition update loop.
            mx::eval(compressedData);
            const uint32_t* dataPtr = compressedData.data<uint32_t>();
            const ui32 lineSize = packed.NumUi32PerDoc;

            // Helper: compute the best split for a single leaf by building histograms
            // for that leaf's docs and calling FindBestSplitPerPartition.
            auto evalLeafLossguide = [&](ui32 leafId) {
                // Build partition array: 0 for docs in this leaf, 1 for others.
                // Then compute histograms for 2 partitions; use only partition 0.
                std::vector<uint32_t> leafPartVec(trainDocs);
                for (ui32 d = 0; d < trainDocs; ++d) {
                    leafPartVec[d] = (lossguideLeafDocVec[d] == leafId) ? 0u : 1u;
                }
                auto leafPartArr = mx::array(
                    reinterpret_cast<const int32_t*>(leafPartVec.data()),
                    {static_cast<int>(trainDocs)}, mx::uint32);

                auto layout2 = ComputePartitionLayout(leafPartArr, trainDocs, 2u);

                // Build histogram arrays lazily then eval in one sync.
                std::vector<mx::array> histArrays2, gradSumArr2, hessSumArr2;
                for (ui32 k = 0; k < approxDim; ++k) {
                    auto statsK = mx::concatenate({
                        mx::reshape(dimGrads[k], {1, static_cast<int>(trainDocs)}),
                        mx::reshape(dimHess[k],  {1, static_cast<int>(trainDocs)})
                    }, 0);
                    statsK = mx::reshape(statsK, {static_cast<int>(2 * trainDocs)});
                    histArrays2.push_back(DispatchHistogram(
                        compressedData, statsK,
                        layout2.DocIndices, layout2.PartOffsets, layout2.PartSizes,
                        packed.Features, packed.NumUi32PerDoc,
                        packed.TotalBinFeatures, 2u, trainDocs));
                    gradSumArr2.push_back(mx::scatter_add_axis(
                        mx::zeros({2}, mx::float32), leafPartArr, dimGrads[k], 0));
                    hessSumArr2.push_back(mx::scatter_add_axis(
                        mx::zeros({2}, mx::float32), leafPartArr, dimHess[k],  0));
                }

                std::vector<mx::array> toEval2;
                for (ui32 k = 0; k < approxDim; ++k) {
                    toEval2.push_back(histArrays2[k]);
                    toEval2.push_back(gradSumArr2[k]);
                    toEval2.push_back(hessSumArr2[k]);
                }
                mx::eval(toEval2);

                // Read to CPU
                std::vector<std::vector<float>> histData2(approxDim);
                std::vector<std::vector<TPartitionStatistics>> partStats2(approxDim);
                for (ui32 k = 0; k < approxDim; ++k) {
                    const float* hPtr = histArrays2[k].data<float>();
                    histData2[k].assign(hPtr, hPtr + 2u * 2u * packed.TotalBinFeatures);
                    partStats2[k].resize(2u);
                    const float* gPtr = gradSumArr2[k].data<float>();
                    const float* hSPtr = hessSumArr2[k].data<float>();
                    partStats2[k][0].Sum    = gPtr[0];
                    partStats2[k][0].Weight = hSPtr[0];
                    partStats2[k][1].Sum    = gPtr[1];
                    partStats2[k][1].Weight = hSPtr[1];
                }

                // Find best split for partition 0 (the leaf's docs).
                // P5: use scaledL2RegLambda (= L2RegLambda * sumAllWeights/docCount).
                auto perPartSplits = FindBestSplitPerPartition(
                    histData2, partStats2,
                    packed.Features, packed.TotalBinFeatures,
                    scaledL2RegLambda, 2u, featureMask,
                    config.RandomStrength, gradRms, &rng,
                    ParseScoreFunction(config.ScoreFunction));

                if (perPartSplits[0].Defined()) {
                    // Enforce optional max-depth limit
                    if (config.MaxDepth > 0 && leafDepth[leafId] >= config.MaxDepth) {
                        return;  // depth limit reached for this leaf
                    }
                    pq.push({perPartSplits[0].Gain, leafId, perPartSplits[0]});
                }
            };

            // Bootstrap: evaluate the root leaf.
            evalLeafLossguide(0u);

            const ui32 maxLeaves = std::max(config.MaxLeaves, 2u);
            while (lossguideNumLeaves < maxLeaves && !pq.empty()) {
                auto top = pq.top();
                pq.pop();

                const ui32 leafId    = top.LeafId;
                const auto& bestSp   = top.Split;
                const ui32 bfsNode   = lossguideLeafBfsIds[leafId];
                const ui32 myDepth   = leafDepth[leafId];
                const ui32 leftBfs   = 2u * bfsNode + 1u;
                const ui32 rightBfs  = 2u * bfsNode + 2u;
                const ui32 rightLeafId = static_cast<ui32>(lossguideLeafBfsIds.size());

                // Record the node split in the sparse map.
                const auto& feat = packed.Features[bestSp.FeatureId];
                TObliviousSplitLevel ns;
                ns.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
                ns.Shift      = feat.Shift;
                ns.Mask       = feat.Mask >> feat.Shift;
                ns.BinThreshold = bestSp.BinId;
                ns.IsOneHot   = feat.OneHotFeature;
                lossguideNodeSplitMap[bfsNode] = ns;
                splits.push_back(ns);  // also append to splits for SaveModelJSON / feature importance
                // DEC-029: populate splitProps so model JSON "splits" is non-empty for Lossguide.
                // Also record the BFS node index so WriteModelJSON can emit bfs_node_index per split,
                // allowing the Python predictor to reconstruct the BFS-node → split map exactly.
                splitProps.push_back(bestSp);
                splitBfsNodeIds.push_back(bfsNode);

                // Register children
                lossguideLeafBfsIds.push_back(rightBfs);
                lossguideLeafBfsIds[leafId] = leftBfs;
                leafDepth.push_back(myDepth + 1u);
                leafDepth[leafId] = myDepth + 1u;
                lossguideNumLeaves++;

                // Update leaf doc assignments
                for (ui32 d = 0; d < trainDocs; ++d) {
                    if (lossguideLeafDocVec[d] != leafId) continue;
                    uint32_t packed_val = dataPtr[d * lineSize + ns.FeatureColumnIdx];
                    uint32_t fv = (packed_val >> ns.Shift) & ns.Mask;
                    uint32_t goRight = ns.IsOneHot
                        ? (fv == ns.BinThreshold ? 1u : 0u)
                        : (fv >  ns.BinThreshold ? 1u : 0u);
                    if (goRight) lossguideLeafDocVec[d] = rightLeafId;
                }

                // Evaluate the two new children
                evalLeafLossguide(leafId);
                evalLeafLossguide(rightLeafId);
            }

            // splits already contains all node splits (appended during the while loop)

            // Build partitions as dense leaf ids for leaf value computation
            partitions = mx::array(
                reinterpret_cast<const int32_t*>(lossguideLeafDocVec.data()),
                {static_cast<int>(trainDocs)}, mx::uint32);
            mx::eval(partitions);

#ifdef CATBOOST_MLX_DEBUG_LEAF
            // P17 — Lossguide partition trace + node-map size (iter=0 only)
            if (iter == 0) {
                printf("[DBG P17] Lossguide iter=0: numLeaves=%u splits.size()=%zu "
                       "splitProps.size()=%zu splitBfsNodeIds.size()=%zu\n",
                       lossguideNumLeaves,
                       splits.size(), splitProps.size(), splitBfsNodeIds.size());
                printf("[DBG P17]   leaf_bfs_ids (first 8): ");
                for (ui32 b = 0; b < std::min((ui32)lossguideLeafBfsIds.size(), 8u); ++b)
                    printf("%u ", lossguideLeafBfsIds[b]);
                printf("\n");
                // Verify partitions range
                const uint32_t* p17Ptr = partitions.data<uint32_t>();
                uint32_t p17Max = 0;
                for (ui32 di = 0; di < trainDocs; ++di) p17Max = std::max(p17Max, p17Ptr[di]);
                printf("[DBG P17]   partitions.max()=%u (expected %u = numLeaves-1)\n",
                       p17Max, lossguideNumLeaves - 1);
                fflush(stdout);
            }
#endif

        } else {

        // ---- Depthwise / SymmetricTree path ----

        double dbgHistMs = 0, dbgSplitMs = 0, dbgLayoutMs = 0, dbgPartMs = 0;
        for (ui32 depth = 0; depth < config.MaxDepth; ++depth) {
            auto tD0 = std::chrono::steady_clock::now();
            ui32 numPartitions = 1u << depth;
#ifdef CATBOOST_MLX_STAGE_PROFILE
            auto _prof_layout_start = std::chrono::steady_clock::now();
#endif
            auto layout = ComputePartitionLayout(partitions, trainDocs, numPartitions);
            auto tD1 = std::chrono::steady_clock::now();
            dbgLayoutMs += std::chrono::duration<double, std::milli>(tD1 - tD0).count();
#ifdef CATBOOST_MLX_STAGE_PROFILE
            {
                mx::eval(layout.DocIndices);
                auto _prof_layout_end = std::chrono::steady_clock::now();
                double _ms = std::chrono::duration<double, std::milli>(
                    _prof_layout_end - _prof_layout_start).count();
                stageProfiler.AccumDepth(NCatboostMlx::EStageId::PartitionLayout,
                                         static_cast<int>(depth), _ms);
            }
#endif

            // Compute per-dim histograms and partition stats.
            // Build all GPU work lazily, then eval in a single sync point.
            std::vector<std::vector<float>> perDimHistData(approxDim);
            std::vector<std::vector<TPartitionStatistics>> perDimPartStats(approxDim);

            // Phase 1: build lazy computation graph (no GPU sync)
            // NOTE (Sprint 16): "no GPU sync" is wrong — DispatchHistogram has
            // mx::eval(histogram) at csv_train.cpp:953 that drains every call.
#ifdef CATBOOST_MLX_STAGE_PROFILE
            auto _phase1_start = std::chrono::steady_clock::now();
#endif
            std::vector<mx::array> histArrays;
            std::vector<mx::array> gradSumArrays;
            std::vector<mx::array> hessSumArrays;
            histArrays.reserve(approxDim);
            gradSumArrays.reserve(approxDim);
            hessSumArrays.reserve(approxDim);
            for (ui32 k = 0; k < approxDim; ++k) {
                auto statsK = mx::concatenate({
                    mx::reshape(dimGrads[k], {1, static_cast<int>(trainDocs)}),
                    mx::reshape(dimHess[k], {1, static_cast<int>(trainDocs)})
                }, 0);
                statsK = mx::reshape(statsK, {static_cast<int>(2 * trainDocs)});

                histArrays.push_back(DispatchHistogram(
                    compressedData, statsK,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    packed.Features, packed.NumUi32PerDoc,
                    packed.TotalBinFeatures, numPartitions, trainDocs
                ));
                gradSumArrays.push_back(mx::scatter_add_axis(
                    mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
                    partitions, dimGrads[k], 0));
                hessSumArrays.push_back(mx::scatter_add_axis(
                    mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
                    partitions, dimHess[k], 0));
            }

            // Count histogram for min-data-in-leaf (lazy)
            std::vector<std::vector<ui32>> countHist;
            std::vector<ui32> partDocCounts;
            std::optional<mx::array> countHistArr;
            bool needCountHist = config.MinDataInLeaf > 1;
            if (needCountHist) {
                auto onesStats = mx::ones({static_cast<int>(2 * trainDocs)}, mx::float32);
                countHistArr = DispatchHistogram(
                    compressedData, onesStats,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    packed.Features, packed.NumUi32PerDoc,
                    packed.TotalBinFeatures, numPartitions, trainDocs
                );
            }

            // Phase 2: single GPU-CPU sync for all histograms + partition stats
#ifdef CATBOOST_MLX_STAGE_PROFILE
            // The Phase 1 loop above ALSO blocks: each DispatchHistogram() call
            // ends with mx::eval(histogram) (csv_train.cpp:953). The "lazy graph"
            // comment is misleading — every iteration of the per-dim loop drains
            // the GPU. Stage 4 measures the full Phase 1+2 wall time.
            auto _prof_hist_start = _phase1_start;  // see below — set before for loop
#endif
            {
                std::vector<mx::array> toEval;
                toEval.reserve(approxDim * 3 + 2);
                for (ui32 k = 0; k < approxDim; ++k) {
                    toEval.push_back(histArrays[k]);
                    toEval.push_back(gradSumArrays[k]);
                    toEval.push_back(hessSumArrays[k]);
                }
                if (needCountHist) {
                    toEval.push_back(*countHistArr);
                    toEval.push_back(layout.PartSizes);
                }
                mx::eval(toEval);
            }
#ifdef CATBOOST_MLX_STAGE_PROFILE
            {
                auto _prof_hist_end = std::chrono::steady_clock::now();
                double _ms = std::chrono::duration<double, std::milli>(
                    _prof_hist_end - _prof_hist_start).count();
                stageProfiler.AccumDepth(NCatboostMlx::EStageId::HistogramBuild,
                                         static_cast<int>(depth), _ms);
            }
#endif

            // Phase 3: read results to CPU
#ifdef CATBOOST_MLX_STAGE_PROFILE
            auto _prof_readback_start = std::chrono::steady_clock::now();
#endif
            for (ui32 k = 0; k < approxDim; ++k) {
                const float* hData = histArrays[k].data<float>();
                perDimHistData[k].assign(hData, hData + numPartitions * 2 * packed.TotalBinFeatures);

                perDimPartStats[k].resize(numPartitions);
                const float* gsPtr = gradSumArrays[k].data<float>();
                const float* hsPtr = hessSumArrays[k].data<float>();
                for (ui32 p = 0; p < numPartitions; ++p) {
                    perDimPartStats[k][p].Sum = gsPtr[p];
                    perDimPartStats[k][p].Weight = hsPtr[p];
                }
            }
            if (needCountHist) {
                const float* chData = countHistArr->data<float>();
                countHist.assign(numPartitions, std::vector<ui32>(packed.TotalBinFeatures, 0));
                for (ui32 p = 0; p < numPartitions; ++p) {
                    const float* partData = chData + p * 2 * packed.TotalBinFeatures;
                    for (ui32 b = 0; b < packed.TotalBinFeatures; ++b) {
                        countHist[p][b] = static_cast<ui32>(partData[b] + 0.5f);
                    }
                }
                const uint32_t* psPtr = layout.PartSizes.data<uint32_t>();
                partDocCounts.assign(psPtr, psPtr + numPartitions);
            }
#ifdef CATBOOST_MLX_STAGE_PROFILE
            {
                auto _prof_readback_end = std::chrono::steady_clock::now();
                double _ms = std::chrono::duration<double, std::milli>(
                    _prof_readback_end - _prof_readback_start).count();
                stageProfiler.AccumDepth(NCatboostMlx::EStageId::CpuReadback,
                                         static_cast<int>(depth), _ms);
            }
#endif

            auto tD2 = std::chrono::steady_clock::now();
            dbgHistMs += std::chrono::duration<double, std::milli>(tD2 - tD1).count();

            const bool isDepthwise = (config.GrowPolicy == "Depthwise" || config.GrowPolicy == "depthwise");

            if (isDepthwise) {
                // Depthwise: find best split per partition.
#ifdef CATBOOST_MLX_STAGE_PROFILE
                auto _prof_split_start = std::chrono::steady_clock::now();
#endif
                // P5: use scaledL2RegLambda (= L2RegLambda * sumAllWeights/docCount).
                auto perPartSplits = FindBestSplitPerPartition(
                    perDimHistData, perDimPartStats,
                    packed.Features, packed.TotalBinFeatures,
                    scaledL2RegLambda, numPartitions, featureMask,
                    config.RandomStrength, gradRms, &rng,
                    ParseScoreFunction(config.ScoreFunction)
                );
                auto tD3 = std::chrono::steady_clock::now();
                dbgSplitMs += std::chrono::duration<double, std::milli>(tD3 - tD2).count();
#ifdef CATBOOST_MLX_STAGE_PROFILE
                {
                    auto _prof_split_end = std::chrono::steady_clock::now();
                    double _ms = std::chrono::duration<double, std::milli>(
                        _prof_split_end - _prof_split_start).count();
                    stageProfiler.AccumDepth(NCatboostMlx::EStageId::SuffixScoring,
                                             static_cast<int>(depth), _ms);
                }
#endif

                bool anyValid = false;
                for (ui32 p = 0; p < numPartitions; ++p) {
                    if (perPartSplits[p].Defined()) anyValid = true;
                    // Build node split descriptor (valid or no-op placeholder).
                    TObliviousSplitLevel nodeSplit;
                    if (perPartSplits[p].Defined()) {
                        const auto& feat = packed.Features[perPartSplits[p].FeatureId];
                        nodeSplit.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
                        nodeSplit.Shift      = feat.Shift;
                        nodeSplit.Mask       = feat.Mask >> feat.Shift;
                        nodeSplit.BinThreshold = perPartSplits[p].BinId;
                        nodeSplit.IsOneHot   = feat.OneHotFeature;
                    } else {
                        // No-op: mask==0 means all docs stay left (bit never set).
                        nodeSplit.FeatureColumnIdx = 0;
                        nodeSplit.Shift = 0; nodeSplit.Mask = 0; nodeSplit.BinThreshold = 0;
                        nodeSplit.IsOneHot = false;
                    }
                    splits.push_back(nodeSplit);
                    // DEC-029: populate splitProps for Depthwise so model JSON "splits" is
                    // non-empty and Python compute_leaf_indices_depthwise can traverse correctly.
                    // For no-op partitions, FeatureId stays at -1 sentinel so feature-importance
                    // code skips them (featIdx >= ds.NumFeatures check).
                    splitProps.push_back(perPartSplits[p]);
                    // DEC-029: compute BFS node index for partition p at depth `depth`.
                    // Partition p is a bit-packed right-turn vector: bit k = direction at depth k.
                    // BFS node is derived by traversing the tree following bits 0..depth-1 of p.
                    // This mapping is needed so the Python predictor can build nodeSplitMap
                    // by BFS index (not by position in the splits list, which is partition order).
                    ui32 bfsNode = 0u;
                    for (ui32 lvl = 0; lvl < depth; ++lvl) {
                        ui32 goRight = (p >> lvl) & 1u;
                        bfsNode = 2u * bfsNode + 1u + goRight;
                    }
                    splitBfsNodeIds.push_back(bfsNode);
                }

                if (!anyValid) break;
                actualTreeDepth = depth + 1;

                // Update partitions for all partitions in one vectorised pass.
                mx::array updateBits = mx::zeros({static_cast<int>(trainDocs)}, mx::uint32);
                for (ui32 p = 0; p < numPartitions; ++p) {
                    const auto& ns = splits[splits.size() - numPartitions + p];
                    if (ns.Mask == 0) continue;
                    auto inPart = mx::astype(
                        mx::equal(partitions, mx::array(static_cast<uint32_t>(p), mx::uint32)),
                        mx::uint32);
                    auto col = mx::reshape(
                        mx::slice(compressedData,
                            {0, static_cast<int>(ns.FeatureColumnIdx)},
                            {static_cast<int>(trainDocs), static_cast<int>(ns.FeatureColumnIdx + 1)}),
                        {static_cast<int>(trainDocs)});
                    auto fv = mx::bitwise_and(
                        mx::right_shift(col, mx::array(static_cast<uint32_t>(ns.Shift), mx::uint32)),
                        mx::array(static_cast<uint32_t>(ns.Mask), mx::uint32));
                    auto gr = ns.IsOneHot
                        ? mx::equal(fv, mx::array(static_cast<uint32_t>(ns.BinThreshold), mx::uint32))
                        : mx::greater(fv, mx::array(static_cast<uint32_t>(ns.BinThreshold), mx::uint32));
                    updateBits = mx::add(updateBits, mx::multiply(mx::astype(gr, mx::uint32), inPart));
                }
                auto bits = mx::left_shift(updateBits,
                    mx::array(static_cast<uint32_t>(depth), mx::uint32));
                partitions = mx::bitwise_or(partitions, bits);
                mx::eval(partitions);
                auto tD4 = std::chrono::steady_clock::now();
                dbgPartMs += std::chrono::duration<double, std::milli>(tD4 - tD3).count();

#ifdef CATBOOST_MLX_DEBUG_LEAF
                // P14 — Depthwise partition trace: partition distribution after each split level.
                if (iter == 0) {
                    const uint32_t* pPtr14 = partitions.data<uint32_t>();
                    uint32_t p14Max = 0;
                    std::vector<ui32> p14BinCount(1u << (depth + 1), 0u);
                    for (ui32 di = 0; di < trainDocs; ++di) {
                        p14Max = std::max(p14Max, pPtr14[di]);
                        if (pPtr14[di] < p14BinCount.size()) p14BinCount[pPtr14[di]]++;
                    }
                    printf("[DBG P14] Depthwise iter=0 after depth=%u: "
                           "numPartitions=%u partitions.max=%u\n",
                           depth, 1u << (depth + 1), p14Max);
                    printf("[DBG P14]   bincount(first 8): ");
                    for (ui32 b = 0; b < std::min((ui32)p14BinCount.size(), 8u); ++b)
                        printf("%u ", p14BinCount[b]);
                    printf("\n");
                    fflush(stdout);
                }
#endif

            } else {
                // Oblivious (SymmetricTree): one split for all partitions at this depth level.
#ifdef CATBOOST_MLX_STAGE_PROFILE
                auto _prof_split_start = std::chrono::steady_clock::now();
#endif
#ifdef COSINE_RESIDUAL_INSTRUMENT
                // S30-T1: arm the cosDen/cosNum double-shadow for this FindBestSplit call
                // when iter==0 and grow_policy==SymmetricTree (the ST anchor cell).
                // S33-PROBE-D: optional override to arm at a different iter via
                //   -DPROBE_D_ARM_AT_ITER=N (default 0). Used to capture iter=2 d=2.
#ifndef PROBE_D_ARM_AT_ITER
#  define PROBE_D_ARM_AT_ITER 0
#endif
                if (iter == PROBE_D_ARM_AT_ITER && ParseScoreFunction(config.ScoreFunction) == EScoreFunction::Cosine) {
                    g_cosInstr.dumpCosDen = true;
                    g_cosInstr.binRecords.clear();
#ifdef PROBE_E_INSTRUMENT
                    g_cosInstr.leafRecords.clear();
#endif
                }
#endif
#ifdef ITER1_AUDIT
                // S31-T3b: arm audit at iter=0 for SymmetricTree (all score functions).
                // Collect parent partition stats and arm candidate accumulation.
                if (iter == 0 && g_iter1Audit.active) {
                    g_iter1Audit.pendingLayer = TIter1AuditState::TLayerRecord{};
                    g_iter1Audit.pendingLayer.depth = static_cast<int>(depth);
                    g_iter1Audit.pendingLayer.partStats.reserve(numPartitions);
                    for (ui32 p = 0; p < numPartitions; ++p) {
                        TIter1AuditState::TLayerRecord::TPartStat ps;
                        ps.partIdx   = p;
                        // Use dim-0 partition stats (K=1 for RMSE; for K>1 we'd need to sum)
                        ps.sumG      = (approxDim > 0) ? perDimPartStats[0][p].Sum    : 0.0;
                        ps.sumH      = (approxDim > 0) ? perDimPartStats[0][p].Weight : 0.0;
                        ps.W         = ps.sumH;  // W = sumH = doc count for unweighted RMSE
                        // leaf_count: sumH should equal doc count for RMSE (hess=1)
                        ps.leafCount = static_cast<uint64_t>(std::round(ps.sumH));
                        g_iter1Audit.pendingLayer.partStats.push_back(ps);
                    }
                    g_iter1Audit.candidateAccum.clear();
                    g_iter1Audit.pendingActive = true;
                }
#endif
#ifdef COSINE_TERM_AUDIT
                // S32-T2: arm term audit at iter=0 depth=0 for SymmetricTree+Cosine only.
                // depth=0 has numPartitions=1, K=1 for RMSE — one (p,k) term per (feat,bin).
                // Disarm after depth=0 write to avoid multi-depth data mixing.
                if (iter == 0 && depth == 0 && !g_termAudit.flushed
                        && ParseScoreFunction(config.ScoreFunction) == EScoreFunction::Cosine) {
                    g_termAudit.active = true;
                    g_termAudit.records.clear();
                }
#endif  // COSINE_TERM_AUDIT
                // P5: use scaledL2RegLambda (= L2RegLambda * sumAllWeights/docCount).
                auto bestSplit = FindBestSplit(
                    perDimHistData, perDimPartStats,
                    packed.Features, packed.TotalBinFeatures,
                    scaledL2RegLambda, numPartitions, featureMask,
                    config.MinDataInLeaf, countHist, partDocCounts,
                    config.MonotoneConstraints,
                    config.RandomStrength, gradRms, &rng,
#ifdef CATBOOST_MLX_DEBUG_LEAF
                    static_cast<int>(iter), static_cast<int>(depth),
#else
                    -1, -1,
#endif
                    // S28-OBLIV-DISPATCH: thread score_function to Oblivious path.
                    // Explicit — no default. Mirrors DW dispatch from 0ea86bde21.
                    ParseScoreFunction(config.ScoreFunction)
                );
#ifdef COSINE_RESIDUAL_INSTRUMENT
                // S30-T1: flush cosDen/cosNum bin records after FindBestSplit returns
                if (!g_cosInstr.binRecords.empty()) {
                    // Write per-depth CSV
                    std::string cosPath = g_cosInstr.outDir + "/cos_accum_seed"
                        + std::to_string(g_cosInstr.seed)
                        + "_depth" + std::to_string(depth) + ".csv";
                    WriteCosAccumCSV(cosPath, g_cosInstr.binRecords);

#ifdef COSINE_D2_INSTRUMENT
                    // S30-D2: also write gain_scalar CSV (M1 measurement) with trimmed schema
                    // (feat_idx, bin, gain_f32, gain_f64, gain_abs_residual only).
                    // One file per depth — Python post-processor merges depths for M1 summary.
                    std::string gainPath = g_cosInstr.outDir + "/gain_scalar_seed"
                        + std::to_string(g_cosInstr.seed)
                        + "_depth" + std::to_string(depth) + ".csv";
                    WriteGainScalarCSV(gainPath, g_cosInstr.binRecords);
#endif  // COSINE_D2_INSTRUMENT

                    // Print per-accumulator summaries for quick gate check
                    std::vector<double> numRes, denRes, gainRes;
                    for (const auto& r : g_cosInstr.binRecords) {
                        numRes.push_back(std::fabs(
                            static_cast<double>(r.cosNum_f32) - r.cosNum_f64));
                        denRes.push_back(std::fabs(
                            static_cast<double>(r.cosDen_f32) - r.cosDen_f64));
                        gainRes.push_back(std::fabs(
                            static_cast<double>(r.gain_f32)   - r.gain_f64));
                    }
                    fprintf(stderr, "[INSTR] ST depth=%u cosNum/cosDen/gain residuals:\n", depth);
                    PrintResidualSummary("[INSTR]   cosNum", numRes);
                    PrintResidualSummary("[INSTR]   cosDen", denRes);
                    PrintResidualSummary("[INSTR]   gain  ", gainRes);
                    g_cosInstr.binRecords.clear();
                    g_cosInstr.dumpCosDen = false;
                }
#ifdef PROBE_E_INSTRUMENT
                if (!g_cosInstr.leafRecords.empty()) {
                    std::string leafPath = g_cosInstr.outDir + "/cos_leaf_seed"
                        + std::to_string(g_cosInstr.seed)
                        + "_depth" + std::to_string(depth) + ".csv";
                    WriteLeafRecordsCSV(leafPath, g_cosInstr.leafRecords);
                    // Quick stderr summary: count of skipped vs total partitions
                    size_t skipped = 0;
                    for (const auto& r : g_cosInstr.leafRecords) if (r.mlxSkipped) ++skipped;
                    fprintf(stderr, "[PROBE-E] depth=%u leaf rows=%zu skipped=%zu (%.1f%%)\n",
                            depth, g_cosInstr.leafRecords.size(), skipped,
                            100.0 * skipped / std::max<size_t>(1, g_cosInstr.leafRecords.size()));
                    g_cosInstr.leafRecords.clear();
                }
#endif
#endif
#ifdef ITER1_AUDIT
                // S31-T3b: flush completed layer record after FindBestSplit returns.
                // pendingLayer.topK is non-empty iff FindBestSplit fired the finalization
                // block (meaning at least one valid candidate was found).
                if (iter == 0 && g_iter1Audit.pendingActive) {
                    g_iter1Audit.pendingActive = false;
                    if (!g_iter1Audit.pendingLayer.topK.empty()) {
                        g_iter1Audit.layers.push_back(g_iter1Audit.pendingLayer);
                        const auto& lay = g_iter1Audit.layers.back();
                        fprintf(stderr, "[AUDIT] depth=%d: winner=(feat=%u, bin=%u, gain=%.8f)"
                                " parts=%zu topK[0]=%.8f\n",
                                lay.depth,
                                lay.winnerFeat,
                                lay.winnerBin,
                                lay.winnerGain,
                                lay.partStats.size(),
                                lay.topK.front().gain);
                    }
                }
#endif
#ifdef COSINE_TERM_AUDIT
                // S32-T2: flush term audit data after depth=0 FindBestSplit.
                // Write CSV and disarm to prevent writing at deeper depths.
                if (iter == 0 && depth == 0 && g_termAudit.active && !g_termAudit.flushed) {
                    g_termAudit.active = false;
                    g_termAudit.flushed = true;
                    std::string termPath = g_termAudit.outDir + "/mlx_terms_seed"
                        + std::to_string(g_termAudit.seed) + "_depth0.csv";
                    WriteCosineTermCSV(termPath, g_termAudit.records);
                    fprintf(stderr, "[TERM_AUDIT] depth=0 flushed — %zu candidates\n",
                            g_termAudit.records.size());
                }
#endif  // COSINE_TERM_AUDIT
                auto tD3 = std::chrono::steady_clock::now();
                dbgSplitMs += std::chrono::duration<double, std::milli>(tD3 - tD2).count();
#ifdef CATBOOST_MLX_STAGE_PROFILE
                {
                    auto _prof_split_end = std::chrono::steady_clock::now();
                    double _ms = std::chrono::duration<double, std::milli>(
                        _prof_split_end - _prof_split_start).count();
                    stageProfiler.AccumDepth(NCatboostMlx::EStageId::SuffixScoring,
                                             static_cast<int>(depth), _ms);
                }
#endif

                if (!bestSplit.Defined()) break;

                // Record split
                const auto& feat = packed.Features[bestSplit.FeatureId];
                TObliviousSplitLevel split;
                split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
                split.Shift = feat.Shift;
                split.Mask = feat.Mask >> feat.Shift;
                split.BinThreshold = bestSplit.BinId;
                split.IsOneHot = feat.OneHotFeature;
                splits.push_back(split);
                splitProps.push_back(bestSplit);
                actualTreeDepth = depth + 1;

                // Update partitions
                auto column = mx::slice(compressedData,
                    {0, static_cast<int>(split.FeatureColumnIdx)},
                    {static_cast<int>(trainDocs), static_cast<int>(split.FeatureColumnIdx + 1)});
                column = mx::reshape(column, {static_cast<int>(trainDocs)});
                auto featureValues = mx::bitwise_and(
                    mx::right_shift(column, mx::array(static_cast<uint32_t>(split.Shift), mx::uint32)),
                    mx::array(static_cast<uint32_t>(split.Mask), mx::uint32));
                auto goRight = feat.OneHotFeature
                    ? mx::equal(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32))
                    : mx::greater(featureValues, mx::array(static_cast<uint32_t>(split.BinThreshold), mx::uint32));
                auto bits = mx::left_shift(mx::astype(goRight, mx::uint32), mx::array(static_cast<uint32_t>(depth), mx::uint32));
                partitions = mx::bitwise_or(partitions, bits);
                mx::eval(partitions);
                auto tD4 = std::chrono::steady_clock::now();
                dbgPartMs += std::chrono::duration<double, std::milli>(tD4 - tD3).count();
            }
        }
        if (iter == 0 && printProgress) {
            printf("  [profile iter0] layout=%.1fms hist=%.1fms split=%.1fms part=%.1fms\n",
                   dbgLayoutMs, dbgHistMs, dbgSplitMs, dbgPartMs);
            fflush(stdout);
        }

        }  // end else (Depthwise / SymmetricTree path)

        // Stop if no splits were found (SymmetricTree/Depthwise) or no leaves were added (Lossguide).
        if (isLossguide) {
            if (lossguideNumLeaves <= 1u) {
                if (printProgress) printf("iter=%u: no valid lossguide split, stopping\n", iter);
                break;
            }
        } else {
            if (splits.empty()) {
                if (printProgress) printf("iter=%u: no valid split, stopping\n", iter);
                break;
            }
        }

        auto tTreeEnd = std::chrono::steady_clock::now();
        result.TreeSearchMs += std::chrono::duration<double, std::milli>(tTreeEnd - tGradEnd).count();

        // Step 3: Estimate leaf values (GPU scatter_add_axis)
        // For oblivious trees: numLeaves = 2^splits.size() (splits.size() == depth).
        // For depthwise trees:  numLeaves = 2^actualTreeDepth (splits.size() == 2^depth - 1 nodes).
        // For lossguide trees:  numLeaves = lossguideNumLeaves.
        const bool isDepthwiseTree = (config.GrowPolicy == "Depthwise" || config.GrowPolicy == "depthwise");
        ui32 numLeaves = isLossguide ? lossguideNumLeaves
                       : isDepthwiseTree ? (1u << actualTreeDepth)
                       : (1u << splits.size());

        auto lrArr = mx::array(config.LearningRate, mx::float32);
        auto l2Arr = mx::array(scaledL2RegLambda, mx::float32);  // P5: scaled by sumAllWeights/docCount
        auto leafTarget = mx::zeros({static_cast<int>(numLeaves)}, mx::float32);

        // Stage 6 (LeafSums) + Stage 7 (LeafValues): scatter_add + Newton step are fused here.
        // We time the whole leaf-value computation block as LeafValues (stage 7);
        // LeafSums (stage 6) is attributed separately as the scatter_add portion.
#ifdef CATBOOST_MLX_STAGE_PROFILE
        auto _prof_leafsums_start = std::chrono::steady_clock::now();
        // _prof_leafvals_start is set after the scatter_add sync, inside each branch below.
        // We declare it here so it is in scope when we compute LeafValues timing after the if/else.
        std::chrono::steady_clock::time_point _prof_leafvals_start{};
#endif
        mx::array leafValues = mx::zeros({1}, mx::float32); // placeholder, overwritten below
#ifdef COSINE_RESIDUAL_INSTRUMENT
        // S30-T1: arm leaf-sum and approx-update dumps at iter=0 for approxDim==1
        if (iter == 0 && approxDim == 1) {
            g_cosInstr.dumpLeaf   = true;
            g_cosInstr.dumpApprox = true;
        }
#endif
        if (approxDim == 1) {
            auto gSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimGrads[0], 0);
            auto hSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimHess[0], 0);
#ifdef CATBOOST_MLX_STAGE_PROFILE
            mx::eval(gSumsArr, hSumsArr);
            {
                auto _t = std::chrono::steady_clock::now();
                double _ms = std::chrono::duration<double, std::milli>(
                    _t - _prof_leafsums_start).count();
                stageProfiler.AccumStage(NCatboostMlx::EStageId::LeafSums, _ms);
                _prof_leafvals_start = _t;
            }
#endif
            leafValues = mx::negative(mx::multiply(lrArr,
                mx::divide(gSumsArr, mx::add(hSumsArr, l2Arr))));

#ifdef COSINE_RESIDUAL_INSTRUMENT
            // S30-T1: leaf-value sum residual — double-shadow of scatter_add_axis gSums/hSums
            // Only instrument at iter=0, approxDim=1 (the ST+Cosine anchor config).
            if (g_cosInstr.dumpLeaf && iter == 0) {
                // Force evaluation so we can read back float32 values
                mx::eval(gSumsArr, hSumsArr, leafValues);
                const float* gF = gSumsArr.data<float>();
                const float* hF = hSumsArr.data<float>();
                const float* lvF = leafValues.data<float>();

                // CPU double-precision scatter_add reference
                mx::eval(partitions, dimGrads[0], dimHess[0]);
                const uint32_t* pPtr  = partitions.data<uint32_t>();
                const float*    gPtr  = dimGrads[0].data<float>();
                const float*    hPtr  = dimHess[0].data<float>();

                std::vector<double> gSums_d(numLeaves, 0.0);
                std::vector<double> hSums_d(numLeaves, 0.0);
                for (ui32 i = 0; i < trainDocs; ++i) {
                    ui32 leaf = pPtr[i];
                    if (leaf < numLeaves) {
                        gSums_d[leaf] += static_cast<double>(gPtr[i]);
                        hSums_d[leaf] += static_cast<double>(hPtr[i]);
                    }
                }

                // Compute double leaf values
                double lr_d = static_cast<double>(config.LearningRate);
                double l2_d = static_cast<double>(config.L2RegLambda);
                std::vector<double> lv_d(numLeaves);
                for (ui32 b = 0; b < numLeaves; ++b) {
                    lv_d[b] = -lr_d * gSums_d[b] / (hSums_d[b] + l2_d);
                }

                // Write leaf-sum CSV
                std::string leafPath = g_cosInstr.outDir + "/leaf_sum_seed"
                    + std::to_string(g_cosInstr.seed) + ".csv";
                std::filesystem::create_directories(
                    std::filesystem::path(leafPath).parent_path());
                FILE* lf = fopen(leafPath.c_str(), "w");
                if (lf) {
                    fprintf(lf, "leaf,gSum_f32,gSum_f64,gSum_residual,"
                                "hSum_f32,hSum_f64,hSum_residual,"
                                "leafVal_f32,leafVal_f64,leafVal_residual\n");
                    std::vector<double> gResiduals, hResiduals, lvResiduals;
                    for (ui32 b = 0; b < numLeaves; ++b) {
                        double gRes = std::fabs(static_cast<double>(gF[b]) - gSums_d[b]);
                        double hRes = std::fabs(static_cast<double>(hF[b]) - hSums_d[b]);
                        double lvRes = std::fabs(static_cast<double>(lvF[b]) - lv_d[b]);
                        fprintf(lf, "%u,%.10g,%.15g,%.6e,%.10g,%.15g,%.6e,%.10g,%.15g,%.6e\n",
                                b,
                                static_cast<double>(gF[b]),  gSums_d[b],  gRes,
                                static_cast<double>(hF[b]),  hSums_d[b],  hRes,
                                static_cast<double>(lvF[b]), lv_d[b],     lvRes);
                        gResiduals.push_back(gRes);
                        hResiduals.push_back(hRes);
                        lvResiduals.push_back(lvRes);
                    }
                    fclose(lf);
                    fprintf(stderr, "[INSTR] wrote %u rows → %s\n", numLeaves, leafPath.c_str());
                    PrintResidualSummary("[INSTR] gSum_residual", gResiduals);
                    PrintResidualSummary("[INSTR] hSum_residual", hResiduals);
                    PrintResidualSummary("[INSTR] leafVal_residual", lvResiduals);
                } else {
                    fprintf(stderr, "[INSTR] ERROR: cannot open %s\n", leafPath.c_str());
                }
                g_cosInstr.dumpLeaf = false;  // only dump once
            }
#endif  // COSINE_RESIDUAL_INSTRUMENT

#ifdef CATBOOST_MLX_DEBUG_LEAF
            // P4/P5 — per-bin sums and leaf values at iter 0
            if (iter == 0) {
                mx::eval(gSumsArr, hSumsArr, leafValues);
                const float* gS = gSumsArr.data<float>();
                const float* hS = hSumsArr.data<float>();
                const float* lV = leafValues.data<float>();
                double gSumTotal = 0, hSumTotal = 0;
                for (ui32 b = 0; b < numLeaves; ++b) {
                    gSumTotal += gS[b]; hSumTotal += hS[b];
                }
                printf("[DBG iter=0] P4 gSumsArr (numLeaves=%u): ", numLeaves);
                for (ui32 b = 0; b < numLeaves && b < 8; ++b)
                    printf("%.4f ", gS[b]);
                printf("...\n");
                printf("[DBG iter=0] P4 hSumsArr: ");
                for (ui32 b = 0; b < numLeaves && b < 8; ++b)
                    printf("%.4f ", hS[b]);
                printf("...\n");
                printf("[DBG iter=0] P4 gSums.sum()=%.4f hSums.sum()=%.4f "
                       "(expected hSum=%.1f = trainDocs)\n",
                       (float)gSumTotal, (float)hSumTotal, (float)trainDocs);

                printf("[DBG iter=0] P5 leafValues: ");
                float absMaxLV = 0;
                for (ui32 b = 0; b < numLeaves && b < 8; ++b) {
                    printf("%.6f ", lV[b]);
                    absMaxLV = std::max(absMaxLV, std::fabs(lV[b]));
                }
                printf("...\n");
                // Hand-compute first leaf as sanity check
                if (hS[0] > 0) {
                    float expected_lv0 = -config.LearningRate * gS[0] / (hS[0] + config.L2RegLambda);
                    printf("[DBG iter=0] P5 expected_lv[0]=-LR*gS[0]/(hS[0]+L2)=%.6f "
                           "actual_lv[0]=%.6f diff=%.2e\n",
                           expected_lv0, lV[0], std::fabs(expected_lv0 - lV[0]));
                }

                // partitions sanity
                mx::eval(partitions);
                const uint32_t* pPtr = partitions.data<uint32_t>();
                uint32_t pMax = 0;
                for (ui32 i = 0; i < trainDocs; ++i) pMax = std::max(pMax, pPtr[i]);
                printf("[DBG iter=0] partitions.max()=%u (expected %u = numLeaves-1)\n",
                       pMax, numLeaves - 1);
                printf("[DBG iter=0] splits.size()=%zu (expected depth=%u)\n",
                       splits.size(), config.MaxDepth);
                fflush(stdout);
            }

            // P15 — Depthwise leaf values + partition populations (iter=0 only)
            if (iter == 0 && isDepthwiseTree) {
                mx::eval(leafValues);
                const float* lV15 = leafValues.data<float>();
                const uint32_t* pP15 = partitions.data<uint32_t>();
                std::vector<ui32> popCount15(numLeaves, 0u);
                for (ui32 di = 0; di < trainDocs; ++di) {
                    if (pP15[di] < numLeaves) popCount15[pP15[di]]++;
                }
                printf("[DBG P15] Depthwise iter=0 leaf_values (numLeaves=%u):\n", numLeaves);
                for (ui32 b = 0; b < numLeaves && b < 16; ++b)
                    printf("  leaf[%2u] = %+.6f  (pop=%u)\n", b, lV15[b], popCount15[b]);
                // SplitProps size: should be 2^depth-1 (non-empty after DEC-029 fix)
                printf("[DBG P15] splitProps.size()=%zu splits.size()=%zu splitBfsNodeIds.size()=%zu\n",
                       splitProps.size(), splits.size(), splitBfsNodeIds.size());
                fflush(stdout);
            }
#endif

        } else {
            std::vector<mx::array> dimLeafVals;
            dimLeafVals.reserve(approxDim);
            std::vector<mx::array> allSums;
            for (ui32 k = 0; k < approxDim; ++k) {
                auto gSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimGrads[k], 0);
                auto hSumsArr = mx::scatter_add_axis(leafTarget, partitions, dimHess[k], 0);
                allSums.push_back(gSumsArr);
                allSums.push_back(hSumsArr);
                dimLeafVals.push_back(mx::negative(mx::multiply(lrArr,
                    mx::divide(gSumsArr, mx::add(hSumsArr, l2Arr)))));
            }
#ifdef CATBOOST_MLX_STAGE_PROFILE
            mx::eval(allSums);
            {
                auto _t = std::chrono::steady_clock::now();
                double _ms = std::chrono::duration<double, std::milli>(
                    _t - _prof_leafsums_start).count();
                stageProfiler.AccumStage(NCatboostMlx::EStageId::LeafSums, _ms);
                _prof_leafvals_start = _t;
            }
#endif
            leafValues = mx::stack(dimLeafVals, 1);  // [numLeaves, approxDim]
        }
#ifdef CATBOOST_MLX_STAGE_PROFILE
        mx::eval(leafValues);
        {
            auto _prof_leafvals_end = std::chrono::steady_clock::now();
            double _ms = std::chrono::duration<double, std::milli>(
                _prof_leafvals_end - _prof_leafvals_start).count();
            stageProfiler.AccumStage(NCatboostMlx::EStageId::LeafValues, _ms);
        }
#endif

        // Post-tree monotone constraint adjustment
        if (!config.MonotoneConstraints.empty() && approxDim == 1) {
            mx::eval(leafValues);
            std::vector<float> lv(leafValues.data<float>(), leafValues.data<float>() + numLeaves);
            for (ui32 level = 0; level < splits.size(); ++level) {
                ui32 featId = splitProps[level].FeatureId;
                if (featId >= config.MonotoneConstraints.size()) continue;
                int constraint = config.MonotoneConstraints[featId];
                if (constraint == 0) continue;

                for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                    if (leaf & (1u << level)) continue;  // only process left child
                    ui32 leftLeaf = leaf;
                    ui32 rightLeaf = leaf | (1u << level);
                    float vl = lv[leftLeaf], vr = lv[rightLeaf];
                    if (constraint == 1 && vr < vl) {
                        float avg = 0.5f * (vl + vr);
                        lv[leftLeaf] = avg; lv[rightLeaf] = avg;
                    } else if (constraint == -1 && vl < vr) {
                        float avg = 0.5f * (vl + vr);
                        lv[leftLeaf] = avg; lv[rightLeaf] = avg;
                    }
                }
            }
            leafValues = mx::array(lv.data(), {static_cast<int>(numLeaves)}, mx::float32);
        }

        // Record tree
        {
            TTreeRecord record;
            record.Splits = splits;
            record.SplitProps = splitProps;
            record.SplitBfsNodeIds = splitBfsNodeIds;  // DEC-029
            record.IsDepthwise = isDepthwiseTree;
            record.IsLossguide = isLossguide;
            if (isLossguide) {
                record.Depth = 0;  // unbalanced tree; depth is not a single number
                record.NumLeaves = lossguideNumLeaves;
                record.LeafBfsIds = lossguideLeafBfsIds;
            } else {
                record.Depth = isDepthwiseTree ? actualTreeDepth : static_cast<ui32>(splits.size());
                record.NumLeaves = numLeaves;
            }
            mx::eval(leafValues);
            const float* lvPtr = leafValues.data<float>();
            ui32 totalFloats = (approxDim == 1) ? numLeaves : numLeaves * approxDim;
            record.LeafValues.assign(lvPtr, lvPtr + totalFloats);
            result.Trees.push_back(std::move(record));

        }

        auto tLeafEnd = std::chrono::steady_clock::now();
        result.LeafMs += std::chrono::duration<double, std::milli>(tLeafEnd - tTreeEnd).count();

        // Step 4: Apply tree to training data
        // For depthwise trees, partitions is already the correct leaf index (same bit-encoding
        // as oblivious trees — the partition update loop sets bit `depth` based on per-node splits).
#ifdef CATBOOST_MLX_STAGE_PROFILE
        auto _prof_apply_start = std::chrono::steady_clock::now();
#endif
        auto docLeafValues = mx::take(leafValues, mx::astype(partitions, mx::int32), 0);
        if (approxDim > 1) {
            docLeafValues = mx::transpose(docLeafValues);  // [K, numDocs]
            cursor = mx::add(cursor, docLeafValues);
        } else {
            cursor = mx::add(mx::reshape(cursor, {static_cast<int>(trainDocs)}), docLeafValues);
        }
        mx::eval(cursor);

#ifdef COSINE_RESIDUAL_INSTRUMENT
        // S30-T1: approx-update residual — double-shadow of cursor += docLeafValues (approxDim==1)
        // Compare per-doc float32 cursor increment vs double reference.
        if (g_cosInstr.dumpApprox && iter == 0 && approxDim == 1) {
            // cursor is already evaluated; docLeafValues is the per-doc leaf-value lookup.
            mx::eval(docLeafValues, leafValues, partitions);
            const float* dlvF = docLeafValues.data<float>();
            const float* lvF  = leafValues.data<float>();
            const uint32_t* pPtr = partitions.data<uint32_t>();

            // Double reference: for each doc, increment = double_leafValues[partition[i]]
            // Build double leaf values from the scalar Newton step (same as dumpLeaf block above,
            // but we may be in a separate code path — recompute from available float32 leafValues).
            // NOTE: If dumpLeaf already computed lv_d, it is out of scope here; recompute
            // as float64 upcast of the float32 leaf values (this measures the cursor-update
            // float32 → float64 step, not the full gSums residual).
            std::vector<double> incResiduals;
            incResiduals.reserve(trainDocs);

            // Write approx-update CSV
            std::string approxPath = g_cosInstr.outDir + "/approx_update_seed"
                + std::to_string(g_cosInstr.seed) + ".csv";
            std::filesystem::create_directories(
                std::filesystem::path(approxPath).parent_path());
            FILE* af = fopen(approxPath.c_str(), "w");
            if (af) {
                fprintf(af, "doc,partition,docLeafVal_f32,docLeafVal_f64,inc_residual\n");
                for (ui32 i = 0; i < trainDocs && i < 10000; ++i) {
                    // Double reference: upcast float32 leaf value for this doc's partition
                    uint32_t leaf = pPtr[i];
                    double dlvRef = (leaf < numLeaves)
                        ? static_cast<double>(lvF[leaf])
                        : 0.0;
                    double res = std::fabs(static_cast<double>(dlvF[i]) - dlvRef);
                    incResiduals.push_back(res);
                    fprintf(af, "%u,%u,%.10g,%.15g,%.6e\n",
                            i, leaf,
                            static_cast<double>(dlvF[i]), dlvRef, res);
                }
                fclose(af);
                fprintf(stderr, "[INSTR] wrote %zu rows → %s\n", incResiduals.size(), approxPath.c_str());
                PrintResidualSummary("[INSTR] approx_update_inc_residual", incResiduals);
            } else {
                fprintf(stderr, "[INSTR] ERROR: cannot open %s\n", approxPath.c_str());
            }
            g_cosInstr.dumpApprox = false;  // only dump once
        }
#endif  // COSINE_RESIDUAL_INSTRUMENT

        result.TreesBuilt++;
#ifdef CATBOOST_MLX_STAGE_PROFILE
        {
            auto _prof_apply_end = std::chrono::steady_clock::now();
            double _ms = std::chrono::duration<double, std::milli>(
                _prof_apply_end - _prof_apply_start).count();
            stageProfiler.AccumStage(NCatboostMlx::EStageId::TreeApply, _ms);
        }
#endif

#ifdef CATBOOST_MLX_DEBUG_LEAF
        // P6 — cursor stats AFTER cursor update (iters 0 and 1)
        if ((iter == 0 || iter == 1) && approxDim == 1) {
            const float* cPtr2 = cursor.data<float>();
            double cMean2=0, cStd2=0; float cMin2=cPtr2[0], cMax2=cPtr2[0];
            for (ui32 i = 0; i < trainDocs; ++i) {
                cMean2 += cPtr2[i];
                cMin2 = std::min(cMin2, cPtr2[i]); cMax2 = std::max(cMax2, cPtr2[i]);
            }
            cMean2 /= trainDocs;
            for (ui32 i = 0; i < trainDocs; ++i) cStd2 += (cPtr2[i]-cMean2)*(cPtr2[i]-cMean2);
            cStd2 = std::sqrt(cStd2 / trainDocs);
            printf("[DBG iter=%u] P6 cursor AFTER apply: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                   iter, (float)cMean2, (float)cStd2, cMin2, cMax2);
            fflush(stdout);
        }
#endif

        // Apply tree to validation data
        if (valDocs > 0) {
            mx::array valLeafIndices = mx::array(0, mx::uint32);
            if (isLossguide) {
                valLeafIndices = ComputeLeafIndicesLossguide(
                    valCompressedData, lossguideNodeSplitMap, lossguideLeafBfsIds,
                    valDocs, lossguideNumLeaves);
            } else if (isDepthwiseTree) {
                valLeafIndices = ComputeLeafIndicesDepthwise(valCompressedData, splits, splitBfsNodeIds, valDocs, actualTreeDepth);
            } else {
                valLeafIndices = ComputeLeafIndices(valCompressedData, splits, valDocs);
            }
            auto valDocLeafValues = mx::take(leafValues, mx::astype(valLeafIndices, mx::int32), 0);
            if (approxDim > 1) {
                valDocLeafValues = mx::transpose(valDocLeafValues);
                valCursor = mx::add(valCursor, valDocLeafValues);
            } else {
                valCursor = mx::add(mx::reshape(valCursor, {static_cast<int>(valDocs)}), valDocLeafValues);
            }
            mx::eval(valCursor);
        }

        auto tApplyEnd = std::chrono::steady_clock::now();
        result.ApplyMs += std::chrono::duration<double, std::milli>(tApplyEnd - tLeafEnd).count();

        // Step 5: Report loss
        auto iterEnd = std::chrono::steady_clock::now();
        auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

        // Compute train loss every iteration (for loss history + final loss tracking)
        {
#ifdef CATBOOST_MLX_STAGE_PROFILE
            auto _prof_loss_start = std::chrono::steady_clock::now();
#endif
            float trainLoss;
            float trainNDCG = 0.0f;
            if (isRanking) {
                auto flatCursor = mx::reshape(cursor, {static_cast<int>(trainDocs)});
                mx::eval(flatCursor);
                const float* predsPtr = flatCursor.data<float>();
                const auto& pairs = (lossType == "pairlogit") ? pairLogitPairs
                    : GenerateYetiRankPairs(trainTargetsVec, trainGroupOffsets, trainNumGroups, rng);
                trainLoss = ComputePairLogitLoss(pairs, predsPtr, trainDocs);
                trainNDCG = ComputeNDCG(trainTargetsVec, predsPtr, trainGroupOffsets, trainNumGroups);
            } else {
                trainLoss = ComputeLossValue(cursor, targetsArr, lossType, lossParam, approxDim, trainDocs);
            }
#ifdef CATBOOST_MLX_STAGE_PROFILE
            {
                auto _prof_loss_end = std::chrono::steady_clock::now();
                double _ms = std::chrono::duration<double, std::milli>(
                    _prof_loss_end - _prof_loss_start).count();
                stageProfiler.AccumStage(NCatboostMlx::EStageId::LossEval, _ms);
            }
#endif
            result.FinalTrainLoss = trainLoss;
            result.TrainLossHistory.push_back(trainLoss);

            // For lossguide, display the number of leaves; for others, display depth.
            const ui32 displayVal = isLossguide ? lossguideNumLeaves
                                  : isDepthwiseTree ? actualTreeDepth
                                  : static_cast<ui32>(splits.size());
            const char* sizeLabel = isLossguide ? "leaves" : "depth";

            float valLoss = 0.0f;
            float valNDCG = 0.0f;
            if (valDocs > 0) {
                if (isRanking) {
                    auto flatVal = mx::reshape(valCursor, {static_cast<int>(valDocs)});
                    mx::eval(flatVal);
                    const float* valPredsPtr = flatVal.data<float>();
                    const auto& vPairs = (lossType == "pairlogit") ? valPairLogitPairs
                        : GenerateYetiRankPairs(valTargetsVec, valGroupOffsets, valNumGroups, rng);
                    valLoss = ComputePairLogitLoss(vPairs, valPredsPtr, valDocs);
                    valNDCG = ComputeNDCG(valTargetsVec, valPredsPtr, valGroupOffsets, valNumGroups);
                } else {
                    valLoss = ComputeLossValue(valCursor, valTargetsArr, lossType, lossParam, approxDim, valDocs);
                }
                result.FinalTestLoss = valLoss;
                result.EvalLossHistory.push_back(valLoss);
            }

            // Print progress (conditional on verbose/interval)
            if (printProgress && (config.Verbose || iter % 10 == 0 || iter == config.NumIterations - 1)) {
                if (valDocs > 0) {
                    if (isRanking) {
                        printf("iter=%u  trees=%u  %s=%u  loss=%.6f  NDCG=%.4f  val_loss=%.6f  val_NDCG=%.4f  time=%lldms\n",
                               iter, result.TreesBuilt, sizeLabel, displayVal, trainLoss, trainNDCG, valLoss, valNDCG, iterMs);
                    } else {
                        printf("iter=%u  trees=%u  %s=%u  train_loss=%.6f  val_loss=%.6f  time=%lldms\n",
                               iter, result.TreesBuilt, sizeLabel, displayVal, trainLoss, valLoss, iterMs);
                    }
                } else {
                    if (isRanking) {
                        printf("iter=%u  trees=%u  %s=%u  loss=%.6f  NDCG=%.4f  time=%lldms\n",
                               iter, result.TreesBuilt, sizeLabel, displayVal, trainLoss, trainNDCG, iterMs);
                    } else {
                        printf("iter=%u  trees=%u  %s=%u  loss=%.6f  time=%lldms\n",
                               iter, result.TreesBuilt, sizeLabel, displayVal, trainLoss, iterMs);
                    }
                }
            }

            // Early stopping check
            if (valDocs > 0 && config.EarlyStoppingPatience > 0) {
                if (valLoss < bestValLoss - 1e-7f) {
                    bestValLoss = valLoss;
                    bestIteration = iter;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                    if (noImprovementCount >= config.EarlyStoppingPatience) {
                        if (printProgress)
                            printf("Early stopping at iter=%u (best val_loss=%.6f at iter=%u)\n",
                                   iter, bestValLoss, bestIteration);
                        break;
                    }
                }
            }
        }

        // Save snapshot at interval
        if (!snapshotPath.empty() && ((iter + 1) % snapshotInterval == 0 || iter == config.NumIterations - 1)) {
            SaveSnapshot(snapshotPath, iter, result.Trees, cursor, trainDocs,
                         valCursor, valDocs, approxDim,
                         bestValLoss, bestIteration, noImprovementCount, rng);
        }

#ifdef CATBOOST_MLX_STAGE_PROFILE
        stageProfiler.EndIter();
#endif
    }

#ifdef CATBOOST_MLX_STAGE_PROFILE
    // Dump stage profile JSON.
    // Output path: CATBOOST_MLX_PROFILE_PATH env var, or
    //              .cache/profiling/sprint16/stage_times_<unix_timestamp>.json by default.
    {
        const char* envPath = std::getenv("CATBOOST_MLX_PROFILE_PATH");
        std::string outPath;
        if (envPath && *envPath) {
            outPath = envPath;
        } else {
            const char* baseDir = ".cache/profiling/sprint16";
            try { std::filesystem::create_directories(baseDir); } catch (...) {}
            auto ts = static_cast<long long>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            outPath = std::string(baseDir) + "/stage_times_" + std::to_string(ts) + ".json";
        }

        std::ostringstream metaOs;
        metaOs << "{"
               << "\"build\":\"CATBOOST_MLX_STAGE_PROFILE=ON\","
               << "\"source\":\"csv_train\","
               << "\"num_iterations\":" << config.NumIterations << ","
               << "\"grow_policy\":\"" << config.GrowPolicy << "\","
               << "\"max_depth\":" << config.MaxDepth << ","
               << "\"approx_dim\":" << approxDim << ","
               << "\"num_docs\":" << trainDocs
               << "}";

        if (stageProfiler.WriteJson(outPath, metaOs.str())) {
            fprintf(stderr, "[stage_profiler] Stage profile written to %s\n", outPath.c_str());
        }
    }
#endif

    result.BestIteration = bestIteration;
    return result;
}

// ============================================================================
// Cross-validation fold creation
// ============================================================================

std::vector<ui32> CreateStratifiedFolds(
    const std::vector<float>& targets, ui32 numFolds,
    const std::string& lossType, std::mt19937& rng
) {
    ui32 numDocs = targets.size();
    std::vector<ui32> foldAssignment(numDocs, 0);

    if (lossType == "logloss" || lossType == "multiclass") {
        // Stratified: group by class, distribute evenly
        std::unordered_map<int, std::vector<ui32>> classIndices;
        for (ui32 d = 0; d < numDocs; ++d) {
            classIndices[static_cast<int>(targets[d])].push_back(d);
        }
        for (auto& [cls, indices] : classIndices) {
            std::shuffle(indices.begin(), indices.end(), rng);
            for (ui32 i = 0; i < indices.size(); ++i) {
                foldAssignment[indices[i]] = i % numFolds;
            }
        }
    } else {
        // Random assignment for regression
        std::vector<ui32> indices(numDocs);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (ui32 i = 0; i < numDocs; ++i) {
            foldAssignment[indices[i]] = i % numFolds;
        }
    }
    return foldAssignment;
}

// ============================================================================
// Main training loop (excluded when building as library via train_api.cpp)
// ============================================================================

#ifndef CATBOOST_MLX_NO_MAIN
int main(int argc, char** argv) {
    auto config = ParseArgs(argc, argv);

#ifdef COSINE_RESIDUAL_INSTRUMENT
    // S30-T1/D2: initialise instrumentation output directory and seed label.
    // Default output (T1):  docs/sprint30/t1-instrument/data/
    // Default output (D2):  docs/sprint30/d2-stack-instrument/data/  (when COSINE_D2_INSTRUMENT)
    // Both paths are overridable via COSINE_RESIDUAL_OUTDIR env var.
    {
        const char* envDir = std::getenv("COSINE_RESIDUAL_OUTDIR");
        if (envDir && *envDir) {
            g_cosInstr.outDir = std::string(envDir);
        } else {
            // Walk up from binary to find the repo root (contains catboost/mlx/ dir)
            std::filesystem::path binPath = std::filesystem::absolute(argv[0]);
            std::filesystem::path repoRoot = binPath.parent_path();
            for (int attempt = 0; attempt < 6; ++attempt) {
                if (std::filesystem::is_directory(repoRoot / "catboost" / "mlx")) break;
                repoRoot = repoRoot.parent_path();
            }
#ifdef COSINE_D2_INSTRUMENT
            g_cosInstr.outDir = (repoRoot / "docs" / "sprint30" / "d2-stack-instrument" / "data").string();
#else
            g_cosInstr.outDir = (repoRoot / "docs" / "sprint30" / "t1-instrument" / "data").string();
#endif
        }
        g_cosInstr.seed = config.RandomSeed;
        fprintf(stderr, "[INSTR] COSINE_RESIDUAL_INSTRUMENT active"
#ifdef COSINE_D2_INSTRUMENT
                " (+D2)"
#endif
                " — outDir=%s seed=%d\n",
                g_cosInstr.outDir.c_str(), g_cosInstr.seed);
        std::filesystem::create_directories(g_cosInstr.outDir);
    }
#endif  // COSINE_RESIDUAL_INSTRUMENT
#ifdef ITER1_AUDIT
    // S31-T3b: initialise iter=1 audit state.
    // The binary is purpose-built; always activate.
    // Output directory: ITER1_AUDIT_OUTDIR env var or default.
    {
        const char* envDir = std::getenv("ITER1_AUDIT_OUTDIR");
        if (envDir && *envDir) {
            g_iter1Audit.outDir = std::string(envDir);
        } else {
            std::filesystem::path binPath = std::filesystem::absolute(argv[0]);
            std::filesystem::path repoRoot = binPath.parent_path();
            for (int attempt = 0; attempt < 6; ++attempt) {
                if (std::filesystem::is_directory(repoRoot / "catboost" / "mlx")) break;
                repoRoot = repoRoot.parent_path();
            }
            g_iter1Audit.outDir = (repoRoot / "docs" / "sprint31" / "t3b-audit" / "data").string();
        }
        g_iter1Audit.seed   = config.RandomSeed;
        g_iter1Audit.active = true;
        std::filesystem::create_directories(g_iter1Audit.outDir);
        fprintf(stderr, "[AUDIT] ITER1_AUDIT active — outDir=%s seed=%d\n",
                g_iter1Audit.outDir.c_str(), g_iter1Audit.seed);
    }
#endif  // ITER1_AUDIT
#ifdef COSINE_TERM_AUDIT
    // S32-T2-INSTRUMENT: initialise per-bin term audit output directory and seed.
    // Output directory: COSINE_TERM_AUDIT_OUTDIR env var or default.
    // Active only at depth=0 iter=0 with SymmetricTree+Cosine — armed at the call site.
    {
        const char* envDir = std::getenv("COSINE_TERM_AUDIT_OUTDIR");
        if (envDir && *envDir) {
            g_termAudit.outDir = std::string(envDir);
        } else {
            std::filesystem::path binPath = std::filesystem::absolute(argv[0]);
            std::filesystem::path repoRoot = binPath.parent_path();
            for (int attempt = 0; attempt < 6; ++attempt) {
                if (std::filesystem::is_directory(repoRoot / "catboost" / "mlx")) break;
                repoRoot = repoRoot.parent_path();
            }
            g_termAudit.outDir = (repoRoot / "docs" / "sprint32" / "t2-terms" / "data").string();
        }
        g_termAudit.seed = config.RandomSeed;
        std::filesystem::create_directories(g_termAudit.outDir);
        fprintf(stderr, "[TERM_AUDIT] COSINE_TERM_AUDIT active — outDir=%s seed=%d\n",
                g_termAudit.outDir.c_str(), g_termAudit.seed);
    }
#endif  // COSINE_TERM_AUDIT

    printf("CatBoost-MLX CSV Training Tool\n");
    printf("==============================\n");

    // Parse loss type (may include parameter like "quantile:0.75" or "huber:1.5")
    auto lossConfig = ParseLossType(config.LossType);

    // Load data
    auto ds = IsBinaryFormat(config.CsvPath)
        ? LoadBinary(config.CsvPath, config.NanMode)
        : LoadCSV(config.CsvPath, config.TargetCol, config.CatFeatureCols, config.NanMode, config.GroupCol, config.WeightCol);
    printf("Loaded: %u rows, %u features from %s\n", ds.NumDocs, ds.NumFeatures, config.CsvPath.c_str());

    // Report categorical features
    ui32 numCat = 0;
    for (ui32 f = 0; f < ds.NumFeatures; ++f) if (ds.IsCategorical[f]) numCat++;
    if (numCat > 0) {
        printf("Categorical features: %u (", numCat);
        bool first = true;
        for (ui32 f = 0; f < ds.NumFeatures; ++f) {
            if (ds.IsCategorical[f]) {
                if (!first) printf(", ");
                printf("f%u:%zu cats", f, ds.CatHashMaps[f].size());
                first = false;
            }
        }
        printf(")\n");
    }

    // Report NaN features
    ui32 numNaN = 0;
    for (ui32 f = 0; f < ds.NumFeatures; ++f) if (ds.HasNaN[f]) numNaN++;
    if (numNaN > 0) {
        printf("Features with NaN: %u\n", numNaN);
    }

    // Detect or validate loss type
    std::string lossType = lossConfig.Type;
    float lossParam = lossConfig.Param;
    if (lossType == "auto") {
        lossType = DetectLossType(ds.Targets);
        printf("Auto-detected loss: %s\n", lossType.c_str());
    }

    bool isRankingLoss = (lossType == "pairlogit" || lossType == "yetirank");
    if (isRankingLoss && ds.NumGroups == 0) {
        fprintf(stderr, "Error: Ranking losses (pairlogit, yetirank) require --group-col\n");
        return 1;
    }
    if (isRankingLoss) {
        printf("Ranking: %u groups, loss=%s\n", ds.NumGroups, lossType.c_str());
    }

    // Determine approx dimension
    ui32 approxDim = 1;
    ui32 numClasses = 0;
    if (lossType == "multiclass") {
        float maxTarget = *std::max_element(ds.Targets.begin(), ds.Targets.end());
        numClasses = static_cast<ui32>(maxTarget) + 1;
        approxDim = numClasses - 1;
        printf("MultiClass: %u classes, approxDim=%u\n", numClasses, approxDim);
    }

    // CTR target encoding for high-cardinality categoricals
    ui32 preCtrNumFeatures = ds.NumFeatures;  // save for eval file validation
    std::vector<TCtrFeature> ctrFeatures;
    if (config.UseCtr) {
        ctrFeatures = ComputeCtrFeatures(ds, lossType, numClasses,
                                          config.CtrPrior, config.MaxOneHotSize, config.RandomSeed);
        if (!ctrFeatures.empty()) {
            printf("CTR features: %zu (replaced high-cardinality categoricals with >%u values)\n",
                   ctrFeatures.size(), config.MaxOneHotSize);
        }
    }

    // Quantize and pack features
    auto quant = QuantizeFeatures(ds, config.MaxBins);
    auto packed = PackFeatures(quant, ds);
    printf("Quantized: %u bin-features, %u uint32 per doc\n",
           packed.TotalBinFeatures, packed.NumUi32PerDoc);
    if (packed.TotalBinFeatures == 0) {
        fprintf(stderr, "Error: No valid features after quantization\n");
        return 1;
    }
#ifdef CATBOOST_MLX_DUMP_BORDERS
    // S31-T2-G2a: dump all feature borders for byte-match probe.
    // Output format: "BORDER\t{feature_idx}\t{border_value}" one per line.
    for (ui32 df = 0; df < static_cast<ui32>(quant.Borders.size()); ++df) {
        for (float bv : quant.Borders[df]) {
            printf("BORDER\t%u\t%.9g\n", df, bv);
        }
    }
    fflush(stdout);
    return 0;
#endif  // CATBOOST_MLX_DUMP_BORDERS
#ifdef CATBOOST_MLX_DEBUG_LEAF
    // P10 (CLI path) — quantization borders for features 0 and 1
    for (ui32 p10_f = 0; p10_f < std::min(static_cast<ui32>(quant.Borders.size()), 2u); ++p10_f) {
        const auto& b = quant.Borders[p10_f];
        printf("[DBG P10] feature_%u borders: count=%zu", p10_f, b.size());
        if (!b.empty()) {
            printf("  min=%.6f  max=%.6f", b.front(), b.back());
            printf("\n[DBG P10] feature_%u first5:", p10_f);
            for (ui32 i = 0; i < std::min((ui32)b.size(), 5u); ++i) printf(" %.6f", b[i]);
            printf("\n[DBG P10] feature_%u last5: ", p10_f);
            ui32 start = b.size() >= 5 ? (ui32)b.size() - 5 : 0;
            for (ui32 i = start; i < (ui32)b.size(); ++i) printf(" %.6f", b[i]);
        }
        printf("\n");
    }
    fflush(stdout);
#endif

    // --- Cross-validation mode ---
    if (config.CVFolds > 1) {
        if (config.EvalFraction > 0.0f) {
            fprintf(stderr, "Error: --cv and --eval-fraction are mutually exclusive\n");
            return 1;
        }

        printf("\n%u-Fold Cross-Validation\n", config.CVFolds);
        printf("---\n");

        std::mt19937 cvRng(config.RandomSeed);

        // For ranking: fold assignment at group level
        std::vector<ui32> foldAssignment;
        if (isRankingLoss && ds.NumGroups > 0) {
            // Assign folds to groups, then expand to per-doc
            std::vector<ui32> groupFolds(ds.NumGroups);
            std::vector<ui32> groupPerm(ds.NumGroups);
            std::iota(groupPerm.begin(), groupPerm.end(), 0);
            std::shuffle(groupPerm.begin(), groupPerm.end(), cvRng);
            for (ui32 i = 0; i < ds.NumGroups; ++i) {
                groupFolds[groupPerm[i]] = i % config.CVFolds;
            }
            foldAssignment.resize(ds.NumDocs);
            for (ui32 g = 0; g < ds.NumGroups; ++g) {
                for (ui32 d = ds.GroupOffsets[g]; d < ds.GroupOffsets[g + 1]; ++d) {
                    foldAssignment[d] = groupFolds[g];
                }
            }
        } else {
            foldAssignment = CreateStratifiedFolds(ds.Targets, config.CVFolds, lossType, cvRng);
        }

        std::vector<float> foldTestLoss(config.CVFolds);
        std::vector<float> foldTrainLoss(config.CVFolds);

        for (ui32 fold = 0; fold < config.CVFolds; ++fold) {
            printf("\n=== Fold %u/%u ===\n", fold + 1, config.CVFolds);

            // Collect train/test indices
            std::vector<ui32> trainIdx, testIdx;
            for (ui32 d = 0; d < ds.NumDocs; ++d) {
                if (foldAssignment[d] == fold) testIdx.push_back(d);
                else trainIdx.push_back(d);
            }

            ui32 foldTrainDocs = trainIdx.size();
            ui32 foldTestDocs = testIdx.size();
            printf("Train: %u docs, Test: %u docs\n", foldTrainDocs, foldTestDocs);

            // Build per-fold packed data from same quantization
            std::vector<uint32_t> trainData(foldTrainDocs * packed.NumUi32PerDoc);
            std::vector<uint32_t> testData(foldTestDocs * packed.NumUi32PerDoc);
            std::vector<float> trainTargets(foldTrainDocs);
            std::vector<float> testTargets(foldTestDocs);
            std::vector<float> foldTrainWeights;

            for (ui32 i = 0; i < foldTrainDocs; ++i) {
                ui32 d = trainIdx[i];
                for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                    trainData[i * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
                trainTargets[i] = ds.Targets[d];
            }
            if (!ds.Weights.empty()) {
                foldTrainWeights.resize(foldTrainDocs);
                for (ui32 i = 0; i < foldTrainDocs; ++i)
                    foldTrainWeights[i] = ds.Weights[trainIdx[i]];
            }
            for (ui32 i = 0; i < foldTestDocs; ++i) {
                ui32 d = testIdx[i];
                for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                    testData[i * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
                testTargets[i] = ds.Targets[d];
            }

            // Build group offsets for train/test subsets (ranking only)
            std::vector<ui32> foldTrainGroupOffsets, foldTestGroupOffsets;
            ui32 foldTrainNumGroups = 0, foldTestNumGroups = 0;
            if (isRankingLoss && ds.NumGroups > 0) {
                // Data within trainIdx/testIdx is already sorted by original group order
                // Rebuild group offsets from the group IDs
                auto buildGroupOffsets = [](const std::vector<ui32>& indices,
                                           const std::vector<ui32>& groupIds,
                                           std::vector<ui32>& offsets, ui32& numGroups) {
                    offsets.clear();
                    numGroups = 0;
                    if (indices.empty()) return;
                    offsets.push_back(0);
                    for (ui32 i = 1; i < indices.size(); ++i) {
                        if (groupIds[indices[i]] != groupIds[indices[i - 1]]) {
                            offsets.push_back(i);
                            numGroups++;
                        }
                    }
                    offsets.push_back(indices.size());
                    numGroups++;
                };
                buildGroupOffsets(trainIdx, ds.GroupIds, foldTrainGroupOffsets, foldTrainNumGroups);
                buildGroupOffsets(testIdx, ds.GroupIds, foldTestGroupOffsets, foldTestNumGroups);
            }

            auto foldCompressed = mx::array(
                reinterpret_cast<const int32_t*>(trainData.data()),
                {static_cast<int>(foldTrainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
            auto foldTargets = mx::array(trainTargets.data(), {static_cast<int>(foldTrainDocs)}, mx::float32);

            auto foldTestCompressed = mx::array(
                reinterpret_cast<const int32_t*>(testData.data()),
                {static_cast<int>(foldTestDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
            auto foldTestTargets = mx::array(testTargets.data(), {static_cast<int>(foldTestDocs)}, mx::float32);
            mx::eval({foldCompressed, foldTargets, foldTestCompressed, foldTestTargets});

            auto foldResult = RunTraining(
                config, foldCompressed, foldTrainDocs, foldTargets,
                foldTestCompressed, foldTestDocs, foldTestTargets,
                packed, lossType, lossParam, approxDim, numClasses,
                config.Verbose,
                trainTargets, foldTrainGroupOffsets, foldTrainNumGroups,
                testTargets, foldTestGroupOffsets, foldTestNumGroups,
                foldTrainWeights
            );

            foldTrainLoss[fold] = foldResult.FinalTrainLoss;
            foldTestLoss[fold] = foldResult.FinalTestLoss;
            printf("Fold %u: train_loss=%.6f  test_loss=%.6f  trees=%u\n",
                   fold + 1, foldResult.FinalTrainLoss, foldResult.FinalTestLoss, foldResult.TreesBuilt);
        }

        // Aggregate results
        float meanTrain = 0, meanTest = 0;
        for (ui32 f = 0; f < config.CVFolds; ++f) {
            meanTrain += foldTrainLoss[f];
            meanTest += foldTestLoss[f];
        }
        meanTrain /= config.CVFolds;
        meanTest /= config.CVFolds;

        float stdTrain = 0, stdTest = 0;
        for (ui32 f = 0; f < config.CVFolds; ++f) {
            stdTrain += (foldTrainLoss[f] - meanTrain) * (foldTrainLoss[f] - meanTrain);
            stdTest += (foldTestLoss[f] - meanTest) * (foldTestLoss[f] - meanTest);
        }
        stdTrain = std::sqrt(stdTrain / config.CVFolds);
        stdTest = std::sqrt(stdTest / config.CVFolds);

        printf("\n===================================\n");
        printf("CV Results (%u folds):\n", config.CVFolds);
        printf("  Train loss: %.6f +/- %.6f\n", meanTrain, stdTrain);
        printf("  Test  loss: %.6f +/- %.6f\n", meanTest, stdTest);
        printf("===================================\n");

        return 0;
    }

    // --- Normal training mode ---
    ui32 trainDocs = ds.NumDocs;
    ui32 valDocs = 0;

    // For ranking, we need per-doc targets and group offsets for train/val subsets
    std::vector<float> trainTargetsVecForRanking;
    std::vector<ui32> trainGroupOffsetsForRanking;
    ui32 trainNumGroupsForRanking = 0;
    std::vector<float> valTargetsVecForRanking;
    std::vector<ui32> valGroupOffsetsForRanking;
    ui32 valNumGroupsForRanking = 0;

    // Mutual exclusivity check
    if (!config.EvalFile.empty() && config.EvalFraction > 0.0f) {
        fprintf(stderr, "Error: --eval-file and --eval-fraction are mutually exclusive\n");
        return 1;
    }

    // External eval file: load, quantize with training borders, pack
    TDataset evalDs;
    TPackedData evalPacked;
    if (!config.EvalFile.empty()) {
        evalDs = IsBinaryFormat(config.EvalFile)
            ? LoadBinary(config.EvalFile, config.NanMode)
            : LoadCSV(config.EvalFile, config.TargetCol, config.CatFeatureCols,
                      config.NanMode, config.GroupCol, config.WeightCol);
        printf("Loaded eval data: %u rows, %u features from %s\n",
               evalDs.NumDocs, evalDs.NumFeatures, config.EvalFile.c_str());
        // Check against pre-CTR feature count (CTR may have added extra features to ds)
        if (evalDs.NumFeatures != preCtrNumFeatures) {
            fprintf(stderr, "Error: Eval data has %u features, training data has %u\n",
                    evalDs.NumFeatures, preCtrNumFeatures);
            return 1;
        }
        // Apply CTR transformation to eval data using training statistics
        if (!ctrFeatures.empty()) {
            ApplyCtrToEvalData(evalDs, ctrFeatures, ds);
            printf("Applied %zu CTR features to eval data (eval now has %u features)\n",
                   ctrFeatures.size(), evalDs.NumFeatures);
        }
        if (evalDs.NumFeatures != ds.NumFeatures) {
            fprintf(stderr, "Error: After CTR, eval has %u features but training has %u\n",
                    evalDs.NumFeatures, ds.NumFeatures);
            return 1;
        }
        auto evalQuant = QuantizeWithBorders(evalDs, quant, ds);
        evalPacked = PackFeatures(evalQuant, evalDs);
        valDocs = evalDs.NumDocs;
        // trainDocs stays as ds.NumDocs (use all training data)
        printf("Using external eval file: %u train, %u val\n", trainDocs, valDocs);
    }

    if (config.EvalFraction > 0.0f && config.EvalFraction < 1.0f) {
        if (isRankingLoss && ds.NumGroups > 0) {
            // Group-aware split: snap to group boundaries
            ui32 valGroups = static_cast<ui32>(ds.NumGroups * config.EvalFraction);
            if (valGroups == 0) valGroups = 1;
            ui32 trainGroups = ds.NumGroups - valGroups;
            if (trainGroups == 0) { trainGroups = 1; valGroups = ds.NumGroups - 1; }
            trainDocs = ds.GroupOffsets[trainGroups];
            valDocs = ds.NumDocs - trainDocs;
            trainNumGroupsForRanking = trainGroups;
            valNumGroupsForRanking = valGroups;
            // Build sub-group offsets
            trainGroupOffsetsForRanking.assign(ds.GroupOffsets.begin(), ds.GroupOffsets.begin() + trainGroups + 1);
            for (ui32 g = 0; g <= valGroups; ++g) {
                valGroupOffsetsForRanking.push_back(ds.GroupOffsets[trainGroups + g] - trainDocs);
            }
            printf("Validation split (group-aware): %u train (%u groups), %u val (%u groups)\n",
                   trainDocs, trainGroups, valDocs, valGroups);
        } else {
            valDocs = static_cast<ui32>(ds.NumDocs * config.EvalFraction);
            trainDocs = ds.NumDocs - valDocs;
            if (valDocs == 0 || trainDocs == 0) {
                fprintf(stderr, "Error: --eval-fraction too small or too large\n");
                return 1;
            }
            printf("Validation split: %u train, %u val (%.0f%%)\n", trainDocs, valDocs, config.EvalFraction * 100);
        }
    }

    // Build target vectors for ranking
    if (isRankingLoss) {
        if (!config.EvalFile.empty()) {
            // External eval file: use all training data for train, evalDs for val
            trainTargetsVecForRanking.assign(ds.Targets.begin(), ds.Targets.end());
            trainGroupOffsetsForRanking = ds.GroupOffsets;
            trainNumGroupsForRanking = ds.NumGroups;
            valTargetsVecForRanking.assign(evalDs.Targets.begin(), evalDs.Targets.end());
            valGroupOffsetsForRanking = evalDs.GroupOffsets;
            valNumGroupsForRanking = evalDs.NumGroups;
        } else {
            trainTargetsVecForRanking.assign(ds.Targets.begin(), ds.Targets.begin() + trainDocs);
            if (valDocs > 0) {
                valTargetsVecForRanking.assign(ds.Targets.begin() + trainDocs, ds.Targets.end());
            }
            // If no eval split, use full group offsets
            if (valDocs == 0) {
                trainGroupOffsetsForRanking = ds.GroupOffsets;
                trainNumGroupsForRanking = ds.NumGroups;
            }
        }
    }

    // Transfer to GPU — split train/val if needed
    mx::array compressedData = mx::array(0);
    mx::array targetsArr = mx::array(0.0f);
    mx::array valCompressedData = mx::array(0);
    mx::array valTargetsArr = mx::array(0.0f);

    if (!config.EvalFile.empty()) {
        // External eval file: training data = all of ds, val data = evalDs (separately packed)
        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(packed.Data.data()),
            {static_cast<int>(trainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(trainDocs)}, mx::float32);

        valCompressedData = mx::array(
            reinterpret_cast<const int32_t*>(evalPacked.Data.data()),
            {static_cast<int>(valDocs), static_cast<int>(evalPacked.NumUi32PerDoc)}, mx::uint32);
        std::vector<float> evalTargetsVec(evalDs.Targets.begin(), evalDs.Targets.end());
        valTargetsArr = mx::array(evalTargetsVec.data(), {static_cast<int>(valDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr, valCompressedData, valTargetsArr});
    } else if (valDocs > 0) {
        std::vector<uint32_t> trainData(trainDocs * packed.NumUi32PerDoc);
        std::vector<uint32_t> valData(valDocs * packed.NumUi32PerDoc);
        std::vector<float> trainTargetsVec(trainDocs);
        std::vector<float> valTargetsVec(valDocs);

        for (ui32 d = 0; d < trainDocs; ++d) {
            for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                trainData[d * packed.NumUi32PerDoc + w] = packed.Data[d * packed.NumUi32PerDoc + w];
            trainTargetsVec[d] = ds.Targets[d];
        }
        for (ui32 d = 0; d < valDocs; ++d) {
            for (ui32 w = 0; w < packed.NumUi32PerDoc; ++w)
                valData[d * packed.NumUi32PerDoc + w] = packed.Data[(trainDocs + d) * packed.NumUi32PerDoc + w];
            valTargetsVec[d] = ds.Targets[trainDocs + d];
        }

        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(trainData.data()),
            {static_cast<int>(trainDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(trainTargetsVec.data(), {static_cast<int>(trainDocs)}, mx::float32);

        valCompressedData = mx::array(
            reinterpret_cast<const int32_t*>(valData.data()),
            {static_cast<int>(valDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        valTargetsArr = mx::array(valTargetsVec.data(), {static_cast<int>(valDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr, valCompressedData, valTargetsArr});
    } else {
        compressedData = mx::array(
            reinterpret_cast<const int32_t*>(packed.Data.data()),
            {static_cast<int>(ds.NumDocs), static_cast<int>(packed.NumUi32PerDoc)}, mx::uint32);
        targetsArr = mx::array(ds.Targets.data(), {static_cast<int>(ds.NumDocs)}, mx::float32);
        mx::eval({compressedData, targetsArr});
    }

    // Print training configuration
    std::string lossDisplay = lossType;
    if (lossType == "quantile") lossDisplay += ":" + std::to_string(lossParam);
    if (lossType == "huber") lossDisplay += ":" + std::to_string(lossParam);
    if (lossType == "tweedie") lossDisplay += ":" + std::to_string(lossParam);
    printf("\nTraining: %u iterations, depth=%u, lr=%.4f, l2=%.2f, loss=%s\n",
           config.NumIterations, config.MaxDepth, config.LearningRate, config.L2RegLambda, lossDisplay.c_str());
    if (config.SubsampleRatio < 1.0f) printf("Row subsample: %.2f\n", config.SubsampleRatio);
    if (config.BootstrapType != "no") printf("Bootstrap: %s", config.BootstrapType.c_str());
    if (config.BootstrapType == "bayesian") printf(" (temperature=%.2f)", config.BaggingTemperature);
    if (config.BootstrapType == "mvs") printf(" (reg=%.2f)", config.MvsReg);
    if (config.BootstrapType != "no") printf("\n");
    if (config.ColsampleByTree < 1.0f) printf("Col subsample: %.2f\n", config.ColsampleByTree);
    if (config.WeightCol >= 0) printf("Sample weight column: %d\n", config.WeightCol);
    if (config.MinDataInLeaf > 1) printf("Min data in leaf: %u\n", config.MinDataInLeaf);
    if (!config.MonotoneConstraints.empty()) {
        printf("Monotone constraints:");
        for (auto c : config.MonotoneConstraints) printf(" %d", c);
        printf("\n");
    }
    if (config.EarlyStoppingPatience > 0) printf("Early stopping patience: %u\n", config.EarlyStoppingPatience);
    printf("---\n");

    auto startTime = std::chrono::steady_clock::now();

    // Slice sample weights for training subset only
    std::vector<float> trainWeights;
    if (!ds.Weights.empty()) {
        trainWeights.assign(ds.Weights.begin(), ds.Weights.begin() + trainDocs);
    }

    auto trainResult = RunTraining(
        config, compressedData, trainDocs, targetsArr,
        valCompressedData, valDocs, valTargetsArr,
        packed, lossType, lossParam, approxDim, numClasses,
        true,  // print progress
        trainTargetsVecForRanking, trainGroupOffsetsForRanking, trainNumGroupsForRanking,
        valTargetsVecForRanking, valGroupOffsetsForRanking, valNumGroupsForRanking,
        trainWeights,
        config.SnapshotPath, config.SnapshotInterval
    );

    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();

    printf("---\n");
    printf("Training complete: %u trees in %.2fs\n", trainResult.TreesBuilt, totalTime / 1000.0);
#ifdef ITER1_AUDIT
    // S31-T3b: flush iter=1 audit JSON after training completes.
    WriteIter1AuditJSON();
#endif
    if (trainResult.TreesBuilt > 0) {
        double n = trainResult.TreesBuilt;
        printf("Phase breakdown (avg per iter): grad=%.1fms tree=%.1fms leaf=%.1fms apply=%.1fms\n",
               trainResult.GradMs/n, trainResult.TreeSearchMs/n, trainResult.LeafMs/n, trainResult.ApplyMs/n);
    }

    // Truncate trees to best iteration when early stopping fired
    if (config.EarlyStoppingPatience > 0 && trainResult.BestIteration > 0 &&
        trainResult.BestIteration + 1 < trainResult.Trees.size()) {
        ui32 keepTrees = trainResult.BestIteration + 1;
        printf("Early stopping: keeping %u/%zu trees (best at iter %u)\n",
               keepTrees, trainResult.Trees.size(), trainResult.BestIteration);
        trainResult.Trees.resize(keepTrees);
        trainResult.TreesBuilt = keepTrees;
    }

    // Feature importance
    if (config.ShowFeatureImportance && !trainResult.Trees.empty()) {
        std::vector<double> featureGain(ds.NumFeatures, 0.0);
        for (const auto& tree : trainResult.Trees) {
            for (ui32 level = 0; level < tree.SplitProps.size(); ++level) {
                ui32 featIdx = tree.SplitProps[level].FeatureId;
                if (featIdx < ds.NumFeatures) {
                    featureGain[featIdx] += tree.SplitProps[level].Gain;
                }
            }
        }
        double totalGain = 0.0;
        for (double g : featureGain) totalGain += g;

        std::vector<ui32> sortedIndices(ds.NumFeatures);
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&](ui32 a, ui32 b) { return featureGain[a] > featureGain[b]; });

        printf("\nFeature Importance (Gain-based):\n");
        printf("%-4s  %-20s  %14s  %6s\n", "Rank", "Feature", "Gain", "%");
        printf("----  --------------------  --------------  ------\n");
        for (ui32 rank = 0; rank < ds.NumFeatures; ++rank) {
            ui32 fi = sortedIndices[rank];
            if (featureGain[fi] <= 0.0) continue;
            std::string name = (fi < ds.FeatureNames.size() && !ds.FeatureNames[fi].empty())
                ? ds.FeatureNames[fi] : ("f" + std::to_string(fi));
            double pct = (totalGain > 0) ? 100.0 * featureGain[fi] / totalGain : 0.0;
            printf("%-4u  %-20s  %14.10f  %5.1f%%\n", rank + 1, name.c_str(), featureGain[fi], pct);
        }
    }

    // Save model
    if (!config.OutputModelPath.empty() && !trainResult.Trees.empty()) {
        SaveModelJSON(config.OutputModelPath, trainResult.Trees, ds, quant, lossType, lossParam,
                      config, approxDim, numClasses, ctrFeatures, trainResult.BasePrediction);
        printf("Model saved to: %s (%u trees, %u features)\n",
               config.OutputModelPath.c_str(), static_cast<ui32>(trainResult.Trees.size()), ds.NumFeatures);
    }

    return 0;
}
#endif // CATBOOST_MLX_NO_MAIN
