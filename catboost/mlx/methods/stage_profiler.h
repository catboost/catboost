#pragma once

// stage_profiler.h — Per-stage wall-clock profiler for CatBoost-MLX (S16-01).
//
// USAGE
//   Activated by -DCATBOOST_MLX_STAGE_PROFILE at compile time.
//   When the macro is NOT defined every class and macro in this header compiles
//   to nothing — guaranteed zero overhead in release builds.
//
// DESIGN
//   TStageProfiler  — collects per-iteration, per-stage timings.
//   TStageTimer     — RAII scope timer; calls mx::eval() at scope exit so the GPU
//                     is drained before the end timestamp is captured.  This gives
//                     a synchronous, attribution-faithful view of each stage —
//                     suitable for bottleneck ranking, not for measuring realistic
//                     pipelined throughput.
//
//   STAGE_TIMER(profiler, stage, arrays)
//     Declares a TStageTimer that evals `arrays` and accumulates into `profiler`.
//     When CATBOOST_MLX_STAGE_PROFILE is undefined, expands to nothing.
//
//   STAGE_TIMER_DEPTH(profiler, stage, depth, arrays)
//     Same, but records into per-depth vectors inside the profiler.
//
// JSON OUTPUT
//   TStageProfiler::WriteJson(path, meta) — emits one JSON file with meta + all
//   per-iteration records.
//
// NOTE
//   The profiler adds mx::eval() calls that are NOT present in production builds.
//   This makes profiling builds slower than production — that is intentional.
//   Do not use profiling builds to measure end-to-end throughput.

#ifdef CATBOOST_MLX_STAGE_PROFILE

#include <mlx/mlx.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace NCatboostMlx {

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Stage identifiers
// ---------------------------------------------------------------------------

enum class EStageId : int {
    Derivatives    = 0,   // ComputeDerivatives — gradients + hessians
    InitPartitions = 1,   // InitPartitions — reset leaf assignment array
    PartitionLayout= 2,   // ComputePartitionLayout — GPU bucket sort (per depth)
    HistogramBuild = 3,   // ComputeHistograms — Metal histogram kernel (per depth)
    SuffixScoring  = 4,   // FindBestSplitGPU — suffix-sum + score kernel (per depth)
    LeafSums       = 5,   // ComputeLeafSumsGPU — Metal leaf accumulator
    LeafValues     = 6,   // ComputeLeafValues — Newton step
    TreeApply      = 7,   // Apply{Oblivious,Depthwise,Lossguide}Tree
    LossEval       = 8,   // ComputeLoss + item<float>() readback
    CpuReadback    = 9,   // .data<float>() + memcpy of histograms/stats to CPU vectors (per depth)
    kCount         = 10
};

static const char* kStageNames[static_cast<int>(EStageId::kCount)] = {
    "derivatives_ms",
    "init_partitions_ms",
    "partition_layout_ms",
    "histogram_ms",
    "suffix_scoring_ms",
    "leaf_sums_ms",
    "leaf_values_ms",
    "tree_apply_ms",
    "loss_eval_ms",
    "cpu_readback_ms",
};

// ---------------------------------------------------------------------------
// Per-iteration record
// ---------------------------------------------------------------------------

struct TDepthRecord {
    int    Depth              = 0;
    double PartitionLayoutMs  = 0.0;
    double HistogramMs        = 0.0;   // total across all approxDim passes
    double HistogramPerDimMax = 0.0;   // max single-dim pass
    double SuffixScoringMs    = 0.0;
    double CpuReadbackMs      = 0.0;   // .data<float>() + memcpy
};

struct TIterRecord {
    int    Iter            = 0;
    double StageMs[static_cast<int>(EStageId::kCount)] = {};
    double IterTotalMs     = 0.0;

    // Depth-level breakdown for stages 3/4/5.
    // Populated by structure searcher; depth vector is appended per depth visited.
    std::vector<TDepthRecord> DepthRecords;
};

// ---------------------------------------------------------------------------
// TStageProfiler
// ---------------------------------------------------------------------------

class TStageProfiler {
public:
    explicit TStageProfiler(int numIterationsHint = 100) {
        Records_.reserve(numIterationsHint);
    }

    // Call at the start of each iteration to open a fresh record.
    void BeginIter(int iter) {
        PushNewIter(iter);
        IterWallStart_ = std::chrono::steady_clock::now();
    }

    // Call at the end of each iteration to close the record.
    void EndIter() {
        if (Records_.empty()) return;
        auto now = std::chrono::steady_clock::now();
        auto& rec = Records_.back();
        rec.IterTotalMs =
            std::chrono::duration<double, std::milli>(now - IterWallStart_).count();
    }

    // Accumulate time into a flat stage slot for the current iteration.
    void AccumStage(EStageId stage, double ms) {
        if (Records_.empty()) return;
        Records_.back().StageMs[static_cast<int>(stage)] += ms;
    }

    // Accumulate time into a per-depth slot for the current iteration.
    // Creates the depth entry if it doesn't exist yet.
    void AccumDepth(EStageId stage, int depth, double ms, double perDimMs = 0.0) {
        if (Records_.empty()) return;
        auto& rec = Records_.back();
        // Grow depth vector to accommodate this depth.
        while (static_cast<int>(rec.DepthRecords.size()) <= depth) {
            TDepthRecord dr;
            dr.Depth = static_cast<int>(rec.DepthRecords.size());
            rec.DepthRecords.push_back(dr);
        }
        auto& dr = rec.DepthRecords[depth];
        switch (stage) {
            case EStageId::PartitionLayout: dr.PartitionLayoutMs  += ms; break;
            case EStageId::HistogramBuild:
                dr.HistogramMs        += ms;
                if (perDimMs > dr.HistogramPerDimMax) dr.HistogramPerDimMax = perDimMs;
                break;
            case EStageId::SuffixScoring:  dr.SuffixScoringMs    += ms; break;
            case EStageId::CpuReadback:    dr.CpuReadbackMs      += ms; break;
            default: break;
        }
        // Also add to the flat stage accumulator for totals.
        rec.StageMs[static_cast<int>(stage)] += ms;
    }

    const std::vector<TIterRecord>& GetRecords() const { return Records_; }

    // Write all recorded data as a JSON file.
    // `meta` is an arbitrary JSON object string (e.g., "{\"grow_policy\":\"SymmetricTree\"}").
    // Returns true on success.
    bool WriteJson(const std::string& path, const std::string& meta = "{}") const {
        std::ofstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "[stage_profiler] Cannot open output path: %s\n", path.c_str());
            return false;
        }

        f << "{\n";
        f << "  \"meta\": " << meta << ",\n";
        f << "  \"stage_names\": [";
        for (int s = 0; s < static_cast<int>(EStageId::kCount); ++s) {
            if (s > 0) f << ", ";
            f << "\"" << kStageNames[s] << "\"";
        }
        f << "],\n";
        f << "  \"iterations\": [\n";

        for (size_t i = 0; i < Records_.size(); ++i) {
            const auto& rec = Records_[i];
            f << "    {\n";
            f << "      \"iter\": " << rec.Iter << ",\n";

            // Flat stage times — always comma-separated; iter_total_ms closes the object.
            for (int s = 0; s < static_cast<int>(EStageId::kCount); ++s) {
                f << "      \"" << kStageNames[s] << "\": "
                  << rec.StageMs[s] << ",\n";
            }

            // Depth-level records (optional)
            if (!rec.DepthRecords.empty()) {
                f << "      \"depth_stages\": [\n";
                for (size_t d = 0; d < rec.DepthRecords.size(); ++d) {
                    const auto& dr = rec.DepthRecords[d];
                    f << "        {"
                      << "\"depth\": " << dr.Depth
                      << ", \"partition_layout_ms\": " << dr.PartitionLayoutMs
                      << ", \"histogram_ms\": " << dr.HistogramMs
                      << ", \"histogram_perdim_max_ms\": " << dr.HistogramPerDimMax
                      << ", \"suffix_scoring_ms\": " << dr.SuffixScoringMs
                      << ", \"cpu_readback_ms\": " << dr.CpuReadbackMs
                      << "}";
                    if (d + 1 < rec.DepthRecords.size()) f << ",";
                    f << "\n";
                }
                f << "      ],\n";
            }

            f << "      \"iter_total_ms\": " << rec.IterTotalMs << "\n";
            f << "    }";
            if (i + 1 < Records_.size()) f << ",";
            f << "\n";
        }

        f << "  ]\n";
        f << "}\n";
        f.flush();
        return f.good();
    }

private:
    void PushNewIter(int iter) {
        TIterRecord rec;
        rec.Iter = iter;
        Records_.push_back(rec);
    }

    std::vector<TIterRecord>                        Records_;
    std::chrono::steady_clock::time_point           IterWallStart_;
};

// ---------------------------------------------------------------------------
// TStageTimer — RAII scope timer
// ---------------------------------------------------------------------------
//
// On construction: captures wall-clock start time.
// On destruction:  mx::eval(arrays) to drain GPU, then captures end time and
//                  accumulates into the profiler.
//
// `depth` == -1  → flat stage accumulation (no depth breakdown).
// `depth` >= 0   → depth-level accumulation + flat total.
// `perDimMs`     → if > 0, passed through to AccumDepth for per-dim max tracking.

class TStageTimer {
public:
    // Constructor for flat stage timer (stages 1, 2, 6, 7, 8, 9).
    TStageTimer(TStageProfiler* profiler,
                EStageId stage,
                std::initializer_list<mx::array> arrays)
        : Profiler_(profiler)
        , Stage_(stage)
        , Depth_(-1)
        , Arrays_(arrays)
        , Start_(std::chrono::steady_clock::now())
    {}

    // Constructor for depth-aware timer (stages 3, 4, 5).
    TStageTimer(TStageProfiler* profiler,
                EStageId stage,
                int depth,
                std::initializer_list<mx::array> arrays)
        : Profiler_(profiler)
        , Stage_(stage)
        , Depth_(depth)
        , Arrays_(arrays)
        , Start_(std::chrono::steady_clock::now())
    {}

    ~TStageTimer() {
        if (!Profiler_) return;
        if (!Arrays_.empty()) {
            mx::eval(Arrays_);
        }
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - Start_).count();
        if (Depth_ < 0) {
            Profiler_->AccumStage(Stage_, ms);
        } else {
            Profiler_->AccumDepth(Stage_, Depth_, ms);
        }
    }

    // Non-copyable, non-movable.
    TStageTimer(const TStageTimer&)            = delete;
    TStageTimer& operator=(const TStageTimer&) = delete;
    TStageTimer(TStageTimer&&)                 = delete;
    TStageTimer& operator=(TStageTimer&&)      = delete;

private:
    TStageProfiler*                        Profiler_;
    EStageId                               Stage_;
    int                                    Depth_;
    std::vector<mx::array>                 Arrays_;
    std::chrono::steady_clock::time_point  Start_;
};

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

// STAGE_TIMER(profiler_ptr, stage_enum, {arrays...})
//   Times a scope. The opening brace MUST be placed by the caller as a C++
//   compound-statement (or the timer goes out of scope immediately).
//   Example:
//     {
//       STAGE_TIMER(profiler_, EStageId::Derivatives, {grad, hess});
//       target.ComputeDerivatives(...);
//     }
#define STAGE_TIMER(profiler, stage, arrays) \
    NCatboostMlx::TStageTimer _stage_timer_##__LINE__((profiler), (stage), arrays)

// STAGE_TIMER_DEPTH(profiler_ptr, stage_enum, depth_int, {arrays...})
//   Depth-aware variant for stages 3/4/5 inside the depth loop.
#define STAGE_TIMER_DEPTH(profiler, stage, depth, arrays) \
    NCatboostMlx::TStageTimer _stage_timer_##__LINE__((profiler), (stage), (depth), arrays)

}  // namespace NCatboostMlx

#else  // CATBOOST_MLX_STAGE_PROFILE not defined

// ---------------------------------------------------------------------------
// Stub definitions — compile to nothing in release builds
// ---------------------------------------------------------------------------

namespace NCatboostMlx {

// Forward-declare enough for function signatures to remain valid when
// the profiler parameter is always TStageProfiler*.
struct TStageProfiler {
    // All methods are no-ops.
    void BeginIter(int) {}
    void EndIter() {}
};

}  // namespace NCatboostMlx

// Macros expand to nothing.
#define STAGE_TIMER(profiler, stage, arrays)              /* no-op */
#define STAGE_TIMER_DEPTH(profiler, stage, depth, arrays) /* no-op */

#endif  // CATBOOST_MLX_STAGE_PROFILE
