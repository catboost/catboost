// parallel_calcer_wrapper_test.cpp
//
// Two test suites:
//   1. Correctness  – parallel results match single-threaded results exactly.
//   2. Performance  – parallel execution is faster than single-threaded for a
//                     large batch on a multi-core machine.
//
// Build (from the model_interface directory):
//   g++ -std=c++17 -O2 -pthread \
//       parallel_calcer_wrapper_test.cpp \
//       -I. \
//       -L/home/den.raskovalov/ProgrammingPrivate/catboost_build/catboost/libs/model_interface \
//       -lcatboostmodel \
//       -Wl,-rpath,/home/den.raskovalov/ProgrammingPrivate/catboost_build/catboost/libs/model_interface \
//       -o parallel_calcer_wrapper_test
//
// Run:
//   ./parallel_calcer_wrapper_test <path_to_model.cbm>

#include "parallel_calcer_wrapper.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny test framework
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "[FAIL] " << msg << "\n";                            \
            ++g_failed;                                                        \
        } else {                                                               \
            std::cout << "[PASS] " << msg << "\n";                            \
            ++g_passed;                                                        \
        }                                                                      \
    } while (false)

#define CHECK_NEAR(a, b, eps, msg)                                             \
    CHECK(std::fabs((a) - (b)) <= (eps), msg)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate a random batch of flat feature vectors. */
static std::vector<std::vector<float>> MakeBatch(size_t docCount,
                                                  size_t featureCount,
                                                  unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> batch(docCount,
                                          std::vector<float>(featureCount));
    for (auto& row : batch) {
        for (auto& v : row) v = dist(rng);
    }
    return batch;
}

/** Wall-clock duration of a callable in milliseconds. */
template <typename F>
static double TimeMs(F&& f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Suite 1 – Correctness
// ---------------------------------------------------------------------------

static void TestCorrectness(const std::string& modelPath) {
    std::cout << "\n=== Suite 1: Correctness ===\n";

    ParallelModelCalcerWrapper calcer(modelPath);
    const size_t featureCount = calcer.GetFloatFeaturesCount();
    const size_t dims         = calcer.GetDimensionsCount();

    std::cout << "  Model: " << calcer.GetTreeCount() << " trees, "
              << featureCount << " float features, "
              << dims << " output dimension(s)\n";

    // --- 1a. Empty batch returns empty result ---
    {
        auto res = calcer.CalcFlatParallel({}, 4);
        CHECK(res.empty(), "1a: empty batch -> empty result");
    }

    // --- 1b. Single-object: CalcFlat(vec) == CalcFlatParallel(batch of 1) ---
    {
        std::vector<float> row(featureCount, 0.5f);
        double single = calcer.CalcFlat(row);

        std::vector<std::vector<float>> batch = {row};
        auto parallel = calcer.CalcFlatParallel(batch, 1);

        CHECK(parallel.size() == dims,
              "1b: result size == DimensionsCount for batch-of-1");
        CHECK_NEAR(single, parallel[0], 1e-9,
                   "1b: single-object CalcFlat matches CalcFlatParallel(1 thread)");
    }

    // --- 1c. numThreads=1 matches numThreads=N for a larger batch ---
    {
        const size_t N = 500;
        auto batch = MakeBatch(N, featureCount);

        auto ref  = calcer.CalcFlatParallel(batch, 1);
        auto par4 = calcer.CalcFlatParallel(batch, 4);
        auto par8 = calcer.CalcFlatParallel(batch, 8);
        auto par0 = calcer.CalcFlatParallel(batch, 0); // hardware_concurrency

        CHECK(ref.size() == N * dims,
              "1c: result size == docCount * dims");
        CHECK(par4.size() == ref.size(),
              "1c: 4-thread result has same size as 1-thread");
        CHECK(par8.size() == ref.size(),
              "1c: 8-thread result has same size as 1-thread");
        CHECK(par0.size() == ref.size(),
              "1c: auto-thread result has same size as 1-thread");

        bool match4 = true, match8 = true, match0 = true;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (std::fabs(ref[i] - par4[i]) > 1e-9) match4 = false;
            if (std::fabs(ref[i] - par8[i]) > 1e-9) match8 = false;
            if (std::fabs(ref[i] - par0[i]) > 1e-9) match0 = false;
        }
        CHECK(match4, "1c: 4-thread results match 1-thread results exactly");
        CHECK(match8, "1c: 8-thread results match 1-thread results exactly");
        CHECK(match0, "1c: auto-thread results match 1-thread results exactly");
    }

    // --- 1d. CalcParallel (float+cat) with no cat features ---
    {
        const size_t N = 200;
        auto floatBatch = MakeBatch(N, featureCount);

        auto ref  = calcer.CalcParallel(floatBatch, {}, 1);
        auto par4 = calcer.CalcParallel(floatBatch, {}, 4);

        CHECK(ref.size() == N * dims,
              "1d: CalcParallel result size correct");

        bool match = true;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (std::fabs(ref[i] - par4[i]) > 1e-9) match = false;
        }
        CHECK(match, "1d: CalcParallel 4-thread matches 1-thread");
    }

    // --- 1e. numThreads clamped to docCount (no crash for threads > docs) ---
    {
        auto batch = MakeBatch(3, featureCount);
        auto ref   = calcer.CalcFlatParallel(batch, 1);
        auto par   = calcer.CalcFlatParallel(batch, 1000); // 1000 > 3

        bool match = true;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (std::fabs(ref[i] - par[i]) > 1e-9) match = false;
        }
        CHECK(match, "1e: numThreads > docCount clamped correctly");
    }

    // --- 1f. InitFromFile path ---
    {
        ParallelModelCalcerWrapper calcer2;
        bool ok = calcer2.InitFromFile(modelPath);
        CHECK(ok, "1f: InitFromFile returns true");
        CHECK(calcer2.GetTreeCount() == calcer.GetTreeCount(),
              "1f: InitFromFile loads same model");
    }
}

// ---------------------------------------------------------------------------
// Suite 2 – Performance
// ---------------------------------------------------------------------------

static void TestPerformance(const std::string& modelPath) {
    std::cout << "\n=== Suite 2: Performance ===\n";

    ParallelModelCalcerWrapper calcer(modelPath);
    const size_t featureCount = calcer.GetFloatFeaturesCount();

    // Use a large batch so thread-spawn overhead is negligible
    const size_t DOC_COUNT = 50'000;
    auto batch = MakeBatch(DOC_COUNT, featureCount, /*seed=*/7);

    // Warm-up (avoid cold-cache effects)
    calcer.CalcFlatParallel(batch, 1);
    calcer.CalcFlatParallel(batch, 4);

    // Measure single-threaded
    double ms1 = 0;
    {
        std::vector<double> result;
        ms1 = TimeMs([&]() {
            result = calcer.CalcFlatParallel(batch, 1);
        });
    }

    // Measure 4-thread
    double ms4 = 0;
    {
        std::vector<double> result;
        ms4 = TimeMs([&]() {
            result = calcer.CalcFlatParallel(batch, 4);
        });
    }

    // Measure auto-thread (hardware_concurrency)
    double msAuto = 0;
    unsigned hwThreads = std::thread::hardware_concurrency();
    {
        std::vector<double> result;
        msAuto = TimeMs([&]() {
            result = calcer.CalcFlatParallel(batch, 0);
        });
    }

    std::cout << "  Batch size : " << DOC_COUNT << " documents\n";
    std::cout << "  1 thread   : " << ms1    << " ms\n";
    std::cout << "  4 threads  : " << ms4    << " ms  (speedup "
              << ms1 / ms4 << "x)\n";
    std::cout << "  " << hwThreads << " threads (auto): " << msAuto
              << " ms  (speedup " << ms1 / msAuto << "x)\n";

    // On a multi-core machine 4 threads should be faster than 1 thread.
    // We use a conservative threshold of 1.5x to avoid flakiness on
    // lightly-loaded or single-core CI machines.
    if (hwThreads >= 4) {
        CHECK(ms4 < ms1 / 1.5,
              "2a: 4-thread is at least 1.5x faster than 1-thread");
    } else {
        std::cout << "  [SKIP] 2a: only " << hwThreads
                  << " hardware threads available, skipping speedup check\n";
    }

    // Auto-thread should be at least as fast as 1-thread
    CHECK(msAuto <= ms1 * 1.1,
          "2b: auto-thread is not slower than 1-thread (within 10% noise)");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path.cbm>\n";
        return 1;
    }
    const std::string modelPath = argv[1];

    try {
        TestCorrectness(modelPath);
        TestPerformance(modelPath);
    } catch (const std::exception& ex) {
        std::cerr << "\n[EXCEPTION] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "\n=== Results: " << g_passed << " passed, "
              << g_failed << " failed ===\n";
    return g_failed > 0 ? 1 : 0;
}
