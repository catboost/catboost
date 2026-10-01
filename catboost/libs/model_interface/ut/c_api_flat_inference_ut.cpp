/**
 * c_api_flat_inference_ut.cpp
 *
 * Tests for the CalcModelPrediction / CalcModelPredictionFlat optimizations:
 *
 *  1. CalcModelPrediction with catFeaturesSize==0 must produce results
 *     identical to CalcModelPredictionFlat (they now share the same code path).
 *
 *  2. CalcModelPredictionFlat must produce results identical to
 *     CalcModelPredictionFlatStaged over the full tree range.
 *
 *  3. Thread-local scratch buffer: concurrent calls on different threads
 *     must produce consistent, correct results (no data races on the
 *     shared ModelCalcerHandle, no aliasing of the thread-local buffer).
 *
 *  4. Single-document fast path (docCount==1) must match the batch path.
 *
 * The tests use the cloudness_small.cbm model which has only float features,
 * making it the canonical case for the catFeaturesSize==0 fast path.
 */

#include <catboost/libs/model_interface/c_api.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/thread.h>

#include <atomic>
#include <cmath>
#include <thread>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* MODEL_PATH =
    "catboost/node-package/test_data/cloudness_small.cbm";

struct TModelGuard {
    ModelCalcerHandle* Handle;

    TModelGuard() {
        Handle = ModelCalcerCreate();
        UNIT_ASSERT_C(Handle, "ModelCalcerCreate returned null");
        bool ok = LoadFullModelFromFile(Handle, MODEL_PATH);
        UNIT_ASSERT_C(ok, TString("Failed to load model: ") + GetErrorString());
    }

    ~TModelGuard() {
        ModelCalcerDelete(Handle);
    }
};

// Build a batch of numDocs feature vectors, each of length numFeatures.
// Values are deterministic: features[doc][feat] = (doc * numFeatures + feat) * 0.1f
static TVector<TVector<float>> MakeFeatures(size_t numDocs, size_t numFeatures) {
    TVector<TVector<float>> features(numDocs, TVector<float>(numFeatures));
    for (size_t d = 0; d < numDocs; ++d) {
        for (size_t f = 0; f < numFeatures; ++f) {
            features[d][f] = static_cast<float>((d * numFeatures + f) * 0.1);
        }
    }
    return features;
}

// Build the const float** pointer array expected by the C API.
static TVector<const float*> MakePtrs(const TVector<TVector<float>>& features) {
    TVector<const float*> ptrs(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        ptrs[i] = features[i].data();
    }
    return ptrs;
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

Y_UNIT_TEST_SUITE(CApiFlatInference) {

    // -----------------------------------------------------------------------
    // 1. CalcModelPrediction(catFeaturesSize=0) == CalcModelPredictionFlat
    // -----------------------------------------------------------------------
    Y_UNIT_TEST(NoCatFeaturesMatchesFlat) {
        TModelGuard m;
        const size_t numFeatures = GetFloatFeaturesCount(m.Handle);
        UNIT_ASSERT_GT(numFeatures, 0u);

        const size_t numDocs = 8;
        auto features = MakeFeatures(numDocs, numFeatures);
        auto ptrs     = MakePtrs(features);

        TVector<double> resultViaCalc(numDocs, 0.0);
        TVector<double> resultViaFlat(numDocs, 0.0);

        // CalcModelPrediction with catFeaturesSize=0 (the common caller pattern)
        bool ok = CalcModelPrediction(
            m.Handle, numDocs,
            ptrs.data(), numFeatures,
            nullptr, 0,
            resultViaCalc.data(), resultViaCalc.size());
        UNIT_ASSERT_C(ok, GetErrorString());

        // CalcModelPredictionFlat (the direct flat path)
        ok = CalcModelPredictionFlat(
            m.Handle, numDocs,
            ptrs.data(), numFeatures,
            resultViaFlat.data(), resultViaFlat.size());
        UNIT_ASSERT_C(ok, GetErrorString());

        for (size_t i = 0; i < numDocs; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                resultViaCalc[i], resultViaFlat[i], 1e-12,
                "Mismatch at doc " << i
                    << ": CalcModelPrediction=" << resultViaCalc[i]
                    << " CalcModelPredictionFlat=" << resultViaFlat[i]);
        }
    }

    // -----------------------------------------------------------------------
    // 2. CalcModelPredictionFlat == CalcModelPredictionFlatStaged(0, treeCount)
    // -----------------------------------------------------------------------
    Y_UNIT_TEST(FlatMatchesStagedFullRange) {
        TModelGuard m;
        const size_t numFeatures = GetFloatFeaturesCount(m.Handle);
        const size_t treeCount   = GetTreeCount(m.Handle);
        UNIT_ASSERT_GT(treeCount, 0u);

        const size_t numDocs = 5;
        auto features = MakeFeatures(numDocs, numFeatures);
        auto ptrs     = MakePtrs(features);

        TVector<double> resultFlat  (numDocs, 0.0);
        TVector<double> resultStaged(numDocs, 0.0);

        bool ok = CalcModelPredictionFlat(
            m.Handle, numDocs,
            ptrs.data(), numFeatures,
            resultFlat.data(), resultFlat.size());
        UNIT_ASSERT_C(ok, GetErrorString());

        ok = CalcModelPredictionFlatStaged(
            m.Handle, numDocs, 0, treeCount,
            ptrs.data(), numFeatures,
            resultStaged.data(), resultStaged.size());
        UNIT_ASSERT_C(ok, GetErrorString());

        for (size_t i = 0; i < numDocs; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                resultFlat[i], resultStaged[i], 1e-12,
                "Mismatch at doc " << i);
        }
    }

    // -----------------------------------------------------------------------
    // 3. Single-document fast path matches batch path
    // -----------------------------------------------------------------------
    Y_UNIT_TEST(SingleDocMatchesBatch) {
        TModelGuard m;
        const size_t numFeatures = GetFloatFeaturesCount(m.Handle);

        // Batch of 1
        auto features = MakeFeatures(1, numFeatures);
        auto ptrs     = MakePtrs(features);

        double resultSingle = 0.0;
        double resultBatch  = 0.0;

        // docCount=1 triggers CalcFlatSingle inside CalcModelPredictionFlatStaged
        bool ok = CalcModelPredictionFlat(
            m.Handle, 1,
            ptrs.data(), numFeatures,
            &resultSingle, 1);
        UNIT_ASSERT_C(ok, GetErrorString());

        // Batch of 3 with the same doc repeated — first result must match
        auto features3 = TVector<TVector<float>>(3, features[0]);
        auto ptrs3     = MakePtrs(features3);
        TVector<double> results3(3, 0.0);

        ok = CalcModelPredictionFlat(
            m.Handle, 3,
            ptrs3.data(), numFeatures,
            results3.data(), results3.size());
        UNIT_ASSERT_C(ok, GetErrorString());

        UNIT_ASSERT_DOUBLES_EQUAL_C(
            resultSingle, results3[0], 1e-12,
            "Single-doc result differs from batch result for the same input");
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            results3[0], results3[1], 1e-12,
            "Identical inputs in batch produced different outputs");
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            results3[1], results3[2], 1e-12,
            "Identical inputs in batch produced different outputs");
    }

    // -----------------------------------------------------------------------
    // 4. Thread safety: concurrent calls on the same handle produce
    //    consistent results. Exercises the thread-local scratch buffer
    //    (each thread has its own; no aliasing possible).
    // -----------------------------------------------------------------------
    Y_UNIT_TEST(ConcurrentCallsAreConsistent) {
        TModelGuard m;
        const size_t numFeatures = GetFloatFeaturesCount(m.Handle);
        const size_t numDocs     = 4;

        auto features = MakeFeatures(numDocs, numFeatures);
        auto ptrs     = MakePtrs(features);

        // Compute reference result on the main thread.
        TVector<double> reference(numDocs, 0.0);
        bool ok = CalcModelPredictionFlat(
            m.Handle, numDocs,
            ptrs.data(), numFeatures,
            reference.data(), reference.size());
        UNIT_ASSERT_C(ok, GetErrorString());

        // Spin up N threads, each running M iterations.
        constexpr int kThreads    = 8;
        constexpr int kIterations = 50;

        std::atomic<int> failures{0};

        auto worker = [&]() {
            TVector<double> result(numDocs, 0.0);
            for (int iter = 0; iter < kIterations; ++iter) {
                bool threadOk = CalcModelPredictionFlat(
                    m.Handle, numDocs,
                    ptrs.data(), numFeatures,
                    result.data(), result.size());
                if (!threadOk) {
                    ++failures;
                    return;
                }
                for (size_t d = 0; d < numDocs; ++d) {
                    if (std::abs(result[d] - reference[d]) > 1e-12) {
                        ++failures;
                        return;
                    }
                }
            }
        };

        TVector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back(worker);
        }
        for (auto& th : threads) {
            th.join();
        }

        UNIT_ASSERT_EQUAL_C(failures.load(), 0,
            failures.load() << " thread(s) produced wrong or failed results");
    }

    // -----------------------------------------------------------------------
    // 5. CalcModelPrediction with catFeaturesSize=0 is consistent across
    //    repeated calls (thread-local buffer reuse correctness).
    // -----------------------------------------------------------------------
    Y_UNIT_TEST(RepeatedCallsConsistent) {
        TModelGuard m;
        const size_t numFeatures = GetFloatFeaturesCount(m.Handle);

        // Alternate between different batch sizes to exercise buffer resize.
        const TVector<size_t> batchSizes = {1, 10, 3, 50, 1, 7};

        for (size_t batchSize : batchSizes) {
            auto features = MakeFeatures(batchSize, numFeatures);
            auto ptrs     = MakePtrs(features);

            TVector<double> resultCalc(batchSize, 0.0);
            TVector<double> resultFlat(batchSize, 0.0);

            bool ok = CalcModelPrediction(
                m.Handle, batchSize,
                ptrs.data(), numFeatures,
                nullptr, 0,
                resultCalc.data(), resultCalc.size());
            UNIT_ASSERT_C(ok, GetErrorString());

            ok = CalcModelPredictionFlat(
                m.Handle, batchSize,
                ptrs.data(), numFeatures,
                resultFlat.data(), resultFlat.size());
            UNIT_ASSERT_C(ok, GetErrorString());

            for (size_t i = 0; i < batchSize; ++i) {
                UNIT_ASSERT_DOUBLES_EQUAL_C(
                    resultCalc[i], resultFlat[i], 1e-12,
                    "Mismatch at batchSize=" << batchSize << " doc=" << i);
            }
        }
    }

} // Y_UNIT_TEST_SUITE(CApiFlatInference)
