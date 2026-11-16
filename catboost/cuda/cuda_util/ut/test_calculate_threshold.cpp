#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/helpers.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>

#include <iostream>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TMvsThresholdCalculationTest) {
    Y_UNIT_TEST(TestMvsThresholdWithFractionalSampleSize) {
        auto stopCudaManagerGuard = StartCudaManager();
        const TVector<float> candidates = {0.1f, 1.0f, 10.0f};
        auto cudaCandidates = TSingleBuffer<float>::Create(TSingleMapping(0, candidates.size()));
        cudaCandidates.Write(candidates);

        const auto thresholds = CalculateMvsThreshold(cudaCandidates, 0.5f);
        UNIT_ASSERT_VALUES_EQUAL(thresholds.size(), 1);
        // min(1, 0.1 / t) + min(1, 1 / t) + min(1, 10 / t) = 1.5.
        UNIT_ASSERT_DOUBLES_EQUAL(thresholds[0], 2.2f, 1e-5f);
    }

    Y_UNIT_TEST(TestMvsThresholdWithRepeatedCandidates) {
        auto stopCudaManagerGuard = StartCudaManager();
        TVector<float> candidates(8192, 100.0f);
        Fill(candidates.begin(), candidates.begin() + 6000, 1.0f);
        auto cudaCandidates = TSingleBuffer<float>::Create(TSingleMapping(0, candidates.size()));
        cudaCandidates.Write(candidates);

        const auto thresholds = CalculateMvsThreshold(cudaCandidates, 0.5f);
        // All 2192 large candidates are selected; the other 6000 share the remaining probability.
        UNIT_ASSERT_DOUBLES_EQUAL(thresholds[0], 6000.0f / (4096 - 2192), 1e-5f);
    }

    Y_UNIT_TEST(TestMvsWeightsAcrossBlocks) {
        auto stopCudaManagerGuard = StartCudaManager();
        TRandom random(0);
        TVector<ui64> seedValues(256 * 256);
        for (auto& seed : seedValues) {
            seed = random.NextUniformL();
        }
        auto seeds = TSingleBuffer<ui64>::Create(TSingleMapping(0, seedValues.size()));
        seeds.Write(seedValues);

        for (const ui32 size : {31, 257, 8192, 8193, 16384, 131073}) {
            const auto mapping = TSingleMapping(0, size);
            auto derivatives = TSingleBuffer<float>::Create(mapping);
            auto weights = TSingleBuffer<float>::Create(mapping);
            FillBuffer(derivatives, 1.0f);
            for (const float takenFraction : {0.5f, 1.0f}) {
                FillBuffer(weights, -1.0f);
                MvsBootstrapRadixSort(seeds, weights, derivatives, takenFraction, 0.0f);
                TVector<float> actualWeights;
                weights.Read(actualWeights);
                for (ui32 objectIdx = 0; objectIdx < size; ++objectIdx) {
                    const float weight = actualWeights[objectIdx];
                    UNIT_ASSERT_C(
                        (takenFraction < 1.0f && weight == 0.0f) || weight == 1.0f / takenFraction,
                        "size=" << size << ", objectIdx=" << objectIdx << ", weight=" << weight
                    );
                }
            }
        }
    }

    Y_UNIT_TEST(TestMvsFullSampleWithZeroDerivatives) {
        auto stopCudaManagerGuard = StartCudaManager();
        auto seeds = TSingleBuffer<ui64>::Create(TSingleMapping(0, 256));
        FillBuffer(seeds, ui64(42));
        const auto mapping = TSingleMapping(0, 8193);
        auto derivatives = TSingleBuffer<float>::Create(mapping);
        auto weights = TSingleBuffer<float>::Create(mapping);
        FillBuffer(derivatives, 0.0f);
        FillBuffer(weights, -1.0f);
        MvsBootstrapRadixSort(seeds, weights, derivatives, 1.0f, 0.0f);
        TVector<float> actualWeights;
        weights.Read(actualWeights);
        for (const float weight : actualWeights) {
            UNIT_ASSERT_VALUES_EQUAL(weight, 1.0f);
        }
    }

    Y_UNIT_TEST(TestMvsThresholdCalculation) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 40;
            const ui32 TILE_SIZE  = 1 << 13;
            TRandom rand(0);

            TVector<float> vecCpu;
            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = rand.NextUniformL() % (k < 20 ? 512 : 10000000);
                vecCpu.resize(size);
                for (ui64 idx = 0; idx < size; ++idx) {
                    vecCpu[idx] = abs(rand.NextUniform()) + 0.01;
                }

                float takenFraction = rand.NextUniform();

                auto cudaVec = TSingleBuffer<float>::Create(TSingleMapping(0, size));
                cudaVec.Write(vecCpu);

                TVector<float> thresholds = CalculateMvsThreshold(cudaVec, takenFraction);
                ui32 blockCount = ::NHelpers::CeilDivide(size, TILE_SIZE);

                TVector<float> prefix(1, 0);
                for (ui32 idx = 0; idx < blockCount; ++idx) {
                    ui32 blockSize = Min(TILE_SIZE, ui32(size - idx * TILE_SIZE));
                    sort(vecCpu.begin() + idx * TILE_SIZE, vecCpu.begin() + idx * TILE_SIZE + blockSize);
                    prefix.resize(1);
                    float prefixSum = 0;
                    ui32 rightPart = 0;

                    for (ui32 j = idx * TILE_SIZE; j < idx * TILE_SIZE + blockSize && j < size; ++j) {
                        float v = vecCpu[j];
                        prefix.push_back(prefix.back() + v);
                        if (v < thresholds[idx]) {
                            prefixSum += v;
                        } else {
                            rightPart += 1;
                        }
                    }

                    float leftPart = thresholds[idx] > 0 ? prefixSum / thresholds[idx] : 0;
                    UNIT_ASSERT(FuzzyEquals(rightPart + leftPart, blockSize * takenFraction, 2.0e-2f));
                }
            }
        }
    }

    Y_UNIT_TEST(TestMvsThresholdCalculationPerformance) {
        TSetLogging inThisScope(ELoggingLevel::Debug);
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            TCudaProfiler& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 size = 100; size < 10000001; size *= 10) {
                for (ui32 k = 0; k < tries; ++k) {
                    TVector<float> vecCpu;
                    for (ui64 i = 0; i < size; ++i) {
                        vecCpu.push_back(abs(rand.NextUniform()) + 0.01);
                    }

                    float takenFraction = rand.NextUniform();

                    auto mapping = TSingleMapping(0, size);
                    auto cudaVec = TSingleBuffer<float>::Create(mapping);
                    cudaVec.Write(vecCpu);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Calculate threshold for #" << size << " elements");
                        TVector<float> thresholds = CalculateMvsThreshold(cudaVec, takenFraction);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestGenerateMvsWeightsPerformance) {
        TSetLogging inThisScope(ELoggingLevel::Debug);
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            TGpuAwareRandom random(0);
            TCudaProfiler& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            ui32 maxSeedCount = 256 * 256;
            TVector<ui64> seedsCpu(maxSeedCount);
            for (ui32 i = 0; i < seedsCpu.size(); ++i) {
                seedsCpu[i] = rand.NextUniformL();
            }
            auto seeds = TSingleBuffer<ui64>::Create(TSingleMapping(0, maxSeedCount));
            seeds.Write(seedsCpu);

            for (ui32 size = 100; size < 10000001; size *= 10) {
                for (ui32 k = 0; k < tries; ++k) {
                    TVector<float> vecCpu;
                    for (ui64 i = 0; i < size; ++i) {
                        vecCpu.push_back(abs(rand.NextUniform()) + 0.01);
                    }

                    float takenFraction = rand.NextUniform();

                    auto mapping = TSingleMapping(0, size);
                    auto cudaVec = TSingleBuffer<float>::Create(mapping);
                    auto result = TSingleBuffer<float>::Create(mapping);
                    cudaVec.Write(vecCpu);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Calculate MVS weights for #" << size << " elements");
                        MvsBootstrapRadixSort(seeds, result, cudaVec, takenFraction, 0.05);
                    }
                }
            }
        }
    }
}
