#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/helpers.h>
#include <catboost/cuda/cuda_util/bootstrap.h>
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
