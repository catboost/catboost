#include <util/random/shuffle.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/segmented_scan.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <library/cpp/testing/unittest/registar.h>
#include <iostream>
#include <catboost/cuda/cuda_util/scan.h>
#include <catboost/cuda/cuda_util/helpers.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TSegmentedScanTest) {
    Y_UNIT_TEST(TestSegmentedScan) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = 1001 + (rand.NextUniformL() % 1000000);

                auto mapping = TSingleMapping(0, size);

                auto input = TSingleBuffer<ui32>::Create(mapping);
                auto output = TSingleBuffer<ui32>::CopyMapping(input);
                auto flags = TSingleBuffer<ui32>::CopyMapping(input);

                TVector<ui32> data(size);
                TVector<ui32> flagsCpu(size);

                std::generate(data.begin(), data.end(), [&]() {
                    return rand.NextUniformL() % 10000;
                });

                const double p = 1000.0 / size;
                std::generate(flagsCpu.begin(), flagsCpu.end(), [&]() {
                    return rand.NextUniform() < p ? 1 : 0;
                });
                flagsCpu[0] = 1;

                input.Write(data);
                flags.Write(flagsCpu);

                SegmentedScanVector(input, flags, output);
                TVector<ui32> result;
                output.Read(result);

                ui32 prefixSum = 0;
                for (ui32 i = 0; i < result.size(); ++i) {
                    if (flagsCpu[i]) {
                        prefixSum = 0;
                    }

                    if (result[i] != prefixSum && i > 0) {
                        CATBOOST_INFO_LOG << flagsCpu[i - 1] << " " << i - 1 << " " << result[i - 1] << " " << data[i - 1] << Endl;
                        CATBOOST_INFO_LOG << flagsCpu[i] << " " << i << " " << result[i] << " " << prefixSum << Endl;
                    }
                    UNIT_ASSERT_EQUAL(result[i], prefixSum);
                    prefixSum += data[i];
                }

                SegmentedScanVector(input, flags, output, true);
                output.Read(result);

                prefixSum = data[0];

                for (ui32 i = 0; i < result.size(); ++i) {
                    if (flagsCpu[i]) {
                        prefixSum = 0;
                    }
                    prefixSum += data[i];
                    UNIT_ASSERT_EQUAL(result[i], prefixSum);
                }
            }
        }
    }

    Y_UNIT_TEST(TestSegmentedScanWithMask) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = 1001 + (rand.NextUniformL() % 1000000);

                auto mapping = TSingleMapping(0, size);

                auto input = TSingleBuffer<ui32>::Create(mapping);
                auto output = TSingleBuffer<ui32>::CopyMapping(input);
                auto flags = TSingleBuffer<ui32>::CopyMapping(input);

                TVector<ui32> data(size);
                TVector<ui32> flagsCpu(size);

                std::generate(data.begin(), data.end(), [&]() {
                    return rand.NextUniformL() % 10000;
                });

                const double p = 1000.0 / size;
                std::generate(flagsCpu.begin(), flagsCpu.end(), [&]() {
                    return (rand.NextUniform() < p ? 1 : 0) << 31;
                });
                flagsCpu[0] = 1 << 31;

                input.Write(data);
                flags.Write(flagsCpu);

                SegmentedScanVector(input, flags, output, false, 1 << 31);
                TVector<ui32> result;
                output.Read(result);

                ui32 prefixSum = 0;
                for (ui32 i = 0; i < result.size(); ++i) {
                    if (flagsCpu[i] >> 31) {
                        prefixSum = 0;
                    }

                    if (result[i] != prefixSum) {
                        CATBOOST_INFO_LOG << flagsCpu[i - 1] << " " << i - 1 << " " << result[i - 1] << " " << data[i - 1] << Endl;
                        CATBOOST_INFO_LOG << flagsCpu[i] << " " << i << " " << result[i] << " " << prefixSum << Endl;
                    }
                    UNIT_ASSERT_EQUAL(result[i], prefixSum);
                    prefixSum += data[i];
                }

                SegmentedScanVector(input, flags, output, true, 1 << 31);
                output.Read(result);

                prefixSum = 0;

                for (ui32 i = 0; i < result.size(); ++i) {
                    if (flagsCpu[i] >> 31) {
                        prefixSum = 0;
                    }
                    prefixSum += data[i];
                    UNIT_ASSERT_EQUAL(result[i], prefixSum);
                }
            }
        }
    }

    Y_UNIT_TEST(TestNonNegativeSegmentedScan) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = 1001 + (rand.NextUniformL() % 1000000);

                auto mapping = TSingleMapping(0, size);

                auto input = TSingleBuffer<float>::Create(mapping);
                auto output = TSingleBuffer<float>::CopyMapping(input);

                TVector<float> data(size);
                TVector<ui32> flagsCpu(size);

                std::generate(data.begin(), data.end(), [&]() {
                    return 1.0 / (1 << (rand.NextUniformL() % 10));
                });

                const double p = 1000.0 / size;
                std::generate(flagsCpu.begin(), flagsCpu.end(), [&]() {
                    return (rand.NextUniform() < p ? 1 : 0);
                });
                flagsCpu[0] = 1;

                for (ui32 i = 0; i < flagsCpu.size(); ++i) {
                    if (flagsCpu[i]) {
                        data[i] = -data[i];
                    }
                }

                input.Write(data);
                TVector<float> result;
                //
                float prefixSum = 0;

                InclusiveSegmentedScanNonNegativeVector(input, output);
                output.Read(result);

                prefixSum = 0;

                for (ui32 i = 0; i < result.size(); ++i) {
                    if (flagsCpu[i]) {
                        prefixSum = 0;
                    }
                    prefixSum += std::abs(data[i]);
                    UNIT_ASSERT_DOUBLES_EQUAL(std::abs(result[i]), prefixSum, 1e-9);
                }
            }
        }
    }

    Y_UNIT_TEST(TestNonNegativeSegmentedScanAndScatter) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = 1001 + (rand.NextUniformL() % 1000000);

                auto mapping = TSingleMapping(0, size);

                auto input = TSingleBuffer<float>::Create(mapping);
                auto output = TSingleBuffer<float>::CopyMapping(input);

                TVector<float> data(size);
                TVector<ui32> indicesCpu(size);

                std::generate(data.begin(), data.end(), [&]() {
                    return 1.0 / (1 << (rand.NextUniformL() % 10));
                });

                const double p = 1000.0 / size;
                for (ui32 i = 0; i < indicesCpu.size(); ++i) {
                    indicesCpu[i] = i;
                };
                Shuffle(indicesCpu.begin(), indicesCpu.end(), rand);
                for (ui32 i = 0; i < indicesCpu.size(); ++i) {
                    indicesCpu[i] |= ((rand.NextUniform() < p ? 1 : 0) << 31);
                }
                indicesCpu[0] |= 1 << 31;

                for (ui32 i = 0; i < indicesCpu.size(); ++i) {
                    if (indicesCpu[i] >> 31) {
                        data[i] = -data[i];
                    }
                }
                auto indices = TSingleBuffer<ui32>::Create(mapping);
                indices.Write(indicesCpu);
                input.Write(data);

                TVector<float> result;

                SegmentedScanAndScatterNonNegativeVector(input, indices, output, false);
                output.Read(result);
                //
                float prefixSum = 0;
                const ui32 mask = 0x3FFFFFFF;
                for (ui32 i = 0; i < result.size(); ++i) {
                    if (indicesCpu[i] >> 31) {
                        prefixSum = 0;
                    }

                    const ui32 scatterIndex = indicesCpu[i] & mask;
                    UNIT_ASSERT_EQUAL(result[scatterIndex], prefixSum);
                    prefixSum += std::abs(data[i]);
                }

                prefixSum = 0;

                SegmentedScanAndScatterNonNegativeVector(input, indices, output, true);
                output.Read(result);

                for (ui32 i = 0; i < result.size(); ++i) {
                    if (indicesCpu[i] >> 31) {
                        prefixSum = 0;
                    }
                    prefixSum += std::abs(data[i]);
                    const ui32 scatterIndex = indicesCpu[i] & mask;
                    const float val = result[scatterIndex];
                    if (std::abs(val - prefixSum) > 1e-9) {
                        CATBOOST_INFO_LOG << scatterIndex << " " << std::abs(val) << " " << prefixSum << Endl;
                        CATBOOST_INFO_LOG << indicesCpu[i - 1] << " " << i - 1 << " " << result[i - 1] << " " << data[i - 1] << Endl;
                        CATBOOST_INFO_LOG << indicesCpu[i] << " " << i << " " << result[i] << " " << prefixSum << Endl;
                    }
                    UNIT_ASSERT_EQUAL(val, prefixSum);
                }
            }
        }
    }

    inline void RunSegmentedScanNonNegativePerformanceTest() {
        {
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            ui64 tries = 20;
            TRandom rand(0);
            for (ui32 i = 10000; i < 10000001; i *= 10) {
                const ui32 size = i;
                auto mapping = TSingleMapping(0, size);

                auto input = TSingleBuffer<float>::Create(mapping);
                auto output = TSingleBuffer<float>::CopyMapping(input);

                TVector<float> data(size);
                TVector<ui32> flagsCpu(size);

                std::generate(data.begin(), data.end(), [&]() {
                    return rand.NextUniformL() % 10000;
                });

                const double p = 5000.0 / size;
                std::generate(flagsCpu.begin(), flagsCpu.end(), [&]() {
                    return rand.NextUniform() < p ? 1 : 0;
                });
                flagsCpu[0] = 1;

                for (ui32 j = 0; j < flagsCpu.size(); ++j) {
                    if (flagsCpu[j]) {
                        data[j] = -data[j];
                    }
                }
                input.Write(data);

                for (ui32 k = 0; k < tries; ++k) {
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Inclusive segmented scan of #" << i << " elements");
                        InclusiveSegmentedScanNonNegativeVector(input, output);
                    }
                }
            }
        }
    }

    //
    template <class T>
    inline void TestSegmentedScanPerformance() {
        {
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            ui64 tries = 10;
            TRandom rand(0);
            for (ui32 i = 10000; i < 10000001; i *= 10) {
                const ui32 size = i;
                auto mapping = TSingleMapping(0, size);

                auto input = TSingleBuffer<ui32>::Create(mapping);
                auto output = TSingleBuffer<ui32>::CopyMapping(input);
                auto flags = TSingleBuffer<ui32>::CopyMapping(input);

                TVector<ui32> data(size);
                TVector<ui32> flagsCpu(size);

                std::generate(data.begin(), data.end(), [&]() {
                    return rand.NextUniformL() % 10000;
                });

                const double p = 5000.0 / size;
                std::generate(flagsCpu.begin(), flagsCpu.end(), [&]() {
                    return rand.NextUniform() < p ? 1 : 0;
                });
                flagsCpu[0] = 1;

                input.Write(data);
                flags.Write(flagsCpu);

                for (ui32 k = 0; k < tries; ++k) {
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Scan of #" << i << " elements");
                        SegmentedScanVector(input, flags, output);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestSegmentedScanPerformanceFloat) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestSegmentedScanPerformance<float>();
        }
    }

    Y_UNIT_TEST(TestSegmentedScanPerformanceInt) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestSegmentedScanPerformance<int>();
        }
    }

    Y_UNIT_TEST(TestSegmentedScanPerformanceUnsignedInt) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestSegmentedScanPerformance<ui32>();
        }
    }

    Y_UNIT_TEST(TestSegmentedScanNonNegativePerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            RunSegmentedScanNonNegativePerformanceTest();
        }
    }
}
