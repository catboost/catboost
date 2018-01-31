#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <library/unittest/registar.h>
#include <iostream>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/helpers.h>

using namespace std;
using namespace NCudaLib;

SIMPLE_UNIT_TEST_SUITE(TReduceTest) {
    SIMPLE_UNIT_TEST(TestReduce) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                ui64 size = rand.NextUniformL() % 10000000;
                double sum = 0;
                for (ui64 i = 0; i < size; ++i) {
                    vec.push_back(8.0 / (1 << (rand.NextUniformL() % 8)));
                    sum += vec.back();
                }

                auto mapping = TSingleMapping(0, size);
                auto reducedMapping = TSingleMapping(0, 1);
                auto cVec = TSingleBuffer<float>::Create(mapping);
                auto resultVec = TSingleBuffer<float>::Create(reducedMapping);
                cVec.Write(vec);
                ReduceVector(cVec, resultVec);

                TVector<float> result;
                resultVec.Read(result);

                UNIT_ASSERT_DOUBLES_EQUAL(result[0] / vec.size(), sum / vec.size(), 1e-5);
            }
        }
    }

    SIMPLE_UNIT_TEST(TestSegmentedReduce) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {2, 4, 7, 15, 30, 55, 77, 110, 140, 255, 1024, 10000}) {
                    TVector<float> vec;
                    ui64 size = 50000 + rand.NextUniformL() % 100;

                    TVector<ui32> segmentOffsets;
                    TVector<double> segmentSums;

                    for (ui32 i = 0; i < size; ++i) {
                        ui32 segmentSize = max(ceil(meanSize * rand.NextUniform()), 1.0);

                        segmentOffsets.push_back(i);
                        segmentSums.push_back(0);

                        for (ui32 j = 0; j < segmentSize; ++j) {
                            vec.push_back(1.0 / (1 << (rand.NextUniformL() % 8)));
                            segmentSums.back() += vec.back();
                        }
                        i += (segmentSize - 1);
                    }
                    size = vec.size();
                    segmentOffsets.push_back(size);

                    auto mapping = TSingleMapping(0, size);
                    auto segmentsMapping = TSingleMapping(0, segmentOffsets.size());
                    auto reducedMapping = TSingleMapping(0, segmentSums.size());
                    auto offsetsVec = TSingleBuffer<ui32>::Create(segmentsMapping);
                    offsetsVec.Write(segmentOffsets);

                    auto cVec = TSingleBuffer<float>::Create(mapping);
                    auto resultVec = TSingleBuffer<float>::Create(reducedMapping);

                    cVec.Write(vec);
                    SegmentedReduceVector(cVec, offsetsVec, resultVec);

                    TVector<ui32> offsetsAfterReduce;
                    offsetsVec.Read(offsetsAfterReduce);

                    TVector<float> reducedOnGpu;
                    resultVec.Read(reducedOnGpu);

                    for (ui32 i = 0; i < segmentSums.size(); ++i) {
                        if (std::abs(segmentSums[i] - reducedOnGpu[i]) > 1e-8) {
                            Dump(cVec, "vector to reduce");
                            Dump(offsetsVec, "vector to reduce");
                            Dump(resultVec, "reduced");
                        }
                        UNIT_ASSERT_DOUBLES_EQUAL(segmentSums[i], reducedOnGpu[i], 1e-8);
                        UNIT_ASSERT_VALUES_EQUAL(offsetsAfterReduce[i], segmentOffsets[i]);
                    }
                }
            }
        }
    }

    SIMPLE_UNIT_TEST(TestSegmentedReducePerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            auto& profiler = GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {2, 4, 7, 15, 30, 55, 77, 110, 140, 255, 512, 1024, 2048, 4096, 10000, 20000}) {
                    TVector<float> vec;
                    ui64 size = 10000000;

                    TVector<ui32> segmentOffsets;
                    TVector<float> segmentSums;

                    for (ui32 i = 0; i < size; ++i) {
                        ui32 segmentSize = max(ceil(meanSize * rand.NextUniform()), 1.0);
                        segmentOffsets.push_back(i);
                        segmentSums.push_back(0);
                        for (ui32 j = 0; j < segmentSize; ++j) {
                            vec.push_back(1.0 / (1 << (rand.NextUniformL() % 8)));
                            segmentSums.back() += vec.back();
                        }
                        i += (segmentSize - 1);
                    }
                    size = vec.size();
                    segmentOffsets.push_back(size);

                    auto mapping = TSingleMapping(0, size);
                    auto segmentsMapping = TSingleMapping(0, segmentOffsets.size());
                    auto reducedMapping = TSingleMapping(0, segmentSums.size());
                    auto offsetsVec = TSingleBuffer<ui32>::Create(segmentsMapping);
                    offsetsVec.Write(segmentOffsets);

                    auto cVec = TSingleBuffer<float>::Create(mapping);
                    auto resultVec = TSingleBuffer<float>::Create(reducedMapping);

                    cVec.Write(vec);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Segmented reduce for mean segmet size = " << meanSize);
                        SegmentedReduceVector(cVec, offsetsVec, resultVec);
                    }
                }
            }
        }
    }
}
