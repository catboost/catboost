#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <library/cpp/testing/unittest/registar.h>
#include <iostream>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TReduceTest) {
    Y_UNIT_TEST(TestReduce) {
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

    Y_UNIT_TEST(TestSegmentedReduce) {
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
                            //                            vec.push_back(1.0f);
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

                    TVector<double> cudaLibReduceImplCpu;
                    auto cudaLibReduceImplGpu = TSingleBuffer<double>::CopyMapping(resultVec);
                    {
                        ComputePartitionStats(cVec, offsetsVec, &cudaLibReduceImplGpu);
                        cudaLibReduceImplGpu.Read(cudaLibReduceImplCpu);
                    }
                    for (ui32 i = 0; i < segmentSums.size(); ++i) {
                        UNIT_ASSERT_DOUBLES_EQUAL(segmentSums[i], reducedOnGpu[i], 1e-8);
                        UNIT_ASSERT_DOUBLES_EQUAL_C(segmentSums[i], cudaLibReduceImplCpu[i], 1e-8, i << " " << segmentSums[i] << " ≠ " << cudaLibReduceImplCpu[i] << " " << segmentSums.size());
                        UNIT_ASSERT_VALUES_EQUAL(offsetsAfterReduce[i], segmentOffsets[i]);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestUpdatePartProps) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {132, 200, 10000, 1000000, 1000000, 10000000}) {
                    TVector<float> vec;

                    TVector<TDataPartition> parts;
                    TVector<ui32> partIds;
                    TVector<double> segmentSums;

                    ui32 i = 0;
                    for (ui32 p = 0; p < 16; ++p) {
                        ui32 segmentSize = i == 0 ? meanSize : max(ceil(meanSize * rand.NextUniform()), 1.0);

                        TDataPartition part;
                        part.Offset = i;
                        part.Size = segmentSize;

                        parts.push_back(part);
                        partIds.push_back(partIds.size());

                        segmentSums.push_back(0);

                        for (ui32 j = 0; j < segmentSize; ++j) {
                            vec.push_back(1.0 / (1 << (rand.NextUniformL() % 8)));
                            segmentSums.back() += vec.back();
                        }
                        i += segmentSize;
                    }
                    ui64 size = vec.size();

                    auto mapping = TMirrorMapping(size);
                    auto partsMapping = TMirrorMapping(parts.size());
                    auto reducedMapping = TMirrorMapping(segmentSums.size());

                    auto partsGpu = TMirrorBuffer<TDataPartition>::Create(partsMapping);
                    auto partIdsGpu = TMirrorBuffer<ui32>::Create(partsMapping);
                    partsGpu.Write(parts);
                    partIdsGpu.Write(partIds);

                    auto cVec = TMirrorBuffer<float>::Create(mapping);
                    auto resultVec = TMirrorBuffer<double>::Create(reducedMapping);

                    cVec.Write(vec);
                    ComputePartitionStats(cVec, partsGpu, partIdsGpu, &resultVec);

                    TVector<double> reducedOnGpu;
                    resultVec.Read(reducedOnGpu);

                    for (ui32 i = 0; i < segmentSums.size(); ++i) {
                        if (std::abs(segmentSums[i] - reducedOnGpu[i]) > 1e-8) {
                            Cout << i << " " << parts[i].Offset << " / " << parts[i].Size << Endl;
                        }
                        UNIT_ASSERT_DOUBLES_EQUAL(segmentSums[i], reducedOnGpu[i], 1e-8);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestUpdatePartPropsPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
            auto& profiler = GetCudaManager().GetProfiler();

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {132, 200, 10000, 1000000, 1000000, 10000000}) {
                    TVector<float> vec;

                    TVector<TDataPartition> parts;
                    TVector<ui32> partIds;
                    TVector<double> segmentSums;

                    ui32 i = 0;
                    for (ui32 p = 0; p < 64; ++p) {
                        ui32 segmentSize = i == 0 ? meanSize : max(ceil(meanSize * rand.NextUniform()), 1.0);

                        TDataPartition part;
                        part.Offset = i;
                        part.Size = segmentSize;
                        parts.push_back(part);
                        partIds.push_back(partIds.size());
                        segmentSums.push_back(0);

                        for (ui32 j = 0; j < segmentSize; ++j) {
                            vec.push_back(1.0 / (1 << (rand.NextUniformL() % 8)));
                            segmentSums.back() += vec.back();
                        }
                        i += segmentSize;
                    }
                    const ui64 size = vec.size();

                    auto mapping = TMirrorMapping(size);
                    auto partsMapping = TMirrorMapping(parts.size());
                    auto reducedMapping = TMirrorMapping(segmentSums.size());
                    auto partsGpu = TMirrorBuffer<TDataPartition>::Create(partsMapping);
                    auto partIdsGpu = TMirrorBuffer<ui32>::Create(partsMapping);
                    partsGpu.Write(parts);
                    partIdsGpu.Write(partIds);

                    auto cVec = TMirrorBuffer<float>::Create(mapping);
                    auto resultVec = TMirrorBuffer<double>::Create(reducedMapping);

                    cVec.Write(vec);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Update part props mean part size = " << meanSize);
                        ComputePartitionStats(cVec, partsGpu, partIdsGpu, &resultVec);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestSegmentedReducePerformance2) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            auto& profiler = GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {200, 10000, 1000000, 1000000, 10000000}) {
                    TVector<float> vec;

                    TVector<ui32> segmentOffsets;
                    TVector<float> segmentSums;

                    ui32 i = 0;
                    for (ui32 p = 0; p < 64; ++p) {
                        ui32 segmentSize = max(ceil(meanSize * rand.NextUniform()), 1.0);
                        segmentOffsets.push_back(i);
                        segmentSums.push_back(0);
                        for (ui32 j = 0; j < segmentSize; ++j) {
                            vec.push_back(1.0 / (1 << (rand.NextUniformL() % 8)));
                            segmentSums.back() += vec.back();
                        }
                        i += (segmentSize);
                    }
                    const ui64 size = vec.size();
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

    Y_UNIT_TEST(TestSegmentedReducePerformance) {
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
                        i += (segmentSize);
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

    Y_UNIT_TEST(TestStatsInLeavesPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            auto& profiler = GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 leafCount : {2, 8, 16, 64}) {
                for (ui32 k = 0; k < tries; ++k) {
                    for (ui32 meanSize : {1000, 100000, 1000000}) {
                        TVector<ui32> segmentOffsets;
                        TVector<float> segmentSums;
                        TVector<float> vec;

                        for (ui32 i = 0; i < leafCount; ++i) {
                            ui32 segmentSize = max(ceil(meanSize * rand.NextUniform()), 1.0);
                            segmentOffsets.push_back(vec.size());
                            segmentSums.push_back(0);
                            for (ui32 j = 0; j < segmentSize; ++j) {
                                vec.push_back(1.0 / (1 << (rand.NextUniformL() % 8)));
                                segmentSums.back() += vec.back();
                            }
                        }
                        size_t size = vec.size();
                        segmentOffsets.push_back(size);

                        auto mapping = TSingleMapping(0, size);
                        auto segmentsMapping = TSingleMapping(0, segmentOffsets.size());
                        auto reducedMapping = TSingleMapping(0, segmentSums.size());
                        auto offsetsVec = TSingleBuffer<ui32>::Create(segmentsMapping);
                        offsetsVec.Write(segmentOffsets);

                        auto cVec = TSingleBuffer<float>::Create(mapping);
                        auto resultVec = TSingleBuffer<float>::Create(reducedMapping);
                        auto resultVecDouble = TSingleBuffer<double>::Create(reducedMapping);

                        cVec.Write(vec);
                        {
                            auto guard = profiler.Profile(TStringBuilder() << "Segmented reduce for  leaf_count  " << leafCount << " meanSize " << meanSize);
                            SegmentedReduceVector(cVec, offsetsVec, resultVec);
                        }

                        {
                            auto guard = profiler.Profile(TStringBuilder() << "Segmented reduce  cuda lib impl for  leaf_count  " << leafCount << " meanSize " << meanSize);
                            ComputePartitionStats(cVec, offsetsVec, &resultVecDouble);
                        }
                    }
                }
            }
        }
    }
}
