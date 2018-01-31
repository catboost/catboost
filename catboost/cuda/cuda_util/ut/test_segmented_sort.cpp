#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/segmented_sort.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <library/unittest/registar.h>
#include <iostream>

using namespace NCudaLib;

SIMPLE_UNIT_TEST_SUITE(TSegmentedSortTest) {
    SIMPLE_UNIT_TEST(TestSegmentedSort) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {2, 4, 7, 15, 30, 55, 77, 110, 140, 255, 1024, 10000}) {
                    const ui32 partCount = 10000 + rand.NextUniformL() % 100;

                    TVector<ui32> segmentOffsets;

                    TVector<ui32> segmentKeys;
                    TVector<ui32> segmentValues;

                    for (ui32 i = 0; i < partCount; ++i) {
                        ui32 segmentSize = std::max<double>(ceil(meanSize * rand.NextUniform()), 2.0);

                        segmentOffsets.push_back(segmentKeys.size());

                        for (ui32 j = 0; j < segmentSize; ++j) {
                            segmentKeys.push_back(rand.NextUniformL());
                            segmentValues.push_back(rand.NextUniformL());
                        }
                    }
                    const ui32 size = segmentKeys.size();
                    segmentOffsets.push_back(size);

                    auto mapping = TSingleMapping(0, size);

                    auto keys = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpKeys = TSingleBuffer<ui32>::Create(mapping);

                    auto values = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpValues = TSingleBuffer<ui32>::Create(mapping);

                    auto segmentsMapping = TSingleMapping(0, segmentOffsets.size());
                    auto offsetsVec = TSingleBuffer<ui32>::Create(segmentsMapping);
                    offsetsVec.Write(segmentOffsets);

                    keys.Write(segmentKeys);
                    values.Write(segmentValues);

                    SegmentedRadixSort(keys, values, tmpKeys, tmpValues, offsetsVec, partCount);

                    TVector<ui32> keysAfterSort;
                    keys.Read(keysAfterSort);

                    for (ui32 i = 0; (i + 1) < (segmentOffsets.size()); ++i) {
                        ui32 start = segmentOffsets[i];
                        ui32 end = segmentOffsets[i + 1];
                        for (ui32 j = start; (j + 1) < end; ++j) {
                            UNIT_ASSERT(keysAfterSort[j] <= keysAfterSort[j + 1]);
                        }
                    }
                }
            }
        }
    }

    SIMPLE_UNIT_TEST(TestSegmentedSortPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 1; //20;
            TRandom rand(0);
            auto& profiler = GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 k = 0; k < tries; ++k) {
                for (ui32 meanSize : {65536}) {
                    TVector<ui32> segmentKeys;
                    TVector<ui32> segmentValues;
                    TVector<ui32> segmentOffsets;
                    ui32 size = 10000000;

                    for (ui32 i = 0; i < size; ++i) {
                        ui32 segmentSize = std::max<double>(ceil(meanSize * rand.NextUniform()), 1.0);
                        segmentOffsets.push_back(i);
                        for (ui32 j = 0; j < segmentSize; ++j) {
                            segmentKeys.push_back(rand.NextUniformL());
                            segmentValues.push_back(rand.NextUniformL());
                        }
                        i += (segmentSize - 1);
                    }
                    size = segmentKeys.size();
                    segmentOffsets.push_back(size);

                    auto mapping = TSingleMapping(0, size);

                    auto keys = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpKeys = TSingleBuffer<ui32>::Create(mapping);

                    auto values = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpValues = TSingleBuffer<ui32>::Create(mapping);

                    auto segmentsMapping = TSingleMapping(0, segmentOffsets.size());
                    auto offsetsVec = TSingleBuffer<ui32>::Create(segmentsMapping);
                    offsetsVec.Write(segmentOffsets);

                    keys.Write(segmentKeys);
                    values.Write(segmentValues);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Segmented sort for mean segment size = " << meanSize);
                        SegmentedRadixSort(keys, values, tmpKeys, tmpValues, offsetsVec, segmentOffsets.size() - 1);
                    }
                }
            }
        }
    }
}
