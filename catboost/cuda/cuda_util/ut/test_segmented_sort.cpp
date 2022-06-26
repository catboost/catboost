#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/segmented_sort.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <library/cpp/testing/unittest/registar.h>
#include <iostream>
#include <util/generic/algorithm.h>
#include <catboost/cuda/cuda_util/fill.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TSegmentedSortTest) {
    Y_UNIT_TEST(TestSegmentedSort) {
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

    Y_UNIT_TEST(TestSegmentedSortNonContinous) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 size = 10000; size < 10000001; size *= 10) {
                for (ui32 k = 0; k < tries; ++k) {
                    const ui32 partCount = 39;

                    TVector<ui32> segmentStarts;
                    TVector<ui32> segmentEnds;

                    TVector<ui32> segmentKeys(size);
                    TVector<ui32> segmentValues(size);

                    Iota(segmentKeys.begin(), segmentKeys.end(), 0);
                    for (ui32 i = 0; i < size; ++i) {
                        segmentValues[i] = (10 * i);
                    }

                    const ui32 segmentSize = 21;

                    for (ui32 i = 0; i < partCount; ++i) {
                        ui32 start = (size / (partCount + 1)) * i;
                        ui32 end = Min(size, start + segmentSize);
                        segmentStarts.push_back(start);
                        segmentEnds.push_back(end);
                    }

                    for (size_t i = 0; i < segmentStarts.size(); ++i) {
                        ui32 firstIdx = segmentStarts[i];
                        ui32 lastIdx = segmentEnds[i];

                        ui32 size = (lastIdx - firstIdx);
                        CB_ENSURE(size > 2);
                        for (ui32 j = 0; j < size / 2; ++j) {
                            std::swap(segmentKeys[firstIdx + j], segmentKeys[firstIdx + size - j - 1]);
                            std::swap(segmentValues[firstIdx + j], segmentValues[firstIdx + size - j - 1]);
                        }
                    }

                    auto mapping = TSingleMapping(0, size);

                    auto keys = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpKeys = TSingleBuffer<ui32>::Create(mapping);

                    auto values = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpValues = TSingleBuffer<ui32>::Create(mapping);

                    FillBuffer(tmpKeys, 0u);
                    FillBuffer(tmpValues, 0u);

                    auto segmentsMapping = TSingleMapping(0, segmentStarts.size());
                    auto segmentStartsGpu = TSingleBuffer<ui32>::Create(segmentsMapping);
                    auto segmentEndGpu = TSingleBuffer<ui32>::Create(segmentsMapping);
                    segmentStartsGpu.Write(segmentStarts);
                    segmentEndGpu.Write(segmentEnds);

                    keys.Write(segmentKeys);
                    values.Write(segmentValues);

                    {
                        auto guard = GetCudaManager().GetProfiler().Profile(TStringBuilder() << "Segmented sort 17 segments #" << size);
                        SegmentedRadixSort(keys, values, tmpKeys, tmpValues, segmentStartsGpu, segmentEndGpu, partCount);
                    }

                    TVector<ui32> keysAfterSort;
                    TVector<ui32> valsAfterSort;
                    keys.Read(keysAfterSort);
                    values.Read(valsAfterSort);

                    for (ui32 i = 0; i < size; ++i) {
                        UNIT_ASSERT_VALUES_EQUAL(keysAfterSort[i], i);
                        UNIT_ASSERT_VALUES_EQUAL(valsAfterSort[i], 10 * i);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestSegmentedSortOneBlock) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            for (ui32 size = 10000; size < 10000001; size *= 10) {
                for (ui32 k = 0; k < tries; ++k) {
                    const ui32 partCount = 1;

                    TVector<ui32> segmentStarts = {0};
                    TVector<ui32> segmentEnds = {size};

                    TVector<ui32> segmentKeys(size);
                    TVector<ui32> segmentValues(size);

                    for (ui32 i = 0; i < size; ++i) {
                        segmentValues[i] = (10 * (size - i - 1));
                        segmentKeys[i] = ((size - i - 1));
                    }

                    auto mapping = TSingleMapping(0, size);

                    auto keys = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpKeys = TSingleBuffer<ui32>::Create(mapping);

                    auto values = TSingleBuffer<ui32>::Create(mapping);
                    auto tmpValues = TSingleBuffer<ui32>::Create(mapping);

                    FillBuffer(tmpKeys, 0u);
                    FillBuffer(tmpValues, 0u);

                    auto segmentsMapping = TSingleMapping(0, segmentStarts.size());
                    auto segmentStartsGpu = TSingleBuffer<ui32>::Create(segmentsMapping);
                    auto segmentEndGpu = TSingleBuffer<ui32>::Create(segmentsMapping);
                    segmentStartsGpu.Write(segmentStarts);
                    segmentEndGpu.Write(segmentEnds);

                    keys.Write(segmentKeys);
                    values.Write(segmentValues);

                    {
                        auto guard = GetCudaManager().GetProfiler().Profile(TStringBuilder() << "Segmented sort 1 segments #" << size);
                        SegmentedRadixSort(keys, values, tmpKeys, tmpValues, segmentStartsGpu, segmentEndGpu, partCount);
                    }

                    TVector<ui32> keysAfterSort;
                    TVector<ui32> valsAfterSort;
                    keys.Read(keysAfterSort);
                    values.Read(valsAfterSort);

                    for (ui32 i = 0; i < size; ++i) {
                        UNIT_ASSERT_VALUES_EQUAL(keysAfterSort[i], i);
                        UNIT_ASSERT_VALUES_EQUAL(valsAfterSort[i], 10 * i);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestSegmentedSortPerformance) {
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
