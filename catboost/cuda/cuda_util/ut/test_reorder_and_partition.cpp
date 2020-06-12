#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/partitions.h>
#include <catboost/cuda/cuda_util/reorder_bins.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/libs/helpers/cpu_random.h>

#include <library/cpp/testing/unittest/registar.h>

#include <iostream>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TReorderTest) {
    void RunReorderTest(ui32 bits = 2) {
        TRandom rand(42);
        const ui32 expCount = 4;
        const ui32 devCount = GetDeviceCount();
        const ui32 binCount = 1 << bits;
        for (ui32 i = 0; i < expCount; ++i) {
            auto binsMapping = TStripeMapping::SplitBetweenDevices(devCount * (rand.NextUniformL() % 1000000));

            TVector<int> offsets(binCount * devCount, 0);
            TVector<int> sizes(binCount * devCount, 0);

            TVector<ui32> binsCpu;
            {
                binsCpu.resize(binsMapping.GetObjectsSlice().Size());
                for (ui32 dev = 0; dev < devCount; ++dev) {
                    TSlice deviceSlice = binsMapping.DeviceSlice(dev);

                    for (ui32 i = deviceSlice.Left; i < deviceSlice.Right; ++i) {
                        binsCpu[i] = (ui32)(rand.NextUniformL() % binCount);
                        sizes[dev * binCount + binsCpu[i]]++;
                    }
                }
            }
            {
                for (ui32 dev = 0; dev < devCount; ++dev) {
                    for (ui32 i = 1; i < binCount; ++i) {
                        offsets[dev * binCount + i] = offsets[dev * binCount + i - 1] + sizes[dev * binCount + i - 1];
                    }
                }
            }

            auto bins = TStripeBuffer<ui32>::Create(binsMapping);
            bins.Write(binsCpu);

            auto indices = TStripeBuffer<ui32>::CopyMapping(bins);
            MakeSequence(indices);
            if (bits == 1) {
                ReorderOneBit(bins, indices, 0);
            } else {
                ReorderBins(bins, indices, 0, bits);
            }

            TVector<TDataPartition> partsCpu;
            auto partsGpuMapping = binsMapping.RepeatOnAllDevices(binCount);
            auto partsGpu = TStripeBuffer<TDataPartition>::Create(partsGpuMapping);
            UpdatePartitionDimensions(bins, partsGpu);
            partsGpu.Read(partsCpu);

            for (ui32 dev = 0; dev < devCount; ++dev) {
                for (ui32 i = 0; i < binCount; ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(partsCpu[dev * binCount + i].Offset, offsets[dev * binCount + i]);
                    UNIT_ASSERT_VALUES_EQUAL(partsCpu[dev * binCount + i].Size, sizes[dev * binCount + i]);
                }
            }

            {
                bins.Read(binsCpu);
                ui32 offset = 0;

                for (ui32 dev = 0; dev < devCount; ++dev) {
                    ui32 devSize = binsMapping.DeviceSlice(dev).Size();
                    for (ui32 i = 1; i < devSize; ++i) {
                        UNIT_ASSERT_VALUES_EQUAL(binsCpu[offset + i] >= binsCpu[offset + i - 1], true);
                    }
                    offset += devSize;
                }
            }
        }
    }

    Y_UNIT_TEST(TestReorder) {
        auto stopCudaManagerGuard = StartCudaManager();
        for (auto bits : {1, 2}) {
            RunReorderTest(bits);
        }
    }
}
