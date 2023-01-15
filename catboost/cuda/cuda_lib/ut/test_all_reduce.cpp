#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/cuda_lib/memory_pool/stack_like_memory_pool.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/reduce_scatter.h>
#include <util/generic/ymath.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TAllReduceTest) {
    const bool performanceOnly = false;
    const int tries = 20;

    void TestAllReduce(const size_t partSize = 64 * 64,
                       const size_t partCountBase = 2000,
                       bool performanceOnly = false,
                       bool compress = false,
                       bool throughMaster = false) {
        TRandom rand(0);
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        auto& profiler = GetProfiler();

        TVector<float> dataCpu;
        TVector<double> reducedDataCpu;
        for (ui32 i = 0; i < dataCpu.size(); ++i) {
            dataCpu[i] = (float)rand.NextUniform();
        }

        for (int tr = 0; tr < tries; ++tr) {
            const size_t randPart = 1 + partCountBase * 0.05;
            const size_t partCount = partCountBase + (tr == 0 ? randPart : rand.NextUniformL() % randPart);

            auto beforeReduceMapping = TStripeMapping::RepeatOnAllDevices(partCount, partSize);
            //            auto singleMapping = TDeviceMapping<MT_SINGLE>::Create(0, partCount * partSize * TCudaManager::GetDeviceCount());

            auto afterReduceMapping = TMirrorMapping(partCount, partSize);

            auto data = TStripeBuffer<float>::Create(beforeReduceMapping);
            auto reduceData = TMirrorBuffer<float>::Create(afterReduceMapping);
            FillBuffer(reduceData, 0.0f);

            dataCpu.resize(beforeReduceMapping.MemorySize());
            reducedDataCpu.clear();
            reducedDataCpu.resize(partCount * partSize);

            if (!performanceOnly) {
                for (ui32 i = 0; i < dataCpu.size(); ++i) {
                    dataCpu[i] = (float)rand.NextUniform();
                    reducedDataCpu[i % reducedDataCpu.size()] += dataCpu[i];
                }
                data.Write(dataCpu);
            }

            {
                TString label = compress ? "AllReduceCompressed" : "AllReduce";
                if (throughMaster) {
                    label += " (through master)";
                }
                label = TStringBuilder() << label << " " << partSize * partCountBase;
                auto guard = profiler.Profile(label);
                if (throughMaster) {
                    AllReduceThroughMaster(data, reduceData, 0, compress);
                } else {
                    AllReduce(data, reduceData, 0, compress);
                }
            }

            if (!performanceOnly) {
                TVector<float> reducedGpu;
                if (!throughMaster) {
                    TVector<float> tmp;
                    data.Read(tmp);
                    for (ui32 i = 0; i < reducedDataCpu.size(); ++i) {
                        UNIT_ASSERT_DOUBLES_EQUAL_C((float)reducedDataCpu[i], tmp[i], 1e-5, " stripped");
                    }
                }
                for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
                    reduceData.DeviceView(dev).Read(reducedGpu);

                    for (ui32 i = 0; i < reducedDataCpu.size(); ++i) {
                        UNIT_ASSERT_DOUBLES_EQUAL_C((float)reducedDataCpu[i], reducedGpu[i], 1e-5, " on dev " << dev);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestAllReduce) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestAllReduce(1, 64, performanceOnly);
        }
    }

    Y_UNIT_TEST(TestAllReduceLatency) {
        for (ui32 i = 10; i <= 10000000; i *= 10) {
            auto stopCudaManagerGuard = StartCudaManager();
            {
                TestAllReduce(1, i, performanceOnly);
            }
        }
    }

    Y_UNIT_TEST(TestAllReduceLatencyThroughHost) {
        for (ui32 i = 10; i <= 10000000; i *= 10) {
            auto stopCudaManagerGuard = StartCudaManager();
            {
                TestAllReduce(1, i, performanceOnly, false, true);
            }
        }
    }

    Y_UNIT_TEST(TestAllReduce4096) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestAllReduce(64 * 64, 20000, performanceOnly);
        }
    }

#if defined(USE_MPI)
    Y_UNIT_TEST(TestAllReduce256Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestAllReduce(16 * 16, 20000, performanceOnly, true);
        }
    }

    Y_UNIT_TEST(TestAllReduce4096Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestAllReduce(64 * 64, 20000, performanceOnly, true);
        }
    }
#endif
}
