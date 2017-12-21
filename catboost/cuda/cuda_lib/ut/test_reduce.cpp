#include <catboost/cuda/cuda_lib/memory_pool/stack_like_memory_pool.h>
#include <library/unittest/registar.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/reduce_scatter.h>

using namespace NCudaLib;

SIMPLE_UNIT_TEST_SUITE(TStripeReduceTest) {
    void TestReduce(const size_t partSize = 64 * 64, const size_t partCountBase = 2000) {
        TRandom rand(0);
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        auto& profiler = GetProfiler();

        const int tries = 10;

        for (int tr = 0; tr < tries; ++tr) {
            const size_t randPart = 1 + partCountBase * 0.05;
            const size_t partCount = partCountBase + (tr == 0 ? randPart : rand.NextUniformL() % randPart);

            auto beforeReduceMapping = TStripeMapping::RepeatOnAllDevices(partCount, partSize);
            //            auto singleMapping = TDeviceMapping<MT_SINGLE>::Create(0, partCount * partSize * TCudaManager::GetDeviceCount());

            auto afterReduceMapping = TStripeMapping::SplitBetweenDevices(partCount, partSize);

            auto data = TStripeBuffer<float>::Create(beforeReduceMapping);

            TVector<float> dataCpu;
            TVector<float> reducedDataCpu;

            dataCpu.resize(beforeReduceMapping.MemorySize());
            reducedDataCpu.resize(partCount * partSize);

            for (ui32 i = 0; i < dataCpu.size(); ++i) {
                dataCpu[i] = (float)rand.NextUniform();
                reducedDataCpu[i % reducedDataCpu.size()] += dataCpu[i];
            }

            data.Write(dataCpu);

            TReducer<decltype(data)> reducer;

            {
                auto guard = profiler.Profile("Reduce");
                reducer(data, afterReduceMapping);
            }

            TVector<float> reducedGpu;
            data.Read(reducedGpu);

            for (ui32 i = 0; i < reducedDataCpu.size(); ++i) {
                UNIT_ASSERT_DOUBLES_EQUAL(reducedDataCpu[i], reducedGpu[i], 1e-5);
            }
        }
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll4x4) {
        StartCudaManager();
        {
            TestReduce(4, 4);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll4x20000) {
        StartCudaManager();
        {
            TestReduce(4, 20000);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll8) {
        StartCudaManager();
        {
            TestReduce(8, 20000);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll128) {
        StartCudaManager();
        {
            TestReduce(128, 20000);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll256) {
        StartCudaManager();
        {
            TestReduce(256, 20000);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll512) {
        StartCudaManager();
        {
            TestReduce(512, 20000);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestReduceOnAll4096) {
        StartCudaManager();
        {
            TestReduce(64 * 64, 20000);
        }
        StopCudaManager();
    }
}
