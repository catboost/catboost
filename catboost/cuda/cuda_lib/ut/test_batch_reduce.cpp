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

Y_UNIT_TEST_SUITE(TBatchStripeReduceTest) {
    const int tries = 100;

    void TestBatchReduce(int batchSize,
                         const size_t partSize = 128 * 4 * 9,
                         bool compress = false) {
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        TSetLogging inThisScope(ELoggingLevel::Debug);
        auto& profiler = GetProfiler();

        for (int tr = 0; tr < tries; ++tr) {
            {
                auto beforeReduceMapping = TStripeMapping::RepeatOnAllDevices(partSize);
                auto afterReduceMapping = TStripeMapping::SplitBetweenDevices(partSize);

                TVector<TStripeBuffer<float>> data;
                TVector<TComputationStream> streams;
                for (int i = 0; i < batchSize; ++i) {
                    data.emplace_back(TStripeBuffer<float>::Create(beforeReduceMapping));
                    FillBuffer(data.back(), 1.0f * i);
                    if (streams.size() < 100) {
                        streams.push_back(GetCudaManager().RequestStream());
                    }
                }

                {
                    TString label = compress ? "ReduceCompressed" : "Reduce";
                    auto guard = profiler.Profile(label);
                    for (int i = 0; i < batchSize; ++i) {
                        auto& toReduce = data[i];
                        TReducer<TStripeBuffer<float>> reducer(streams[i % 100].GetId());
                        reducer(toReduce, afterReduceMapping, compress);
                    }
                }
            }
            {
                auto beforeReduceMapping = TStripeMapping::RepeatOnAllDevices(batchSize * partSize);
                auto afterReduceMapping = TStripeMapping::SplitBetweenDevices(batchSize * partSize);

                TStripeBuffer<float> data = TStripeBuffer<float>::Create(beforeReduceMapping);
                {
                    TString label = compress ? "ReduceAllBatchCompressed" : "ReduceBatch";
                    auto guard = profiler.Profile(label);
                    TReducer<TStripeBuffer<float>> reducer;
                    reducer(data, afterReduceMapping, compress);
                }
            }
        }
    }

    Y_UNIT_TEST(TestBatchReduce1) {
        auto stopCudaManagerGuard = StartCudaManager();
        TestBatchReduce(1);
    }

    Y_UNIT_TEST(TestBatchReduce10) {
        auto stopCudaManagerGuard = StartCudaManager();
        TestBatchReduce(10);
    }

    Y_UNIT_TEST(TestBatchReduce100) {
        auto stopCudaManagerGuard = StartCudaManager();
        TestBatchReduce(100);
    }

    Y_UNIT_TEST(TestBatchReduce1000) {
        auto stopCudaManagerGuard = StartCudaManager();
        TestBatchReduce(1000);
    }
}
