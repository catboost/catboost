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
//TODO(noxoomo): remove duplication for this and tree-reduce tests
Y_UNIT_TEST_SUITE(TRingStripeReduceTest) {
    const bool performanceOnly = false;
    const int tries = 20;

    void TestReduce(const size_t partSize = 64 * 64,
                    const size_t partCountBase = 2000,
                    bool performanceOnly = false,
                    bool compress = false,
                    bool reduceSingle = false) {
        TRandom rand(0);
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        auto& profiler = GetProfiler();

        TVector<float> dataCpu;
        TVector<float> reducedDataCpu;
        for (ui32 i = 0; i < dataCpu.size(); ++i) {
            dataCpu[i] = (float)rand.NextUniform();
        }

        for (int tr = 0; tr < tries; ++tr) {
            const size_t randPart = 1 + partCountBase * 0.05;
            const size_t partCount = partCountBase + (tr == 0 ? randPart : rand.NextUniformL() % randPart);

            auto beforeReduceMapping = TStripeMapping::RepeatOnAllDevices(partCount, partSize);
            //            auto singleMapping = TDeviceMapping<MT_SINGLE>::Create(0, partCount * partSize * TCudaManager::GetDeviceCount());

            auto afterReduceMapping = TStripeMapping::SplitBetweenDevices(partCount, partSize);
            if (reduceSingle) {
                TMappingBuilder<TStripeMapping> builder;
                builder.SetSizeAt(0, partCount);
                afterReduceMapping = builder.Build(partSize);
            }

            auto data = TStripeBuffer<float>::Create(beforeReduceMapping);

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

            TReducer<decltype(data), EReduceAlgorithm::Ring> reducer;
            {
                TString label = compress ? "ReduceCompressed" : "Reduce";
                if (reduceSingle) {
                    label += " (to single)";
                }
                auto guard = profiler.Profile(label);
                reducer(data, afterReduceMapping, compress);
            }

            if (!performanceOnly) {
                TVector<float> reducedGpu;
                data.Read(reducedGpu);

                for (ui32 i = 0; i < reducedDataCpu.size(); ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(reducedDataCpu[i], reducedGpu[i], 1e-5);
                }
            }
        }
    }

    Y_UNIT_TEST(TestReduceOnAll4x4) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(4, 4, performanceOnly);
        }
    }
#if defined(USE_MPI)
    Y_UNIT_TEST(TestReduceOnAll4x4Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(4, 4, performanceOnly);
        }
    }
#endif

    Y_UNIT_TEST(TestReduceOnAll4x20000) {
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(4, 20000, performanceOnly);
        }
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(4, 20000, performanceOnly, false, true);
        }
    }

#if defined(USE_MPI)
    Y_UNIT_TEST(TestReduceOnAll4x20000Compressed) {
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(4, 20000, performanceOnly, true);
        }
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(4, 20000, performanceOnly, true, true);
        }
    }
#endif

    Y_UNIT_TEST(TestReduceOnAll8) {
        {
            auto sopCudaManagerGuard = StartCudaManager();
            TestReduce(8, 20000, performanceOnly);
        }
        {
            auto sopCudaManagerGuard = StartCudaManager();
            TestReduce(8, 20000, performanceOnly, false, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll128) {
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(128, 20000, performanceOnly);
        }
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(128, 20000, performanceOnly, false, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll256) {
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(256, 20000, performanceOnly);
        }
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(256, 20000, performanceOnly, false, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll512) {
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(512, 20000, performanceOnly);
        }
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(512, 20000, performanceOnly, false, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll4096) {
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(64 * 64, 20000, performanceOnly);
        }
        {
            auto stopCudaManagerGuard = StartCudaManager();
            TestReduce(64 * 64, 20000, performanceOnly, false, true);
        }
    }

#if defined(USE_MPI)
    Y_UNIT_TEST(TestReduceOnAll8Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(8, 20000, performanceOnly, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll128Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(128, 20000, performanceOnly, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll256Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(256, 20000, performanceOnly, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll512Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(512, 20000, performanceOnly, true);
        }
    }

    Y_UNIT_TEST(TestReduceOnAll4096Compressed) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestReduce(64 * 64, 20000, performanceOnly, true);
        }
    }

#endif
}
