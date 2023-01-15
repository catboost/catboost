#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <iostream>
#include <thread>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/utils/countdown_latch.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>
#include <library/cpp/threading/local_executor/local_executor.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TCudaManagerTest) {
    Y_UNIT_TEST(TestKernelDDOS) {
        {
            ui32 count = 100000;
            auto stopCudaManagerGuard = StartCudaManager();
            auto cudaVec1 = TCudaBuffer<float, TMirrorMapping>::Create(TMirrorMapping(GetDeviceCount()));
            auto cudaVec2 = TCudaBuffer<float, TStripeMapping>::Create(TStripeMapping::SplitBetweenDevices(GetDeviceCount()));

            for (ui32 id = 0; id < count; ++id) {
                FillBuffer(cudaVec2, 1.0f);
                Reshard(cudaVec2, cudaVec1);
            }
        }
    }

    Y_UNIT_TEST(TestCreateStreams) {
        for (ui32 i = 0; i < 3; ++i) {
            auto stopCudaManagerGuard = StartCudaManager();
            auto cudaVec = TCudaBuffer<float, TMirrorMapping>::Create(TMirrorMapping(2));

            TVector<TComputationStream> streams;
            for (ui32 id = 0; id < 100; ++id) {
                streams.push_back(RequestStream());
                FillBuffer(cudaVec, 1.0f, streams.back().GetId());
            }
        }
    }

    Y_UNIT_TEST(TestFillChild) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TCudaProfiler& profiler = manager.GetProfiler();
            profiler.SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
            auto profileGuard = profiler.Profile(TStringBuilder() << "Total time");

            ui64 tries = 1000;
            const ui64 size = 1000000;

            RunPerDeviceSubtasks([&](ui32 device) {
                TRandom rand(device);

                for (ui32 k = 0; k < tries; ++k) {
                    TVector<float> data(size);
                    TVector<float> tmp;
                    std::generate(data.begin(), data.end(), [&]() {
                        return rand.NextUniform();
                    });
                    auto& childProfiler = GetCudaManager().GetProfiler();

                    TSingleMapping mapping = TSingleMapping(device, size);

                    auto cudaVec = TCudaBuffer<float, TSingleMapping>::Create(mapping);
                    {
                        auto guard = childProfiler.Profile(TStringBuilder() << "Write #" << size << " floats");
                        cudaVec.Write(data);
                    }
                    {
                        auto guard = childProfiler.Profile(TStringBuilder() << "FillBuffer #" << size << " floats");
                        FillBuffer(cudaVec, 1.0f);
                    }
                    {
                        auto guard = childProfiler.Profile(TStringBuilder() << "Read #" << size << " floats");
                        cudaVec.Read(tmp);
                    }

                    UNIT_ASSERT_EQUAL(tmp.size(), size);
                    for (ui32 i = 0; i < tmp.size(); ++i) {
                        UNIT_ASSERT_EQUAL(tmp[i], 1.0f);
                    }
                }
            });
        }
    }

    Y_UNIT_TEST(TestRequestStreamChild) {
        ui64 tries = 3;

        for (ui32 i = 0; i < tries; ++i) {
            auto stopCudaManagerGuard = StartCudaManager();

            for (int i = 0; i < 10; ++i) {
                RunPerDeviceSubtasks([&](ui32) {
                    TVector<TComputationStream> streams;
                    const ui32 streamCount = 8;
                    for (ui32 i = 0; i < streamCount; ++i) {
                        streams.push_back(GetCudaManager().RequestStream());
                    }
                });
            }
        }
    }

    Y_UNIT_TEST(TestMakeSeqStripeChild) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();
        {
            NPar::LocalExecutor().RunAdditionalThreads(manager.GetDeviceCount());

            TCudaProfiler& profiler = manager.GetProfiler();
            profiler.SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
            auto profileGuard = profiler.Profile(TStringBuilder() << "Total time");

            ui64 tries = 1000;
            const ui64 size = 1000000;

            TStripeMapping mapping = TStripeMapping::SplitBetweenDevices(size);
            auto buffer = TStripeBuffer<ui32>::Create(mapping);

            RunPerDeviceSubtasks([&](ui32 device) {
                TRandom rand(device);

                for (ui32 k = 0; k < tries; ++k) {
                    const TSlice& deviceSlice = mapping.DeviceSlice(device);
                    TVector<ui32> data(deviceSlice.Size());
                    TVector<ui32> tmp;
                    std::generate(data.begin(), data.end(), [&]() {
                        return rand.NextUniformL();
                    });
                    auto& childProfiler = GetCudaManager().GetProfiler();

                    auto cudaVec = buffer.DeviceView(device);
                    {
                        auto guard = childProfiler.Profile(TStringBuilder() << "Write #" << size << " ui32");
                        cudaVec.Write(data);
                    }
                    {
                        auto guard = childProfiler.Profile(TStringBuilder() << "MakeSeq #" << size << " ui32");
                        MakeSequence(cudaVec);
                    }
                    {
                        auto guard = childProfiler.Profile(TStringBuilder() << "Read #" << size << " ui32");
                        cudaVec.Read(tmp);
                    }

                    UNIT_ASSERT_EQUAL(tmp.size(), deviceSlice.Size());
                    for (ui32 i = 0; i < tmp.size(); ++i) {
                        UNIT_ASSERT_EQUAL(tmp[i], i);
                    }
                }
            });
        }
    }
}
