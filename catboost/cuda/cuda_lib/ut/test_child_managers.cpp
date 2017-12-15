#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <library/unittest/registar.h>
#include <iostream>
#include <thread>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <library/threading/local_executor/local_executor.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/utils/countdown_latch.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

using namespace NCudaLib;

SIMPLE_UNIT_TEST_SUITE(TChildManagerTest) {
    SIMPLE_UNIT_TEST(TestFill) {
        auto& manager = NCudaLib::GetCudaManager();
        StartCudaManager();
        {
            NPar::LocalExecutor().RunAdditionalThreads(manager.GetDeviceCount());

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
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestMakeSeqStripe) {
        auto& manager = NCudaLib::GetCudaManager();
        StartCudaManager();
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
        StopCudaManager();
    }
}
