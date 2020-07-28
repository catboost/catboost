#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <library/cpp/testing/unittest/registar.h>
#include <iostream>
#include <thread>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TFillTest) {
    Y_UNIT_TEST(TestMakeSequence) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 5;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = 2 + rand.NextUniformL() % 1000000;
                ui32 dev = rand.NextUniformL() % GetDeviceCount();
                TSingleMapping mapping(dev, size);
                auto cVec = TCudaBuffer<ui32, TSingleMapping>::Create(mapping);
                MakeSequence(cVec);
                TVector<ui32> result;
                cVec.Read(result);

                for (ui32 i = 0; i < result.size(); ++i) {
                    UNIT_ASSERT_EQUAL(result[i], i);
                }
            }

            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = 2 + rand.NextUniformL() % 1000000;

                TStripeMapping stripeMapping = TStripeMapping::SplitBetweenDevices(size);
                auto cVec = TCudaBuffer<ui32, TStripeMapping>::Create(stripeMapping);
                MakeSequence(cVec);
                TVector<ui32> result;
                cVec.Read(result);

                UNIT_ASSERT_EQUAL(result.size(), size);

                for (ui32 devId : cVec.NonEmptyDevices()) {
                    TSlice devSlice = stripeMapping.DeviceSlice(devId);
                    for (ui32 i = devSlice.Left; i < devSlice.Right; ++i) {
                        UNIT_ASSERT_EQUAL(result[i], i - devSlice.Left);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestFill) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 5;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                const ui64 size = 2 + rand.NextUniformL() % 1000000;
                TVector<float> data(size);
                TVector<float> tmp;
                std::generate(data.begin(), data.end(), [&]() {
                    return rand.NextUniform();
                });

                TStripeMapping mapping = TStripeMapping::SplitBetweenDevices(data.size());
                auto cudaVec = TCudaBuffer<float, TStripeMapping>::Create(mapping);
                cudaVec.Write(data);
                FillBuffer(cudaVec, 1.0f);
                cudaVec.Read(tmp);

                UNIT_ASSERT_EQUAL(tmp.size(), size);
                for (ui32 i = 0; i < tmp.size(); ++i) {
                    UNIT_ASSERT_EQUAL(tmp[i], 1.0f);
                }
            }
        }
    }
}
