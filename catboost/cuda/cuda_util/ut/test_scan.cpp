#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/scan.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <library/cpp/testing/unittest/registar.h>
#include <iostream>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TScanTest) {
    Y_UNIT_TEST(TestScanStripe) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);
            for (ui32 k = 0; k < tries; ++k) {
                ui64 size = rand.NextUniformL() % 1000000;

                TStripeMapping stripeMapping = TStripeMapping::SplitBetweenDevices(size);

                auto input = TStripeBuffer<ui32>::Create(stripeMapping);
                auto output = TStripeBuffer<ui32>::CopyMapping(input);

                TVector<ui32> data(size);
                std::generate(data.begin(), data.end(), [&]() {
                    return rand.NextUniformL() % 10000;
                });
                input.Write(data);

                ScanVector(input, output);
                TVector<ui32> result;
                output.Read(result);

                for (ui32 dev : input.NonEmptyDevices()) {
                    ui32 prefixSum = 0;
                    const auto slice = stripeMapping.DeviceSlice(dev);

                    for (auto i = slice.Left; i < slice.Right; ++i) {
                        UNIT_ASSERT_EQUAL(result[i], prefixSum);
                        prefixSum += data[i];
                    }
                }

                ScanVector(input, output, true);
                output.Read(result);

                for (ui32 dev : input.NonEmptyDevices()) {
                    ui32 prefixSum = 0;
                    const auto slice = stripeMapping.DeviceSlice(dev);

                    for (auto i = slice.Left; i < slice.Right; ++i) {
                        prefixSum += data[i];
                        UNIT_ASSERT_EQUAL(result[i], prefixSum);
                    }
                }
            }
        }
    }

    template <class T>
    inline void TestScanPerformance() {
        {
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            ui64 tries = 10;
            TRandom rand(0);
            for (ui32 i = 10; i < 10000001; i *= 10) {
                ui64 size = i;
                TSingleMapping mapping = TSingleMapping(0, size);
                auto input = TSingleBuffer<T>::Create(mapping);
                TVector<T> data(size);
                std::generate(data.begin(), data.end(), [&]() {
                    return static_cast<T>(rand.NextUniformL() % 10000);
                });
                input.Write(data);
                auto output = TSingleBuffer<T>::CopyMapping(input);

                for (ui32 k = 0; k < tries; ++k) {
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Scan of #" << i << " elements");
                        ScanVector(input, output, true);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestScanPerformanceFloat) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestScanPerformance<float>();
        }
    }

    Y_UNIT_TEST(TestScanPerformanceInt) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestScanPerformance<int>();
        }
    }

    Y_UNIT_TEST(TestScanPerformanceUnsignedInt) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestScanPerformance<ui32>();
        }
    }
}
