#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/libs/helpers/cpu_random.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/random/shuffle.h>

#include <iostream>
#include <numeric>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TTransformTest) {
    Y_UNIT_TEST(TestMultiply) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                ui64 size = rand.NextUniformL() % 10000000;
                for (ui64 i = 0; i < size; ++i) {
                    vec.push_back(rand.NextUniform());
                }

                float scale = (float)rand.NextUniform();

                auto mapping = TStripeMapping::SplitBetweenDevices(size);
                auto cVec = TCudaBuffer<float, TStripeMapping>::Create(mapping);
                cVec.Write(vec);

                MultiplyVector(cVec, scale);
                TVector<float> result;
                cVec.Read(result);

                for (ui32 i = 0; i < vec.size(); ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(result[i], scale * vec[i], 1e-5);
                }
            }

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                TVector<float> vec2;
                ui64 size = rand.NextUniformL() % 10000000;
                for (ui64 i = 0; i < size; ++i) {
                    vec.push_back(rand.NextUniform());
                    vec2.push_back(rand.NextUniform());
                }
                auto mapping = TStripeMapping::SplitBetweenDevices(size);

                auto cVec = TStripeBuffer<float>::Create(mapping);
                auto cVec2 = TStripeBuffer<float>::CopyMapping(cVec);

                cVec.Write(vec);
                cVec2.Write(vec2);

                MultiplyVector(cVec, cVec2);
                TVector<float> result;
                cVec.Read(result);

                for (ui32 i = 0; i < vec.size(); ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(result[i], vec2[i] * vec[i], 1e-5);
                }
            }
        }
    }

    Y_UNIT_TEST(TestSubtract) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                TVector<float> vec2;
                ui64 size = rand.NextUniformL() % 10000000;
                for (ui64 i = 0; i < size; ++i) {
                    vec.push_back(rand.NextUniform());
                    vec2.push_back(rand.NextUniform());
                }
                auto mapping = TStripeMapping::SplitBetweenDevices(size);

                auto cVec = TStripeBuffer<float>::Create(mapping);
                auto cVec2 = TStripeBuffer<float>::CopyMapping(cVec);

                cVec.Write(vec);
                cVec2.Write(vec2);

                SubtractVector(cVec, cVec2);
                TVector<float> result;
                cVec.Read(result);

                for (ui32 i = 0; i < vec.size(); ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(result[i], vec[i] - vec2[i], 1e-5);
                }
            }
        }
    }

    Y_UNIT_TEST(TestAdd) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                TVector<float> vec2;
                ui64 size = rand.NextUniformL() % 10000000;
                for (ui64 i = 0; i < size; ++i) {
                    vec.push_back(rand.NextUniform());
                    vec2.push_back(rand.NextUniform());
                }
                auto mapping = TStripeMapping::SplitBetweenDevices(size);

                auto cVec = TStripeBuffer<float>::Create(mapping);
                auto cVec2 = TStripeBuffer<float>::CopyMapping(cVec);
                cVec.Write(vec);
                cVec2.Write(vec2);

                AddVector(cVec, cVec2);
                TVector<float> result;
                cVec.Read(result);

                for (ui32 i = 0; i < vec.size(); ++i) {
                    UNIT_ASSERT_DOUBLES_EQUAL(result[i], vec[i] + vec2[i], 1e-5);
                }
            }
        }
    }

    Y_UNIT_TEST(TestScatterAndGather) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10000;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> src;
                TVector<float> dstScatter;
                TVector<float> dstGather;
                TVector<ui32> index;
                ui64 size = rand.NextUniformL() % 100000;
                index.resize(size);
                dstGather.resize(size);
                dstScatter.resize(size);
                std::iota(index.begin(), index.end(), 0);
                Shuffle(index.begin(), index.end(), rand);

                for (ui64 i = 0; i < size; ++i) {
                    src.push_back((float)rand.NextUniform());
                    dstScatter[index[i]] = src.back();
                }
                for (ui64 i = 0; i < size; ++i) {
                    dstGather[i] = src[index[i]];
                }

                auto cudaSrc = TSingleBuffer<float>::Create(TSingleMapping(0, size));
                auto cudaDstScatter = TSingleBuffer<float>::CopyMapping(cudaSrc);
                auto cudaDstGather = TSingleBuffer<float>::CopyMapping(cudaSrc);
                auto cudaIndex = TSingleBuffer<ui32>::CopyMapping(cudaSrc);

                auto streamWrite = RequestStream();
                cudaSrc.Write(src, streamWrite.GetId());
                cudaIndex.Write(index, streamWrite.GetId());

                auto streamScatter = RequestStream();
                auto streamGather = RequestStream();

                Gather(cudaDstGather, cudaSrc, cudaIndex, streamGather.GetId());
                Scatter(cudaDstScatter, cudaSrc, cudaIndex, streamScatter.GetId());

                {
                    TVector<float> tmp;
                    cudaDstGather.Read(tmp, streamGather.GetId());
                    for (ui32 i = 0; i < size; ++i) {
                        UNIT_ASSERT_DOUBLES_EQUAL(tmp[i], dstGather[i], 1e-20f);
                    }
                }

                {
                    TVector<float> tmp;
                    cudaDstScatter.Read(tmp, streamScatter.GetId());
                    for (ui32 i = 0; i < size; ++i) {
                        UNIT_ASSERT_DOUBLES_EQUAL(tmp[i], dstScatter[i], 1e-20f);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestDotProductMirror) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 1;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                TVector<float> vec2;
                ui64 size = rand.NextUniformL() % 10000000;
                double dotProd = 0;
                for (ui64 i = 0; i < size; ++i) {
                    const double x = rand.NextUniform();
                    const double y = rand.NextUniform();
                    vec.push_back(x);
                    vec2.push_back(y);
                    dotProd += x * y;
                }

                TMirrorMapping mapping = TMirrorMapping(vec.size());
                auto cVec = TMirrorBuffer<float>::Create(mapping);
                auto cVec2 = TMirrorBuffer<float>::CopyMapping(cVec);

                cVec.Write(vec);
                cVec2.Write(vec2);

                const float result = DotProduct(cVec, cVec2);
                UNIT_ASSERT_DOUBLES_EQUAL(result, dotProd, 1e-4 * vec.size());
            }
        }
    }

    Y_UNIT_TEST(TestDotProductSingle) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 10;
            TRandom rand(0);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                TVector<float> vec2;
                ui64 size = rand.NextUniformL() % 10000000;
                double dotProd = 0;
                for (ui64 i = 0; i < size; ++i) {
                    const double x = rand.NextUniform();
                    const double y = rand.NextUniform();
                    vec.push_back(x);
                    vec2.push_back(y);
                    dotProd += x * y;
                }

                auto mapping = TSingleMapping(0, vec.size());
                auto cVec = TSingleBuffer<float>::Create(mapping);
                auto cVec2 = TSingleBuffer<float>::CopyMapping(cVec);

                cVec.Write(vec);
                cVec2.Write(vec2);

                const float result = DotProduct(cVec, cVec2);
                UNIT_ASSERT_DOUBLES_EQUAL(result, dotProd, 1e-4 * vec.size());
            }
        }
    }

    Y_UNIT_TEST(TestDotProductStripe) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 100;
            TRandom rand(0);

            NCudaLib::TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);

            for (ui32 k = 0; k < tries; ++k) {
                TVector<float> vec;
                TVector<float> vec2;
                ui64 size = (rand.NextUniformL() % 10000000) * GetCudaManager().GetDeviceCount();
                double dotProd = 0;

                for (ui64 i = 0; i < size; ++i) {
                    const double x = rand.NextUniform();
                    const double y = rand.NextUniform();
                    vec.push_back(x);
                    vec2.push_back(y);
                    dotProd += x * y;
                }

                auto mapping = TStripeMapping::SplitBetweenDevices(vec.size());
                auto cVec = TStripeBuffer<float>::Create(mapping);
                auto cVec2 = TStripeBuffer<float>::CopyMapping(cVec);

                cVec.Write(vec);
                cVec2.Write(vec2);

                auto guard = profiler.Profile("DotProduct");
                const float result = DotProduct(cVec, cVec2);

                UNIT_ASSERT_DOUBLES_EQUAL(result, dotProd, 1e-4 * vec.size());
            }
        }
    }

    Y_UNIT_TEST(TestPowVector) {
        const auto stopCudaManagerGuard = StartCudaManager();
        const ui64 seed = 0;
        const size_t size = 1000000;
        const float base = 2.f;

        TRandom rand(seed);
        TVector<float> exponents;
        exponents.yresize(size);
        for (auto& exponent : exponents) {
            exponent = rand.NextUniform();
        }

        TVector<float> cpuPow;
        cpuPow.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            cpuPow[i] = pow(base, exponents[i]);
        }

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);
        auto tmp = TStripeBuffer<float>::Create(mapping);
        tmp.Write(exponents);

        PowVector(tmp, base);

        TVector<float> gpuPow;
        tmp.Read(gpuPow);

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL(cpuPow[i], gpuPow[i], 1e-5);
        }
    }

    Y_UNIT_TEST(TestPowVector2) {
        const auto stopCudaManagerGuard = StartCudaManager();
        const ui64 seed = 0;
        const size_t size = 1000000;
        const float base = 2.f;

        TRandom rand(seed);
        TVector<float> exponents;
        exponents.yresize(size);
        for (auto& exponent : exponents) {
            exponent = rand.NextUniform();
        }

        TVector<float> cpuPow;
        cpuPow.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            cpuPow[i] = pow(base, exponents[i]);
        }

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);
        auto deviceExponent = TStripeBuffer<float>::Create(mapping);
        auto devicePow = TStripeBuffer<float>::Create(mapping);

        deviceExponent.Write(exponents);

        PowVector(deviceExponent, base, devicePow);

        TVector<float> gpuPow;
        devicePow.Read(gpuPow);

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL(cpuPow[i], gpuPow[i], 1e-5);
        }
    }
}
