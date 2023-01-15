#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/methods/langevin_utils.h>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>

#include <util/system/info.h>
#include <numeric>

Y_UNIT_TEST_SUITE(TAddingLangevinNoiseTest) {
    using TVec = TCudaBuffer<float, NCudaLib::TStripeMapping>;

    void FillDerivatives(TVector<float>& data, TRandom& random) {
        std::generate(data.begin(), data.end(), [&]() {
            return random.NextUniform();
        });
    }

    TVec MakeCudaBuffer(const TVector<float>& source) {
        auto dataMapping = NCudaLib::TStripeMapping::SplitBetweenDevices(source.size());
        auto buffer = TStripeBuffer<float>::Create(dataMapping);
        buffer.Write(source);

        return buffer;
    }

    void AddLangevinNoiseOnGPU(TVec* data,
                               float diffusionTemperature,
                               float learningRate,
                               TGpuAwareRandom& random) {
        auto& seeds = random.GetGpuSeeds<NCudaLib::TStripeMapping>();
        AddLangevinNoise(seeds,
                         data,
                         diffusionTemperature,
                         learningRate);
    }

    double ComputeVectorStd(const TVector<float>& data) {
        double mean = Accumulate(data.begin(), data.end(), 0) / data.size();
        double variance = 0;

        std::for_each(data.begin(), data.end(), [&] (float n) {
            variance += (n - mean) * (n - mean);
        });

        return sqrt(variance / data.size());
    }

    double ComputeCpuStd(const TVector<float>& derivatives, float diffusionTemperature, float learningRate, ui32 seed) {
        TVector<double> cpuDerivatives(derivatives.size());
        for (ui32 i = 0; i < derivatives.size(); ++i) {
            cpuDerivatives[i] = derivatives[i];
        }

        NPar::TLocalExecutor executor;
        AddLangevinNoiseToDerivatives(diffusionTemperature, learningRate, seed, &cpuDerivatives, &executor);

        TVector<float> diff(derivatives.size());
        for (ui32 i = 0; i < cpuDerivatives.size(); ++i) {
            diff[i] = cpuDerivatives[i] - derivatives[i];
        }

        return ComputeVectorStd(diff);
    }

    void TestAddingLangevinNoise(ui32 objectsCount,
                                 ui32 seed,
                                 float diffusionTemperature,
                                 float learningRate) {
        TGpuAwareRandom random(seed);

        {
            TVector<float> derivatives(objectsCount);
            FillDerivatives(derivatives, random);
            auto derivativesBuffer = MakeCudaBuffer(derivatives);

            //
            AddLangevinNoiseOnGPU(&derivativesBuffer, diffusionTemperature, learningRate, random);

            //
            TVector<float> gpuDerivatives;
            derivativesBuffer.Read(gpuDerivatives);

            //
            UNIT_ASSERT_EQUAL(derivatives.size(), gpuDerivatives.size());
            constexpr double PRECISION = 1e-6;
            TVector<float> diff(objectsCount);
            for (ui32 i = 0; i < gpuDerivatives.size(); ++i) {
                UNIT_ASSERT(std::fabs(derivatives[i] - gpuDerivatives[i]) > PRECISION);
                diff[i] = gpuDerivatives[i] - derivatives[i];
            }

            //
            double diffStdOnGpu = ComputeVectorStd(diff);
            double diffStdOnCpu = ComputeCpuStd(derivatives, diffusionTemperature, learningRate, seed);

            constexpr double STD_PRECISION = 1e-1;
            UNIT_ASSERT_DOUBLES_EQUAL(diffStdOnGpu, diffStdOnCpu, STD_PRECISION);
        }
    }

    Y_UNIT_TEST(LangevinTest) {
        auto stopCudaManagerGuard = StartCudaManager();
        for (ui32 objectsCount : {128, 1024, 4096, 10000}) {
            for (ui64 seed : {42, 100}) {
                for (float diffusionTemperature : {10000, 50000}) {
                    for (float learningRate: {0.001, 0.1}) {
                        TestAddingLangevinNoise(objectsCount, seed, diffusionTemperature, learningRate);
                    }
                }
            }
        }
    }
}
