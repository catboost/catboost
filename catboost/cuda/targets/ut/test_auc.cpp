#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/targets/auc.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/metrics/auc.h>
#include <catboost/libs/metrics/sample.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <util/system/info.h>
#include <util/generic/ymath.h>

using namespace std;
using namespace NCudaLib;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(TAucTest) {
    void GenerateSamples(TRandom & random, double rate, ui64 size, TVector<float>* classes, TVector<float>* predictions, TVector<float>* weights) {
        classes->resize(size);
        predictions->resize(size);
        weights->resize(size);

        double positiveApproxMean = 1;
        double negativeApproxMean = -1;

        for (ui64 i = 0; i < size; ++i) {
            (*classes)[i] = random.NextUniformL() % 2;
            (*weights)[i] = (1.0 + random.NextUniformL() % 3) / 4;

            const double classifierClass = (random.NextUniform() < rate) ? (*classes)[i] : 1.0 - (*classes)[i];

            double mean = classifierClass ? positiveApproxMean : negativeApproxMean;
            (*predictions)[i] = random.NextGaussian() + mean;
        }
    }

    inline void TestAucImpl(ui64 seed, ui32 docCount, double rate = 0.7) {
        TRandom random(seed);

        auto stopCudaManagerGuard = StartCudaManager();
        {
            TVector<float> targets;
            TVector<float> cursor;
            TVector<float> weights;

            auto docsMapping = TStripeMapping::SplitBetweenDevices(docCount);

            GenerateSamples(random, rate, docCount, &targets, &cursor, &weights);
            TVector<NMetrics::TSample> samples;
            for (ui64 i = 0; i < targets.size(); ++i) {
                samples.push_back(NMetrics::TSample(targets[i], cursor[i], weights[i]));
            }
            auto targetsGpu = TStripeBuffer<float>::Create(docsMapping);
            auto weightsGpu = TStripeBuffer<float>::Create(docsMapping);
            auto cursorGpu = TStripeBuffer<float>::Create(docsMapping);

            targetsGpu.Write(targets);
            weightsGpu.Write(weights);
            cursorGpu.Write(cursor);

            double aucGpu = ComputeAUC(targetsGpu, weightsGpu, cursorGpu);
            double aucCpu = CalcAUC(&samples);

            UNIT_ASSERT_DOUBLES_EQUAL(aucGpu, aucCpu, 1e-5);
        }
    }

    Y_UNIT_TEST(TestAuc) {
        ui64 seed = 0;
        for (ui32 i = 0; i < 10; ++i) {
            for (double rate : {0.6, 0.7})
                for (ui32 docCount : {1000, 10000, 10000000}) {
                    TestAucImpl(++seed, docCount, rate);
                }
        }
    }
}
