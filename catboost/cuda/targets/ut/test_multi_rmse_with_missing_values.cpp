#include <library/cpp/testing/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/targets/multiclass_kernels.h>
#include <catboost/cuda/ut_helpers/test_utils.h>

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/metrics/metric.h>

#include <catboost/private/libs/options/loss_description.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>

#include <limits>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TMultiRMSEWithMissingValuesTests) {
    static const float MISSING = std::numeric_limits<float>::quiet_NaN();

    template <class T, class TMapping>
    static inline void AssertDoubleEqual(const TVector<float>& ref, const TCudaBuffer<T, TMapping>& gpu, double eps,
                                         TString messagePrefix) {
        TVector<T> gpuValues;
        gpu.Read(gpuValues);

        UNIT_ASSERT_VALUES_EQUAL_C(ref.size(), gpuValues.size(), messagePrefix);
        for (ui32 i = 0; i < gpuValues.size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                static_cast<double>(ref[i]),
                static_cast<double>(gpuValues[i]),
                eps,
                TStringBuilder() << messagePrefix << " " << i << " " << gpuValues[i] << " " << ref[i]);
        }
    }

    //target, approx and derivatives are stored dimension by dimension, doc-th document of dim-th
    //dimension is at dim * docCount + doc
    static inline ui32 Idx(ui32 doc, ui32 dim, ui32 docCount) {
        return dim * docCount + doc;
    }

    void TestMultiRMSEWithMissingValuesImpl(ui64 seed, ui32 docCount, ui32 targetCount, double missingRate) {
        TRandom random(seed);

        auto stopCudaManagerGuard = StartCudaManager();
        {
            TVector<float> targets(docCount * targetCount);
            TVector<float> cursor(docCount * targetCount);
            TVector<float> weights(docCount);

            auto docsMapping = TStripeMapping::SplitBetweenDevices(docCount);

            double totalWeight = 0;
            for (ui32 doc = 0; doc < docCount; ++doc) {
                weights[doc] = 1.0f / (1 << (random.NextUniformL() % 3));
                totalWeight += weights[doc];
            }
            for (ui32 dim = 0; dim < targetCount; ++dim) {
                for (ui32 doc = 0; doc < docCount; ++doc) {
                    const bool isMissing = random.NextUniform() < missingRate;
                    targets[Idx(doc, dim, docCount)] = isMissing ? MISSING : (float)random.NextGaussian();
                    cursor[Idx(doc, dim, docCount)] = (float)random.NextGaussian();
                }
            }

            double funcValueRef = 0;
            TVector<float> derRef(docCount * targetCount);
            TVector<double> sumErrorsRef(targetCount);
            TVector<double> sumWeightsRef(targetCount);

            for (ui32 dim = 0; dim < targetCount; ++dim) {
                for (ui32 doc = 0; doc < docCount; ++doc) {
                    const float target = targets[Idx(doc, dim, docCount)];
                    if (std::isnan(target)) {
                        continue;
                    }
                    const float weight = weights[doc];
                    const float diff = target - cursor[Idx(doc, dim, docCount)];
                    funcValueRef -= weight * diff * diff;
                    derRef[Idx(doc, dim, docCount)] = diff * weight;
                    sumErrorsRef[dim] += weight * diff * diff;
                    sumWeightsRef[dim] += weight;
                }
            }

            auto targetsGpu = TStripeBuffer<float>::Create(docsMapping, targetCount);
            auto approxGpu = TStripeBuffer<float>::Create(docsMapping, targetCount);
            auto weightsGpu = TStripeBuffer<float>::Create(docsMapping);
            targetsGpu.Write(targets);
            approxGpu.Write(cursor);
            weightsGpu.Write(weights);

            auto funcValue = TStripeBuffer<float>::Create(TStripeMapping::RepeatOnAllDevices(1));
            auto der = TStripeBuffer<float>::CopyMappingAndColumnCount(approxGpu);

            MultiRMSEWithMissingValuesValueAndDer<TStripeMapping>(
                targetsGpu.ConstCopyView(),
                weightsGpu.ConstCopyView(),
                approxGpu.ConstCopyView(),
                (const TStripeBuffer<ui32>*)nullptr,
                &funcValue,
                &der);

            double value = 0;
            for (auto val : ReadReduce(funcValue)) {
                value += val;
            }
            UNIT_ASSERT_DOUBLES_EQUAL_C(value / totalWeight, funcValueRef / totalWeight, 1e-4,
                                        TStringBuilder() << value << " " << funcValueRef);

            const double eps = 1e-4;
            AssertDoubleEqual(derRef, der, eps, "der");

            //the leaves estimator passes a buffer with row + 1 columns, so nothing above the
            //diagonal element may be written
            for (ui32 row = 0; row < targetCount; ++row) {
                const float notWritten = -1.0f;
                auto der2 = TStripeBuffer<float>::CopyMappingAndColumnCount(approxGpu);
                FillBuffer(der2, notWritten);

                MultiRMSEWithMissingValuesSecondDerRow<TStripeMapping>(
                    targetsGpu.ConstCopyView(),
                    weightsGpu.ConstCopyView(),
                    row,
                    &der2);

                TVector<float> der2Ref(docCount * targetCount, notWritten);
                for (ui32 dim = 0; dim < targetCount; ++dim) {
                    for (ui32 doc = 0; doc < docCount; ++doc) {
                        if (dim < row) {
                            der2Ref[Idx(doc, dim, docCount)] = 0.0f;
                        } else if (dim == row) {
                            const float target = targets[Idx(doc, dim, docCount)];
                            der2Ref[Idx(doc, dim, docCount)] = std::isnan(target) ? 0.0f : weights[doc];
                        }
                    }
                }
                AssertDoubleEqual(der2Ref, der2, eps, TStringBuilder() << "der2_" << row);
            }

            //the metric hands its stats to the CPU metric to compute the final value, so both the
            //layout and the values have to match
            auto statsGpu = TStripeBuffer<float>::Create(TStripeMapping::RepeatOnAllDevices(2 * targetCount));
            MultiRMSEWithMissingValuesStats<TStripeMapping>(
                targetsGpu.ConstCopyView(),
                weightsGpu.ConstCopyView(),
                approxGpu.ConstCopyView(),
                &statsGpu);
            const TVector<float> stats = ReadReduce(statsGpu);

            UNIT_ASSERT_VALUES_EQUAL(stats.size(), 2 * targetCount);
            for (ui32 dim = 0; dim < targetCount; ++dim) {
                UNIT_ASSERT_DOUBLES_EQUAL_C(sumErrorsRef[dim], stats[2 * dim],
                                            eps * Max(1.0, sumErrorsRef[dim]),
                                            TStringBuilder() << "sum errors " << dim);
                UNIT_ASSERT_DOUBLES_EQUAL_C(sumWeightsRef[dim], stats[2 * dim + 1],
                                            eps * Max(1.0, sumWeightsRef[dim]),
                                            TStringBuilder() << "sum weights " << dim);
            }

            const auto metric = std::move(
                CreateMetric(ELossFunction::MultiRMSEWithMissingValues, TLossParams(), targetCount)[0]);

            TVector<TVector<double>> cpuApproxStorage(targetCount, TVector<double>(docCount));
            TVector<TVector<float>> cpuTargetStorage(targetCount, TVector<float>(docCount));
            TVector<TConstArrayRef<double>> cpuApprox(targetCount);
            TVector<TConstArrayRef<float>> cpuTarget(targetCount);
            for (ui32 dim = 0; dim < targetCount; ++dim) {
                for (ui32 doc = 0; doc < docCount; ++doc) {
                    cpuApproxStorage[dim][doc] = cursor[Idx(doc, dim, docCount)];
                    cpuTargetStorage[dim][doc] = targets[Idx(doc, dim, docCount)];
                }
                cpuApprox[dim] = cpuApproxStorage[dim];
                cpuTarget[dim] = cpuTargetStorage[dim];
            }

            NPar::TLocalExecutor executor;
            const TMetricHolder cpuStats = dynamic_cast<const IMultiTargetEval&>(*metric).Eval(
                cpuApprox, {}, cpuTarget, weights, 0, docCount, executor);

            UNIT_ASSERT_VALUES_EQUAL(cpuStats.Stats.size(), stats.size());

            TMetricHolder gpuStats(stats.size());
            for (ui32 i = 0; i < stats.size(); ++i) {
                gpuStats.Stats[i] = stats[i];
            }
            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(cpuStats), metric->GetFinalError(gpuStats), eps);
        }
    }

    Y_UNIT_TEST(TestMultiRMSEWithMissingValues) {
        for (ui32 targetCount : {2, 5}) {
            for (ui32 docCount : {100, 1000, 134532}) {
                for (double missingRate : {0.0, 0.3, 0.9}) {
                    TestMultiRMSEWithMissingValuesImpl(10 * targetCount + docCount, docCount, targetCount,
                                                       missingRate);
                }
            }
        }
    }
}
