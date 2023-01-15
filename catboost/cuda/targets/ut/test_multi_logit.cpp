#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/cuda/targets/multiclass_kernels.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <util/system/info.h>
#include <util/generic/ymath.h>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TMultiLogitTests) {
    template <class T1, class T2, class TMapping>
    static inline void AssertDoubleEqual(const TVector<T1>& ref, const TCudaBuffer<T2, TMapping>& gpu, double eps, TString messagePrefix) {
        TVector<T2> tmp;
        gpu.Read(tmp);

        UNIT_ASSERT_EQUAL_C(ref.size(), tmp.size(), TStringBuilder() << messagePrefix << " " << ref.size() << " " << tmp.size());
        for (ui32 i = 0; i < tmp.size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(static_cast<double>(ref[i]), static_cast<double>(tmp[i]), eps, TStringBuilder() << messagePrefix << " " << i << " " << tmp[i] << " " << ref[i]);
        }
    }

    void TestMultiLogitImpl(ui64 seed, ui32 docCount, ui32 numClasses, double minApprox = -50, double maxApprox = 50) {
        TRandom random(seed);

        auto stopCudaManagerGuard = StartCudaManager();
        {
            TVector<float> targets;
            TVector<float> cursor;
            TVector<float> weights;

            auto docsMapping = TStripeMapping::SplitBetweenDevices(docCount);

            double funcValueRef = 0;
            double totalWeight = 0;
            TVector<float> derRef;
            derRef.resize(docCount * (numClasses - 1));

            TVector<TVector<float>> der2Ref(numClasses - 1);

            for (ui32 doc = 0; doc < docCount; ++doc) {
                targets.push_back(random.NextUniformL() % numClasses);
                weights.push_back(1.0f / (1 << (random.NextUniformL() % 3)));
                //                weights.push_back(1.0);
                totalWeight += weights.back();
                for (ui32 i = 0; i < (numClasses - 1); ++i) {
                    cursor.push_back(random.NextUniform() * (maxApprox - minApprox) + minApprox);
                }
            }

            for (ui32 i = 0; i < numClasses - 1; ++i) {
                der2Ref[i].resize(docCount * (numClasses - 1));
            }

            for (ui32 doc = 0; doc < docCount; ++doc) {
                const float weight = weights[doc];
                TVector<float> softmaxes(numClasses);

                float maxValue = 0;
                for (ui32 i = 0; i < numClasses - 1; ++i) {
                    softmaxes[i] = cursor[doc + docCount * i];
                    maxValue = Max<float>(maxValue, softmaxes[i]);
                }
                float denum = 0;

                const ui32 clazz = static_cast<ui32>(targets[doc]);

                funcValueRef += weight * (softmaxes[clazz] - maxValue);

                for (ui32 i = 0; i < numClasses; ++i) {
                    softmaxes[i] = exp(softmaxes[i] - maxValue);
                    denum += softmaxes[i];
                }

                funcValueRef -= weight * log(denum);

                for (ui32 i = 0; i < (numClasses - 1); ++i) {
                    const float pi = softmaxes[i] / denum;
                    derRef[doc + i * docCount] = -weight * pi;

                    if (i == clazz) {
                        derRef[doc + i * docCount] += weight;
                    }

                    for (ui32 j = 0; j < (numClasses - 1); ++j) {
                        const float pj = softmaxes[j] / denum;
                        if (j < i) {
                            der2Ref[i][doc + j * docCount] = -weight * pi * pj;
                        }
                        if (i == j) {
                            der2Ref[i][doc + j * docCount] = weight * pi * (1.0f - pi);
                        }
                    }
                }
            }

            auto targetsGpu = TStripeBuffer<float>::Create(docsMapping);
            auto approxGpu = TStripeBuffer<float>::Create(docsMapping, numClasses - 1);
            auto weightsGpu = TStripeBuffer<float>::Create(docsMapping);
            targetsGpu.Write(targets);
            weightsGpu.Write(weights);
            approxGpu.Write(cursor);

            auto funcValue = TStripeBuffer<float>::Create(TStripeMapping::RepeatOnAllDevices(1));
            auto der = TStripeBuffer<float>::CopyMappingAndColumnCount(approxGpu);

            MultiLogitValueAndDer<TStripeMapping>(targetsGpu, weightsGpu, approxGpu, nullptr, numClasses, &funcValue, &der);
            double value = 0;
            {
                TVector<float> funcValCpu;
                funcValue.Read(funcValCpu);
                for (auto val : funcValCpu) {
                    value += val;
                }
            }
            UNIT_ASSERT_DOUBLES_EQUAL_C(value / totalWeight, funcValueRef / totalWeight, 1e-4, TStringBuilder() << value << " " << funcValueRef);

            const double eps = 1e-4;
            AssertDoubleEqual(derRef, der, eps, "der");

            for (ui32 row = 0; row < numClasses - 1; ++row) {
                FillBuffer(der, 0.0f);
                MultiLogitSecondDerRow(targetsGpu, weightsGpu, approxGpu, numClasses, row, &der);
                AssertDoubleEqual(der2Ref[row], der, eps, TStringBuilder() << "der2_" << row);
            }
        }
    }

    Y_UNIT_TEST(TestMultilogit) {
        for (ui32 numClasses : {2, 5, 17}) {
            for (ui32 docCount : {100, 1000, 134532}) {
                TestMultiLogitImpl(10 * numClasses + docCount, docCount, numClasses);
            }
        }
    }
}
