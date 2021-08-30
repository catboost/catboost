#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/cuda/targets/query_cross_entropy_kernels.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <util/system/info.h>
#include <util/generic/ymath.h>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TQueryCrossEntropyTests) {
    template <class T, class TMapping>
    static inline void AssertEqual(const TVector<T>& ref, const TCudaBuffer<T, TMapping>& gpu) {
        TVector<T> tmp;
        gpu.Read(tmp);

        UNIT_ASSERT_EQUAL(ref.size(), tmp.size());
        for (ui32 i = 0; i < tmp.size(); ++i) {
            UNIT_ASSERT_EQUAL_C(ref[i], tmp[i], TStringBuilder() << i << " " << tmp[i] << " " << ref[i]);
        }
    }

    template <class T1, class T2, class TMapping>
    static inline void AssertDoubleEqual(const TVector<T1>& ref, const TCudaBuffer<T2, TMapping>& gpu, double eps, TString messagePrefix) {
        TVector<T2> tmp;
        gpu.Read(tmp);

        UNIT_ASSERT_EQUAL_C(ref.size(), tmp.size(), TStringBuilder() << messagePrefix << " " << ref.size() << " " << tmp.size());
        for (ui32 i = 0; i < tmp.size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(static_cast<double>(ref[i]), static_cast<double>(tmp[i]), eps, TStringBuilder() << messagePrefix << " " << i << " " << tmp[i] << " " << ref[i]);
        }
    }

    void TestMakeFlagsImpl(ui64 seed) {
        TRandom random(seed);

        auto stopCudaManagerGuard = StartCudaManager();
        {
            TVector<float> targets;
            TVector<ui32> queryOffsets;

            const ui32 queryCount = static_cast<const ui32>(random.NextUniformL() % 100000);
            NCudaLib::TStripeMapping queriesMapping = TStripeMapping::SplitBetweenDevices(queryCount);
            NCudaLib::TMappingBuilder<TStripeMapping> docMapping;

            TVector<bool> flags;
            TVector<ui32> trueClassCount;
            ui32 dev = 0;
            ui32 devOffset = 0;
            for (ui32 qid = 0; qid < queryCount; ++qid) {
                queryOffsets.push_back(devOffset);
                ui32 querySize = 1 + (random.NextUniformL() % 16);
                const bool isSingleClassQuery = ((random.NextUniformL()) % 3 == 0) || querySize == 1; //;

                ui32 trueCount = 0;
                for (ui32 i = 0; i < querySize; ++i) {
                    double target = isSingleClassQuery ? 0 : random.NextUniform();
                    targets.push_back(target);
                    flags.push_back(isSingleClassQuery);
                    trueCount += (target > 0.5);
                }
                for (ui32 i = 0; i < querySize; ++i) {
                    trueClassCount.push_back(trueCount);
                }
                devOffset += querySize;

                if (queriesMapping.DeviceSlice(dev).Right == (qid + 1)) {
                    queryOffsets.push_back(devOffset);
                    docMapping.SetSizeAt(dev, devOffset);
                    ++dev;
                    devOffset = 0;
                }
            }

            auto gpuQueryOffset = TStripeBuffer<ui32>::Create(queriesMapping.Transform([&](const TSlice slice) -> ui64 {
                return slice.Size() + 1;
            }));
            gpuQueryOffset.Write(queryOffsets);

            auto targetsGpu = TStripeBuffer<float>::Create(docMapping.Build());
            auto gpuFlags = TStripeBuffer<bool>::CopyMapping(targetsGpu);
            auto gpuTrueClassCount = TStripeBuffer<ui32>::CopyMapping(targetsGpu);
            UNIT_ASSERT_EQUAL_C(targetsGpu.GetObjectsSlice().Size(), targets.size(), TStringBuilder() << targetsGpu.GetObjectsSlice().Size() << " " << targets.size());
            targetsGpu.Write(targets);

            TStripeBuffer<ui32> loadIndices = TStripeBuffer<ui32>::CopyMapping(targetsGpu);
            MakeSequence(loadIndices);

            //
#define CHECK(k)                                                                                                                                            \
    FillBuffer(gpuFlags, false);                                                                                                                            \
    FillBuffer(gpuTrueClassCount, 0);                                                                                                                       \
    MakeIsSingleClassQueryFlags(targetsGpu.ConstCopyView(), loadIndices.ConstCopyView(), gpuQueryOffset.ConstCopyView(), k, &gpuFlags, &gpuTrueClassCount); \
    AssertEqual(flags, gpuFlags);

            CHECK(2)
            CHECK(4)
            CHECK(8)
            CHECK(16)
        }
    }

    static inline double BestQueryShift(const float* cursor, const float* targets, const float* weights, ui32 size) {
        double bestShift = 0;
        double left = -20;
        double right = 20;

        for (int i = 0; i < 50; ++i) {
            double der = 0;
            for (ui32 doc = 0; doc < size; ++doc) {
                const double expApprox = exp(cursor[doc] + bestShift);
                const double p = (std::isfinite(expApprox) ? (expApprox / (1.0 + expApprox)) : 1.0);
                der += weights[doc] * (targets[doc] - p);
            }

            if (der > 0) {
                left = bestShift;
            } else {
                right = bestShift;
            }

            bestShift = (left + right) / 2;
        }
        return bestShift;
    }

    void TestQueryLogitGradientImpl(ui64 seed, double alpha) {
        TRandom random(seed);
        const ui32 queryCount = 1000000 + static_cast<const ui32>(random.NextUniformL() % 10000);

        auto stopCudaManagerGuard = StartCudaManager();
        {
            TVector<float> targets;
            TVector<float> cursor;
            TVector<float> weights;
            TVector<bool> flags;
            TVector<ui32> qids;

            TVector<ui32> queryOffsets;

            NCudaLib::TStripeMapping queriesMapping = TStripeMapping::SplitBetweenDevices(queryCount);
            NCudaLib::TMappingBuilder<TStripeMapping> docMapping;

            ui32 dev = 0;

            ui32 devOffset = 0;
            ui32 localQid = 0;

            double funcValueRef = 0;
            double totalWeight = 0;

            TVector<float> derRef;
            TVector<float> der2llpRef;
            TVector<float> der2llmaxRef;
            TVector<float> groupDer2Ref;

            for (ui32 qid = 0; qid < queryCount; ++qid) {
                queryOffsets.push_back(devOffset);
                ui32 querySize = 1 + (random.NextUniformL() % 5);
                ui32 cpuQueryOffset = targets.size();

                const double queryProb = random.NextUniform();
                TVector<double> queryTargets;

                bool isSingleClassQuery = true;
                for (ui32 i = 0; i < querySize; ++i) {
                    queryTargets.push_back((random.NextUniform() > queryProb ? 1.0f : 0.0f));
                }

                for (ui32 i = 0; i < querySize; ++i) {
                    if (Abs(queryTargets[i] - queryTargets[0]) > 1e-5) {
                        isSingleClassQuery = false;
                    }
                }

                for (ui32 i = 0; i < querySize; ++i) {
                    const double target = queryTargets[i];
                    targets.push_back(target);
                    weights.push_back(random.NextUniform());
                    cursor.push_back(2 * random.NextUniform());
                    flags.push_back(isSingleClassQuery);
                    qids.push_back(localQid);
                }

                devOffset += querySize;
                ++localQid;

                {
                    double bestShift = BestQueryShift(cursor.data() + cpuQueryOffset,
                                                      targets.data() + cpuQueryOffset,
                                                      weights.data() + cpuQueryOffset,
                                                      querySize);

                    double groupDer2 = 0;
                    for (ui32 i = 0; i < querySize; ++i) {
                        const double approx = cursor[cpuQueryOffset + i];
                        const double target = targets[cpuQueryOffset + i];
                        const double w = weights[cpuQueryOffset + i];
                        const double expApprox = exp(approx);
                        const double shiftedExpApprox = exp(approx + bestShift);

                        const double prob = std::isfinite(expApprox) ? expApprox / (1.0 + expApprox) : 1.0;
                        const double shiftedProb = std::isfinite(shiftedExpApprox) ? shiftedExpApprox / (1.0 + shiftedExpApprox) : 1.0;

                        {
                            const double logExpValPlusOne = std::isfinite(expApprox) ? log(1 + expApprox) : approx;
                            const double llp = w * (target * approx - logExpValPlusOne);
                            funcValueRef += (1.0 - alpha) * llp;
                        }

                        if (!isSingleClassQuery) {
                            const double shiftedApprox = approx + bestShift;
                            const double logExpValPlusOne = std::isfinite(shiftedExpApprox) ? log(1 + shiftedExpApprox) : shiftedApprox;
                            const double llmax = w * (target * shiftedApprox - logExpValPlusOne);
                            funcValueRef += alpha * llmax;
                        }
                        totalWeight += w;

                        const double derllp = w * (1.0 - alpha) * (target - prob);
                        const double derllmax = isSingleClassQuery ? 0 : w * alpha * (target - shiftedProb);
                        derRef.push_back((derllp + derllmax));

                        const double der2llp = w * (1.0 - alpha) * prob * (1.0 - prob);
                        const double der2llmax = isSingleClassQuery ? 0 : w * alpha * shiftedProb * (1.0 - shiftedProb);
                        der2llpRef.push_back(der2llp);
                        der2llmaxRef.push_back(der2llmax);
                        groupDer2 += der2llmax;
                    }
                    groupDer2Ref.push_back(groupDer2);
                }

                if (queriesMapping.DeviceSlice(dev).Right == (qid + 1)) {
                    docMapping.SetSizeAt(dev, devOffset);
                    queryOffsets.push_back(devOffset);
                    ++dev;
                    devOffset = 0;
                    localQid = 0;
                }
            }

            auto gpuQueryOffsets = TStripeBuffer<ui32>::Create(queriesMapping.Transform([&](const TSlice slice) -> ui64 {
                return slice.Size() + 1;
            }));
            gpuQueryOffsets.Write(queryOffsets);

            auto gpuTargets = TStripeBuffer<float>::Create(docMapping.Build());
            auto gpuWeights = TStripeBuffer<float>::CopyMapping(gpuTargets);
            auto gpuCursor = TStripeBuffer<float>::CopyMapping(gpuTargets);
            auto gpuQids = TStripeBuffer<ui32>::CopyMapping(gpuTargets);
            auto gpuFlags = TStripeBuffer<bool>::CopyMapping(gpuTargets);
            ui32 approxScaleSize = 0;
            auto approxScale = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(1));
            auto trueClassCount = TStripeBuffer<ui32>::CopyMapping(gpuTargets);

            gpuTargets.Write(targets);
            gpuWeights.Write(weights);
            gpuCursor.Write(cursor);
            gpuQids.Write(qids);
            gpuFlags.Write(flags);
            approxScale.Write(cursor);
            trueClassCount.Write(TVector<ui32>(cursor.size(), 0));

            auto funcValueGpu = TStripeBuffer<float>::Create(TStripeMapping::RepeatOnAllDevices(1));
            auto der = TStripeBuffer<float>::CopyMapping(gpuTargets);
            auto der2llp = TStripeBuffer<float>::CopyMapping(gpuTargets);
            auto der2llmax = TStripeBuffer<float>::CopyMapping(gpuTargets);
            auto groupDer2 = TStripeBuffer<float>::Create(queriesMapping);

            QueryCrossEntropy<TStripeMapping>(alpha,
                                              /*defaultScale*/ 1.0f,
                                              approxScaleSize,
                                              gpuTargets.AsConstBuf(),
                                              gpuWeights.AsConstBuf(),
                                              gpuCursor.AsConstBuf(),
                                              gpuQids,
                                              gpuFlags,
                                              gpuQueryOffsets,
                                              approxScale.AsConstBuf(),
                                              trueClassCount,
                                              &funcValueGpu,
                                              &der,
                                              &der2llp,
                                              &der2llmax,
                                              &groupDer2);

            const double eps = 1e-5;
            AssertDoubleEqual(derRef, der, eps, "der");
            AssertDoubleEqual(der2llpRef, der2llp, eps, "der2llp");
            AssertDoubleEqual(der2llmaxRef, der2llmax, eps, "der2llmax");
            AssertDoubleEqual(groupDer2Ref, groupDer2, eps, "groupDer2");

            double value = 0;
            {
                TVector<float> funcValCpu;
                funcValueGpu.Read(funcValCpu);
                for (auto val : funcValCpu) {
                    value += val;
                }
            }
            UNIT_ASSERT_DOUBLES_EQUAL_C(value / totalWeight, funcValueRef / totalWeight, 1e-5, TStringBuilder() << value << " " << funcValueRef);
        }
    }

    Y_UNIT_TEST(TestMakeFlags) {
        TestMakeFlagsImpl(0);
    }

    Y_UNIT_TEST(TestLLp) {
        TestQueryLogitGradientImpl(0, 0.0);
    }

    Y_UNIT_TEST(TestLLmax) {
        TestQueryLogitGradientImpl(1, 1.0);
    }

    Y_UNIT_TEST(TestMixture) {
        TestQueryLogitGradientImpl(2, 0.9);
    }
}
