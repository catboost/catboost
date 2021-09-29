#include "approx_updater_helpers.h"

#include "learn_context.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>

#include <util/generic/cast.h>


using namespace NCB;

template <bool StoreExpApprox>
static void UpdateLearnAvrgApprox(
    ui32 learnSampleCount,
    const TVector<TIndexType>& indices,
    const TVector<TVector<double>>& treeDelta,
    TLearnProgress* learnProgress,
    NPar::ILocalExecutor* localExecutor,
    TVector<TVector<double>>* trainFoldApprox
) {
    TConstArrayRef<ui32> learnPermutationRef(
        learnProgress->AveragingFold.GetLearnPermutationArray());
    TConstArrayRef<TIndexType> indicesRef(indices);

    TVector<TVector<double>> expTreeDelta(treeDelta);
    ExpApproxIf(StoreExpApprox, &expTreeDelta);

    auto& avrgFoldApprox = learnProgress->AveragingFold.BodyTailArr[0].Approx;
    for (ui32 dimIdx : xrange(treeDelta.size())) {
        TArrayRef<double> avrgFoldApproxRef(avrgFoldApprox[dimIdx]);
        TArrayRef<double> avrgApproxRef(learnProgress->AvrgApprox[dimIdx]);
        TArrayRef<double> trainFoldApproxRef;
        if (trainFoldApprox) {
            trainFoldApproxRef = (*trainFoldApprox)[dimIdx];
        }
        TConstArrayRef<double> expTreeDeltaRef(expTreeDelta[dimIdx]);
        TConstArrayRef<double> treeDeltaRef(treeDelta[dimIdx]);
        DispatchGenericLambda(
            [=] (auto isAveragingFoldPermuted, auto haveTrainFoldApprox) {
                NPar::ParallelFor(
                    *localExecutor,
                    0,
                    learnSampleCount,
                    [=] (ui32 idx) {
                        avrgFoldApproxRef[idx] = UpdateApprox<StoreExpApprox>(
                            avrgFoldApproxRef[idx],
                            expTreeDeltaRef[indicesRef[idx]]);
                        if constexpr (isAveragingFoldPermuted) {
                            avrgApproxRef[learnPermutationRef[idx]] += treeDeltaRef[indicesRef[idx]];
                            if constexpr (haveTrainFoldApprox) {
                                trainFoldApproxRef[learnPermutationRef[idx]] = avrgFoldApproxRef[idx];
                            }
                        } else {
                            avrgApproxRef[idx] += treeDeltaRef[indicesRef[idx]];
                            if constexpr (haveTrainFoldApprox) {
                                trainFoldApproxRef[idx] = avrgFoldApproxRef[idx];
                            }
                        }
                    });
            },
            learnProgress->IsAveragingFoldPermuted, trainFoldApproxRef.size() > 0);
    }
}

template <bool StoreExpApprox>
static void UpdateAvrgApprox(
    ui32 learnSampleCount,
    const TVector<TIndexType>& indices,
    const TVector<TVector<double>>& treeDelta,
    TConstArrayRef<TTrainingDataProviderPtr> testData, // can be empty
    TLearnProgress* learnProgress,
    NPar::ILocalExecutor* localExecutor,
    TVector<TVector<double>>* trainFoldApprox
) {
    Y_ASSERT(learnProgress->AveragingFold.BodyTailArr.ysize() == 1);
    const TVector<size_t>& testOffsets = CalcTestOffsets(learnSampleCount, testData);

    localExecutor->ExecRange(
        [&](int setIdx){
            if (setIdx == 0) { // learn data set
                if (learnSampleCount == 0) {
                    return;
                }
                Y_ASSERT(learnProgress->AvrgApprox[0].size() == learnSampleCount);
                Y_ASSERT(learnProgress->AveragingFold.BodyTailArr.ysize() == 1);

                UpdateLearnAvrgApprox<StoreExpApprox>(
                    learnSampleCount,
                    indices,
                    treeDelta,
                    learnProgress,
                    localExecutor,
                    trainFoldApprox);
            } else { // test data set
                const int testIdx = setIdx - 1;
                const size_t testSampleCount = testData[testIdx]->GetObjectCount();
                TConstArrayRef<TIndexType> indicesRef(indices.data() + testOffsets[testIdx], testSampleCount);
                const auto updateTestApprox = [=](
                    TConstArrayRef<double> delta,
                    TArrayRef<double> approx,
                    size_t idx
                ) {
                    approx[idx] += delta[indicesRef[idx]];
                };
                Y_ASSERT(learnProgress->TestApprox[testIdx][0].size() == testSampleCount);
                UpdateApprox(updateTestApprox, treeDelta, &learnProgress->TestApprox[testIdx], localExecutor);
            }
        },
        0,
        1 + SafeIntegerCast<int>(testData.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

void UpdateAvrgApprox(
    bool storeExpApprox,
    ui32 learnSampleCount,
    const TVector<TIndexType>& indices,
    const TVector<TVector<double>>& treeDelta,
    TConstArrayRef<TTrainingDataProviderPtr> testData, // can be empty
    TLearnProgress* learnProgress,
    NPar::ILocalExecutor* localExecutor,
    TVector<TVector<double>>* trainFoldApprox
) {
    DispatchGenericLambda(
        [&] (auto storeExpApprox) {
            ::UpdateAvrgApprox<storeExpApprox>(
                learnSampleCount,
                indices,
                treeDelta,
                testData,
                learnProgress,
                localExecutor,
                trainFoldApprox);
        },
        storeExpApprox);
}
