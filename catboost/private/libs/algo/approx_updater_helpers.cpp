#include "approx_updater_helpers.h"

#include "learn_context.h"

#include <util/generic/cast.h>


using namespace NCB;


template <bool StoreExpApprox>
static void UpdateAvrgApprox(
    ui32 learnSampleCount,
    const TVector<TIndexType>& indices,
    const TVector<TVector<double>>& treeDelta,
    TConstArrayRef<TTrainingDataProviderPtr> testData, // can be empty
    TLearnProgress* learnProgress,
    NPar::TLocalExecutor* localExecutor
) {
    Y_ASSERT(learnProgress->AveragingFold.BodyTailArr.ysize() == 1);
    const TVector<size_t>& testOffsets = CalcTestOffsets(learnSampleCount, testData);

    localExecutor->ExecRange(
        [&](int setIdx){
            if (setIdx == 0) { // learn data set
                if (learnSampleCount == 0) {
                    return;
                }
                TConstArrayRef<TIndexType> indicesRef(indices);
                const auto updateApprox = [=](
                    TConstArrayRef<double> delta,
                    TArrayRef<double> approx,
                    size_t idx
                ) {
                    approx[idx] = UpdateApprox<StoreExpApprox>(approx[idx], delta[indicesRef[idx]]);
                };
                TVector<TVector<double>> expTreeDelta(treeDelta);
                ExpApproxIf(StoreExpApprox, &expTreeDelta);
                TFold::TBodyTail& bt = learnProgress->AveragingFold.BodyTailArr[0];
                Y_ASSERT(bt.Approx[0].ysize() == bt.TailFinish);
                UpdateApprox(updateApprox, expTreeDelta, &bt.Approx, localExecutor);

                TConstArrayRef<ui32> learnPermutationRef(
                    learnProgress->AveragingFold.GetLearnPermutationArray());
                const auto updateAvrgApprox = [=](
                    TConstArrayRef<double> delta,
                    TArrayRef<double> approx,
                    size_t idx
                ) {
                    approx[learnPermutationRef[idx]] += delta[indicesRef[idx]];
                };
                Y_ASSERT(learnProgress->AvrgApprox[0].size() == learnSampleCount);
                UpdateApprox(updateAvrgApprox, treeDelta, &learnProgress->AvrgApprox, localExecutor);
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
    NPar::TLocalExecutor* localExecutor
) {
    if (storeExpApprox) {
        ::UpdateAvrgApprox<true>(learnSampleCount, indices, treeDelta, testData, learnProgress, localExecutor);
    } else {
        ::UpdateAvrgApprox<false>(learnSampleCount, indices, treeDelta, testData, learnProgress, localExecutor);
    }
}
