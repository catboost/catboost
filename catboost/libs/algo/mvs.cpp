#include "mvs.h"

#include "fold.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>


inline static double GetSingleProbability(double derivativeAbsoluteValue, double threshold) {
    return (derivativeAbsoluteValue > threshold) ? 1.0 : (derivativeAbsoluteValue / threshold);
}

static double CalculateLastIterMeanLeafValue(const TVector<TVector<TVector<double>>>& leafValues) {
    const ui32 lastIter = leafValues.ysize();
    const TVector<double>& lastIterValues = leafValues[lastIter - 1][0]; // one dimensional approx
    double sumOfAbsoluteValues = 0.0;
    for (double value : lastIterValues) {
        sumOfAbsoluteValues += Abs(value);
    }
    return sumOfAbsoluteValues / lastIterValues.ysize();
}

static double CalculateMeanGradValue(TConstArrayRef<double> derivatives, ui32 cnt, NPar::TLocalExecutor* localExecutor) {
        NPar::TLocalExecutor::TExecRangeParams blockParams(0, cnt);
        blockParams.SetBlockCount(CB_THREAD_LIMIT);
        TVector<double> gradSumInBlock(blockParams.GetBlockCount(), 0.0);
        localExecutor->ExecRange(
            [&](ui32 blockId) {
                const ui32 blockOffset = blockId * blockParams.GetBlockSize();
                const ui32 blockSize = Min(
                    static_cast<ui32>(blockParams.GetBlockSize()),
                    cnt - blockOffset
                );
                const ui32 blockFinish = blockOffset + blockSize;
                const auto derivativesBlockBegin = derivatives.begin() + blockOffset;
                const auto derivativesBlockEnd = derivatives.begin() + blockFinish;
                for (auto it = derivativesBlockBegin; it != derivativesBlockEnd; ++it) {
                    gradSumInBlock[blockId] += Abs(*it);
                }
            },
            0,
            blockParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        const double sumOfGradients = Accumulate(gradSumInBlock.begin(), gradSumInBlock.end(), 0.0);
        return sumOfGradients / cnt;
}

double TMvsSampler::GetLambda(
    TConstArrayRef<double> derivatives,
    const TVector<TVector<TVector<double>>>& leafValues,
    NPar::TLocalExecutor* localExecutor) const {

    if (Lambda.Defined()) {
        return Lambda.GetRef();
    }
    const double mean = (!leafValues.empty())
        ? CalculateLastIterMeanLeafValue(leafValues)
        : CalculateMeanGradValue(derivatives, SampleCount, localExecutor);
    return mean * mean;
}

double TMvsSampler::CalculateThreshold(
    TVector<double>::iterator candidatesBegin,
    TVector<double>::iterator candidatesEnd,
    double sumOfSmallCurrent,
    ui32 numberOfLargeCurrent,
    double sampleSize) const {

    double threshold = *candidatesBegin;
    auto middleBegin = std::partition(candidatesBegin, candidatesEnd, [threshold](double candidate) {
        return candidate < threshold;
    });
    auto middleEnd = std::partition(middleBegin, candidatesEnd, [threshold](double candidate) {
        return candidate <= threshold;
    });

    double sumOfSmallUpdate = Accumulate(candidatesBegin, middleBegin, 0.0);
    ui32 numberOfLargeUpdate = candidatesEnd - middleEnd;
    ui32 numberOfMiddle = middleEnd - middleBegin;
    double sumOfMiddle = numberOfMiddle * threshold;

    double estimatedSampleSize = 
        (sumOfSmallCurrent + sumOfSmallUpdate) / threshold + numberOfLargeCurrent + numberOfLargeUpdate + numberOfMiddle;
    if (estimatedSampleSize > sampleSize) {
        if (middleEnd != candidatesEnd) {
            sumOfSmallCurrent += sumOfMiddle + sumOfSmallUpdate;
            return CalculateThreshold(middleEnd, candidatesEnd, sumOfSmallCurrent, numberOfLargeCurrent, sampleSize);
        } else {
            return (sumOfSmallCurrent + sumOfSmallUpdate + sumOfMiddle) / (sampleSize - numberOfLargeCurrent);
        }
    } else {
        if (middleBegin != candidatesBegin) {
            numberOfLargeCurrent += numberOfLargeUpdate + numberOfMiddle;
            return CalculateThreshold(candidatesBegin, middleBegin, sumOfSmallCurrent, numberOfLargeCurrent, sampleSize);
        } else {
            return sumOfSmallCurrent / (sampleSize - numberOfLargeCurrent - numberOfMiddle - numberOfLargeUpdate);
        }
    }
}

void TMvsSampler::GenSampleWeights(
    EBoostingType boostingType,
    const TVector<TVector<TVector<double>>>& leafValues,
    TRestorableFastRng64* rand,
    NPar::TLocalExecutor* localExecutor,
    TFold* fold) const {

    if (SampleRate == 1.0f) {
        Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1.0f);
    } else {
        CB_ENSURE_INTERNAL(
            fold->BodyTailArr[0].WeightedDerivatives.size() == 1,
            "MVS bootstrap mode is not implemented for multi-dimensional approxes"
        );
        TVector<double> tailDerivatives;
        TConstArrayRef<double> derivatives = fold->BodyTailArr[0].WeightedDerivatives[0];
        if (boostingType == EBoostingType::Ordered) {
            tailDerivatives.yresize(SampleCount);
            localExecutor->ExecRange(
                [&](ui32 bodyTailId) {
                    const TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
                    TConstArrayRef<double> bodyTailDerivatives = bt.WeightedDerivatives[0];
                    if (bodyTailId == 0) {
                        Copy(
                            bodyTailDerivatives.begin(),
                            bodyTailDerivatives.begin() + bt.TailFinish,
                            tailDerivatives.begin()
                        );
                    } else {
                        Copy(
                            bodyTailDerivatives.begin() + bt.BodyFinish,
                            bodyTailDerivatives.begin() + bt.TailFinish,
                            tailDerivatives.begin() + bt.BodyFinish
                        );
                    }
                },
                0,
                fold->BodyTailArr.size(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
            derivatives = tailDerivatives;
        }

        double lambda = GetLambda(derivatives, leafValues, localExecutor);

        NPar::TLocalExecutor::TExecRangeParams blockParams(0, SampleCount);
        blockParams.SetBlockCount(CB_THREAD_LIMIT);
        const ui64 randSeed = rand->GenRand();
        localExecutor->ExecRange(
            [&](ui32 blockId) {
                TRestorableFastRng64 prng(randSeed + blockId);
                prng.Advance(10); // reduce correlation between RNGs in different threads
                const ui32 blockOffset = blockId * blockParams.GetBlockSize();
                const ui32 blockSize = Min(
                    static_cast<ui32>(blockParams.GetBlockSize()),
                    SampleCount - blockOffset
                );
                const ui32 blockFinish = blockOffset + blockSize;

                TVector<double> thresholdCandidates(blockSize);
                Transform(
                    derivatives.begin() + blockOffset,
                    derivatives.begin() + blockFinish,
                    thresholdCandidates.begin(),
                    [lambda](double grad) {
                        return sqrt(grad * grad + lambda);
                    });
                double threshold = CalculateThreshold(
                    thresholdCandidates.begin(),
                    thresholdCandidates.end(),
                    0,
                    0,
                    SampleRate * blockSize);
                for (ui32 i = blockOffset; i < blockFinish; ++i) {
                    const double grad = derivatives[i];
                    const double probability = GetSingleProbability(sqrt(grad * grad + lambda), threshold);
                    if (probability > std::numeric_limits<double>::epsilon()) {
                        const double weight = 1 / probability;
                        double r = prng.GenRandReal1();
                        fold->SampleWeights[i] = weight * (r < probability);
                    } else {
                        fold->SampleWeights[i] = 0;
                    }
                }
            },
            0,
            blockParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}
