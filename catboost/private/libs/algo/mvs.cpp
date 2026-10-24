#include "mvs.h"

#include "fold.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>


inline static double GetSingleProbability(double derivativeAbsoluteValue, double threshold) {
    return (derivativeAbsoluteValue > threshold) ? 1.0 : (derivativeAbsoluteValue / threshold);
}

static double CalculateLastIterMeanLeafValue(const TVector<TVector<TVector<double>>>& leafValues) {
    const auto approxDimension = leafValues.back().size();
    const auto numLeaves = leafValues.back()[0].size();
    double sumOverLeaves = 0;
    const auto& lastIterValues = leafValues.back();
    for (auto leaf : xrange(numLeaves)) {
        double w2 = 0;
        for (auto dim : xrange(approxDimension)) {
            const double leafValue = lastIterValues[dim][leaf];
            w2 += leafValue * leafValue;
        }
        sumOverLeaves += sqrt(w2);
    }
    return sumOverLeaves / numLeaves;
}

static double CalculateMeanGradValue(const TVector<TConstArrayRef<double>>& derivatives, ui32 cnt, NPar::ILocalExecutor* localExecutor) {
        NPar::ILocalExecutor::TExecRangeParams blockParams(0, cnt);
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
                for (auto idx = blockOffset; idx < blockFinish; ++idx) {
                    double grad2 = 0;
                    for (auto dim : xrange(derivatives.size())) {
                        const double der = derivatives[dim][idx];
                        grad2 += der * der;
                    }
                    gradSumInBlock[blockId] += sqrt(grad2);
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
    const TVector<TConstArrayRef<double>>& derivatives,
    const TVector<TVector<TVector<double>>>& leafValues,
    NPar::ILocalExecutor* localExecutor) const {

    if (Lambda.Defined()) {
        return Lambda.GetRef();
    }
    const double mean = (!leafValues.empty())
        ? CalculateLastIterMeanLeafValue(leafValues)
        : CalculateMeanGradValue(derivatives, SampleCount, localExecutor);
    return mean * mean;
}

// Gives the same result as std::partition, including the order of elements (the threshold depends on it
// through the pivot choice and the summation order), but has no data-dependent branches,
// which are mispredicted for random candidates.
// std::partition swaps the k-th element of [begin, middle) not satisfying the predicate with the k-th
// element from the end of [middle, end) satisfying it, where middle = begin + (count satisfying the predicate).
// buffer must have room for (end - begin) elements.
template <typename TPredicate>
static TVector<double>::iterator PartitionBranchless(
    TVector<double>::iterator begin,
    TVector<double>::iterator end,
    TPredicate predicate,
    ui32* buffer) {

    const ui32 size = end - begin;
    ui32 middle = 0;
    for (ui32 idx : xrange(size)) {
        middle += predicate(begin[idx]);
    }
    ui32 leftCount = 0;
    for (ui32 idx : xrange(middle)) {
        buffer[leftCount] = idx;
        leftCount += !predicate(begin[idx]);
    }
    ui32* right = buffer + leftCount;
    ui32 rightCount = 0;
    for (ui32 idx = size; idx > middle; --idx) {
        right[rightCount] = idx - 1;
        rightCount += predicate(begin[idx - 1]);
    }
    Y_ASSERT(leftCount == rightCount);
    for (ui32 pairIdx : xrange(leftCount)) {
        std::swap(begin[buffer[pairIdx]], begin[right[pairIdx]]);
    }
    return begin + middle;
}

double TMvsSampler::CalculateThreshold(
    TVector<double>::iterator candidatesBegin,
    TVector<double>::iterator candidatesEnd,
    double sumOfSmallCurrent,
    ui32 numberOfLargeCurrent,
    double sampleSize,
    ui32* partitionBuffer) const {

    double threshold = *candidatesBegin;
    auto middleBegin = PartitionBranchless(
        candidatesBegin,
        candidatesEnd,
        [threshold](double candidate) {
            return candidate < threshold;
        },
        partitionBuffer);
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
            return CalculateThreshold(middleEnd, candidatesEnd, sumOfSmallCurrent, numberOfLargeCurrent, sampleSize, partitionBuffer);
        } else {
            return (sumOfSmallCurrent + sumOfSmallUpdate + sumOfMiddle) / (sampleSize - numberOfLargeCurrent);
        }
    } else {
        if (middleBegin != candidatesBegin) {
            numberOfLargeCurrent += numberOfLargeUpdate + numberOfMiddle;
            return CalculateThreshold(candidatesBegin, middleBegin, sumOfSmallCurrent, numberOfLargeCurrent, sampleSize, partitionBuffer);
        } else {
            return sumOfSmallCurrent / (sampleSize - numberOfLargeCurrent - numberOfMiddle - numberOfLargeUpdate);
        }
    }
}

void TMvsSampler::GenSampleWeights(
    EBoostingType boostingType,
    const TVector<TVector<TVector<double>>>& leafValues,
    TRestorableFastRng64* rand,
    NPar::ILocalExecutor* localExecutor,
    TFold* fold) const {

    if (SampleRate == 1.0f) {
        Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1.0f);
    } else {
        const auto approxDimension = fold->GetApproxDimension();
        TVector<TVector<double>> tailDerivatives;
        TVector<TConstArrayRef<double>> derivatives(approxDimension);
        for (auto dim : xrange(approxDimension)) {
            derivatives[dim] = fold->BodyTailArr[0].WeightedDerivatives[dim];
        }
        if (boostingType == EBoostingType::Ordered) {
            tailDerivatives.resize(approxDimension);
            for (auto dim : xrange(approxDimension)) {
                tailDerivatives[dim].yresize(SampleCount);
            }
            localExecutor->ExecRange(
                [&](ui32 bodyTailId) {
                    const TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
                    for (auto dim : xrange(approxDimension)) {
                        TConstArrayRef<double> bodyTailDerivatives = bt.WeightedDerivatives[dim];
                        if (bodyTailId == 0) {
                            Copy(
                                bodyTailDerivatives.begin(),
                                bodyTailDerivatives.begin() + bt.TailFinish,
                                tailDerivatives[dim].begin()
                            );
                        } else {
                            Copy(
                                bodyTailDerivatives.begin() + bt.BodyFinish,
                                bodyTailDerivatives.begin() + bt.TailFinish,
                                tailDerivatives[dim].begin() + bt.BodyFinish
                            );
                        }
                    }
                },
                0,
                SafeIntegerCast<int>(fold->BodyTailArr.size()),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
            for (auto dim : xrange(approxDimension)) {
                derivatives[dim] = tailDerivatives[dim];
            }
        }

        double lambda = GetLambda(derivatives, leafValues, localExecutor);

        NPar::ILocalExecutor::TExecRangeParams blockParams(0, SampleCount);
        blockParams.SetBlockSize(BlockSize);
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

                TVector<double> thresholdCandidates(blockSize, lambda);
                for (auto dim : xrange(approxDimension)) {
                    TConstArrayRef<double> derivativesRef(derivatives[dim].begin() + blockOffset, blockSize);
                    for (auto idx : xrange(blockSize)) {
                        const double der = derivativesRef[idx];
                        thresholdCandidates[idx] += der * der;
                    }
                }
                for (auto& value : thresholdCandidates) {
                    value = sqrt(value);
                }
                TVector<ui32> partitionBuffer;
                partitionBuffer.yresize(blockSize);
                double threshold = CalculateThreshold(
                    thresholdCandidates.begin(),
                    thresholdCandidates.end(),
                    0,
                    0,
                    SampleRate * blockSize,
                    partitionBuffer.data());
                for (ui32 i = blockOffset; i < blockFinish; ++i) {
                    double grad2 = 0;
                    for (auto dim : xrange(approxDimension)) {
                        const double der = derivatives[dim][i];
                        grad2 += der * der;
                    }
                    const double probability = GetSingleProbability(sqrt(grad2 + lambda), threshold);
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
