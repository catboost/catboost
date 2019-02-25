#include "mvs.h"
#include <catboost/libs/options/restrictions.h>
#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>


inline static double GetSingleProbability(double derivativeAbsoluteValue, double threshold) {
    return (derivativeAbsoluteValue > threshold) ? 1.0 : (derivativeAbsoluteValue / threshold);
}

void TMvsSampler::GenSampleWeights(
    TFold& fold,
    EBoostingType boostingType,
    TRestorableFastRng64* rand,
    NPar::TLocalExecutor* localExecutor) const {

    if (GetHeadFraction() == 1.0f) {
        Fill(fold.SampleWeights.begin(), fold.SampleWeights.end(), 1.0f);
    } else {
        TVector<ui32> docIndices(SampleCount);
        TVector<double> tailDerivatives;
        Iota(docIndices.begin(), docIndices.end(), 0);
        CB_ENSURE_INTERNAL(fold.BodyTailArr[0].WeightedDerivatives.size() == 1, "MVS bootstrap mode is not implemented for multi-dimensional approxes");
        const double* derivatives = fold.BodyTailArr[0].WeightedDerivatives[0].data();
        if (boostingType == EBoostingType::Ordered) {
            tailDerivatives.yresize(SampleCount);
            localExecutor->ExecRange(
                [&](ui32 bodyTailId) {
                    const TFold::TBodyTail& bt = fold.BodyTailArr[bodyTailId];
                    const double* bodyTailDerivatives = bt.WeightedDerivatives[0].data();
                    if (bodyTailId == 0) {
                        Copy(bodyTailDerivatives, bodyTailDerivatives + bt.TailFinish, tailDerivatives.begin());
                    } else {
                        Copy(bodyTailDerivatives + bt.BodyFinish, bodyTailDerivatives + bt.TailFinish, tailDerivatives.begin() + bt.BodyFinish);
                    }
                },
                0,
                fold.BodyTailArr.size(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
            derivatives = tailDerivatives.data();
        }
        TVector<double> sampleThresholds(CB_THREAD_LIMIT);
        NPar::TLocalExecutor::TExecRangeParams blockParams(0, SampleCount);
        blockParams.SetBlockCount(CB_THREAD_LIMIT);
        localExecutor->ExecRange(
            [&](ui32 blockId) {
                const ui32 blockOffset = blockId * blockParams.GetBlockSize();
                const ui32 blockSize = Min(static_cast<ui32>(blockParams.GetBlockSize()), SampleCount - blockOffset);
                const ui32 blockFinish = blockOffset + blockSize;
                ui32 headCount = Min(static_cast<ui32>(GetHeadFraction() * blockSize), blockSize);
                NthElement(
                    docIndices.begin() + blockOffset,
                    docIndices.begin() + blockOffset + headCount,
                    docIndices.begin() + blockFinish,
                    [derivatives](ui32 lhs, ui32 rhs) {
                        return Abs(derivatives[lhs]) > Abs(derivatives[rhs]);
                    }
                );
                sampleThresholds[blockId] = Abs(derivatives[docIndices[blockOffset + headCount]]);
            },
            0,
            blockParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
        double threshold = Accumulate(sampleThresholds.begin(), sampleThresholds.end(), 0.0) / sampleThresholds.size();

        const ui64 randSeed = rand->GenRand();
        localExecutor->ExecRange(
            [&](ui32 blockId) {
                TRestorableFastRng64 prng(randSeed + blockId);
                prng.Advance(10); // reduce correlation between RNGs in different threads
                const ui32 blockOffset = blockId * blockParams.GetBlockSize();
                const ui32 blockSize = Min(static_cast<ui32>(blockParams.GetBlockSize()), SampleCount - blockOffset);
                const ui32 blockFinish = blockOffset + blockSize;
                for (ui32 i = blockOffset; i < blockFinish; ++i) {
                    const double probability = GetSingleProbability(Abs(derivatives[i]), threshold);
                    if (probability > std::numeric_limits<double>::epsilon()) {
                        const double weight = 1 / probability;
                        double r = prng.GenRandReal1();
                        fold.SampleWeights[i] = weight * (r < probability);
                    } else {
                        fold.SampleWeights[i] = 0;
                    }
                }
            },
            0,
            blockParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}
