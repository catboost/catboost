#pragma once

#include "approx_util.h"
#include "restorable_rng.h"
#include "target_classifier.h"
#include "train_data.h"
#include "online_ctr.h"
#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/helpers/clear_array.h>
#include <catboost/libs/model/projection.h>

#include <util/generic/vector.h>
#include <util/random/shuffle.h>
#include <util/generic/ymath.h>


static int SelectMinBatchSize(int sampleCount) {
    return sampleCount > 500 ? Min<int>(100, sampleCount / 50) : 1;
}

struct TFold {
    struct TBodyTail {
        yvector<yvector<double>> Approx;
        yvector<yvector<double>> Derivatives;
        // TODO(annaveronika): make a single vector<vector> for all BodyTail
        yvector<yvector<double>> WeightedDer;

        int BodyFinish;
        int TailFinish;

        TBodyTail()
            : BodyFinish(0)
            , TailFinish(0)
        {
        }
    };

    yvector<float> LearnWeights;
    yvector<int> LearnPermutation; // index in original array
    yvector<TBodyTail> BodyTailArr;
    yvector<float> LearnTarget;
    yvector<float> SampleWeights;
    yvector<yvector<int>> LearnTargetClass;
    yvector<int> TargetClassesCount;
    size_t EffectiveDocCount = 0;

    TOnlineCTRHash& GetCtrs(const TProjection& proj) {
        return HasSingleFeature(proj) ? OnlineSingleCtrs : OnlineCTR;
    }

    const TOnlineCTRHash& GetCtrs(const TProjection& proj) const {
        return HasSingleFeature(proj) ? OnlineSingleCtrs : OnlineCTR;
    }

    TOnlineCTR& GetCtrRef(const TProjection& proj) {
        return GetCtrs(proj)[proj];
    }

    const TOnlineCTR& GetCtr(const TProjection& proj) const {
        return GetCtrs(proj).at(proj);
    }

    void DropEmptyCTRs() {
        yvector<TProjection> emptyProjections;
        for (auto& projCtr : OnlineSingleCtrs) {
            if (projCtr.second.Feature.empty()) {
                emptyProjections.emplace_back(projCtr.first);
            }
        }
        for (auto& projCtr : OnlineCTR) {
            if (projCtr.second.Feature.empty()) {
                emptyProjections.emplace_back(projCtr.first);
            }
        }
        for (const auto& proj : emptyProjections) {
            GetCtrs(proj).erase(proj);
        }
    }

    void AssignTarget(const yvector<float>& target,
                      const yvector<TTargetClassifier>& targetClassifiers) {
        AssignPermuted(target, &LearnTarget);
        int learnSampleCount = LearnPermutation.ysize();

        int ctrCount = targetClassifiers.ysize();
        LearnTargetClass.assign(ctrCount, yvector<int>(learnSampleCount));
        TargetClassesCount.resize(ctrCount);
        for (int ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
            for (int z = 0; z < learnSampleCount; ++z) {
                LearnTargetClass[ctrIdx][z] = targetClassifiers[ctrIdx].GetTargetClass(LearnTarget[z]);
            }
            TargetClassesCount[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
        }
    }

    void AssignPermuted(const yvector<float>& source, yvector<float>* dest) const {
        int learnSampleCount = LearnPermutation.ysize();
        yvector<float>& destination = *dest;
        destination.resize(learnSampleCount);
        for (int z = 0; z < learnSampleCount; ++z) {
            int i = LearnPermutation[z];
            destination[z] = source[i];
        }
    }

    int GetApproxDimension() const {
        return BodyTailArr[0].Approx.ysize();
    }

    void TrimOnlineCTR(size_t maxOnlineCTRFeatures) {
        if (OnlineCTR.size() > maxOnlineCTRFeatures) {
            OnlineCTR.clear();
        }
    }

    void SaveApproxes(IOutputStream* s) const {
        const ui64 bodyTailCount = BodyTailArr.size();
        ::Save(s, bodyTailCount);
        for (ui64 i = 0; i < bodyTailCount; ++i) {
            ::Save(s, BodyTailArr[i].Approx);
        }
    }
    void LoadApproxes(IInputStream* s) {
        ui64 bodyTailCount;
        ::Load(s, bodyTailCount);
        CB_ENSURE(bodyTailCount == BodyTailArr.size());
        for (ui64 i = 0; i < bodyTailCount; ++i) {
            ::Load(s, BodyTailArr[i].Approx);
        }
    }
private:
    TOnlineCTRHash OnlineSingleCtrs;
    TOnlineCTRHash OnlineCTR;
    static bool HasSingleFeature(const TProjection& proj) {
        return proj.BinFeatures.ysize() + proj.CatFeatures.ysize() == 1;
    }
};

static void InitFromBaseline(const int beginIdx, const int endIdx,
                        const yvector<yvector<double>>& baseline,
                        const yvector<int>& learnPermutation,
                        bool storeExpApproxes,
                        yvector<yvector<double>>* approx) {
    const int learnSampleCount = learnPermutation.ysize();
    const int approxDimension = approx->ysize();
    for (int docId = beginIdx; docId < endIdx; ++docId) {
        int initialIdx = docId;
        if (docId < learnSampleCount) {
            initialIdx = learnPermutation[docId];
        }
        yvector<yvector<double>> tempBaseline(1, baseline[initialIdx]);
        ExpApproxIf(storeExpApproxes, &tempBaseline);
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*approx)[dim][docId] = tempBaseline[0][dim];
        }
    }
}


inline TFold BuildLearnFold(const TTrainData& data,
                            const yvector<TTargetClassifier>& targetClassifiers,
                            bool shuffle,
                            int permuteBlockSize,
                            int approxDimension,
                            double multiplier,
                            bool storeExpApproxes,
                            TRestorableFastRng64& rand) {
    TFold ff;
    ff.LearnPermutation.resize(data.LearnSampleCount);
    std::iota(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), 0);
    if (shuffle) {
        if (permuteBlockSize == 1) { // shortcut for speed
            Shuffle(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), rand);
        } else {
            const int blocksCount = (data.LearnSampleCount + permuteBlockSize - 1) / permuteBlockSize;
            yvector<int> blockedPermute(blocksCount);
            std::iota(blockedPermute.begin(), blockedPermute.end(), 0);
            Shuffle(blockedPermute.begin(), blockedPermute.end(), rand);

            int currentIdx = 0;
            for (int i = 0; i < blocksCount; ++i) {
                const int blockStartIdx = blockedPermute[i] * permuteBlockSize;
                const int blockEndIndx = Min(blockStartIdx + permuteBlockSize, data.LearnSampleCount);
                for (int j = blockStartIdx; j < blockEndIndx; ++j) {
                    ff.LearnPermutation[currentIdx + j - blockStartIdx] = j;
                }
                currentIdx += blockEndIndx - blockStartIdx;
            }
        }
    }

    ff.AssignTarget(data.Target, targetClassifiers);

    if (!data.Weights.empty()) {
        ff.AssignPermuted(data.Weights, &ff.LearnWeights);
    }
    ff.EffectiveDocCount = data.LearnSampleCount;
    int leftPartLen = SelectMinBatchSize(data.LearnSampleCount);
    while (leftPartLen < data.LearnSampleCount) {
        TFold::TBodyTail bt;
        bt.BodyFinish = leftPartLen;
        bt.TailFinish = Min(ceil(leftPartLen * multiplier), ff.LearnPermutation.ysize() + 0.);
        bt.Approx.resize(approxDimension, yvector<double>(bt.TailFinish, GetNeutralApprox(storeExpApproxes)));
        if (!data.Baseline[0].empty()) {
            InitFromBaseline(leftPartLen, bt.TailFinish, data.Baseline, ff.LearnPermutation, storeExpApproxes, &bt.Approx);
        }
        bt.Derivatives.resize(approxDimension, yvector<double>(bt.TailFinish));
        bt.WeightedDer.resize(approxDimension, yvector<double>(bt.TailFinish));
        ff.BodyTailArr.emplace_back(std::move(bt));
        leftPartLen = bt.TailFinish;
    }
    return ff;
}

inline TFold BuildAveragingFold(const TTrainData& data,
                                const yvector<TTargetClassifier>& targetClassifiers,
                                bool shuffle,
                                int approxDimension,
                                bool storeExpApproxes,
                                TRestorableFastRng64& rand) {
    TFold ff;
    ff.LearnPermutation.resize(data.LearnSampleCount);
    std::iota(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), 0);

    if (shuffle) {
        Shuffle(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), rand);
    }

    ff.AssignTarget(data.Target, targetClassifiers);

    if (!data.Weights.empty()) {
        ff.AssignPermuted(data.Weights, &ff.LearnWeights);
    }
    ff.EffectiveDocCount = data.GetSampleCount();
    TFold::TBodyTail bt;
    bt.BodyFinish = data.LearnSampleCount;
    bt.TailFinish = data.LearnSampleCount;
    bt.Approx.resize(approxDimension, yvector<double>(data.GetSampleCount(), GetNeutralApprox(storeExpApproxes)));
    bt.WeightedDer.resize(approxDimension, yvector<double>(data.GetSampleCount()));
    if (!data.Baseline[0].empty()) {
        InitFromBaseline(0, data.GetSampleCount(), data.Baseline, ff.LearnPermutation, storeExpApproxes, &bt.Approx);
    }
    ff.BodyTailArr.emplace_back(std::move(bt));
    return ff;
}
