#include "fold.h"
#include "train_data.h"
#include "helpers.h"

#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>

static int UpdateSize(int size, const TVector<TQueryInfo>& queryInfo, const TVector<int>& queryIndices, int learnSampleCount) {
    size = Min(size, learnSampleCount);
    if (!queryInfo.empty()) {
        size = queryInfo[queryIndices[size - 1]].End;
    }
    return size;
}

static int SelectMinBatchSize(int learnSampleCount) {
    int size = learnSampleCount > 500 ? Min<int>(100, learnSampleCount / 50) : 1;
    return size;
}

static double SelectTailSize(int oldSize, double multiplier) {
    int size = ceil(oldSize * multiplier);
    return size;
}

static void InitFromBaseline(
    const int beginIdx,
    const int endIdx,
    const TVector<TVector<double>>& baseline,
    const TVector<int>& learnPermutation,
    bool storeExpApproxes,
    TVector<TVector<double>>* approx
) {
    const int learnSampleCount = learnPermutation.ysize();
    const int approxDimension = approx->ysize();
    for (int dim = 0; dim < approxDimension; ++dim) {
        TVector<double> tempBaseline(baseline[dim]);
        ExpApproxIf(storeExpApproxes, &tempBaseline);
        for (int docId = beginIdx; docId < endIdx; ++docId) {
            int initialIdx = docId;
            if (docId < learnSampleCount) {
                initialIdx = learnPermutation[docId];
            }
            (*approx)[dim][docId] = tempBaseline[initialIdx];
        }
    }
}

static void ShuffleData(const TTrainData& data, int permuteBlockSize, TRestorableFastRng64& rand, TFold* fold) {
    if (permuteBlockSize == 1 || !data.QueryId.empty()) {
        Shuffle(data.QueryId, rand, &fold->LearnPermutation);
        fold->PermutationBlockSize = 1;
    } else {
        const int blocksCount = (data.LearnSampleCount + permuteBlockSize - 1) / permuteBlockSize;
        TVector<int> blockedPermute(blocksCount);
        std::iota(blockedPermute.begin(), blockedPermute.end(), 0);
        Shuffle(blockedPermute.begin(), blockedPermute.end(), rand);

        int currentIdx = 0;
        for (int i = 0; i < blocksCount; ++i) {
            const int blockStartIdx = blockedPermute[i] * permuteBlockSize;
            const int blockEndIndx = Min(blockStartIdx + permuteBlockSize, data.LearnSampleCount);
            for (int j = blockStartIdx; j < blockEndIndx; ++j) {
                fold->LearnPermutation[currentIdx + j - blockStartIdx] = j;
            }
            currentIdx += blockEndIndx - blockStartIdx;
        }
        fold->PermutationBlockSize = permuteBlockSize;
    }
}

TFold BuildDynamicFold(
    const TTrainData& data,
    const TVector<TTargetClassifier>& targetClassifiers,
    bool shuffle,
    int permuteBlockSize,
    int approxDimension,
    double multiplier,
    bool storeExpApproxes,
    TRestorableFastRng64& rand
) {
    TFold ff;
    ff.SampleWeights.resize(data.LearnSampleCount, 1);
    ff.LearnPermutation.resize(data.LearnSampleCount);

    std::iota(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), 0);
    if (shuffle) {
        ShuffleData(data, permuteBlockSize, rand, &ff);
    } else {
        ff.PermutationBlockSize = data.LearnSampleCount;
    }

    ff.AssignTarget(data.Target, targetClassifiers);

    if (!data.Weights.empty()) {
        ff.AssignPermuted(data.Weights, &ff.LearnWeights);
    }

    TVector<int> invertPermutation = InvertPermutation(ff.LearnPermutation);

    TVector<int> queryIndices;
    if (!data.QueryId.empty()) {
        if (shuffle) {
            TVector<ui32> queriesId;
            ff.AssignPermuted(data.QueryId, &queriesId);
            UpdateQueriesInfo(queriesId, 0, data.LearnSampleCount, &ff.LearnQueriesInfo);
            UpdateQueriesPairs(data.Pairs, 0, data.LearnPairsCount, invertPermutation, &ff.LearnQueriesInfo);
        } else {
            ff.LearnQueriesInfo.insert(ff.LearnQueriesInfo.end(), data.QueryInfo.begin(), data.QueryInfo.begin() + data.LearnQueryCount);
        }
        queryIndices = GetQueryIndicesForDocs(ff.LearnQueriesInfo, data.LearnSampleCount);
    }

    ff.EffectiveDocCount = data.LearnSampleCount;

    int leftPartLen = UpdateSize(SelectMinBatchSize(data.LearnSampleCount), ff.LearnQueriesInfo, queryIndices, data.LearnSampleCount);
    while (ff.BodyTailArr.empty() || leftPartLen < data.LearnSampleCount) {
        TFold::TBodyTail bt;

        bt.BodyFinish = leftPartLen;
        bt.TailFinish = UpdateSize(SelectTailSize(leftPartLen, multiplier), ff.LearnQueriesInfo, queryIndices, data.LearnSampleCount);
        if (!data.QueryId.empty()) {
            bt.BodyQueryFinish = queryIndices[bt.BodyFinish - 1] + 1;
            bt.TailQueryFinish = queryIndices[bt.TailFinish - 1] + 1;
        }

        bt.Approx.resize(approxDimension, TVector<double>(bt.TailFinish, GetNeutralApprox(storeExpApproxes)));
        if (!data.Baseline.empty()) {
            InitFromBaseline(leftPartLen, bt.TailFinish, data.Baseline, ff.LearnPermutation, storeExpApproxes, &bt.Approx);
        }
        bt.Derivatives.resize(approxDimension, TVector<double>(bt.TailFinish));
        bt.WeightedDer.resize(approxDimension, TVector<double>(bt.TailFinish));
        ff.BodyTailArr.emplace_back(std::move(bt));
        leftPartLen = bt.TailFinish;
    }
    return ff;
}

TFold BuildPlainFold(
    const TTrainData& data,
    const TVector<TTargetClassifier>& targetClassifiers,
    bool shuffle,
    int permuteBlockSize,
    int approxDimension,
    bool storeExpApproxes,
    TRestorableFastRng64& rand
) {
    TFold ff;
    ff.SampleWeights.resize(data.LearnSampleCount, 1);
    ff.LearnPermutation.resize(data.LearnSampleCount);

    std::iota(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), 0);
    if (shuffle) {
        ShuffleData(data, permuteBlockSize, rand, &ff);
    } else {
        ff.PermutationBlockSize = data.LearnSampleCount;
    }

    ff.AssignTarget(data.Target, targetClassifiers);

    if (!data.Weights.empty()) {
        ff.AssignPermuted(data.Weights, &ff.LearnWeights);
    }

    TVector<int> invertPermutation = InvertPermutation(ff.LearnPermutation);

    if (shuffle) {
        TVector<ui32> queriesId;
        if (!data.QueryId.empty()) {
            ff.AssignPermuted(data.QueryId, &queriesId);
        }
        UpdateQueriesInfo(queriesId, 0, data.LearnSampleCount, &ff.LearnQueriesInfo);
        UpdateQueriesPairs(data.Pairs, 0, data.LearnPairsCount, invertPermutation, &ff.LearnQueriesInfo);
    } else {
        ff.LearnQueriesInfo.insert(ff.LearnQueriesInfo.end(), data.QueryInfo.begin(), data.QueryInfo.begin() + data.LearnQueryCount);
    }

    ff.EffectiveDocCount = data.GetSampleCount();

    TFold::TBodyTail bt;

    bt.BodyFinish = data.LearnSampleCount;
    bt.TailFinish = data.LearnSampleCount;
    bt.BodyQueryFinish = data.LearnQueryCount;
    bt.TailQueryFinish = data.LearnQueryCount;

    bt.Approx.resize(approxDimension, TVector<double>(data.GetSampleCount(), GetNeutralApprox(storeExpApproxes)));
    bt.Derivatives.resize(approxDimension, TVector<double>(data.GetSampleCount()));
    bt.WeightedDer.resize(approxDimension, TVector<double>(data.GetSampleCount()));
    if (!data.Baseline.empty()) {
        InitFromBaseline(0, data.GetSampleCount(), data.Baseline, ff.LearnPermutation, storeExpApproxes, &bt.Approx);
    }
    ff.BodyTailArr.emplace_back(std::move(bt));
    return ff;
}


void TFold::DropEmptyCTRs() {
    TVector<TProjection> emptyProjections;
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

void TFold::AssignTarget(const TVector<float>& target, const TVector<TTargetClassifier>& targetClassifiers) {
    AssignPermuted(target, &LearnTarget);
    int learnSampleCount = LearnPermutation.ysize();

    int ctrCount = targetClassifiers.ysize();
    LearnTargetClass.assign(ctrCount, TVector<int>(learnSampleCount));
    TargetClassesCount.resize(ctrCount);
    for (int ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
        for (int z = 0; z < learnSampleCount; ++z) {
            LearnTargetClass[ctrIdx][z] = targetClassifiers[ctrIdx].GetTargetClass(LearnTarget[z]);
        }
        TargetClassesCount[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
    }
}

void TFold::SaveApproxes(IOutputStream* s) const {
    const ui64 bodyTailCount = BodyTailArr.size();
    ::Save(s, bodyTailCount);
    for (ui64 i = 0; i < bodyTailCount; ++i) {
        ::Save(s, BodyTailArr[i].Approx);
    }
}

void TFold::LoadApproxes(IInputStream* s) {
    ui64 bodyTailCount;
    ::Load(s, bodyTailCount);
    CB_ENSURE(bodyTailCount == BodyTailArr.size());
    for (ui64 i = 0; i < bodyTailCount; ++i) {
        ::Load(s, BodyTailArr[i].Approx);
    }
}
