#include "fold.h"
#include "dataset.h"
#include "helpers.h"

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/restorable_rng.h>

static int UpdateSize(int size, const TVector<TQueryInfo>& queryInfo, const TVector<int>& queryIndices, int learnSampleCount) {
    size = Min(size, learnSampleCount);
    if (!queryInfo.empty()) {
        size = queryInfo[queryIndices[size - 1]].End;
    }
    return size;
}

static int SelectMinBatchSize(int learnSampleCount) {
    return learnSampleCount > 500 ? Min<int>(100, learnSampleCount / 50) : 1;
}

static double SelectTailSize(int oldSize, double multiplier) {
    return ceil(oldSize * multiplier);
}

static void InitFromBaseline(
    const int beginIdx,
    const int endIdx,
    const TVector<TVector<double>>& baseline,
    const TVector<size_t>& learnPermutation,
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

static void ShuffleData(const TDataset& learnData, int permuteBlockSize, TRestorableFastRng64& rand, TFold* fold) {
    const int learnSampleCount = learnData.GetSampleCount();
    if (permuteBlockSize == 1 || !learnData.QueryId.empty()) {
        Shuffle(learnData.QueryId, rand, &fold->LearnPermutation);
        fold->PermutationBlockSize = 1;
    } else {
        const int blocksCount = (learnSampleCount + permuteBlockSize - 1) / permuteBlockSize;
        TVector<int> blockedPermute(blocksCount);
        std::iota(blockedPermute.begin(), blockedPermute.end(), 0);
        Shuffle(blockedPermute.begin(), blockedPermute.end(), rand);

        int currentIdx = 0;
        for (int i = 0; i < blocksCount; ++i) {
            const int blockStartIdx = blockedPermute[i] * permuteBlockSize;
            const int blockEndIndx = Min(blockStartIdx + permuteBlockSize, learnSampleCount);
            for (int j = blockStartIdx; j < blockEndIndx; ++j) {
                fold->LearnPermutation[currentIdx + j - blockStartIdx] = j;
            }
            currentIdx += blockEndIndx - blockStartIdx;
        }
        fold->PermutationBlockSize = permuteBlockSize;
    }
}

TFold BuildDynamicFold(
    const TDataset& learnData,
    const TVector<TTargetClassifier>& targetClassifiers,
    bool shuffle,
    int permuteBlockSize,
    int approxDimension,
    double multiplier,
    bool storeExpApproxes,
    bool hasPairwiseWeights,
    TRestorableFastRng64& rand
) {
    const int learnSampleCount = learnData.GetSampleCount();

    TFold ff;
    ff.SampleWeights.resize(learnSampleCount, 1);
    ff.LearnPermutation.resize(learnSampleCount);

    std::iota(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), 0);
    if (shuffle) {
        ShuffleData(learnData, permuteBlockSize, rand, &ff);
    } else {
        ff.PermutationBlockSize = learnSampleCount;
    }

    ff.AssignTarget(learnData.Target, targetClassifiers);

    if (!learnData.Weights.empty()) {
        ff.AssignPermuted(learnData.Weights, &ff.LearnWeights);
    }

    TVector<size_t> invertPermutation = InvertPermutation(ff.LearnPermutation);

    TVector<int> queryIndices;
    if (!learnData.QueryId.empty()) {
        if (shuffle) {
            TVector<TGroupId> groupIds;
            TVector<ui32> subgroupId;
            ff.AssignPermuted(learnData.QueryId, &groupIds);
            if (!learnData.SubgroupId.empty()) {
                ff.AssignPermuted(learnData.SubgroupId, &subgroupId);
            }
            UpdateQueriesInfo(groupIds, subgroupId, 0, learnSampleCount, &ff.LearnQueriesInfo);
            UpdateQueriesPairs(learnData.Pairs, invertPermutation, &ff.LearnQueriesInfo);
        } else {
            ff.LearnQueriesInfo.insert(ff.LearnQueriesInfo.end(), learnData.QueryInfo.begin(), learnData.QueryInfo.end());
        }
        queryIndices = GetQueryIndicesForDocs(ff.LearnQueriesInfo, learnSampleCount);
    }
    TVector<float> pairwiseWeights;
    if (hasPairwiseWeights) {
        pairwiseWeights.resize(learnSampleCount);
        CalcPairwiseWeights(ff.LearnQueriesInfo, ff.LearnQueriesInfo.ysize(), &pairwiseWeights);
    }

    int leftPartLen = UpdateSize(SelectMinBatchSize(learnSampleCount), ff.LearnQueriesInfo, queryIndices, learnSampleCount);
    while (ff.BodyTailArr.empty() || leftPartLen < learnSampleCount) {
        TFold::TBodyTail bt;

        bt.BodyFinish = leftPartLen;
        bt.TailFinish = UpdateSize(SelectTailSize(leftPartLen, multiplier), ff.LearnQueriesInfo, queryIndices, learnSampleCount);
        if (!learnData.QueryId.empty()) {
            bt.BodyQueryFinish = queryIndices[bt.BodyFinish - 1] + 1;
            bt.TailQueryFinish = queryIndices[bt.TailFinish - 1] + 1;
        }

        bt.Approx.resize(approxDimension, TVector<double>(bt.TailFinish, GetNeutralApprox(storeExpApproxes)));
        if (!learnData.Baseline.empty()) {
            InitFromBaseline(leftPartLen, bt.TailFinish, learnData.Baseline, ff.LearnPermutation, storeExpApproxes, &bt.Approx);
        }
        bt.WeightedDerivatives.resize(approxDimension, TVector<double>(bt.TailFinish));
        bt.SampleWeightedDerivatives.resize(approxDimension, TVector<double>(bt.TailFinish));
        if (hasPairwiseWeights) {
            bt.PairwiseWeights.resize(bt.TailFinish);
            bt.PairwiseWeights.insert(bt.PairwiseWeights.begin(), pairwiseWeights.begin(), pairwiseWeights.begin() + bt.TailFinish);
            bt.SamplePairwiseWeights.resize(bt.TailFinish);
        }
        ff.BodyTailArr.emplace_back(std::move(bt));
        leftPartLen = bt.TailFinish;
    }
    return ff;
}

TFold BuildPlainFold(
    const TDataset& learnData,
    const TVector<TTargetClassifier>& targetClassifiers,
    bool shuffle,
    int permuteBlockSize,
    int approxDimension,
    bool storeExpApproxes,
    bool hasPairwiseWeights,
    TRestorableFastRng64& rand
) {
    const int learnSampleCount = learnData.GetSampleCount();

    TFold ff;
    ff.SampleWeights.resize(learnSampleCount, 1);
    ff.LearnPermutation.resize(learnSampleCount);

    std::iota(ff.LearnPermutation.begin(), ff.LearnPermutation.end(), 0);
    if (shuffle) {
        ShuffleData(learnData, permuteBlockSize, rand, &ff);
    } else {
        ff.PermutationBlockSize = learnSampleCount;
    }

    ff.AssignTarget(learnData.Target, targetClassifiers);

    if (!learnData.Weights.empty()) {
        ff.AssignPermuted(learnData.Weights, &ff.LearnWeights);
    }

    TVector<size_t> invertPermutation = InvertPermutation(ff.LearnPermutation);

    if (shuffle) {
        TVector<TGroupId> groupIds;
        TVector<ui32> subgroupId;
        if (!learnData.QueryId.empty()) {
            ff.AssignPermuted(learnData.QueryId, &groupIds);
        }
        if (!learnData.SubgroupId.empty()) {
            ff.AssignPermuted(learnData.SubgroupId, &subgroupId);
        }
        UpdateQueriesInfo(groupIds, subgroupId, 0, learnSampleCount, &ff.LearnQueriesInfo);
        UpdateQueriesPairs(learnData.Pairs, invertPermutation, &ff.LearnQueriesInfo);
    } else {
        ff.LearnQueriesInfo.insert(ff.LearnQueriesInfo.end(), learnData.QueryInfo.begin(), learnData.QueryInfo.end());
    }

    TFold::TBodyTail bt;

    bt.BodyFinish = learnSampleCount;
    bt.TailFinish = learnSampleCount;
    bt.BodyQueryFinish = learnData.GetQueryCount();
    bt.TailQueryFinish = learnData.GetQueryCount();

    bt.Approx.resize(approxDimension, TVector<double>(learnSampleCount, GetNeutralApprox(storeExpApproxes)));
    bt.WeightedDerivatives.resize(approxDimension, TVector<double>(learnSampleCount));
    bt.SampleWeightedDerivatives.resize(approxDimension, TVector<double>(learnSampleCount));
    if (hasPairwiseWeights) {
        bt.PairwiseWeights.resize(learnSampleCount);
        CalcPairwiseWeights(ff.LearnQueriesInfo, bt.TailQueryFinish, &bt.PairwiseWeights);
        bt.SamplePairwiseWeights.resize(learnSampleCount);
    }
    if (!learnData.Baseline.empty()) {
        InitFromBaseline(0, learnSampleCount, learnData.Baseline, ff.LearnPermutation, storeExpApproxes, &bt.Approx);
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
