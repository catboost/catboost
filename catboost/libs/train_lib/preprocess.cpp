#include "preprocess.h"

// TODO(nikitxskv): Is this a bottleneck? switch to vector+unique vs vector+sort+unique?
static bool IsCorrectQueryIdsFormat(const TVector<ui32>& queryIds, int begin, int end) {
    THashSet<ui32> queryGroupIds;
    ui32 lastId = queryIds.empty() ? 0 : queryIds[0];
    for (int i = begin; i < end; ++i) {
        ui32 id = queryIds[i];
        if (id != lastId) {
            if (queryGroupIds.has(id)) {
                return false;
            }
            queryGroupIds.insert(lastId);
            lastId = id;
        }
    }
    return true;
}

static bool ArePairsGroupedByQuery(const TVector<ui32>& queryId, const TVector<TPair>& pairs) {
    for (const auto& pair : pairs) {
        if (queryId[pair.WinnerId] != queryId[pair.LoserId]) {
            return false;
        }
    }
    return true;
}

static void CheckTarget(const TVector<float>& target, int learnSampleCount, ELossFunction lossFunction) {
    if (lossFunction == ELossFunction::Logloss) {
        float minTarget = *MinElement(target.begin(), target.begin() + learnSampleCount);
        float maxTarget = *MaxElement(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(minTarget == 0, "All train targets are greater than border");
        CB_ENSURE(maxTarget == 1, "All train targets are smaller than border");
    }

    if (lossFunction == ELossFunction::CrossEntropy) {
        float minTarget = *MinElement(target.begin(), target.begin() + learnSampleCount);
        float maxTarget = *MaxElement(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(minTarget >= 0, "Min target less than 0: " + ToString(minTarget));
        CB_ENSURE(maxTarget <= 1, "Max target greater than 1: " + ToString(minTarget));
    }

    if (lossFunction == ELossFunction::QuerySoftMax) {
        float minTarget = *MinElement(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(minTarget >= 0, "Min target less than 0: " + ToString(minTarget));
    }

    if (IsMultiClassError(lossFunction)) {
        CB_ENSURE(AllOf(target, [](float x) { return floor(x) == x && x >= 0; }), "if loss-function is MultiClass then each target label should be nonnegative integer");
    }

    if (lossFunction != ELossFunction::PairLogit) {
        float minTarget = *MinElement(target.begin(), target.begin() + learnSampleCount);
        float maxTarget = *MaxElement(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(minTarget != maxTarget, "All train targets are equal");
    }
}

static void CheckBaseline(ELossFunction lossFunction,
                   const TVector<TVector<double>>& trainBaseline,
                   const TVector<TVector<double>>& testBaseline,
                   int testDocs) {
    bool trainHasBaseline = trainBaseline.ysize() != 0;
    bool testHasBaseline = trainHasBaseline;
    if (testDocs != 0) {
        testHasBaseline = testBaseline.ysize() != 0;
    }
    if (trainHasBaseline && !testHasBaseline) {
        CB_ENSURE(false, "Baseline for test is not provided");
    }
    if (testHasBaseline && !trainHasBaseline) {
        CB_ENSURE(false, "Baseline for train is not provided");
    }
    if (trainHasBaseline && testHasBaseline && testDocs != 0) {
        CB_ENSURE(trainBaseline.ysize() == testBaseline.ysize(), "Baseline dimensions differ.");
    }
    if (trainHasBaseline) {
        CB_ENSURE((trainBaseline.ysize() > 1) == IsMultiClassError(lossFunction), "Loss-function is MultiClass iff baseline dimension > 1");
    }

    if (testDocs != 0) {
        CB_ENSURE(testBaseline.ysize() == trainBaseline.ysize(), "train pool baseline dimension == " << trainBaseline.ysize() << " and test pool baseline dimension == " << testBaseline.ysize());
    }
}

void PreprocessAndCheck(const NCatboostOptions::TLossDescription& lossDescription,
                        int learnSampleCount,
                        const TVector<ui32>& queryId,
                        const TVector<TPair>& pairs,
                        const TVector<float>& classWeights,
                        TVector<float>* weights,
                        TVector<float>* target) {
    CB_ENSURE(learnSampleCount != 0, "Train dataset is empty");

    if (lossDescription.GetLossFunction() == ELossFunction::Logloss) {
        PrepareTargetBinary(NCatboostOptions::GetLogLossBorder(lossDescription), target);
    }

    float minWeight = *MinElement(weights->begin(), weights->begin() + learnSampleCount);
    float maxWeight = *MaxElement(weights->begin(), weights->begin() + learnSampleCount);
    CB_ENSURE(minWeight >= 0, "Has negative weight: " + ToString(minWeight));
    CB_ENSURE(maxWeight > 0, "All weights are 0");

    if (lossDescription.GetLossFunction() == ELossFunction::PairLogit) {
        CB_ENSURE(minWeight == maxWeight, "Pairwise loss doesn't support document weights");
    }

    if (!classWeights.empty()) {
        // TODO(annaveronika): check class weight not negative.
        int dataSize = target->ysize();
        for (int i = 0; i < dataSize; ++i) {
            CB_ENSURE(target->at(i) < classWeights.ysize(), "class " + ToString((*target)[i]) + " is missing in class weights");
            (*weights)[i] *= classWeights[(*target)[i]];
        }
    }

    CheckTarget(*target, learnSampleCount, lossDescription.GetLossFunction());

    bool hasQuery = !queryId.empty();
    if (hasQuery) {
        bool isGroupIdCorrect = IsCorrectQueryIdsFormat(queryId, 0, learnSampleCount);
        if (learnSampleCount < target->ysize()) {
            isGroupIdCorrect &= queryId[learnSampleCount - 1] != queryId[learnSampleCount];
        }
        CB_ENSURE(isGroupIdCorrect, "If GroupId is provided then train Pool & Test Pool should be grouped by GroupId and should have different GroupId");
    }
    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        CB_ENSURE(!queryId.empty(), "You should provide QueryId for Pairwise Errors." );
        CB_ENSURE(ArePairsGroupedByQuery(queryId, pairs), "Pairs should have same QueryId");
    }

}

TTrainData BuildTrainData(ELossFunction lossFunction, const TPool& learnPool, const TPool& testPool) {
    TTrainData trainData;
    trainData.LearnSampleCount = learnPool.Docs.GetDocCount();
    trainData.Target.reserve(learnPool.Docs.GetDocCount() + testPool.Docs.GetDocCount());
    trainData.Pairs.reserve(learnPool.Pairs.size() + testPool.Pairs.size());
    trainData.Pairs.insert(trainData.Pairs.end(), learnPool.Pairs.begin(), learnPool.Pairs.end());
    trainData.Pairs.insert(trainData.Pairs.end(), testPool.Pairs.begin(), testPool.Pairs.end());
    for (int pairInd = learnPool.Pairs.ysize(); pairInd < trainData.Pairs.ysize(); ++pairInd) {
        trainData.Pairs[pairInd].WinnerId += trainData.LearnSampleCount;
        trainData.Pairs[pairInd].LoserId += trainData.LearnSampleCount;
    }
    trainData.Target = learnPool.Docs.Target;
    trainData.Weights = learnPool.Docs.Weight;
    trainData.QueryId = learnPool.Docs.QueryId;
    trainData.Baseline = learnPool.Docs.Baseline;
    trainData.Target.insert(trainData.Target.end(), testPool.Docs.Target.begin(), testPool.Docs.Target.end());
    trainData.Weights.insert(trainData.Weights.end(), testPool.Docs.Weight.begin(), testPool.Docs.Weight.end());
    trainData.QueryId.insert(trainData.QueryId.end(), testPool.Docs.QueryId.begin(), testPool.Docs.QueryId.end());

    CheckBaseline(lossFunction, learnPool.Docs.Baseline, testPool.Docs.Baseline, testPool.Docs.GetDocCount());

    for (int dim = 0; dim < testPool.Docs.GetBaselineDimension(); ++dim) {
        trainData.Baseline[dim].insert(trainData.Baseline[dim].end(), testPool.Docs.Baseline[dim].begin(), testPool.Docs.Baseline[dim].end());
    }
    return trainData;
}
