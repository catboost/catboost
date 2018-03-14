#include "preprocess.h"

#include <catboost/libs/metrics/metric.h>

static int CountGroups(const TVector<ui32>& queryIds) {
    if (queryIds.empty()) {
        return 0;
    }
    int result = 1;
    ui32 id = queryIds[0];
    for (int i = 1; i < queryIds.ysize(); ++i) {
       if (queryIds[i] != id) {
           result++;
           id = queryIds[i];
       }
    }
    return result;
}

static bool AreQueriesGrouped(const TVector<ui32>& queryIds) {
    int groupCount = CountGroups(queryIds);

    auto queryIdsCopy = queryIds;
    Sort(queryIdsCopy.begin(), queryIdsCopy.end());
    int sortedGroupCount = CountGroups(queryIdsCopy);
    return groupCount == sortedGroupCount;
}

static bool ArePairsGroupedByQuery(const TVector<ui32>& queryId, const TVector<TPair>& pairs) {
    for (const auto& pair : pairs) {
        if (queryId[pair.WinnerId] != queryId[pair.LoserId]) {
            return false;
        }
    }
    return true;
}

void CheckTrainTarget(const TVector<float>& target, int learnSampleCount, ELossFunction lossFunction) {
    CheckTarget(target, lossFunction);
    if (lossFunction == ELossFunction::Logloss) {
        float minTarget = *MinElement(target.begin(), target.begin() + learnSampleCount);
        float maxTarget = *MaxElement(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(minTarget == 0, "All train targets are greater than border");
        CB_ENSURE(maxTarget == 1, "All train targets are smaller than border");
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

    CheckTrainTarget(*target, learnSampleCount, lossDescription.GetLossFunction());

    bool hasQuery = !queryId.empty();
    if (hasQuery) {
        bool isGroupIdCorrect = AreQueriesGrouped(queryId);
        if (learnSampleCount < target->ysize()) {
            isGroupIdCorrect &= queryId[learnSampleCount - 1] != queryId[learnSampleCount];
        }
        CB_ENSURE(isGroupIdCorrect, "Train and eval group ids should have distinct group ids. And group ids in train and eval should be grouped.");
    }
    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        CB_ENSURE(!queryId.empty(), "You should provide GroupId for Pairwise Errors." );
        CB_ENSURE(ArePairsGroupedByQuery(queryId, pairs), "Two objects in a pair should have same GroupId");
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
    trainData.SubgroupId = learnPool.Docs.SubgroupId;
    trainData.Baseline = learnPool.Docs.Baseline;
    trainData.Target.insert(trainData.Target.end(), testPool.Docs.Target.begin(), testPool.Docs.Target.end());
    trainData.Weights.insert(trainData.Weights.end(), testPool.Docs.Weight.begin(), testPool.Docs.Weight.end());
    trainData.QueryId.insert(trainData.QueryId.end(), testPool.Docs.QueryId.begin(), testPool.Docs.QueryId.end());
    trainData.SubgroupId.insert(trainData.SubgroupId.end(), testPool.Docs.SubgroupId.begin(), testPool.Docs.SubgroupId.end());

    CheckBaseline(lossFunction, learnPool.Docs.Baseline, testPool.Docs.Baseline, testPool.Docs.GetDocCount());

    for (int dim = 0; dim < testPool.Docs.GetBaselineDimension(); ++dim) {
        trainData.Baseline[dim].insert(trainData.Baseline[dim].end(), testPool.Docs.Baseline[dim].begin(), testPool.Docs.Baseline[dim].end());
    }
    return trainData;
}
