#include "preprocess.h"

#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/metrics/metric.h>

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

static bool IsCorrectQueryIdsFormat(const TVector<ui32>& queryIds) {
    return IsCorrectQueryIdsFormat(queryIds, 0, queryIds.size());
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
                   const TVector<TVector<double>>& testBaseline) {
    size_t testDocs = testBaseline.size() ? testBaseline[0].size() : 0;
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

void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                TTrainData& data) {
    if (lossDescription.GetLossFunction() == ELossFunction::Logloss) {
        PrepareTargetBinary(NCatboostOptions::GetLogLossBorder(lossDescription), &data.Target);
    }

    if (!classWeights.empty()) {
        // TODO(annaveronika): check class weight not negative.
        int dataSize = data.Target.ysize();
        for (int i = 0; i < dataSize; ++i) {
            CB_ENSURE(data.Target[i] < classWeights.ysize(), "class " + ToString(data.Target[i]) + " is missing in class weights");
            data.Weights[i] *= classWeights[data.Target[i]];
        }
    }
}

TTrainData BuildTrainData(const TPool& pool) {
    TTrainData data;
    data.Target = pool.Docs.Target;
    data.Weights = pool.Docs.Weight;
    data.QueryId = pool.Docs.QueryId;
    data.Baseline = pool.Docs.Baseline;
    data.Pairs = pool.Pairs;
    return data;
}

void CheckConsistency1(ELossFunction lossFunction,
                       const TTrainData& learnData,
                       const TTrainData& testData) {
    CheckBaseline(lossFunction, learnData.Baseline, testData.Baseline);
}

void CheckConsistency2(const NCatboostOptions::TLossDescription& lossDescription,
                       const TTrainData& learnData,
                       const TTrainData& testData) {
    CB_ENSURE(learnData.Target.size() > 0, "Train dataset is empty");

    TMinMax<float> weight(learnData.Weights);
    CB_ENSURE(weight.Min >= 0, "Has negative weight: " + ToString(weight.Min));
    CB_ENSURE(weight.Max > 0, "All weights are 0");

    if (lossDescription.GetLossFunction() == ELossFunction::PairLogit) {
        CB_ENSURE(weight.Min == weight.Max, "Pairwise loss doesn't support document weights");
    }

    CheckTrainTarget(learnData.Target, learnData.Target.size(), lossDescription.GetLossFunction());

    bool learnHasQuery = !learnData.QueryId.empty();
    bool testHasQuery = !testData.QueryId.empty();
    CB_ENSURE(learnHasQuery == testHasQuery, "If GroupId is provided then both train and test pool should be grouped by GroupId");

    if (learnHasQuery) {
        CB_ENSURE(IsCorrectQueryIdsFormat(learnData.QueryId), "Train pool should be grouped by GroupId");
        CB_ENSURE(IsCorrectQueryIdsFormat(testData.QueryId), "Test pool should be grouped by GroupId");
        CB_ENSURE(learnData.QueryId.back() != testData.QueryId.front(), " Train and test pools should have different GroupId");
    }

    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        CB_ENSURE(learnHasQuery, "You should provide GroupId for Pairwise Errors." );
        CB_ENSURE(ArePairsGroupedByQuery(learnData.QueryId, learnData.Pairs), "Pairs should have same QueryId");
        CB_ENSURE(ArePairsGroupedByQuery(testData.QueryId, testData.Pairs), "Pairs should have same QueryId");
    }
}
