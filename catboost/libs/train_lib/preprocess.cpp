#include "preprocess.h"

#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/metrics/metric.h>

static int CountGroups(const TVector<TGroupId>& queryIds) {
    if (queryIds.empty()) {
        return 0;
    }
    int result = 1;
    TGroupId id = queryIds[0];
    for (int i = 1; i < queryIds.ysize(); ++i) {
       if (queryIds[i] != id) {
           result++;
           id = queryIds[i];
       }
    }
    return result;
}

static bool AreQueriesGrouped(const TVector<TGroupId>& queryIds) {
    int groupCount = CountGroups(queryIds);

    auto queryIdsCopy = queryIds;
    Sort(queryIdsCopy.begin(), queryIdsCopy.end());
    int sortedGroupCount = CountGroups(queryIdsCopy);
    return groupCount == sortedGroupCount;
}

static bool ArePairsGroupedByQuery(const TVector<TGroupId>& queryId, const TVector<TPair>& pairs) {
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
        auto targetBounds = CalcMinMax(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(targetBounds.Min == 0, "All train targets are greater than border");
        CB_ENSURE(targetBounds.Max == 1, "All train targets are smaller than border");
    }

    if (lossFunction != ELossFunction::PairLogit) {
        auto targetBounds = CalcMinMax(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(targetBounds.Min != targetBounds.Max, "All train targets are equal");
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

void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                TDataset& learnOrTestData) {
    auto& data = learnOrTestData;
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

TDataset BuildTrainData(const TPool& pool) {
    TDataset data;
    data.Target = pool.Docs.Target;
    data.Weights = pool.Docs.Weight;
    data.QueryId = pool.Docs.QueryId;
    data.SubgroupId = pool.Docs.SubgroupId;
    data.Baseline = pool.Docs.Baseline;
    data.Pairs = pool.Pairs;
    return data;
}

void CheckConsistency(const NCatboostOptions::TLossDescription& lossDescription,
                      const TDataset& learnData,
                      const TDataset& testData) {
    CB_ENSURE(learnData.Target.size() > 0, "Train dataset is empty");

    CheckBaseline(lossDescription.GetLossFunction(), learnData.Baseline, testData.Baseline);

    TMinMax<float> weightBounds = CalcMinMax(learnData.Weights);
    CB_ENSURE(weightBounds.Min >= 0, "Has negative weight: " + ToString(weightBounds.Min));
    CB_ENSURE(weightBounds.Max > 0, "All weights are 0");

    if (lossDescription.GetLossFunction() == ELossFunction::PairLogit) {
        if (weightBounds.Min != weightBounds.Max) {
            MATRIXNET_WARNING_LOG << "Pairwise loss doesn't support document weights. They will be ignored in optimization. If a custom metric is specified then they will be used for custom metric calculation." << Endl;
        }
    }

    CheckTrainTarget(learnData.Target, learnData.Target.size(), lossDescription.GetLossFunction());

    bool learnHasQuery = !learnData.QueryId.empty();
    bool testHasQuery = !testData.QueryId.empty();

    if (learnHasQuery) {
        CB_ENSURE(AreQueriesGrouped(learnData.QueryId), "Train pool should be grouped by GroupId");
        if (testHasQuery) {
            CB_ENSURE(AreQueriesGrouped(testData.QueryId), "Test pool should be grouped by GroupId");
            CB_ENSURE(learnData.QueryId.back() != testData.QueryId.front(), " Train and test pools should have different GroupId");
        }
    }

    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        CB_ENSURE(!learnData.Pairs.empty() || lossDescription.GetLossFunction() == ELossFunction::YetiRank, "You should provide learn pairs for Pairwise Errors.");
        CB_ENSURE(learnHasQuery, "You should provide GroupId for Pairwise Errors.");
        CB_ENSURE(ArePairsGroupedByQuery(learnData.QueryId, learnData.Pairs), "Pairs should have same QueryId");
        CB_ENSURE(ArePairsGroupedByQuery(testData.QueryId, testData.Pairs), "Pairs should have same QueryId");
    }
}
