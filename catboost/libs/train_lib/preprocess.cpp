#include "preprocess.h"

#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/helpers/restorable_rng.h>

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

static void CheckGroupWeightCorrectness(const TVector<float>& groupWeight, const TVector<TGroupId>& groupId) {
    TGroupId previousGroupId = groupId[0];
    float previousGroupWeight = groupWeight[0];
    for (int i = 1; i < groupId.ysize(); ++i) {
        if (previousGroupId == groupId[i] && (previousGroupWeight == groupWeight[i])) {
            continue;
        } else if (previousGroupId != groupId[i]) {
            previousGroupId = groupId[i];
            previousGroupWeight = groupWeight[i];
        } else {
            CB_ENSURE(false, "Objects from the same group should have the same QueryWeight.");
        }
    }
}

void CheckTrainTarget(const TVector<float>& target, int learnSampleCount, ELossFunction lossFunction, bool allowConstLabel) {
    CheckTarget(target, lossFunction);
    if (lossFunction == ELossFunction::Logloss) {
        auto targetBounds = CalcMinMax(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE(targetBounds.Min == 0, "All train targets are greater than border");
        CB_ENSURE(targetBounds.Max == 1, "All train targets are smaller than border");
    }

    if (lossFunction != ELossFunction::PairLogit) {
        auto targetBounds = CalcMinMax(target.begin(), target.begin() + learnSampleCount);
        CB_ENSURE((targetBounds.Min != targetBounds.Max) || allowConstLabel, "All train targets are equal");
    }
}

static void CheckTrainBaseline(ELossFunction lossFunction, const TVector<TVector<double>>& trainBaseline) {
    if (trainBaseline.ysize() > 1) {
        CB_ENSURE(IsMultiClassError(lossFunction), "Loss-function is MultiClass iff baseline dimension > 1");
    }
}

static void CheckTestBaseline(
    ELossFunction lossFunction,
    const TVector<TVector<double>>& trainBaseline,
    const TVector<TVector<double>>& testBaseline
) {
    size_t testDocs = testBaseline.size() ? testBaseline[0].size() : 0;
    bool trainHasBaseline = trainBaseline.ysize() != 0;
    bool testHasBaseline = testDocs == 0 ? trainHasBaseline : testBaseline.ysize() != 0;
    if (trainHasBaseline) {
        CB_ENSURE(testHasBaseline, "Baseline for test is not provided");
    }
    if (testHasBaseline) {
        CB_ENSURE(trainHasBaseline, "Baseline for train is not provided");
    }
    if (trainBaseline.ysize() > 1) {
        CB_ENSURE(IsMultiClassError(lossFunction), "Loss-function is MultiClass iff baseline dimension > 1");
    }
    if (testDocs != 0) {
        CB_ENSURE(trainBaseline.ysize() == testBaseline.ysize(), "Baseline dimension differ: train " << trainBaseline.ysize() << " vs test " << testBaseline.ysize());
    }
}

void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                const TLabelConverter& labelConverter,
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

    if (IsMultiClassError(lossDescription.GetLossFunction())) {
        PrepareTargetCompressed(labelConverter, &data.Target);
    }
}

void CheckLearnConsistency(
    const NCatboostOptions::TLossDescription& lossDescription,
    bool allowConstLabel,
    const TDataset& learnData
) {
    CB_ENSURE(learnData.Target.size() > 0, "Train dataset is empty");

    CheckTrainBaseline(lossDescription.GetLossFunction(), learnData.Baseline);

    TMinMax<float> weightBounds = CalcMinMax(learnData.Weights);
    CB_ENSURE(weightBounds.Min >= 0, "Has negative weight: " + ToString(weightBounds.Min));
    CB_ENSURE(weightBounds.Max > 0, "All weights are 0");

    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        if (weightBounds.Min != weightBounds.Max && !learnData.HasGroupWeight) {
            MATRIXNET_WARNING_LOG << "Pairwise losses don't support document weights. They will be ignored in optimization. If a custom metric is specified then they will be used for custom metric calculation." << Endl;
        }
    }

    CheckTrainTarget(learnData.Target, learnData.Target.size(), lossDescription.GetLossFunction(), allowConstLabel);

    bool learnHasQuery = !learnData.QueryId.empty();

    if (learnHasQuery) {
        CB_ENSURE(AreQueriesGrouped(learnData.QueryId), "Train pool should be grouped by GroupId");
    }

    if (learnData.HasGroupWeight) {
        CheckGroupWeightCorrectness(learnData.Weights, learnData.QueryId);
    }

    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        CB_ENSURE(!learnData.Pairs.empty() || ShouldGenerateYetiRankPairs(lossDescription.GetLossFunction()),
            "You should provide learn pairs for Pairwise Errors.");
        CB_ENSURE(learnHasQuery, "You should provide GroupId for Pairwise Errors.");
        CB_ENSURE(ArePairsGroupedByQuery(learnData.QueryId, learnData.Pairs), "Pairs should have same QueryId");
    }
}

void CheckTestConsistency(const NCatboostOptions::TLossDescription& lossDescription,
                          const TDataset& learnData,
                          const TDataset& testData) {
    CheckTestBaseline(lossDescription.GetLossFunction(), learnData.Baseline, testData.Baseline);

    bool learnHasQuery = !learnData.QueryId.empty();
    bool testHasQuery = !testData.QueryId.empty();

    if (learnHasQuery && testHasQuery) {
        CB_ENSURE(AreQueriesGrouped(testData.QueryId), "Test pool should be grouped by GroupId");
        CB_ENSURE(learnData.QueryId.back() != testData.QueryId.front(), " Train and test pools should have different GroupId");
    }

    if (testData.HasGroupWeight) {
        CheckGroupWeightCorrectness(testData.Weights, testData.QueryId);
    }

    if (IsPairwiseError(lossDescription.GetLossFunction())) {
        CB_ENSURE(ArePairsGroupedByQuery(testData.QueryId, testData.Pairs), "Pairs should have same QueryId");
    }
}

void UpdateUndefinedRandomSeed(const NCatboostOptions::TOutputFilesOptions& outputOptions, NJson::TJsonValue* updatedJsonParams) {
    const TString snapshotFilename = TOutputFiles::AlignFilePath(outputOptions.GetTrainDir(), outputOptions.GetSnapshotFilename(), /*namePrefix=*/"");
    if (NFs::Exists(snapshotFilename)) {
        TIFStream inputStream(snapshotFilename);

        TString unusedLabel;
        TRestorableFastRng64 unusedRng(0);
        TString serializedTrainParams;
        NJson::TJsonValue restoredJsonParams;
        try {
            ::LoadMany(&inputStream, unusedLabel, unusedRng, serializedTrainParams);
            ReadJsonTree(serializedTrainParams, &restoredJsonParams);
            CB_ENSURE(restoredJsonParams.Has("random_seed"), "Snapshot is broken.");
        } catch (...) {
            CB_ENSURE(false, "Can't load progress from snapshot file: " << snapshotFilename <<
                    " Exception: " << CurrentExceptionMessage() << Endl);
            return;
        }

        if (!(*updatedJsonParams)["flat_params"].Has("random_seed") && !restoredJsonParams["flat_params"].Has("random_seed")) {
            (*updatedJsonParams)["random_seed"] = restoredJsonParams["random_seed"];
        }
    }
}
