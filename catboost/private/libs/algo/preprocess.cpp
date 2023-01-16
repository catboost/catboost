#include "preprocess.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/defaults_helper.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/xrange.h>
#include <util/system/fs.h>


using namespace NCB;


static void CheckTestBaseline(
    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> trainBaseline,
    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> testBaseline,
    size_t testIdx) {

    size_t testDocs = testBaseline ? (*testBaseline)[0].size() : 0;
    if (trainBaseline) {
        CB_ENSURE(testBaseline, "Baseline for test is not provided");
    }
    if (testBaseline) {
        CB_ENSURE(trainBaseline, "Baseline for train is not provided");
    }
    if (testDocs != 0) {
        CB_ENSURE(
            trainBaseline->size() == testBaseline->size(),
            "Baseline dimension differ: train: " << trainBaseline->size() << " vs test["
             << testIdx << "]: " << testBaseline->size()
        );
    }
}


void CheckConsistency(const TTrainingDataProviders& data) {
    auto learnBaseline = data.Learn->TargetData->GetBaseline();
    for (auto testIdx : xrange(data.Test.size())) {
        CheckTestBaseline(learnBaseline, data.Test[testIdx]->TargetData->GetBaseline(), testIdx);
    }
}

void UpdateUndefinedRandomSeed(
    ETaskType taskType,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    NJson::TJsonValue* updatedJsonParams,
    std::function<void(TIFStream*, TString&)> paramsLoader) {

    const TString snapshotFilename = TOutputFiles::AlignFilePath(
        outputOptions.GetTrainDir(),
        outputOptions.GetSnapshotFilename(),
        /*namePrefix=*/ ""
    );
    if (outputOptions.SaveSnapshot() && NFs::Exists(snapshotFilename)) {
        TString serializedTrainParams;
        NJson::TJsonValue restoredJsonParams;
        try {
            TProgressHelper(ToString(taskType)).CheckedLoad(
                snapshotFilename,
                [&](TIFStream* inputStream) {
                    paramsLoader(inputStream, serializedTrainParams);
                }
            );
            ReadJsonTree(serializedTrainParams, &restoredJsonParams);
            CB_ENSURE(restoredJsonParams.Has("random_seed"), "Snapshot is broken.");
        } catch (const TCatBoostException&) {
            throw;
        } catch (...) {
            CATBOOST_WARNING_LOG << "Can't load progress from snapshot file: " << snapshotFilename <<
                " Exception: " << CurrentExceptionMessage() << Endl;
            return;
        }

        if (!(*updatedJsonParams)["flat_params"].Has("random_seed") &&
            !restoredJsonParams["flat_params"].Has("random_seed"))
        {
            (*updatedJsonParams)["random_seed"] = restoredJsonParams["random_seed"];
        }
    }
}

void UpdateUndefinedClassLabels(
    const TVector<NJson::TJsonValue>& classLabels,
    NJson::TJsonValue* updatedJsonParams) {

    if (!updatedJsonParams->Has("data_processing_options")) {
        updatedJsonParams->InsertValue("data_processing_options", NJson::TJsonValue());
    }
    if (classLabels.empty()) {
        return;
    }
    (*updatedJsonParams)["data_processing_options"]["class_names"] = {};
    for (const auto& classLabel : classLabels) {
        (*updatedJsonParams)["data_processing_options"]["class_names"].AppendValue(classLabel);
    }
}

static void CheckTimestampsInEachGroup(
    TConstArrayRef<TGroupBounds> groupsBounds,
    TConstArrayRef<ui64> timestamps
) {
    for (auto groupBounds : groupsBounds) {
        if (groupBounds.GetSize()) {
            ui64 groupTimestamp = timestamps[groupBounds.Begin];
            for (auto objectIdx : xrange(groupBounds.Begin + 1, groupBounds.End)) {
                CB_ENSURE(
                    groupTimestamp == timestamps[objectIdx],
                    "timestamps[" << objectIdx << "] = " << timestamps[objectIdx]
                    << " is not equal to the timestamp of group's first element "
                    << " (timestamps[" << groupBounds.Begin << "] = " << groupTimestamp << ")."
                    << " CatBoost supports training only with groups with the same timestamp for each element."
                );
            }
        }
    }
}

TDataProviderPtr ReorderByTimestampLearnDataIfNeeded(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    TDataProviderPtr learnData,
    NPar::ILocalExecutor* localExecutor) {

    if (catBoostOptions.DataProcessingOptions->HasTimeFlag &&
        learnData->MetaInfo.HasTimestamp &&
        (learnData->ObjectsData->GetOrder() != EObjectsOrder::Ordered))
    {
        auto objectsGrouping = learnData->ObjectsData->GetObjectsGrouping();

        if (!objectsGrouping->IsTrivial()) {
            CheckTimestampsInEachGroup(
                objectsGrouping->GetNonTrivialGroups(),
                *learnData->ObjectsData->GetTimestamp()
            );
        }

        auto objectsPermutation = CreateOrderByKey<ui32>(*learnData->ObjectsData->GetTimestamp());

        return learnData->GetSubset(
            GetSubset(
                objectsGrouping,
                TArraySubsetIndexing<ui32>(std::move(objectsPermutation)),
                EObjectsOrder::Ordered
            ),
            ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
            localExecutor
        );
    }
    return learnData;
}


static bool NeedShuffle(
    const ui32 catFeatureCount,
    const ui32 docCount,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions) {

    if (catBoostOptions.DataProcessingOptions->HasTimeFlag) {
        return false;
    }

    if (catFeatureCount == 0) {
        NCatboostOptions::TCatBoostOptions updatedOpitons(catBoostOptions);
        UpdateBoostingTypeOption(docCount, &updatedOpitons);
        if (updatedOpitons.BoostingOptions->BoostingType == EBoostingType::Ordered) {
            return true;
        } else {
            return false;
        }
    } else {
        return true;
    }
}

TDataProviderPtr ShuffleLearnDataIfNeeded(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    TDataProviderPtr learnData,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand) {

    if (NeedShuffle(
        learnData->MetaInfo.FeaturesLayout->GetCatFeatureCount(),
        learnData->ObjectsData->GetObjectCount(),
        catBoostOptions
    )) {
        auto objectsGroupingSubset = NCB::Shuffle(learnData->ObjectsGrouping, 1, rand);
        return learnData->GetSubset(
            objectsGroupingSubset,
            ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
            localExecutor
        );
    }
    return learnData;
}
