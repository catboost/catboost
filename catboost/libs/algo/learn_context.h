#pragma once

#include "online_ctr.h"
#include "features_layout.h"
#include "fold.h"
#include "ctr_helper.h"
#include "split.h"
#include "calc_score_cache.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/helpers/restorable_rng.h>

#include <library/json/json_reader.h>
#include <library/threading/local_executor/local_executor.h>

#include <library/par/par.h>

#include <util/generic/noncopyable.h>
#include <util/generic/hash_set.h>


struct TLearnProgress {
    TVector<TFold> Folds;
    TFold AveragingFold;
    TVector<TVector<double>> AvrgApprox;
    TVector<TVector<double>> TestApprox;

    TVector<TCatFeature> CatFeatures;
    TVector<TFloatFeature> FloatFeatures;
    int ApproxDimension = 1;
    TString SerializedTrainParams; // TODO(kirillovs): do something with this field

    TVector<TSplitTree> TreeStruct;
    TVector<TTreeStats> TreeStats;
    TVector<TVector<TVector<double>>> LeafValues; // [numTree][dim][bucketId]

    TVector<TVector<double>> LearnErrorsHistory;
    TVector<TVector<double>> TestErrorsHistory;
    TVector<TVector<double>> TimeHistory;

    THashSet<std::pair<ECtrType, TProjection>> UsedCtrSplits;

    ui32 PoolCheckSum = 0;

    void Save(IOutputStream* s) const;
    void Load(IInputStream* s);
};

class TCommonContext : public TNonCopyable {
public:
    TCommonContext(const NJson::TJsonValue& jsonParams,
                   const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                   const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                   int featureCount,
                   const std::vector<int>& catFeatures,
                   const TVector<TString>& featureId)
        : Params(NCatboostOptions::LoadOptions(jsonParams))
        , ObjectiveDescriptor(objectiveDescriptor)
        , EvalMetricDescriptor(evalMetricDescriptor)
        , Layout(featureCount, catFeatures, featureId)
        , CatFeatures(catFeatures.begin(), catFeatures.end()) {
        LocalExecutor.RunAdditionalThreads(Params.SystemOptions->NumThreads - 1);
        CB_ENSURE(static_cast<ui32>(LocalExecutor.GetThreadCount()) == Params.SystemOptions->NumThreads - 1);
    }

public:
    NCatboostOptions::TCatBoostOptions Params;
    const TMaybe<TCustomObjectiveDescriptor> ObjectiveDescriptor;
    const TMaybe<TCustomMetricDescriptor> EvalMetricDescriptor;
    TFeaturesLayout Layout;
    THashSet<int> CatFeatures;
    TCtrHelper CtrsHelper;
    // TODO(asaitgalin): local executor should be shared by all contexts
    NPar::TLocalExecutor LocalExecutor;
};

class TOutputFiles {
public:
    TOutputFiles(const NCatboostOptions::TOutputFilesOptions& params,
                 const TString& namesPrefix) {
        InitializeFiles(params, namesPrefix);
    }
    TString NamesPrefix;
    TString TimeLeftLogFile;
    TString LearnErrorLogFile;
    TString TestErrorLogFile;
    TString SnapshotFile;
    TString MetaFile;
    TString JsonLogFile;
    TString ProfileLogFile;
    static TString AlignFilePath(const TString& baseDir, const TString& fileName, const TString& namePrefix = "");

private:
    void InitializeFiles(const NCatboostOptions::TOutputFilesOptions& params, const TString& namesPrefix);
};

/************************************************************************/
/* Class for storing learn specific data structures like:               */
/* prng, learn progress and target classifiers                          */
/************************************************************************/
class TLearnContext : public TCommonContext {
public:
    TLearnContext(const NJson::TJsonValue& jsonParams,
                  const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                  const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                  const NCatboostOptions::TOutputFilesOptions& outputOptions,
                  int featureCount,
                  const std::vector<int>& catFeatures,
                  const TVector<TString>& featuresId,
                  const TString& fileNamesPrefix = "")
        : TCommonContext(jsonParams, objectiveDescriptor, evalMetricDescriptor, featureCount, catFeatures, featuresId)
        , Rand(Params.RandomSeed)
        , OutputOptions(outputOptions)
        , Files(outputOptions, fileNamesPrefix)
        , RootEnvironment(nullptr)
        , SharedTrainData(nullptr)
        , Profile((int)Params.BoostingOptions->IterationCount) {
        LearnProgress.SerializedTrainParams = ToString(Params);
        ETaskType taskType = Params.GetTaskType();
        CB_ENSURE(taskType == ETaskType::CPU, "Error: except learn on CPU task type, got " << taskType);
    }
    ~TLearnContext();

    void OutputMeta();
    void InitContext(const TDataset& learnData, const TDataset* testData);
    void SaveProgress();
    bool TryLoadProgress();

public:
    TRestorableFastRng64 Rand;
    TLearnProgress LearnProgress;
    NCatboostOptions::TOutputFilesOptions OutputOptions;
    TOutputFiles Files;

    TCalcScoreFold SmallestSplitSideDocs;
    TCalcScoreFold SampledDocs;
    TBucketStatsCache PrevTreeLevelStats;
    TObj<NPar::IRootEnvironment> RootEnvironment;
    TObj<NPar::IEnvironment> SharedTrainData;
    TProfileInfo Profile;
};

NJson::TJsonValue GetJsonMeta(
    int iterationCount,
    const TString& optionalExperimentName,
    const TVector<const IMetric*>& metrics,
    const TVector<TString>& learnSetNames,
    const TVector<TString>& testSetNames,
    ELaunchMode launchMode
);
