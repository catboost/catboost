#pragma once

#include "online_ctr.h"
#include <catboost/libs/data_new/features_layout.h>
#include "fold.h"
#include "ctr_helper.h"
#include "split.h"
#include "calc_score_cache.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/helpers/restorable_rng.h>

#include <library/json/json_reader.h>
#include <library/threading/local_executor/local_executor.h>

#include <library/par/par.h>

#include <util/generic/noncopyable.h>
#include <util/generic/hash_set.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>


struct TLearnProgress {
    TVector<TFold> Folds;
    TFold AveragingFold;
    TVector<TVector<double>> AvrgApprox;          //       [dim][docIdx]
    TVector<TVector<TVector<double>>> TestApprox; // [test][dim][docIdx]
    TVector<TVector<double>> BestTestApprox;      //       [dim][docIdx]

    TVector<TCatFeature> CatFeatures;
    TVector<TFloatFeature> FloatFeatures;

    int ApproxDimension = 1;
    TLabelConverter LabelConverter;

    TString SerializedTrainParams; // TODO(kirillovs): do something with this field

    TVector<TSplitTree> TreeStruct;
    TVector<TTreeStats> TreeStats;
    TVector<TVector<TVector<double>>> LeafValues; // [numTree][dim][bucketId]

    TMetricsAndTimeLeftHistory MetricsAndTimeHistory;

    THashSet<std::pair<ECtrType, TProjection>> UsedCtrSplits;

    ui32 PoolCheckSum = 0;

    void Save(IOutputStream* s) const;
    void Load(IInputStream* s);
};

class TCommonContext : public TNonCopyable {
public:
    TCommonContext(const NCatboostOptions::TCatBoostOptions& params,
                   const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                   const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                   int featureCount,
                   const TVector<int>& catFeatures,
                   const TVector<TString>& featureId)
        : Params(params)
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
    NCB::TFeaturesLayout Layout;
    THashSet<int> CatFeatures;
    TCtrHelper CtrsHelper;
    // TODO(asaitgalin): local executor should be shared by all contexts
    NPar::TLocalExecutor LocalExecutor;
};



/************************************************************************/
/* Class for storing learn specific data structures like:               */
/* prng, learn progress and target classifiers                          */
/************************************************************************/
class TLearnContext : public TCommonContext {
public:
    TLearnContext(const NCatboostOptions::TCatBoostOptions& params,
                  const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                  const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                  const NCatboostOptions::TOutputFilesOptions& outputOptions,
                  int featureCount,
                  const TVector<int>& catFeatures,
                  const TVector<TString>& featuresId,
                  const TString& fileNamesPrefix = "")
        : TCommonContext(params, objectiveDescriptor, evalMetricDescriptor, featureCount, catFeatures, featuresId)
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
    void InitContext(const TDataset& learnData, const TDatasetPtrs& testDataPtrs);
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

