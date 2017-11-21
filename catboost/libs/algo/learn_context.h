#pragma once

#include "calc_score_cache.h"
#include "online_ctr.h"
#include "priors.h"
#include "features_layout.h"
#include "fold.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/params/params.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>

#include <library/json/json_reader.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/noncopyable.h>

struct TLearnProgress {
    TVector<TFold> Folds;
    TFold AveragingFold;
    TVector<TVector<double>> AvrgApprox;

    TVector<TCatFeature> CatFeatures;
    TVector<TFloatFeature> FloatFeatures;
    int ApproxDimension = 1;
    TString SerializedTrainParams; // TODO(kirillovs): do something with this field

    TVector<TSplitTree> TreeStruct;
    TVector<TVector<TVector<double>>> LeafValues; // [numTree][dim][bucketId]

    TVector<TVector<double>> LearnErrorsHistory, TestErrorsHistory;
    bool StoreExpApprox;

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
        : Params(jsonParams, objectiveDescriptor, evalMetricDescriptor, &ResultingParams)
        , Layout(featureCount, catFeatures, featureId)
        , CatFeatures(catFeatures.begin(), catFeatures.end()) {
        LocalExecutor.RunAdditionalThreads(Params.ThreadCount - 1);
        Priors.Init(Params.CtrParams.DefaultPriors, Params.CtrParams.DefaultCounterPriors, Params.CtrParams.PerFeaturePriors, Params.CtrParams.Ctrs, Layout);
    }

public:
    // TODO(kirillovs): remove ResultingParams when protobuf or flatbuf params
    // serializer will be implemented
    NJson::TJsonValue ResultingParams;

    TFitParams Params;
    TFeaturesLayout Layout;
    THashSet<int> CatFeatures;
    TPriors Priors;
    // TODO(asaitgalin): local executor should be shared by all contexts
    NPar::TLocalExecutor LocalExecutor;
};

class TOutputFiles {
public:
    TOutputFiles(const TFitParams& params, const TString& namesPrefix) {
        InitializeFiles(params, namesPrefix);
    }
    TString NamesPrefix;
    TString TimeLeftLogFile;
    TString LearnErrorLogFile;
    TString TestErrorLogFile;
    TString SnapshotFile;
    TString MetaFile;
    static TString AlignFilePath(const TString& baseDir, const TString& fileName, const TString& namePrefix = "");

private:
    void InitializeFiles(const TFitParams& params, const TString& namesPrefix);
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
                  int featureCount,
                  const std::vector<int>& catFeatures,
                  const TVector<TString>& featuresId,
                  const TString& fileNamesPrefix = "")
        : TCommonContext(jsonParams, objectiveDescriptor, evalMetricDescriptor, featureCount, catFeatures, featuresId)
        , Rand(Params.RandomSeed)
        , Files(Params, fileNamesPrefix)
        , TimeLeftLog(Params.AllowWritingFiles ? new TOFStream(Files.TimeLeftLogFile) : nullptr)
        , Profile(Params.DetailedProfile, Params.Iterations, TimeLeftLog.Get()) {
        LearnProgress.SerializedTrainParams = ToString(ResultingParams);
        LearnProgress.StoreExpApprox = Params.StoreExpApprox;
    }

    void OutputMeta();
    void InitData(const TTrainData& data);
    void SaveProgress();
    bool TryLoadProgress();

public:
    TRestorableFastRng64 Rand;
    TLearnProgress LearnProgress;
    TOutputFiles Files;

    TSmallestSplitSideFold ParamsUsedWithStatsFromPrevTree;
    TStatsFromPrevTree StatsFromPrevTree;

private:
    THolder<TOFStream> TimeLeftLog;

public:
    TProfileInfo Profile;

};
