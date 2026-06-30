#pragma once

#include "calc_score_cache.h"
#include "ctr_helper.h"
#include "fold.h"
#include "online_ctr.h"
#include "split.h"

#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/private/libs/algo_helpers/scratch_cache.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/model/fwd.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/noncopyable.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>


namespace NPar {
    class ILocalExecutor;
}

namespace NCatboostOptions {
    struct TBinarizationOptions;
}


struct TFoldsCreationParams {
    bool IsOrderedBoosting;
    int LearningFoldCount;
    ui32 FoldPermutationBlockSize;
    bool StoreExpApproxes;
    bool HasPairwiseWeights;
    float FoldLenMultiplier;
    bool IsAverageFoldPermuted;
    TMaybe<TVector<double>> StartingApprox;
    ELossFunction LossFunction;

public:
    TFoldsCreationParams(
        const NCatboostOptions::TCatBoostOptions& params,
        const NCB::TQuantizedObjectsDataProvider& learnObjectsData,
        const TMaybe<TVector<double>>& startingApprox,
        bool isForWorkerLocalData);

    ui32 CalcCheckSum(const NCB::TObjectsGrouping& objectsGrouping, NPar::ILocalExecutor* localExecutor) const;
};


struct TLearnProgress {
    TVector<TFold> Folds;
    TFold AveragingFold;
    TVector<TVector<double>> AvrgApprox;          //       [dim][docIdx]
    TVector<TVector<TVector<double>>> TestApprox; // [test][dim][docIdx]
    TVector<TVector<double>> BestTestApprox;      //       [dim][docIdx]
    TMaybe<TVector<double>> StartingApprox;

    /* folds and approx data can become invalid if ShrinkModel is called
     * then this data has to be recalculated if training continuation is called
     */
    bool IsFoldsAndApproxDataValid = true;

    /* used to update AveragingFold faster
     * if AveragingFold is not permuted, we do not use its LearnPermutation
     */
    bool IsAveragingFoldPermuted = false;

    /* used for training continuation
     * if FoldCreationParamsCheckSum is the same it means Folds and AveragingFold can be reused for
     * training continuation (although Target-related data is updated)
     */
    ui32 FoldCreationParamsCheckSum = 0;

    TVector<TCatFeature> CatFeatures;
    TVector<TFloatFeature> FloatFeatures;
    TVector<TTextFeature> TextFeatures;
    TVector<TEmbeddingFeature> EmbeddingFeatures;

    int ApproxDimension = 1;
    TLabelConverter LabelConverter;
    EHessianType HessianType = EHessianType::Symmetric;
    bool EnableSaveLoadApprox = true;

    TString SerializedTrainParams; // TODO(kirillovs): do something with this field

    TVector<std::variant<TSplitTree, TNonSymmetricTreeStructure>> TreeStruct;
    TVector<TTreeStats> TreeStats;
    TVector<TVector<TVector<double>>> LeafValues; // [numTree][dim][bucketId]
    /* Vector of multipliers that were applied to approxes at each iteration.
     * Needed to properly scale leaf values as well at the end of training.
     */
    TVector<double> ModelShrinkHistory; // [numTree]

    ui32 InitTreesSize = 0; // can be non-0 if it is a continuation of training.

    TMetricsAndTimeLeftHistory MetricsAndTimeHistory;

    THashSet<std::pair<ECtrType, TProjection>> UsedCtrSplits;

    ui32 LearnAndTestQuantizedFeaturesCheckSum = 0;

    // separate = not a continuation with tree data in this object.
    ui32 SeparateInitModelTreesSize = 0;
    ui32 SeparateInitModelCheckSum = 0;

    TRestorableFastRng64 Rand;

    TVector<bool> UsedFeatures;
    TMap<ui32, TVector<bool>> UsedFeaturesPerObject;

    NCB::TCombinedEstimatedFeaturesContext EstimatedFeaturesContext;
public:
    TLearnProgress();
    TLearnProgress(
        bool isForWorkerLocalData,
        bool isSingleHost,
        const NCB::TTrainingDataProviders& data,
        int approxDimension,
        const TLabelConverter& labelConverter, // unused if isForWorkerLocalData
        ui64 randomSeed,
        TMaybe<const TRestorableFastRng64*> initRand,
        const TFoldsCreationParams& foldsCreationParams,
        bool datasetsCanContainBaseline,
        const TVector<TTargetClassifier>& targetClassifiers,
        ui32 featuresCheckSum,
        ui32 foldCreationParamsCheckSum,
        const NCatboostOptions::TBinarizationOptions& estimatedFeaturesQuantizationOptions,

        // can be non-empty only if there is single fold
        TMaybe<NCB::TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
        const NCatboostOptions::TObliviousTreeLearnerOptions& trainOptions,
        TMaybe<TFullModel*> initModel,
        NCB::TDataProviders initModelApplyCompatiblePools,
        NPar::ILocalExecutor* localExecutor);

    // call after fold initizalization
    void SetSeparateInitModel(
        const TFullModel& initModel,
        const NCB::TDataProviders& initModelApplyCompatiblePools,
        bool isOrderedBoosting,
        bool storeExpApproxes,
        NPar::ILocalExecutor* localExecutor);

    void PrepareForContinuation();

    void Save(IOutputStream* s) const;
    void Load(IInputStream* s);

    ui32 GetCurrentTrainingIterationCount() const;
    ui32 GetCompleteModelTreesSize() const; // includes init model size if it's a continuation training
    ui32 GetInitModelTreesSize() const;

    NCB::TQuantizedEstimatedFeaturesInfo GetOnlineEstimatedFeaturesInfo() const;
};

class TCommonContext : public TNonCopyable {
public:
    TCommonContext(
        const NCatboostOptions::TCatBoostOptions& params,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        NCB::TFeaturesLayoutPtr layout,
        NPar::ILocalExecutor* localExecutor
    )
        : Params(params)
        , ObjectiveDescriptor(objectiveDescriptor)
        , EvalMetricDescriptor(evalMetricDescriptor)
        , Layout(layout)
        , LocalExecutor(localExecutor)
    {}

public:
    NCatboostOptions::TCatBoostOptions Params;
    const TMaybe<TCustomObjectiveDescriptor> ObjectiveDescriptor;
    const TMaybe<TCustomMetricDescriptor> EvalMetricDescriptor;
    NCB::TFeaturesLayoutPtr Layout;
    TCtrHelper CtrsHelper;
    // TODO(asaitgalin): local executor should be shared by all contexts. MLTOOLS-2451.
    NPar::ILocalExecutor* LocalExecutor;
};



/************************************************************************/
/* Class for storing learn specific data structures like:               */
/* prng, learn progress and target classifiers                          */
/************************************************************************/
class TLearnContext : public TCommonContext {
public:
    TLearnContext(
        const NCatboostOptions::TCatBoostOptions& params,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const NCB::TTrainingDataProviders& data,

        // can be non-empty only if there is single fold
        TMaybe<NCB::TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
        const TLabelConverter& labelConverter,
        const TMaybe<TVector<double>>& startingApprox,
        TMaybe<const TRestorableFastRng64*> initRand,
        TMaybe<TFullModel*> initModel,
        THolder<TLearnProgress> initLearnProgress, // will be modified if not non-nullptr
        NCB::TDataProviders initModelApplyCompatiblePools,
        NPar::ILocalExecutor* localExecutor,
        const TString& fileNamesPrefix = "");

    void SaveProgress(std::function<void(IOutputStream*)> onSaveSnapshot = [] (IOutputStream* /*snapshot*/) {});
    bool TryLoadProgress(std::function<bool(IInputStream*)> onLoadSnapshot = [] (IInputStream* /*snapshot*/) { return true; });
    bool UseTreeLevelCaching() const;
    bool GetHasWeights() const;

public:
    THolder<TLearnProgress> LearnProgress;
    NCatboostOptions::TOutputFilesOptions OutputOptions;
    TOutputFiles Files;

    TCalcScoreFold SmallestSplitSideDocs;
    TCalcScoreFold SampledDocs;
    TBucketStatsCache PrevTreeLevelStats;
    TProfileInfo Profile;

    NCB::TScratchCache ScratchCache;

private:
    bool UseTreeLevelCachingFlag;
    bool HasWeights;
};

bool NeedToUseTreeLevelCaching(
    const NCatboostOptions::TCatBoostOptions& params,
    ui32 maxBodyTailCount,
    ui32 approxDimension);

bool UseAveragingFoldAsFoldZero(const TLearnContext& ctx);
