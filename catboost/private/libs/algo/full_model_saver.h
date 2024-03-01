#pragma once

#include "online_ctr.h"
#include "projection.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/model/fwd.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/private/libs/options/cat_feature_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/text_features/text_processing_collection.h>
#include <catboost/private/libs/embedding_features/embedding_processing_collection.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/system/types.h>

#include <functional>



struct TDatasetDataForFinalCtrs;
struct TMetricsAndTimeLeftHistory;

namespace NCatboostOptions {
    class TCatBoostOptions;
    class TOutputFilesOptions;
}

namespace NCB {
    class TClassificationTargetHelper;
}


namespace NCB {

    class TCoreModelToFullModelConverter {
    private:
        using TGetCoreModelFunc = std::function<TFullModel&()>;

        using TGetBinarizedDataFunc = std::function<
            void(const TFullModel&, TDatasetDataForFinalCtrs*, const THashMap<TFeatureCombination, TProjection>**)
        >;

    public:
        TCoreModelToFullModelConverter(
            const NCatboostOptions::TCatBoostOptions& options,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TClassificationTargetHelper& classificationTargetHelper,
            ui64 ctrLeafCountLimit,
            bool storeAllSimpleCtrs,
            EFinalCtrComputationMode finalCtrComputationMode,
            EFinalFeatureCalcersComputationMode finalFeatureCalcerComputationMode
        );

        TCoreModelToFullModelConverter& WithCoreModelFrom(TFullModel* coreModel);

        TCoreModelToFullModelConverter& WithObjectsDataFrom(TObjectsDataProviderPtr learnObjectsData);

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            TDatasetDataForFinalCtrs&& datasetDataForFinalCtrs,
            THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection
        );

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            const TTrainingDataProviders& trainingData,
            THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection,
            const TVector<TTargetClassifier>& targetClassifiers
        );

        TCoreModelToFullModelConverter& WithPerfectHashedToHashedCatValuesMap(
            const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap
        );

        TCoreModelToFullModelConverter& WithFeatureEstimators(
            TFeatureEstimatorsPtr featureEstimators
        );

        TCoreModelToFullModelConverter& WithMetrics(const TMetricsAndTimeLeftHistory& metrics);

        void Do(
            bool requiresStaticCtrProvider,
            TFullModel* dstModel,
            NPar::ILocalExecutor* localExecutor,
            const TVector<TTargetClassifier>* targetClassifiers
        );

        void Do(
            const TString& fullModelPath,
            const TVector<EModelType>& formats,
            bool addFileFormatExtension = false,
            NPar::ILocalExecutor* localExecutor = nullptr,
            const TVector<TTargetClassifier>* targetClassifiers = nullptr
        );

    private:
        void DoImpl(
            bool requiresStaticCtrProvider,
            TFullModel* fullModel,
            NPar::ILocalExecutor* localExecutor,
            const TVector<TTargetClassifier>* targetClassifiers
        );

        void CalcFinalCtrs(
            const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
            const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
            const TVector<TModelCtrBase>& ctrBases,
            std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
        );

    private:
        ui32 NumThreads;
        EFinalCtrComputationMode FinalCtrComputationMode;
        EFinalFeatureCalcersComputationMode FinalFeatureCalcerComputationMode;
        ui64 CpuRamLimit;

        /* these two params are explicit here because we can't get them from CatFeatureParams as
         * they are CPU-only
         */
        ui64 CtrLeafCountLimit;
        bool StoreAllSimpleCtrs;

        const NCatboostOptions::TCatBoostOptions& Options;
        const NCatboostOptions::TOutputFilesOptions& outputOptions;
        const TClassificationTargetHelper& ClassificationTargetHelper;

        TFullModel* CoreModel = nullptr;
        const NCB::TPerfectHashedToHashedCatValuesMap* PerfectHashedToHashedCatValuesMap = nullptr;
        const TMetricsAndTimeLeftHistory* MetricsAndTimeHistory = nullptr;
        TFeatureEstimatorsPtr FeatureEstimators = nullptr;

        TGetBinarizedDataFunc GetBinarizedDataFunc;
        TObjectsDataProviderPtr LearnObjectsData;
    };

    void CreateProcessingCollections(
        const TFeatureEstimators& featureEstimators,
        const TTextDigitizers& textDigitizers,
        const TVector<TEstimatedFeature>& estimatedFeatures,
        TTextProcessingCollection* textProcessingCollection,
        TEmbeddingProcessingCollection* embeddingProcessingCollection,
        TVector<TEstimatedFeature>* reorderedEstimatedFeatures,
        NPar::ILocalExecutor* localExecutor
    );

    void ExportFullModel(
        const TFullModel& fullModel,
        const TString& fullModelPath,

        // if specified - all categorical feature values must be present in this dataset
        const TMaybe<const TObjectsDataProvider*> allLearnObjectsData,
        TConstArrayRef<EModelType> formats,
        bool addFileFormatExtension = false
    );
}
