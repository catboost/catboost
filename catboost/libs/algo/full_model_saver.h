#pragma once

#include "online_ctr.h"
#include "projection.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/libs/options/cat_feature_options.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/system/types.h>

#include <functional>


class TCtrValueTable;
struct TDatasetDataForFinalCtrs;
struct TFullModel;

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
            EFinalCtrComputationMode finalCtrComputationMode
        );

        TCoreModelToFullModelConverter& WithCoreModelFrom(TFullModel* coreModel);

        TCoreModelToFullModelConverter& WithObjectsDataFrom(TObjectsDataProviderPtr learnObjectsData);

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            TDatasetDataForFinalCtrs&& datasetDataForFinalCtrs,
            THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection
        );

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            TTrainingForCPUDataProviders&& trainingData,
            THashMap<TFeatureCombination, TProjection>&& featureCombinationToProjection,
            const TVector<TTargetClassifier>& targetClassifiers
        );

        TCoreModelToFullModelConverter& WithPerfectHashedToHashedCatValuesMap(
            const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap
        );

        void Do(bool requiresStaticCtrProvider, TFullModel* dstModel);

        void Do(
            const TString& fullModelPath,
            const TVector<EModelType>& formats,
            bool addFileFormatExtension = false
        );

    private:
        void DoImpl(bool requiresStaticCtrProvider, TFullModel* fullModel);

        void CalcFinalCtrs(
            const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
            const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjectionMap,
            const TVector<TModelCtrBase>& ctrBases,
            std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
        );

    private:
        ui32 NumThreads;
        EFinalCtrComputationMode FinalCtrComputationMode;
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
        TGetBinarizedDataFunc GetBinarizedDataFunc;
        TObjectsDataProviderPtr LearnObjectsData;
    };

    void ExportFullModel(
        const TFullModel& fullModel,
        const TString& fullModelPath,

        // if specified - all categorical feature values must be present in this dataset
        const TMaybe<const TObjectsDataProvider*> allLearnObjectsData,
        TConstArrayRef<EModelType> formats,
        bool addFileFormatExtension = false
    );
}
