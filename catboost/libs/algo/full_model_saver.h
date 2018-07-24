#pragma once

#include "dataset.h"
#include "online_ctr.h"
#include "projection.h"

#include <catboost/libs/data/pool.h>

#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/ctr_value_table.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/model/target_classifier.h>

#include <catboost/libs/options/cat_feature_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/load_options.h>

#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/system/types.h>

#include <functional>


namespace NCB {

    class TCoreModelToFullModelConverter {
    private:
        using TGetCoreModelFunc = std::function<TFullModel&()>;

        using TGetBinarizedDataFunc = std::function<
            void(const TFullModel&, TDatasetDataForFinalCtrs*, const THashMap<TFeatureCombination, TProjection>**)
        >;

    public:
        TCoreModelToFullModelConverter(
            ui32 numThreads,
            EFinalCtrComputationMode finalCtrComputationMode,
            ui64 ctrLeafCountLimit,
            bool storeAllSimpleCtrs,
            const NCatboostOptions::TCatFeatureParams& catFeatureParams
        );

        TCoreModelToFullModelConverter& WithCoreModelFrom(TFullModel* coreModel);

        TCoreModelToFullModelConverter& WithCoreModelFrom(const TString& coreModelPath);

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            const TDatasetDataForFinalCtrs& datasetDataForFinalCtrs,
            const THashMap<TFeatureCombination, TProjection>& featureCombinationToProjection
        );

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            const TClearablePoolPtrs& pools,
            const TVector<TTargetClassifier>& targetClassifiers
        );

        TCoreModelToFullModelConverter& WithBinarizedDataComputedFrom(
            const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
            const TVector<TString>& classNames,
            const TVector<TTargetClassifier>& targetClassifiers
        );

        void Do(TFullModel* dstModel, bool requiresStaticCtrProvider);

        void Do(const TString& fullModelPath);

    private:
        void CalcFinalCtrs(
            const TFullModel& coreModel,
            const TVector<TModelCtrBase>& ctrBases,
            std::function<void(TCtrValueTable&& table)>&& asyncCtrValueTableCallback
        );

    private:
        ui32 NumThreads;
        EFinalCtrComputationMode FinalCtrComputationMode;

        /* these two params are explicit here because we can't get them from CatFeatureParams as
         * they are CPU-only
         */
        ui64 CtrLeafCountLimit;
        bool StoreAllSimpleCtrs;

        const NCatboostOptions::TCatFeatureParams& CatFeatureParams;

        TGetCoreModelFunc GetCoreModelFunc;
        TGetBinarizedDataFunc GetBinarizedDataFunc;
    };
}
