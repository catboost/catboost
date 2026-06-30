#pragma once

#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/data/quantization.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/vector.h>

#include <util/system/types.h>


namespace NCB {

    struct TBinarizedFeatureStatistics {
        TVector<float> Borders; // empty in case of one-hot features
        TVector<ui32> BinarizedFeature;
        TVector<float> MeanTarget;
        TVector<float> MeanWeightedTarget;
        TVector<float> MeanPrediction;
        TVector<size_t> ObjectsPerBin;
        TVector<double> PredictionsOnVaryingFeature;
    };

    struct TFeatureTypeAndInternalIndex {
        EFeatureType Type;
        int Index;
    };

    TVector<TBinarizedFeatureStatistics> GetBinarizedStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const TVector<size_t>& catFeaturesNums,
        const TVector<size_t>& floatFeaturesNums,
        const EPredictionType predictionType,
        const int threadCount);

    ui32 GetCatFeaturePerfectHash(
        const TFullModel& model,
        const TStringBuf& value,
        const size_t featureNum);

    TFeatureTypeAndInternalIndex GetFeatureTypeAndInternalIndex(const TFullModel& model, const int flatFeatureIndex);

    TVector<TString> GetCatFeatureValues(const TDataProvider& dataset, const size_t flatFeatureIndex);

}
