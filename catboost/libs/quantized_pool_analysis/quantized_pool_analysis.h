#pragma once

#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/index_calcer.h>
#include <catboost/libs/data_new/quantization.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/target/data_providers.h>

#include <util/generic/vector.h>


namespace NCB {

    struct TBinarizedFloatFeatureStatistics {
        TVector<float> Borders;
        TVector<ui8> BinarizedFeature;
        TVector<float> MeanTarget;
        TVector<float> MeanPrediction;
        TVector<size_t> ObjectsPerBin;
        TVector<double> PredictionsOnVaryingFeature;
    };

    struct TBinarizedOneHotFeatureStatistics{
        TVector<int> BinarizedFeature;
        TVector<float> MeanTarget;
        TVector<float> MeanPrediction;
        TVector<size_t> ObjectsPerBin;
        TVector<double> PredictionsOnVaryingFeature;
    };

    struct TFeatureTypeAndInternalIndex {
        EFeatureType Type;
        int Index;
    };

    TBinarizedFloatFeatureStatistics GetBinarizedFloatFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const EPredictionType predictionType);

    TBinarizedOneHotFeatureStatistics GetBinarizedOneHotFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const EPredictionType predictionType);

    ui32 GetCatFeaturePerfectHash(
        const TFullModel& model,
        const TStringBuf& value,
        const size_t featureNum);

    TFeatureTypeAndInternalIndex GetFeatureTypeAndInternalIndex(const TFullModel& model, const int flatFeatureIndex);

}
