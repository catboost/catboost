#pragma once

#include <catboost/libs/data/objects.h>
#include <catboost/libs/data/quantized_features_info.h>

#include <library/cpp/grid_creator/binarization.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

namespace NPar {
    class TLocalExecutor;
}

namespace NCB {
    class TFeaturesLayout;
}


NCB::TQuantizedFeaturesInfoPtr PrepareQuantizationParameters(
    const NCB::TFeaturesLayout& featuresLayout,
    const TString& plainJsonParamsAsString
);


class TNanModeAndBordersBuilder {
public:
    TNanModeAndBordersBuilder(NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo);

    bool HasFeaturesToCalc() const {
        return !FeatureIndicesToCalc.empty();
    }

    // call before Finish and preferably before adding samples
    void SetSampleSize(i32 sampleSize);

    void AddSample(TConstArrayRef<double> objectData);

    void CalcBordersWithoutNans(i32 threadCount);

    /* updates parameters in quantizedFeaturesInfo passed to constructor
     * @param hasNans is an array with flatFeatureIdx index, can be empty
     */
    void Finish(TConstArrayRef<i8> hasNans);

private:
    size_t SampleSize = 0;
    TVector<ui32> FeatureIndicesToCalc;
    NCB::TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    TVector<TVector<float>> Data; // [featureIdxToCalc]

    // need to save them until full hasNans information becomes available
    TVector<NSplitSelection::TQuantization> QuantizationWithoutNans; // [featureIdxToCalc]
    TVector<bool> HasNans; // [featureIdxToCalc]
};


NCB::TQuantizedObjectsDataProviderPtr Quantize(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TRawObjectsDataProviderPtr* rawObjectsDataProvider, // moved into
    NPar::TLocalExecutor* localExecutor
);


void GetActiveFeaturesIndices(
    NCB::TFeaturesLayoutPtr featuresLayout,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TVector<i32>* ui8FlatIndices,
    TVector<i32>* ui16FlatIndices,
    TVector<i32>* ui32FlatIndices
);
