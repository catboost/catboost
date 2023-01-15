#pragma once

#include <catboost/libs/data/objects.h>
#include <catboost/libs/data/quantized_features_info.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

#include <jni.h>


class TNanModeAndBordersBuilder {
public:
    TNanModeAndBordersBuilder(
        const TString& plainJsonParamsAsString,
        i32 featureCount,
        const TVector<TString>& featureNames,
        i32 sampleSize
    ) throw (yexception);

    void AddSample(TConstArrayRef<double> objectData) throw (yexception);

    // returns with updated parameters
    NCB::TQuantizedFeaturesInfoPtr Finish(i32 threadCount) throw (yexception);

private:
    size_t SampleSize;
    TVector<ui32> FeatureIndicesToCalc;
    NCB::TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    TVector<TVector<float>> Data; // [featureIdxToCalc]
};


NCB::TQuantizedObjectsDataProviderPtr Quantize(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TRawObjectsDataProviderPtr* rawObjectsDataProvider, // moved into
    int threadCount
) throw (yexception);

