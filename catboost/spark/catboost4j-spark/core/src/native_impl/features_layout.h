#pragma once

#include <catboost/libs/data/features_layout.h>

#include <catboost/private/libs/options/enums.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

// separate function because SWIG support for constructor overloading seems to be broken
NCB::TFeatureMetaInfo MakeFeatureMetaInfo(
    EFeatureType type,
    const TString& name,
    bool isSparse = false,
    bool isIgnored = false,
    bool isAvailable = true // isIgnored = true overrides this parameter
) throw (yexception);

// separate function because SWIG support for constructor overloading seems to be broken
// data is moved into - poor substitute to && because SWIG does not support it
NCB::TFeaturesLayout MakeFeaturesLayout(TVector<NCB::TFeatureMetaInfo>* data) throw (yexception);

NCB::TFeaturesLayout MakeFeaturesLayout(
    const int featureCount,
    const TVector<TString>& featureNames,
    const TVector<i32>& ignoredFeatures
) throw (yexception);

NCB::TFeaturesLayoutPtr CloneWithSelectedFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    TConstArrayRef<i32> selectedFeatures
) throw (yexception);

template <EFeatureType FeatureType>
TVector<i32> GetAvailableFeatures(const NCB::TFeaturesLayout& featuresLayout) throw (yexception) {
    TVector<i32> result;
    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (NCB::TFeatureIdx<FeatureType> idx) {
            result.push_back(SafeIntegerCast<i32>(*idx));
        }
    );

    return result;
}

template <EFeatureType FeatureType>
TVector<i32> GetAvailableFeaturesFlatIndices(const NCB::TFeaturesLayout& featuresLayout) throw (yexception) {
    TVector<i32> result;
    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (NCB::TFeatureIdx<FeatureType> idx) {
            result.push_back(SafeIntegerCast<i32>(featuresLayout.GetExternalFeatureIdx(*idx, FeatureType)));
        }
    );

    return result;
}
