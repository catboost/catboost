#pragma once

#include <catboost/libs/data/quantized_features_info.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>


NCB::TQuantizedFeaturesInfoPtr MakeQuantizedFeaturesInfo(
    const NCB::TFeaturesLayout& featuresLayout
) throw(yexception);

NCB::TQuantizedFeaturesInfoPtr MakeEstimatedQuantizedFeaturesInfo(i32 featureCount) throw(yexception);

void UpdateCatFeaturesInfo(
    TConstArrayRef<i32> catFeaturesUniqValueCounts, // [flatFeatureIdx]
    bool isInitialization,
    NCB::TQuantizedFeaturesInfo* quantizedFeaturesInfo
) throw(yexception);

i32 CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
) throw(yexception);

// returned vector is indexed by flatFeatureIdx, non-categorical or non-available features will contain 0
TVector<i32> GetCategoricalFeaturesUniqueValuesCounts(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
) throw(yexception);


void DbgDump(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo, const TString& fileName);
