#pragma once

#include <catboost/libs/data/quantized_features_info.h>

#include <util/generic/fwd.h>
#include <util/generic/yexception.h>


NCB::TQuantizedFeaturesInfoPtr MakeQuantizedFeaturesInfo(
    const NCB::TFeaturesLayout& featuresLayout
) throw(yexception);


void DbgDump(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo, const TString& fileName);
