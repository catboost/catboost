#pragma once

#include "objects.h"

#include <catboost/libs/model/model.h>

#include <util/generic/hash.h>
#include <util/system/types.h>


namespace NCB {

    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        THashMap<ui32, ui32>* columnIndexesReorderMap);

    TVector<TVector<ui8>> GetFloatFeaturesBordersRemap(  // [flatFeatureIdx][poolFeatureBin]
        const TFullModel& model,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo);
}
