#pragma once

#include "objects.h"

#include <catboost/libs/model/model.h>

#include <util/generic/hash.h>
#include <util/system/types.h>


namespace NCB {
    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TFeaturesLayout& datasetFeaturesLayout,

        // modelFlatFeatureIdx -> dataFlatFeatureIdx, only for features used in the model
        THashMap<ui32, ui32>* columnIndexesReorderMap);

    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,

        // modelFlatFeatureIdx -> dataFlatFeatureIdx, only for features used in the model
        THashMap<ui32, ui32>* columnIndexesReorderMap);

    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData);

    TVector<ui8> GetFloatFeatureBordersRemap(  // [poolFeatureBin]
        const TFloatFeature& feature,
        ui32 datasetFlatFeatureIdx,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo);

    TVector<TVector<ui8>> GetFloatFeaturesBordersRemap(  // [modelFlatFeatureIdx][poolFeatureBin]
        const TFullModel& model,

        // modelFlatFeatureIdx -> dataFlatFeatureIdx, only for features used in the model
        const THashMap<ui32, ui32>& columnIndexesReorderMap,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo);

    TVector<TVector<ui32>> GetCatFeaturesBinToHashedValueRemap(  // [modelFlatFeatureIdx][poolFeatureBin]
        const TFullModel& model,

        // modelFlatFeatureIdx -> dataFlatFeatureIdx, only for features used in the model
        const THashMap<ui32, ui32>& columnIndexesReorderMap,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo
    );
}
