#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/exception.h>


namespace NCB {

    inline const float* GetRawFeatureDataBeginPtr(
        const TRawObjectsDataProvider& rawObjectsData,
        ui32 consecutiveSubsetBegin,
        ui32 flatFeatureIdx) {

        const auto featuresLayout = rawObjectsData.GetFeaturesLayout();
        const ui32 internalFeatureIdx = featuresLayout->GetInternalFeatureIdx(flatFeatureIdx);
        if (featuresLayout->GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float) {
            return (*(*(**rawObjectsData.GetFloatFeature(internalFeatureIdx)).GetArrayData().GetSrc()
                )).data() + consecutiveSubsetBegin;
        } else {
            return reinterpret_cast<const float*>((*(*(**rawObjectsData.GetCatFeature(internalFeatureIdx))
                .GetArrayData().GetSrc())).data()) + consecutiveSubsetBegin;
        }
    }

    inline const ui8* GetQuantizedForCpuFloatFeatureDataBeginPtr(
        const TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
        ui32 consecutiveSubsetBegin,
        ui32 flatFeatureIdx)
    {
        const auto featuresLayout = *quantizedObjectsData.GetFeaturesLayout();
        CB_ENSURE_INTERNAL(
            featuresLayout.GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float,
            "Mismatched feature type"
        );
        return quantizedObjectsData.GetFloatFeatureRawSrcData(flatFeatureIdx) + consecutiveSubsetBegin;
    }

    template <class TDataProvidersTemplate>
    inline ui32 GetConsecutiveSubsetBegin(const TDataProvidersTemplate& objectsData) {
        const auto maybeConsecutiveSubsetBegin =
            objectsData.GetFeaturesArraySubsetIndexing().GetConsecutiveSubsetBegin();
        CB_ENSURE_INTERNAL(
            maybeConsecutiveSubsetBegin,
            "Only consecutive feature data is supported for apply"
        );
        return *maybeConsecutiveSubsetBegin;
    }

    inline ui8 QuantizedFeaturesFloatAccessor(
        const TVector<TVector<ui8>>& floatBinsRemap,
        TConstArrayRef<TConstArrayRef<ui8>> repackedFeatures,
        const TVector<TMaybe<TPackedBinaryIndex>>& packedIndexes,
        const TFloatFeature& floatFeature,
        size_t index)
    {
        auto& packIdx = packedIndexes[floatFeature.FeatureIndex];
        if (packIdx.Defined()) {
            TBinaryFeaturesPack bitIdx = packIdx->BitIdx;
            ui8 binaryFeatureValue = (repackedFeatures[floatFeature.FlatFeatureIndex][index] >> bitIdx) & 1;
            return floatBinsRemap[floatFeature.FlatFeatureIndex][binaryFeatureValue];
        } else {
            return floatBinsRemap[floatFeature.FlatFeatureIndex][repackedFeatures[floatFeature.FlatFeatureIndex][index]];
        }
    }
}
