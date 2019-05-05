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
        ui32 flatFeatureIdx) {

        const auto featuresLayout = *quantizedObjectsData.GetFeaturesLayout();
        CB_ENSURE_INTERNAL(
            featuresLayout.GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float,
            "Mismatched feature type");
        return quantizedObjectsData.GetFloatFeatureRawSrcData(flatFeatureIdx) + consecutiveSubsetBegin;
    }

    template <class TDataProvidersTemplate>
    inline ui32 GetConsecutiveSubsetBegin(const TDataProvidersTemplate& objectsData) {
        const auto maybeConsecutiveSubsetBegin =
            objectsData.GetFeaturesArraySubsetIndexing().GetConsecutiveSubsetBegin();
        CB_ENSURE_INTERNAL(
            maybeConsecutiveSubsetBegin,
            "Only consecutive feature data is supported for apply");
        return *maybeConsecutiveSubsetBegin;
    }

    inline ui8 QuantizedFeaturesFloatAccessor(
        const TVector<TVector<ui8>>& floatBinsRemap,
        TConstArrayRef<TExclusiveFeaturesBundle> bundlesMetaData,
        TConstArrayRef<TConstArrayRef<ui8>> repackedFeatures,
        const TVector<TMaybe<TExclusiveBundleIndex>>& bundledIndexes,
        const TVector<TMaybe<TPackedBinaryIndex>>& packedIndexes,
        const TFloatFeature& floatFeature,
        size_t index) {

        const auto& bundleIdx = bundledIndexes[floatFeature.FeatureIndex];
        const auto& packIdx = packedIndexes[floatFeature.FeatureIndex];

        ui8 unremappedFeatureBin;
        if (bundleIdx.Defined()) {
            const auto& bundleMetaData = bundlesMetaData[bundleIdx->BundleIdx];
            const auto& bundlePart = bundleMetaData.Parts[bundleIdx->InBundleIdx];
            auto boundsInBundle = bundlePart.Bounds;

            auto getBundleValueFunction = [&](const auto* bundlesData) {
                return GetBinFromBundle<ui8>(bundlesData[index], boundsInBundle);
            };
            const ui8* rawBundlesData = repackedFeatures[floatFeature.FlatFeatureIndex].data();

            switch (bundleMetaData.SizeInBytes) {
                case 1:
                    unremappedFeatureBin = getBundleValueFunction(rawBundlesData);
                    break;
                case 2:
                    unremappedFeatureBin = getBundleValueFunction((const ui16*)rawBundlesData);
                    break;
                default:
                    CB_ENSURE_INTERNAL(
                        false,
                        "unsupported Bundle SizeInBytes = " << bundleMetaData.SizeInBytes);
            }
        } else if (packIdx.Defined()) {
            TBinaryFeaturesPack bitIdx = packIdx->BitIdx;
            unremappedFeatureBin = (repackedFeatures[floatFeature.FlatFeatureIndex][index] >> bitIdx) & 1;
        } else {
            unremappedFeatureBin = repackedFeatures[floatFeature.FlatFeatureIndex][index];
        }
        return floatBinsRemap[floatFeature.FlatFeatureIndex][unremappedFeatureBin];
    }
}
