#include "feature_layout_doc_parallel.h"
#include "feature_layout_feature_parallel.h"
#include "feature_layout_single.h"

#include <catboost/libs/helpers/math_utils.h>

namespace NCatboostCuda {
    template struct TGpuFeaturesBlockDescription<NCudaLib::TSingleMapping, NCudaLib::TSingleMapping>;
    template struct TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TStripeMapping>;
    template struct TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TMirrorMapping>;

    template struct TCudaFeaturesLayoutHelper<TFeatureParallelLayout>;
    template struct TCudaFeaturesLayoutHelper<TDocParallelLayout>;

    TMap<ui32, ui32> TCpuGrid::BuildInverseIndex(const TVector<ui32>& features) {
        TMap<ui32, ui32> inverseFeatures;
        for (ui32 i = 0; i < features.size(); ++i) {
            inverseFeatures[features[i]] = i;
        }
        return inverseFeatures;
    }

    TFoldsHistogram TCpuGrid::ComputeFoldsHistogram(const TSlice& featuresSlice) const {
        TFoldsHistogram result;
        for (ui32 f = featuresSlice.Left; f < featuresSlice.Right; ++f) {
            const ui32 foldCount = Folds[f];
            if (foldCount > 0) {
                result.Counts[NCB::IntLog2(foldCount)]++;
            }
        }
        return result;
    }

    TFoldsHistogram TCpuGrid::ComputeFoldsHistogram() const {
        return ComputeFoldsHistogram(TSlice(0, FeatureIds.size()));
    }

    TMap<ui32, ui32> TCpuGrid::ComputeFoldOffsets() const {
        TMap<ui32, ui32> offsets;
        ui32 cursor = 0;
        for (ui32 i = 0; i < FeatureIds.size(); ++i) {
            const ui32 f = FeatureIds[i];
            offsets[f] = cursor;
            cursor += Folds[i];
        }
        return offsets;
    }

    ui32 TBinarizationInfoProvider::GetFoldsCount(ui32 featureId) const {
        const ui32 binCount = Manager->GetBinCount(featureId);
        if (DataProvider && binCount && Manager->IsCat(featureId)) {
            ui32 dataProviderId = Manager->GetDataProviderId(featureId);
            auto catFeatureIdx = DataProvider->MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(dataProviderId);
            ui32 count = DataProvider->ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(catFeatureIdx).OnAll;
            return count > 2 ? count : count - 1;
        }
        return binCount ? binCount - 1 : 0;
    }

    double TBinarizationInfoProvider::GetGroupingLevel(ui32 featureId) const {
        const ui32 binCount = Manager->GetBinCount(featureId);

        //-x <= 128 does not use atomics and
        // should be groupped by binarization level for best performance
        //+1 bin cause 129 bins, 128 borders
        if (binCount <= 129 || Manager->IsCtr(featureId)) {
            return binCount * 1.0 / 256;
        }

        return 1.0 + binCount * 1.0 / 256;
    }
}
