#pragma once

#include "gpu_structures.h"
#include "grid_policy.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/data/feature.h>

namespace NCatboostCuda
{
    template<class TFeaturesMapping>
    struct TCompressedIndexMappingTrait
    {
        using TCompressedIndexMapping = NCudaLib::TStripeMapping;
    };

    template<>
    struct TCompressedIndexMappingTrait<NCudaLib::TSingleMapping>
    {
        using TCompressedIndexMapping = NCudaLib::TSingleMapping;
    };

    struct TCatBoostPoolLayout
    {
        using TFeaturesMapping = NCudaLib::TStripeMapping;
        using TSampleMapping = NCudaLib::TMirrorMapping;
    };

    struct TSingleDevPoolLayout
    {
        using TFeaturesMapping = NCudaLib::TSingleMapping;
        using TSampleMapping = NCudaLib::TSingleMapping;
    };

    using TByteFeatureGridPolicy = TGridPolicy<8, 1>;
    using TBinaryFeatureGridPolicy = TGridPolicy<1, 8>;
    using THalfByteFeatureGridPolicy = TGridPolicy<4, 2>;

    template<class TGridPolicy_,
            class TLayoutPolicy = TCatBoostPoolLayout>
    class TGpuBinarizedDataSet: public TMoveOnly,
                                public TGuidHolder
    {
    public:
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSampleMapping = typename TLayoutPolicy::TSampleMapping;
        using TCompressedIndexMapping = typename TCompressedIndexMappingTrait<TFeaturesMapping>::TCompressedIndexMapping;
        using TGridPolicy = TGridPolicy_;

        const TCudaBuffer<TCFeature, TFeaturesMapping>& GetGrid() const
        {
            return Grid;
        }

        ui32 GetFeatureId(TCFeature feature) const
        {
            return GetFeatureId(feature.Index);
        }

        bool NotEmpty() const
        {
            return GetFeatureIds().size() > 0;
        }

        TCFeature GetFeatureByGlobalId(ui32 featureId) const
        {
            Y_ASSERT(HasFeature(featureId));
            CB_ENSURE(HasFeature(featureId));
            return HostFeatures.at(LocalFeatureIndex.at(featureId));
        }

        TCFeature GetFeatureByLocalId(ui32 featureId) const
        {
            return HostFeatures.at(featureId);
        }

        ui32 GetFeatureId(const ui32 localId) const
        {
            return FeatureIds[localId];
        }

        bool HasFeature(ui32 featureId) const
        {
            return LocalFeatureIndex.has(featureId);
        }

        const TVector<ui32>& GetFeatureIds() const
        {
            return FeatureIds;
        }

        ui32 GetFeatureCount() const
        {
            return static_cast<ui32>(FeatureIds.size());
        }

        const TCudaBuffer<ui32, TCompressedIndexMapping>& GetCompressedIndex() const
        {
            return CompressedIndex;
        }

        ui32 FeatureCount() const
        {
            return static_cast<ui32>(HostFeatures.size());
        }

        const TVector<TCFeature>& GetHostFeatures() const
        {
            return HostFeatures;
        }

        const TSampleMapping& GetDocumentsMapping() const
        {
            return DocsMapping;
        }

        NCudaLib::TDistributedObject<ui32> GetDataSetSize() const
        {
            auto sizes = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>();
            for (auto dev : GetDocumentsMapping().NonEmptyDevices())
            {
                sizes.Set(dev, GetDocumentsMapping().DeviceSlice(dev).Size());
            }
            return sizes;
        }

        const TCudaBuffer<TCBinFeature, TFeaturesMapping>& GetBinaryFeatures() const
        {
            return BinaryFeatures;
        };

        NCudaLib::TDistributedObject<ui32> GetBinFeatureCount() const
        {
            auto sizes = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>();

            for (auto dev : BinaryFeatures.NonEmptyDevices())
            {
                sizes.Set(dev, BinaryFeatures.GetMapping().DeviceSlice(dev).Size());
            }
            return sizes;
        }

        TBinarySplit CreateSplit(ui32 localIdx,
                                 ui32 binId) const
        {
            auto feature = HostFeatures[localIdx];
            return {FeatureIds.at(localIdx), binId,
                    feature.OneHotFeature
                    ? EBinSplitType::TakeBin
                    : EBinSplitType::TakeGreater};
        }

        const TVector<TCBinFeature>& GetHostBinaryFeatures() const
        {
            return HostBinFeatures;
        }

    private:
        TVector<ui32> FeatureIds;
        TMap<ui32, ui32> LocalFeatureIndex;
        TCudaBuffer<ui32, TCompressedIndexMapping> CompressedIndex;
        //features
        TCudaBuffer<TCFeature, TFeaturesMapping> Grid;
        TVector<TCFeature> HostFeatures;

        TCudaBuffer<TCBinFeature, TFeaturesMapping> BinaryFeatures;
        TVector<TCBinFeature> HostBinFeatures;
        TSampleMapping DocsMapping;

        template<class>
        friend
        class TGpuBinarizedDataSetBuilderHelper;

        template<class, class>
        friend
        class TGpuBinarizedDataSetBuilder;

        friend class TTreeCtrDataSetBuilder;
    };
}
