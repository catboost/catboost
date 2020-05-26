#pragma once

#include "gpu_structures.h"
#include "grid_policy.h"
#include "folds_histogram.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/libs/data/data_provider.h>

namespace NCatboostCuda {
    //damn proxy for learn set one-hots and exclusive feature bundles
    class TBinarizationInfoProvider {
    public:
        ui32 GetFoldsCount(ui32 featureId) const;

        double GetGroupingLevel(ui32 featureId) const;

        bool IsEffectivelyOneHot(ui32 featureId) const {
            return Manager->IsCat(featureId) || Manager->IsFeatureBundle(featureId);
        }
        bool SkipInSplitSearch(ui32 featureId) const {
            return Manager->PresentInExclusiveFeatureBundle(featureId);
        }

        bool SkipFirstBucketInOneHot(ui32 featureId) const {
            if (Manager->IsCat(featureId)) {
                return false; // TODO(kirillovs): would be true for wide one-hots
            } else if (Manager->IsFeatureBundle(featureId)) {
                for (auto& part : Manager->GetFeatureBundleForFeatureId(featureId).Parts) {
                    if (part.FeatureType == EFeatureType::Categorical) {
                        // we don't need to skip zero bucket if any one-hot feature is in bundle as zero is a regular bucket
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        explicit TBinarizationInfoProvider(const TBinarizedFeaturesManager& manager,
                                           const NCB::TTrainingDataProvider* provider = nullptr)
            : Manager(&manager)
            , DataProvider(provider)
        {
        }

    private:
        const TBinarizedFeaturesManager* Manager;
        const NCB::TTrainingDataProvider* DataProvider;
    };

    struct TCpuGrid {
        TVector<ui32> AllFeatureIds;
        TVector<ui32> AllFolds;
        TVector<bool> SkipInSplitSearch;

        TVector<ui32> FeatureIds;
        TVector<ui32> Folds;
        TVector<bool> IsOneHot;
        TVector<bool> SkipFirstBucketWhileScoring;
        TMap<ui32, ui32> InverseFeatures;

        template <class TFeaturesBinarizationDescription>
        TCpuGrid(const TFeaturesBinarizationDescription& info,
                 const TVector<ui32>& features)
            : AllFeatureIds(features)
        {
            AllFolds.reserve(features.size());
            Folds.reserve(features.size());
            IsOneHot.reserve(features.size());
            for (ui32 i = 0; i < features.size(); ++i) {
                IsOneHot.push_back(info.IsEffectivelyOneHot(features[i]));
                const bool skipInSplitSearch = info.SkipInSplitSearch(features[i]);
                SkipInSplitSearch.push_back(skipInSplitSearch);
                const ui32 folds = info.GetFoldsCount(features[i]);
                AllFolds.push_back(folds);
                if (skipInSplitSearch) {
                    continue;
                }
                FeatureIds.push_back(features[i]);
                Folds.push_back(folds);
                SkipFirstBucketWhileScoring.push_back(info.SkipFirstBucketInOneHot(features[i]));
            }
            InverseFeatures = BuildInverseIndex(AllFeatureIds);
        }

        TCpuGrid Subgrid(const TVector<ui32>& indices) const {
            TCpuGrid grid;
            grid.AllFeatureIds = AllFeatureIds;
            for (ui32 idx : indices) {
                grid.FeatureIds.push_back(FeatureIds[idx]);
                grid.Folds.push_back(Folds[idx]);
                grid.IsOneHot.push_back(IsOneHot[idx]);
                grid.SkipInSplitSearch.push_back(SkipInSplitSearch[idx]);
            }
            grid.InverseFeatures = TCpuGrid::BuildInverseIndex(grid.FeatureIds);
            return grid;
        }

        static TMap<ui32, ui32> BuildInverseIndex(const TVector<ui32>& features);

        ui32 FoldCount(ui32 featureId) const {
            return Folds[InverseFeatures.at(featureId)];
        }

        TFoldsHistogram ComputeFoldsHistogram(const TSlice& featuresSlice) const;

        TFoldsHistogram ComputeFoldsHistogram() const;

        TMap<ui32, ui32> ComputeFoldOffsets() const;

    private:
        TCpuGrid() = default;
    };

    //block of compressed features for one policy
    //what features we have and hot to access one
    template <class TFeaturesMapping,
              class TSamplesMapping>
    struct TGpuFeaturesBlockDescription {
        TCpuGrid Grid;
        NCudaLib::TDistributedObject<ui64> CIndexSizes = NCudaLib::GetCudaManager().CreateDistributedObject<ui64>(0);
        NCudaLib::TDistributedObject<ui64> CIndexOffsets = NCudaLib::GetCudaManager().CreateDistributedObject<ui64>(0);
        TCudaBuffer<TCFeature, TFeaturesMapping> CudaFeaturesDevice;
        //TCFeatures on each device
        TVector<NCudaLib::TDistributedObject<TCFeature>> CudaFeaturesHost;
        NCudaLib::TDistributedObject<TFoldsHistogram> FoldsHistogram = CreateDistributedObject<TFoldsHistogram>();
        TSamplesMapping Samples;

        NCudaLib::TDistributedObject<ui32> BinFeatureCount = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(
            0);
        TVector<TCBinFeature> BinFeatures;

        //for statistics
        TFeaturesMapping HistogramsMapping;
        //for best splits
        //i don't like it here
        TCudaBuffer<TCBinFeature, TFeaturesMapping> BinFeaturesForBestSplits;

        explicit TGpuFeaturesBlockDescription(TCpuGrid&& grid)
            : Grid(std::move(grid))
        {
        }

        const NCudaLib::TDistributedObject<TCFeature>& GetTCFeature(ui32 featureId) const {
            CB_ENSURE(Grid.InverseFeatures.contains(featureId));
            return CudaFeaturesHost[Grid.InverseFeatures.at(featureId)];
        }

        const NCudaLib::TDistributedObject<TFoldsHistogram>& GetFoldsHistogram() const {
            return FoldsHistogram;
        }
    };

    extern template struct TGpuFeaturesBlockDescription<NCudaLib::TSingleMapping, NCudaLib::TSingleMapping>;
    extern template struct TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TStripeMapping>;
    extern template struct TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TMirrorMapping>;

    template <class TPoolLayout>
    struct TCudaFeaturesLayoutHelper;

    //cuda-manager has 1 active device (mainly for child managers) and we use it
    struct TSingleDevLayout {
        using TFeaturesMapping = NCudaLib::TSingleMapping;
        using TBinFeaturesMapping = NCudaLib::TSingleMapping;
        using TSamplesMapping = NCudaLib::TSingleMapping;
        using TCompressedIndexMapping = NCudaLib::TSingleMapping;
        using TPartStatsMapping = NCudaLib::TSingleMapping;
        using TFeatureWeightsMapping = NCudaLib::TSingleMapping;
    };

    struct TFeatureParallelLayout {
        using TFeaturesMapping = NCudaLib::TStripeMapping;
        using TSamplesMapping = NCudaLib::TMirrorMapping;
        using TCompressedIndexMapping = NCudaLib::TStripeMapping;
        using TFeatureWeightsMapping = NCudaLib::TMirrorMapping;
    };

    struct TDocParallelLayout {
        using TFeaturesMapping = NCudaLib::TStripeMapping;
        using TSamplesMapping = NCudaLib::TStripeMapping;
        using TCompressedIndexMapping = NCudaLib::TStripeMapping;
        using TPartStatsMapping = NCudaLib::TMirrorMapping;
        using TFeatureWeightsMapping = NCudaLib::TMirrorMapping;
    };
}
