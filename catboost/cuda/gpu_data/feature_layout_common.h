#pragma once

#include "gpu_structures.h"
#include "grid_policy.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>

namespace NCatboostCuda {
    //damn proxy for learn set one-hots
    class TBinarizationInfoProvider {
    public:
        ui32 GetFoldsCount(ui32 featureId) const {
            const ui32 binCount = Manager->GetBinCount(featureId);
            if (DataProvider && binCount && IsOneHot(featureId)) {
                ui32 dataProviderId = Manager->GetDataProviderId(featureId);
                ui32 count = dynamic_cast<const ICatFeatureValuesHolder&>(DataProvider->GetFeatureById(dataProviderId)).GetUniqueValues();
                return count > 2 ? count : count - 1;
            }
            return binCount ? binCount - 1 : 0;
        }

        double GetGroupingLevel(ui32 featureId) const {
            const ui32 binCount = Manager->GetBinCount(featureId);

            //-x <= 128 does not use atomics and
            // should be groupped by binarization level for best performance
            //+1 bin cause 129 bins, 128 borders
            if (binCount <= 129 || Manager->IsCtr(featureId)) {
                return binCount * 1.0 / 256;
            }

            //for features with binCount > 128 heuristic to group most sparse
            //features together as this'll increase register cache hit
            //and reduce atomic conflicts
            if (DataProvider && !Manager->IsCtr(featureId)) {
                const ui32 dataProviderId = Manager->GetDataProviderId(featureId);

                if (!DataProvider->HasFeatureId(dataProviderId)) {
                    return 2.0;
                }
                const IFeatureValuesHolder& featureValuesHolder = DataProvider->GetFeatureById(dataProviderId);
                if (featureValuesHolder.GetType() == EFeatureValuesType::Float) {
                    return 2.0;
                } else {
                    CB_ENSURE(featureValuesHolder.GetType() == EFeatureValuesType::BinarizedFloat);
                }

                return 1.0 + dynamic_cast<const TCompressedValuesHolderImpl&>(DataProvider->GetFeatureById(dataProviderId)).SparsityLevel();
            }
            return 2.0;
        }

        bool IsOneHot(ui32 featureId) const {
            return Manager->IsCat(featureId);
        }

        explicit TBinarizationInfoProvider(const TBinarizedFeaturesManager& manager,
                                           const TDataProvider* provider = nullptr)
            : Manager(&manager)
            , DataProvider(provider)
        {
        }

    private:
        const TBinarizedFeaturesManager* Manager;
        const TDataProvider* DataProvider;
    };

    struct TCpuGrid {
        TVector<ui32> FeatureIds;
        TVector<ui32> Folds;
        TVector<bool> IsOneHot;
        TMap<ui32, ui32> InverseFeatures;

        template <class TFeaturesBinarizationDescription>
        TCpuGrid(const TFeaturesBinarizationDescription& info,
                 const TVector<ui32>& features)
            : FeatureIds(features)
        {
            Folds.reserve(features.size());
            IsOneHot.reserve(features.size());
            for (ui32 i = 0; i < features.size(); ++i) {
                IsOneHot.push_back(info.IsOneHot(features[i]));
                const ui32 folds = info.GetFoldsCount(features[i]);
                Folds.push_back(folds);
            }
            InverseFeatures = BuildInverseIndex(FeatureIds);
        }

        inline static TMap<ui32, ui32> BuildInverseIndex(const TVector<ui32>& features) {
            TMap<ui32, ui32> inverseFeatures;
            for (ui32 i = 0; i < features.size(); ++i) {
                inverseFeatures[features[i]] = i;
            }
            return inverseFeatures;
        };

        ui32 FoldCount(ui32 featureId) const {
            return Folds[InverseFeatures.at(featureId)];
        }

        TMap<ui32, ui32> ComputeFoldOffsets() const {
            TMap<ui32, ui32> offsets;
            ui32 cursor = 0;
            for (ui32 i = 0; i < FeatureIds.size(); ++i) {
                const ui32 f = FeatureIds[i];
                offsets[f] = cursor;
                cursor += Folds[i];
            }
            return offsets;
        }
    };

    class TCudaFeaturesHelper {
    public:
        TCudaFeaturesHelper(const TCpuGrid& grid)
            : Grid(grid)
        {
        }

        TVector<TCBinFeature> BuildBinaryFeatures(const TSlice& featuresSlice) const {
            TVector<TCBinFeature> result;
            CB_ENSURE(featuresSlice.Right <= Grid.FeatureIds.size());

            for (ui32 i = featuresSlice.Left; i < featuresSlice.Right; ++i) {
                const ui32 folds = Grid.Folds[i];
                TCBinFeature binFeatureBase;
                binFeatureBase.FeatureId = Grid.FeatureIds[i];

                for (ui32 fold = 0; fold < folds; ++fold) {
                    TCBinFeature binFeature = binFeatureBase;
                    binFeature.BinId = fold;
                    result.push_back(binFeature);
                }
            }
            return result;
        }

        template <EFeaturesGroupingPolicy Policy>
        ui64 CompressedIndexSize(const TSlice& featuresSlice,
                                 ui64 docCount) const {
            using THelper = TCompressedIndexHelper<Policy>;
            const ui64 featuresPerInt = THelper::FeaturesPerInt();
            return ((featuresSlice.Size() + featuresPerInt - 1) / featuresPerInt) * docCount;
        }

        //creates TCFeatures mapping for continuous block and return block size (how many ui32 it uses)
        template <EFeaturesGroupingPolicy Policy>
        ui64 AddDeviceFeatures(const TSlice& featuresSlice,
                               ui64 loadOffset,
                               ui64 docCount,
                               TVector<TCFeature>* features) const {
            using THelper = TCompressedIndexHelper<Policy>;
            const ui32 featuresPerInt = THelper::FeaturesPerInt();
            const ui32 mask = THelper::Mask();
            const ui32 maxFolds = THelper::MaxFolds();

            ui32 foldOffset = 0;
            for (ui32 i = 0; i < featuresSlice.Size(); ++i) {
                const ui32 feature = featuresSlice.Left + i;
                const ui32 shift = THelper::Shift(i);
                const ui64 cindexOffset = loadOffset + (i / featuresPerInt) * docCount;
                const ui32 foldCount = Grid.Folds[feature];
                CB_ENSURE(foldCount <= maxFolds, TStringBuilder() << "Fold count " << foldCount << " max folds " << maxFolds << " (" << Policy << ")");
                TCFeature cudaFeature{cindexOffset,
                                      mask,
                                      shift,
                                      foldOffset,
                                      foldCount,
                                      Grid.IsOneHot[feature]};
                foldOffset += foldCount;
                (*features).push_back(cudaFeature);
            }
            if (featuresSlice.Size() == 0) {
                return 0;
            }
            //how many int lines we use
            ui64 size = (features->back().Offset - loadOffset + docCount);
            CB_ENSURE(size == CompressedIndexSize<Policy>(featuresSlice, docCount));
            return size;
        }

    private:
        const TCpuGrid& Grid;
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
        TSamplesMapping Samples;

        NCudaLib::TDistributedObject<ui32> BinFeatureCount = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(0);
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
            CB_ENSURE(Grid.InverseFeatures.has(featureId));
            return CudaFeaturesHost[Grid.InverseFeatures.at(featureId)];
        }
    };

    template <class TPoolLayout>
    struct TCudaFeaturesLayoutHelper;

}
