#pragma once

#include "feature_layout_common.h"

namespace NCatboostCuda {
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
}
