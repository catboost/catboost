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
            CB_ENSURE(featuresSlice.Right <= Grid.AllFeatureIds.size());
            ui32 innerFeatureId = 0;
            for (ui32 i = 0; i < featuresSlice.Left; ++i) {
                innerFeatureId += !Grid.SkipInSplitSearch[i];
            }
            for (ui32 i = featuresSlice.Left; i < featuresSlice.Right; ++i) {
                if (Grid.SkipInSplitSearch[i]) {
                    continue;
                }
                const ui32 folds = Grid.Folds[innerFeatureId];
                TCBinFeature binFeatureBase;
                binFeatureBase.FeatureId = Grid.FeatureIds[innerFeatureId];

                for (ui32 fold = 0; fold < folds; ++fold) {
                    TCBinFeature binFeature = binFeatureBase;
                    binFeature.BinId = fold;
                    if (fold == 0 && Grid.SkipFirstBucketWhileScoring[innerFeatureId]) {
                        binFeature.SkipInScoreCount = true;
                    }
                    result.push_back(binFeature);
                }
                ++innerFeatureId;
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
                               TVector<TCFeature>* allFeatures,
                               TVector<TCFeature>* features) const {
            using THelper = TCompressedIndexHelper<Policy>;
            const ui32 featuresPerInt = THelper::FeaturesPerInt();
            const ui32 mask = THelper::Mask();
            const ui32 maxFolds = THelper::MaxFolds();

            ui32 foldOffset = 0;
            ui32 innerFeartureId = 0;
            // roll through skipped features
            for (ui32 i = 0; i < featuresSlice.Left; ++i) {
                innerFeartureId += !Grid.SkipInSplitSearch[i];
            }
            for (ui32 i = 0; i < featuresSlice.Size(); ++i) {
                const ui32 feature = featuresSlice.Left + i;
                const ui32 shift = THelper::Shift(i);
                const ui64 cindexOffset = loadOffset + (i / featuresPerInt) * docCount;
                const ui32 foldCount = Grid.AllFolds[feature];
                CB_ENSURE(foldCount <= maxFolds, TStringBuilder() << "Fold count " << foldCount << " max folds " << maxFolds << " (" << Policy << ")");
                TCFeature cudaFeature{cindexOffset,
                                      mask,
                                      shift,
                                      foldOffset,
                                      foldCount,
                                      Grid.IsOneHot[feature],
                                      Grid.SkipInSplitSearch[feature] ? false : Grid.SkipFirstBucketWhileScoring[innerFeartureId]
                                      };
                foldOffset += foldCount;
                (*allFeatures).push_back(cudaFeature);
                if (!Grid.SkipInSplitSearch[feature]) {
                    (*features).push_back(cudaFeature);
                    ++innerFeartureId;
                }
            }
            if (featuresSlice.Size() == 0) {
                return 0;
            }
            //how many int lines we use
            ui64 size = (allFeatures->back().Offset - loadOffset + docCount);
            CB_ENSURE(size == CompressedIndexSize<Policy>(featuresSlice, docCount));
            return size;
        }

    private:
        const TCpuGrid& Grid;
    };
}
