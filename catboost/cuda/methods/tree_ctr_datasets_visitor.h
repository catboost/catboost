#pragma once

#include "tree_ctrs_dataset.h"
#include "pointwise_optimization_subsets.h"

#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/splitter.h>
#include <catboost/private/libs/options/oblivious_tree_options.h>

namespace NCatboostCuda {
    class TTreeCtrDataSetVisitor {
    public:
        TTreeCtrDataSetVisitor(TBinarizedFeaturesManager& featuresManager,
                               const ui32 foldCount,
                               const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                               const TOptimizationSubsets<NCudaLib::TMirrorMapping>& subsets);

        TTreeCtrDataSetVisitor& SetBestGain(double score);

        TTreeCtrDataSetVisitor& SetScoreStdDevAndSeed(double scoreStdDev,
                                                      ui64 seed);

        bool HasSplit() {
            return BestDevice >= 0;
        }

        void Accept(const TTreeCtrDataSet& ctrDataSet,
                    const TMirrorBuffer<const TPartitionStatistics>& partStats,
                    const TMirrorBuffer<ui32>& ctrDataSetInverseIndices,
                    const TMirrorBuffer<ui32>& subsetDocs,
                    const TMirrorBuffer<const float>& featureWeights,
                    double scoreBeforeSplit,
                    ui32 maxUniqueValues,
                    float modelSizeReg);

        TBestSplitProperties CreateBestSplitProperties();

        TSingleBuffer<const ui64> GetBestSplitBits() const;

    private:
        void EnsureHasBestProps() const {
            CB_ENSURE(BestDevice >= 0, "Error: no good split in visitor found");
            CB_ENSURE(BestBin <= 255);
        }

        void CacheCtrBorders(const TMap<TCtr, TVector<float>>& bordersMap);

        TVector<ui32> GetCtrsBordersToCacheIds(const TVector<TCtr>& ctrs);

        bool IsNeedToCacheBorders(const TCtr& ctr);

        void UpdateBestSplit(const TTreeCtrDataSet& dataSet,
                             const TMirrorBuffer<ui32>& inverseIndices,
                             const TBestSplitProperties& bestSplitProperties);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const ui32 FoldCount;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        const TOptimizationSubsets<NCudaLib::TMirrorMapping>& Subsets;

        TAdaptiveLock Lock;

        double BestScore;
        double BestGain;
        double ScoreStdDev = 0;
        ui32 BestBin;
        int BestDevice;
        TCtr BestCtr;

        TVector<TVector<float>> BestBorders;
        TVector<TSingleBuffer<ui64>> BestSplits;
        TVector<ui64> Seeds;
    };
}
