#pragma once

#include "split_properties_helper.h"
#include "structure_searcher_options.h"
#include "compute_by_blocks_helper.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/data/leaf_path.h>

namespace NCatboostCuda {
    class TGreedySearchHelper {
    public:
        TGreedySearchHelper(const TDocParallelDataSet& dataSet,
                            const TBinarizedFeaturesManager& featuresManager,
                            const TTreeStructureSearcherOptions& options,
                            ui32 statCount,
                            TGpuAwareRandom& random)
            : FeaturesManager(featuresManager)
            , Options(options)
            , SplitPropsHelper(dataSet,
                               featuresManager,
                               GetComputeByBlocksHelper(dataSet, options, statCount))
            , Random(random)
        {
        }

        TPointsSubsets CreateInitialSubsets(const IWeakObjective& objective);

        void ComputeOptimalSplits(TPointsSubsets* subsets);

        /* Try to split and compute histograms
         * On terminate condition (like after points split we reach max-depth/max-leaf-count/min-leaf-size/etc)
         * write to resultPtrs optimal values
          */
        bool SplitLeaves(TPointsSubsets* subsets,
                         TVector<TLeafPath>* resultLeaves,
                         TVector<double>* resultsLeafWeights,
                         TVector<TVector<float>>* resultsLeafValues);

    private:
        bool IsTerminalLeaf(const TPointsSubsets& subsets, ui32 leafId);

        bool ShouldTerminate(const TPointsSubsets& subsets);

        void MarkTerminal(const TVector<ui32>& ids, TPointsSubsets* subsets);

        bool AreAllTerminal(const TPointsSubsets& subsets,
                            const TVector<ui32>& leaves);

        void SelectLeavesToSplit(const TPointsSubsets& subsets,
                                 TVector<ui32>* leavesToSplit);

        void SelectLeavesToVisit(const TPointsSubsets& subsets,
                                 TVector<ui32>* leavesToVisit);

        bool IsObliviousSplit() const {
            return Options.Policy == EGrowPolicy::SymmetricTree;
        }

        bool HaveFixedSplits(ui32 depth) const;

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const TTreeStructureSearcherOptions& Options;
        TSplitPropertiesHelper SplitPropsHelper;
        TGpuAwareRandom& Random;
        double ScoreStdDev = 0;
    };

}
