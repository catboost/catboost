#pragma once

#include "split_properties_helper.h"
#include "structure_searcher_options.h"
#include "greedy_search_helper.h"
#include "model_builder.h"
#include "weak_objective_impl.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/methods/helpers.h>
#include <catboost/private/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/targets/weak_objective.h>

namespace NCatboostCuda {
    template <class TTreeModel>
    class TGreedyTreeLikeStructureSearcher {
    public:
        using TDataSet = TDocParallelDataSet;
        using TVec = TStripeBuffer<float>;
        using TSampelsMapping = NCudaLib::TStripeMapping;
        using TModel = TTreeModel;

    public:
        TGreedyTreeLikeStructureSearcher(const TBinarizedFeaturesManager& featuresManager,
                                         const TTreeStructureSearcherOptions& searcherOptions)
            : FeaturesManager(featuresManager)
            , SearcherOptions(searcherOptions)
        {
        }

        template <class TObjective>
        TModel Fit(const TDataSet& dataSet, const TObjective& objective) {
            TWeakObjective<TObjective> weakObjectiveWrapper(objective);
            return FitImpl(dataSet, weakObjectiveWrapper);
        }

    private:
        TModel FitImpl(const TDataSet& dataSet,
                       const IWeakObjective& objective) {
            TGreedySearchHelper searchHelper(dataSet,
                                             FeaturesManager,
                                             SearcherOptions,
                                             objective.GetDim() + 1,
                                             objective.GetRandom());

            TPointsSubsets subsets = searchHelper.CreateInitialSubsets(objective);

            TVector<TLeafPath> leaves;
            TVector<double> leavesWeights;
            TVector<TVector<float>> leavesValues;

            while (true) {
                searchHelper.ComputeOptimalSplits(&subsets);

                if (!searchHelper.SplitLeaves(&subsets,
                                              &leaves,
                                              &leavesWeights,
                                              &leavesValues))
                {
                    break;
                }
            }

            return BuildTreeLikeModel<TModel>(leaves, leavesWeights, leavesValues);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        TTreeStructureSearcherOptions SearcherOptions;
    };
}
