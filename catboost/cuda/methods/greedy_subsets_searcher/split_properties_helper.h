#pragma once

#include "structure_searcher_options.h"
#include "compute_by_blocks_helper.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/data/leaf_path.h>
#include <catboost/cuda/targets/weak_objective.h>

namespace NCatboostCuda {
    enum class EHistogramsType {
        Zeroes,
        PreviousPath,
        CurrentPath
    };

    struct TLeaf {
        TLeafPath Path;
        ui32 Size = 0;

        EHistogramsType HistogramsType = EHistogramsType::Zeroes;
        TBestSplitProperties BestSplit;
        bool IsTerminal = false;
    };


    struct TPointsSubsets {
        /* this are ders + weights that will be splitted between leaves during search */
        TOptimizationTarget Target;

        //how to access this leaves
        TStripeBuffer<TDataPartition> Partitions;

        //this leaf sizes
        NCudaLib::TCudaBuffer<TDataPartition, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost> PartitionsCpu;
        //sum of stats in leaves for each devices
        TStripeBuffer<double> PartitionStats;

        //stripped between devices final histograms (already reduced)
        TStripeBuffer<float> Histograms;
        TStripeBuffer<const TCBinFeature> BinFeatures;

        //existed leaves
        TVector<TLeaf> Leaves; //existed leaves

        ui32 GetStatCount() const {
            return static_cast<int>(Target.StatsToAggregate.GetColumnCount());
        }

        TStripeBuffer<const TDataPartition> CurrentParts() const {
            return NCudaLib::ParallelStripeView(Partitions, TSlice(0, Leaves.size()));
        }

        TStripeBuffer<const double> CurrentPartStats() const {
            return NCudaLib::ParallelStripeView(PartitionStats, TSlice(0, Leaves.size()));
        }

        NCudaLib::TCudaBuffer<const TDataPartition, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost> CurrentPartsCpu() const {
            return NCudaLib::ParallelStripeView(PartitionsCpu, TSlice(0, Leaves.size()));
        }
    };

    class TSplitPropertiesHelper {
    public:
        TSplitPropertiesHelper(const TDocParallelDataSet& dataSet,
                               const TBinarizedFeaturesManager& featuresManager,
                               TComputeSplitPropertiesByBlocksHelper& helper)
            : DataSet(dataSet)
            , FeaturesManager(featuresManager)
            , ComputeByBlocksHelper(helper)
            , MaxStreamCount(2)
        {
        }

        TPointsSubsets CreateInitialSubsets(TOptimizationTarget&& target,
                                            ui32 maxLeaves);

        /* Lazy call on demand */
        void BuildNecessaryHistograms(TPointsSubsets* subsets);

        void MakeSplit(const TVector<ui32>& leafIds,
                       TPointsSubsets* subsets);

        const TDocParallelDataSet& GetDataSet() const {
            return DataSet;
        }

    private:
        ui32 GetStream(size_t i) const {
            if (Streams.size()) {
                return Streams[i % Streams.size()].GetId();
            } else {
                return 0;
            }
        }

        void ComputeSplitProperties(ELoadFromCompressedIndexPolicy loadPolicy,
                                    const TVector<ui32>& leavesToCompute,
                                    TPointsSubsets* subsetsPtr);

        void ZeroLeavesHistograms(const TVector<ui32>& leaves,
                                  TPointsSubsets* subsets);

        void SubstractHistograms(const TVector<ui32>& from,
                                 const TVector<ui32>& what,
                                 TPointsSubsets* subsets);

    private:
        const TDocParallelDataSet& DataSet;
        const TBinarizedFeaturesManager& FeaturesManager;
        TComputeSplitPropertiesByBlocksHelper& ComputeByBlocksHelper;
        ui32 MaxStreamCount;

        TVector<TComputationStream> Streams;
    };

}
