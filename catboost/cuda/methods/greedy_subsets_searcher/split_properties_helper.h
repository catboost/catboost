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

        void UpdateBestSplit(const TBestSplitProperties& best) {
            BestSplit = best;
        }

        bool IsTerminal = false;
    };

    struct TSplitCandidate {
        int LeafIdx = -1;
        double Score = 0;

        bool operator<(const TSplitCandidate& rhs) const;
        bool operator>(const TSplitCandidate& rhs) const;
        bool operator<=(const TSplitCandidate& rhs) const;
        bool operator>=(const TSplitCandidate& rhs) const;
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

        TMirrorBuffer<float> FeatureWeights;

        //existed leaves
        TVector<TLeaf> Leaves; //existed leaves

        ui32 GetStatCount() const {
            return static_cast<int>(Target.StatsToAggregate.GetColumnCount());
        }

        TStripeBuffer<const TDataPartition> CurrentParts() const {
            return NCudaLib::ParallelStripeView(Partitions, TSlice(0, Leaves.size())).AsConstBuf();
        }

        TStripeBuffer<const double> CurrentPartStats() const {
            return NCudaLib::ParallelStripeView(PartitionStats, TSlice(0, Leaves.size())).AsConstBuf();
        }

        NCudaLib::TCudaBuffer<const TDataPartition, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost> CurrentPartsCpu() const {
            return NCudaLib::ParallelStripeView(PartitionsCpu, TSlice(0, Leaves.size())).AsConstBuf();
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
            , MaxStreamCount(helper.GetStreamCount())
        {
            if (MaxStreamCount > 1) {
                for (ui32 i = 0; i < MaxStreamCount; ++i) {
                    Streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                }
            } else {
                Streams.push_back(NCudaLib::GetCudaManager().DefaultStream());
            }
        }

        TPointsSubsets CreateInitialSubsets(TOptimizationTarget&& target,
                                            ui32 maxLeaves,
                                            TConstArrayRef<float> featureWeights);

        /* Lazy call on demand */
        void BuildNecessaryHistograms(TPointsSubsets* subsets);

        void MakeSplit(const TVector<ui32>& leafIds,
                       TPointsSubsets* subsets,
                       TVector<ui32>* leftIds,
                       TVector<ui32>* rightIds);

        void MakeSplit(const TVector<ui32>& leafIds,
                       TPointsSubsets* subsets) {
            TVector<ui32> leftIds;
            TVector<ui32> rightIds;
            MakeSplit(leafIds, subsets, &leftIds, &rightIds);
        }

        const TDocParallelDataSet& GetDataSet() const {
            return DataSet;
        }

    private:
        void MakeSplit(const ui32 leafId,
                       TPointsSubsets* subsets,
                       TVector<ui32>* leftIds,
                       TVector<ui32>* rightIds);

    private:
        bool IsOnlyDefaultStream() const {
            return Streams.size() == 0 || (Streams.size() == 1 && Streams.back().GetId() == 0);
        }

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
