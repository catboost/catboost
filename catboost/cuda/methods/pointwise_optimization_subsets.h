#pragma once

#include "weak_target_helpers.h"
#include "pointwise_kernels.h"
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_util/reorder_bins.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/partitions.h>
#include <catboost/cuda/gpu_data/splitter.h>

namespace NCatboostCuda {
    template <class>
    struct TSubsetsHelper;

    template <class TMapping = NCudaLib::TMirrorMapping,
              bool IsConst = false>
    struct TOptimizationSubsets {
        template <class T>
        using TBuffer = typename NCudaLib::TMaybeConstView<IsConst>::template TBuffer<T, TMapping>;

        TBuffer<ui32> Bins;
        TBuffer<ui32> Indices;
        TBuffer<TDataPartition> Partitions;
        TBuffer<TPartitionStatistics> PartitionStats;

        TBuffer<float> WeightedTarget;
        TBuffer<float> Weights;

        ui32 FoldCount = 0;
        ui32 CurrentDepth = 0;
        ui32 FoldBits = 0;

        TOptimizationSubsets<NCudaLib::TSingleMapping, true> DeviceView(ui32 dev) const {
            TOptimizationSubsets<NCudaLib::TSingleMapping, true> result;
            result.Bins = Bins.DeviceView(dev).AsConstBuf();
            result.Indices = Indices.DeviceView(dev).AsConstBuf();
            result.Partitions = Partitions.DeviceView(dev).AsConstBuf();
            result.PartitionStats = PartitionStats.DeviceView(dev).AsConstBuf();
            result.WeightedTarget = WeightedTarget.DeviceView(dev).AsConstBuf();
            result.Weights = Weights.DeviceView(dev).AsConstBuf();

            result.FoldCount = FoldCount;
            result.CurrentDepth = CurrentDepth;
            result.FoldBits = FoldBits;
            return result;
        };

    private:
        template <class>
        friend struct TSubsetsHelper;
    };

    template <class TMapping, class TL2>
    inline void UpdateSubsetsStats(const TL2& source,
                                   TOptimizationSubsets<TMapping, false>* subsetsPtr) {
        auto& subsets = *subsetsPtr;
        auto currentParts = TSubsetsHelper<TMapping>::CurrentPartsView(subsets);
        subsets.PartitionStats.Reset(currentParts.GetMapping());
        UpdatePartitionDimensions(subsets.Bins, currentParts);

        GatherTarget(subsets.WeightedTarget,
                     subsets.Weights,
                     source,
                     subsets.Indices);

        UpdatePartitionStats(subsets.PartitionStats,
                             currentParts,
                             subsets.WeightedTarget,
                             subsets.Weights);
    }

    template <>
    struct TSubsetsHelper<NCudaLib::TMirrorMapping> {
        template <class TTarget>
        static void Split(const TTarget& sourceTarget,
                          const TCudaBuffer<ui32, NCudaLib::TMirrorMapping>& nextLevelDocBins,
                          const TCudaBuffer<ui32, NCudaLib::TMirrorMapping>& docMap,
                          TOptimizationSubsets<NCudaLib::TMirrorMapping, false>* subsets) {
            auto& profiler = NCudaLib::GetProfiler();
            {
                auto guard = profiler.Profile(TStringBuilder() << "Update bins");
                UpdateBins(subsets->Bins, nextLevelDocBins, docMap, subsets->CurrentDepth, subsets->FoldBits);
            }
            {
                auto guard = profiler.Profile(TStringBuilder() << "Reorder bins");
                ReorderBins(subsets->Bins, subsets->Indices,
                            subsets->CurrentDepth + subsets->FoldBits,
                            1);
            }
            ++subsets->CurrentDepth;
            UpdateSubsetsStats(sourceTarget,
                               subsets);
        }

        static TMirrorBuffer<TDataPartition> CurrentPartsView(TOptimizationSubsets<NCudaLib::TMirrorMapping, false>& subsets) {
            auto currentSlice = TSlice(0, static_cast<ui64>(1ULL << (subsets.CurrentDepth + subsets.FoldBits)));
            return subsets.Partitions.SliceView(currentSlice);
        }

        template <bool IsConst>
        static TMirrorBuffer<const TDataPartition> CurrentPartsView(const TOptimizationSubsets<NCudaLib::TMirrorMapping, IsConst>& subsets) {
            auto currentSlice = TSlice(0, static_cast<ui64>(1 << (subsets.CurrentDepth + subsets.FoldBits)));
            return subsets.Partitions.SliceView(currentSlice);
        }
    };

    template <>
    struct TSubsetsHelper<NCudaLib::TStripeMapping> {
        static void Split(const TL2Target<NCudaLib::TStripeMapping>& sourceTarget,
                          const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& cindex,
                          const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& docsForBins,
                          const NCudaLib::TDistributedObject<TCFeature>& feature, ui32 bin,
                          TOptimizationSubsets<NCudaLib::TStripeMapping, false>* subsets);

        static TStripeBuffer<const TDataPartition> CurrentPartsView(const TOptimizationSubsets<NCudaLib::TStripeMapping>& subsets) {
            auto currentSlice = TSlice(0, static_cast<ui64>(1ULL << (subsets.CurrentDepth + subsets.FoldBits)));
            return NCudaLib::ParallelStripeView(subsets.Partitions,
                                                currentSlice).AsConstBuf();
        }

        static TStripeBuffer<TDataPartition> CurrentPartsView(TOptimizationSubsets<NCudaLib::TStripeMapping>& subsets) {
            auto currentSlice = TSlice(0, static_cast<ui64>(1ULL << (subsets.CurrentDepth + subsets.FoldBits)));
            return NCudaLib::ParallelStripeView(subsets.Partitions,
                                                currentSlice);
        }

        static TOptimizationSubsets<NCudaLib::TStripeMapping> CreateSubsets(const ui32 maxDepth,
                                                                            const TL2Target<NCudaLib::TStripeMapping>& src);
    };

}
