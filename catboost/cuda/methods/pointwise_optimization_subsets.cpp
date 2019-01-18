#include "pointwise_optimization_subsets.h"

namespace NCatboostCuda {
    TOptimizationSubsets<NCudaLib::TStripeMapping>
    TSubsetsHelper<NCudaLib::TStripeMapping>::CreateSubsets(const ui32 maxDepth,
                                                            const TL2Target<NCudaLib::TStripeMapping>& src) {
        TOptimizationSubsets<NCudaLib::TStripeMapping, false> subsets;
        subsets.Bins.Reset(src.WeightedTarget.GetMapping());
        subsets.Indices.Reset(src.WeightedTarget.GetMapping());

        subsets.CurrentDepth = 0;
        subsets.FoldCount = 0;
        subsets.FoldBits = 0;
        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(maxPartCount));
        subsets.PartitionStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(maxPartCount));

        FillBuffer(subsets.Bins, 0u);
        MakeSequence(subsets.Indices);

        UpdateSubsetsStats(src,
                           &subsets);
        return subsets;
    }

    void TSubsetsHelper<NCudaLib::TStripeMapping>::Split(const TL2Target<NCudaLib::TStripeMapping>& sourceTarget,
                                                         const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& cindex,
                                                         const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& docsForBins,
                                                         const NCudaLib::TDistributedObject<TCFeature>& feature,
                                                         ui32 bin,
                                                         TOptimizationSubsets<NCudaLib::TStripeMapping, false>* subsets) {
        auto& profiler = NCudaLib::GetProfiler();
        {
            auto guard = profiler.Profile(TStringBuilder() << "Update bins");
            UpdateBinFromCompressedIndex(cindex,
                                         feature,
                                         bin,
                                         docsForBins,
                                         subsets->CurrentDepth + subsets->FoldBits,
                                         subsets->Bins);
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

}
