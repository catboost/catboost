#pragma once

#include "pairwise_optimization_subsets.h"
#include "blocked_histogram_helper.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/reduce_scatter.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/methods/pointwise_kernels.h>
#include <catboost/cuda/methods/histograms_helper.h>

namespace NCatboostCuda {
    struct TBinaryFeatureSplitResults {
        TStripeBuffer<TCBinFeature> BinFeatures;
        TStripeBuffer<float> Scores;
        TStripeBuffer<float> Solutions;

        //optional, will be used for tests only
        THolder<TStripeBuffer<float>> LinearSystems;
        THolder<TStripeBuffer<float>> SqrtMatrices;

        void ReadBestSolution(ui32 idx, TVector<float>* resultPtr) {
            const ui32 rowSize = Solutions.GetMapping().SingleObjectSize();
            auto& result = *resultPtr;

            Solutions
                .CreateReader()
                .SetReadSlice((TSlice(idx, (idx + 1))))
                .Read(result);

            CB_ENSURE(result.size() == rowSize);
        }
    };

    class TComputePairwiseScoresHelper: public TMoveOnly, public TGuidHolder {
    private:
        struct TTempData {
            TVector<TStripeBuffer<float>> LinearSystems;
            TVector<TStripeBuffer<float>> SqrtMatrices;
        };

        using TGpuDataSet = typename TSharedCompressedIndex<TDocParallelLayout>::TCompressedDataSet;
        using TFeaturesMapping = typename TDocParallelLayout::TFeaturesMapping;
        using TSamplesMapping = typename TDocParallelLayout::TSamplesMapping;

    public:
        TComputePairwiseScoresHelper(EFeaturesGroupingPolicy policy,
                                     const TGpuDataSet& dataSet,
                                     const TPairwiseOptimizationSubsets& subsets,
                                     ui32 maxDepth,
                                     double l2Reg,
                                     double nonDiagReg)
            : Policy(policy)
            , DataSet(dataSet)
            , Subsets(subsets)
            , MaxDepth(maxDepth)
            , NeedPointwiseWeights(subsets.GetPairwiseTarget().PointDer2OrWeights.GetObjectsSlice().Size() > 0)
            , LambdaDiag(l2Reg)
            , LambdaNonDiag(nonDiagReg)
        {
            Y_VERIFY(MaxDepth < 8);
            ResetHistograms();
        }

        TComputePairwiseScoresHelper& Compute(TScopedCacheHolder& scoresCacheHolder,
                                              TBinaryFeatureSplitResults* result) {
            if (DataSet.GetGridSize(Policy) == 0) {
                return *this;
            }

            const TMirrorBuffer<const TCBinFeature>& binaryFeatures = DataSet.GetCacheHolder().Cache(DataSet, Policy, [&]() -> TMirrorBuffer<const TCBinFeature> {
                TMirrorBuffer<TCBinFeature> mirrorBinFeatures;
                mirrorBinFeatures.Reset(NCudaLib::TMirrorMapping(DataSet.GetBinFeatures(Policy).size()));
                mirrorBinFeatures.Write(DataSet.GetBinFeatures(Policy));
                return mirrorBinFeatures.ConstCopyView();
            });

            ++CurrentBit;
            Y_ASSERT(CurrentBit >= 0);
            if (static_cast<ui32>(CurrentBit) != Subsets.GetCurrentDepth() || CurrentBit == 0) {
                //            {
                BuildFromScratch = true;
                CurrentBit = Subsets.GetCurrentDepth();
            }

            if (BuildFromScratch) {
                ResetHistograms();
            }

            auto& manager = NCudaLib::GetCudaManager();

            const ui32 rowSize = (1u << (CurrentBit + 1));
            const ui32 singleMatrixSize = rowSize * (rowSize + 1) / 2;
            const ui32 singleLinearSystemSize = singleMatrixSize + rowSize;
            auto& profiler = NCudaLib::GetProfiler();
            auto guard = profiler.Profile(TStringBuilder() << "Compute scores (" << Policy << ") for  #" << DataSet.GetGridSize(Policy)
                                                           << " features, depth " << CurrentBit);

            TBlockedHistogramsHelper blockedHelper(Policy,
                                                   static_cast<ui32>(CurrentBit),
                                                   DataSet.GetCpuGrid(Policy));

            const ui32 blockCount = blockedHelper.GetBlockCount();
            const ui32 workingBlockCount = Min<ui32>(blockCount, MaxStreamCount);

            for (ui32 i = Streams.size(); i < workingBlockCount; ++i) {
                Streams.push_back(manager.RequestStream());
            }

            //it's temporary data, will be allocated for the biggest block
            //deallocation should be done after all computations to achieve compute/copy overlaps
            //linear systems are in ABCD ABCD ABCD ABCD layout
            //sqrt matrices are reduce-scattered data: A B C D
            //tempData will live while scoresCache is not destroyed
            auto& tempData = scoresCacheHolder.Cache(*this, 0, [&]() -> TTempData {
                TTempData data;
                data.LinearSystems.resize(workingBlockCount);
                data.SqrtMatrices.resize(workingBlockCount);
                {
                    using TMappingBuilder = NCudaLib::TMappingBuilder<NCudaLib::TStripeMapping>;

                    TVector<TMappingBuilder> linearSystemMapping(workingBlockCount);
                    TVector<TMappingBuilder> sqrtMatricesMapping(workingBlockCount);

                    for (ui32 blockId = 0; blockId < blockCount; ++blockId) {
                        const ui32 workingBlockId = blockId % Streams.size();
                        auto& linearSystemBlockMappingBuilder = linearSystemMapping[workingBlockId];
                        auto& matrixBlockMappingBuilder = sqrtMatricesMapping[workingBlockId];

                        auto reduceMapping = blockedHelper.ReduceMapping(blockId);

                        for (auto dev : reduceMapping.NonEmptyDevices()) {
                            matrixBlockMappingBuilder.UpdateMaxSizeAt(dev, reduceMapping.DeviceSlice(dev).Size() *
                                                                               singleMatrixSize);
                        }

                        const ui32 linearSystemMemoryConsumption =
                            blockedHelper.GetBinFeatureCount(blockId) * singleMatrixSize;

                        for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
                            linearSystemBlockMappingBuilder.UpdateMaxSizeAt(dev,
                                                                            linearSystemMemoryConsumption);
                        }
                    }

                    for (ui32 workingBlock = 0; workingBlock < workingBlockCount; ++workingBlock) {
                        data.LinearSystems[workingBlock].Reset(linearSystemMapping[workingBlock].Build());
                        data.SqrtMatrices[workingBlock].Reset(sqrtMatricesMapping[workingBlock].Build());
                    }
                }
                return data;
            });

            auto flatResultsMapping = blockedHelper.GetFlatResultsMapping();

            //vector equal to leaf count
            result->Solutions.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
                return slice.Size();
            },
                                                                 rowSize));

            //one per bin feature
            result->BinFeatures.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
                return slice.Size();
            }));

            //one per score
            result->Scores.Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
                return slice.Size();
            }));

            if (result->LinearSystems) {
                result->LinearSystems->Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
                    return slice.Size();
                },
                                                                          singleLinearSystemSize));
            }

            if (result->SqrtMatrices) {
                result->SqrtMatrices->Reset(flatResultsMapping.Transform([&](const TSlice slice) -> ui64 {
                    return slice.Size();
                },
                                                                         singleMatrixSize));
            }

            for (ui32 blockId = 0; blockId < blockCount; ++blockId) {
                const ui32 groupId = blockId % Streams.size();
                const ui32 streamId = Streams[groupId].GetId();

                auto& linearSystem = tempData.LinearSystems[groupId];
                auto& sqrtMatrix = tempData.SqrtMatrices[groupId];

                auto blockGrid = blockedHelper.GetFeatures(DataSet.GetGrid(Policy),
                                                           blockId);

                auto blockFoldsHist = blockedHelper.ComputeFoldsHistogram(blockId);
                auto blockBinFeaturesSlice = blockedHelper.GetBinFeatureSlice(blockId);

                const ui32 blockBinFeaturesCount = blockBinFeaturesSlice.Size();

                const auto& gatheredByLeavesTarget = Subsets.GetPairwiseTarget();

                if (NeedPointwiseWeights) {
                    Y_VERIFY(gatheredByLeavesTarget.PointDer2OrWeights.GetObjectsSlice().Size(),
                             "No weights, use hist1 instead");

                    ComputeBlockHistogram2(Policy,
                                           blockGrid,
                                           blockFoldsHist,
                                           blockBinFeaturesSlice,
                                           DataSet.GetCompressedIndex(),
                                           gatheredByLeavesTarget.PointWeightedDer,
                                           gatheredByLeavesTarget.PointDer2OrWeights,
                                           gatheredByLeavesTarget.Docs,
                                           Subsets.GetPointPartitions(),
                                           static_cast<ui32>(1 << CurrentBit),
                                           PointwiseHistograms,
                                           HistogramLineSize /* = total number of bin features */,
                                           BuildFromScratch,
                                           streamId);

                } else {
                    Y_VERIFY(gatheredByLeavesTarget.PointDer2OrWeights.GetObjectsSlice().Size() == 0,
                             "There are weights, use hist2 instead");
                    Y_VERIFY(PointwiseHistograms.GetMapping().SingleObjectSize() == 1);

                    ComputeBlockHistogram1(Policy,
                                           blockGrid,
                                           blockBinFeaturesSlice,
                                           DataSet.GetCompressedIndex(),
                                           gatheredByLeavesTarget.PointWeightedDer,
                                           gatheredByLeavesTarget.Docs,
                                           Subsets.GetPointPartitions(),
                                           static_cast<ui32>(1 << CurrentBit),
                                           PointwiseHistograms,
                                           HistogramLineSize /* = total number of bin features */,
                                           BuildFromScratch,
                                           streamId);
                }

                {
                    auto pairHistGuard = profiler.Profile(TStringBuilder() << "Pairwise hist (" << Policy << ") for  #" << blockGrid.GetObjectsSlice().Size()
                                                                           << " binary features, depth " << CurrentBit);

                    ComputeBlockPairwiseHist2(Policy,
                                              blockGrid,
                                              blockBinFeaturesSlice,
                                              DataSet.GetCompressedIndex(),
                                              gatheredByLeavesTarget.PairDer2OrWeights,
                                              gatheredByLeavesTarget.Pairs,
                                              Subsets.GetPairPartitions(),
                                              Subsets.GetPairPartStats(),
                                              CurrentBit,
                                              HistogramLineSize /* = total number of bin features */,
                                              BuildFromScratch,
                                              PairwiseHistograms,
                                              streamId);
                }

                //histograms are flat
                //linear system is for block
                //Reset will not reallocate memory as usual
                linearSystem.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(blockBinFeaturesCount,
                                                                                singleLinearSystemSize));
                FillBuffer(linearSystem, 0.0f, streamId);

                MakeLinearSystem(PointwiseHistograms,
                                 Subsets.GetPointPartitionStats(),
                                 PairwiseHistograms,
                                 HistogramLineSize /* = total number of bin features */,
                                 blockBinFeaturesSlice,
                                 linearSystem,
                                 streamId);

                {
                    auto reducedMapping = blockedHelper.ReduceMapping(blockId);
                    auto reducedLinearSystemsMapping = reducedMapping.Transform([&](const TSlice slice) -> ui64 {
                        return slice.Size();
                    },
                                                                                singleLinearSystemSize);

                    ReduceScatter(linearSystem,
                                  reducedLinearSystemsMapping,
                                  IsReduceCompressed(),
                                  streamId);
                }

                {
                    auto sqrtMatrixMapping = blockedHelper.ReduceMapping(blockId).Transform([&](const TSlice slice) -> ui64 {
                        return slice.Size();
                    },
                                                                                            singleMatrixSize);
                    sqrtMatrix.Reset(sqrtMatrixMapping);
                }

                auto flatResultsSlice = blockedHelper.GetFlatResultsSlice(blockId);

                //cholesky decomposition is done in-place on provided buffers + we need to store original system for score computation
                //linear system and sqrtMatrix are  zero-index
                //solutions are flat indexing for all blocks (so we could find best result in one kernel)

                PrepareSystemForCholesky(linearSystem,
                                         sqrtMatrix,
                                         flatResultsSlice,
                                         result->Solutions,
                                         streamId);

                Regularize(sqrtMatrix,
                           LambdaNonDiag, //bayesian -F weights adjust
                           LambdaDiag,    //classic l2 adjust
                           streamId);

                //if only pairwise ders, then we don't need last row
                const bool removeLastRow = !NeedPointwiseWeights;

                //will mutate sqrtMatrix
                CholeskySolver(sqrtMatrix,
                               flatResultsSlice,
                               result->Solutions,
                               removeLastRow,
                               streamId);

                ComputeScores(binaryFeatures,
                              blockedHelper.GetBinFeatureAccessorSlice(blockId),
                              linearSystem,
                              flatResultsSlice,
                              result->Solutions,
                              result->BinFeatures,
                              result->Scores,
                              streamId);

                if (result->LinearSystems) {
                    CopyReducedTempResult(linearSystem,
                                          flatResultsSlice,
                                          *result->LinearSystems,
                                          streamId);
                }

                if (result->SqrtMatrices) {
                    CopyReducedTempResult(sqrtMatrix,
                                          flatResultsSlice,
                                          *result->SqrtMatrices,
                                          streamId);
                }
            }
            Synchronized = false;
            BuildFromScratch = false;
            return *this;
        }

        void EnsureCompute() {
            if (!Synchronized) {
                for (const auto& stream : Streams) {
                    stream.Synchronize();
                }
                Synchronized = true;
            }
        };

    private:
        void ResetHistograms() {
            const ui32 binFeatureCount = DataSet.GetBinFeatures(Policy).size();

            auto pairwiseHistMapping = DataSet.GetHistogramsMapping(Policy).Transform([&](const TSlice& binFeaturesSlice) -> ui64 {
                ui32 maxParts = (1 << (MaxDepth - 1));
                Y_VERIFY(binFeatureCount == binFeaturesSlice.Size());
                return maxParts * maxParts * binFeaturesSlice.Size();
            },
                                                                                      4);

            auto pointwiseHistMapping = DataSet.GetHistogramsMapping(Policy).Transform([&](const TSlice& binFeaturesSlice) -> ui64 {
                Y_VERIFY(binFeatureCount == binFeaturesSlice.Size());
                ui32 maxParts = (1 << (MaxDepth - 1));
                return maxParts * binFeaturesSlice.Size();
            },
                                                                                       (NeedPointwiseWeights ? 2 : 1));

            HistogramLineSize = binFeatureCount;

            PairwiseHistograms.Reset(pairwiseHistMapping);
            PointwiseHistograms.Reset(pointwiseHistMapping);

            FillBuffer(PairwiseHistograms, 0.0f);
            FillBuffer(PointwiseHistograms, 0.0f);

            NCudaLib::GetCudaManager().DefaultStream().Synchronize();
        }

    private:
        EFeaturesGroupingPolicy Policy;
        const TGpuDataSet& DataSet;
        const TPairwiseOptimizationSubsets& Subsets;
        ui32 MaxDepth;
        int CurrentBit = -1;
        bool BuildFromScratch = true;
        //if we need to add pointwise der2 (or weights): need for llmax-like targets
        bool NeedPointwiseWeights = false;
        double LambdaDiag = 0.0;
        double LambdaNonDiag = 0.1;

        TCudaBuffer<float, TFeaturesMapping> PairwiseHistograms;
        TCudaBuffer<float, TFeaturesMapping> PointwiseHistograms;

        //memory used by one part. parts are accessed via partId * partSize + binFeatureId * histCount
        ui64 HistogramLineSize = 0;

        TVector<TComputationStream> Streams;
        bool Synchronized = true;

        const ui32 MaxStreamCount = 8;
    };

}
