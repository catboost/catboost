#pragma once

#include "pointwise_kernels.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

struct TL2Target {
    TMirrorBuffer<float> WeightedTarget;
    TMirrorBuffer<float> Weights;
};

void GatherTarget(TL2Target& to, const TL2Target& from,
                  const TMirrorBuffer<ui32>& indices) {
    auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Gather target and weights");

    to.Weights.Reset(from.Weights.GetMapping());
    to.WeightedTarget.Reset(from.WeightedTarget.GetMapping());

    CB_ENSURE(to.Weights.GetObjectsSlice() == from.Weights.GetObjectsSlice());
    CB_ENSURE(to.Weights.GetObjectsSlice() == indices.GetObjectsSlice());

    Gather(to.WeightedTarget, from.WeightedTarget, indices);
    Gather(to.Weights, from.Weights, indices);
}

template <class TMapping>
inline TBestSplitProperties BestSplit(const TCudaBuffer<TBestSplitProperties, TMapping>& optimalSplits) {
    yvector<TBestSplitProperties> best;
    optimalSplits.Read(best);
    TBestSplitProperties minScr = best[0];

    for (auto scr : best) {
        if (scr.Score < minScr.Score) {
            minScr = scr;
        }
    }
    return minScr;
}

//TODO(noxoomo): class with private fields…
struct TOptimizationSubsets {
    TL2Target* Src;

    TMirrorBuffer<ui32> Bins;
    TMirrorBuffer<ui32> Indices;
    TMirrorBuffer<TDataPartition> Partitions;
    TL2Target GatheredTarget;

    ui32 FoldCount = 0;
    ui32 CurrentDepth = 0;
    ui32 FoldBits = 0;

    void Split(const TMirrorBuffer<ui32>& nextLevelDocBins,
               const TMirrorBuffer<ui32>& docMap) {
        auto& profiler = NCudaLib::GetProfiler();
        {
            auto guard = profiler.Profile(TStringBuilder() << "Update bins");
            UpdateBins(Bins, nextLevelDocBins, docMap, CurrentDepth, FoldBits);
        }
        {
            auto guard = profiler.Profile(TStringBuilder() << "Reorder bins");
            ReorderBins(Bins, Indices, CurrentDepth + FoldBits, 1);
        }
        ++CurrentDepth;
        Update();
    }

    void Update() {
        auto currentParts = CurrentPartsView();
        UpdatePartitionDimensions(Bins, currentParts);
        GatherTarget(GatheredTarget, *Src, Indices);
    }

    TMirrorBuffer<const TPartitionStatistics> ComputePartitionStats() {
        auto currentParts = CurrentPartsView();
        auto partStats = TMirrorBuffer<TPartitionStatistics>::CopyMapping(currentParts);
        UpdatePartitionStats(partStats, currentParts,
                             GatheredTarget.WeightedTarget, GatheredTarget.Weights);
        return partStats.ConstCopyView();
    }

    TMirrorBuffer<TDataPartition> CurrentPartsView() {
        auto currentSlice = TSlice(0, static_cast<ui64>(1 << (CurrentDepth + FoldBits)));
        return Partitions.SliceView(currentSlice);
    }

    TMirrorBuffer<const TDataPartition> CurrentPartsView() const {
        auto currentSlice = TSlice(0, static_cast<ui64>(1 << (CurrentDepth + FoldBits)));
        return Partitions.SliceView(currentSlice);
    }
};

template <class TGridPolicy,
          class TLayoutPolicy = TCatBoostPoolLayout>
class TScoreHelper: public TMoveOnly {
public:
    using TGpuDataSet = TGpuBinarizedDataSet<TGridPolicy, TLayoutPolicy>;
    using TFeaturesMapping = typename TGpuDataSet::TFeaturesMapping;

public:
    TScoreHelper(const TGpuDataSet& dataSet,
                 ui32 foldCount,
                 ui32 maxDepth,
                 EScoreFunction score = EScoreFunction::Correlation,
                 double l2 = 1.0,
                 bool normalize = false,
                 bool requestStream = true)
        : DataSet(&dataSet)
        , Stream(requestStream ? NCudaLib::GetCudaManager().RequestStream() : NCudaLib::GetCudaManager().DefaultStream())
        , FoldCount(foldCount)
        , MaxDepth(maxDepth)
        , ScoreFunction(score)
        , L2(l2)
        , Normalize(normalize)
    {
        auto histMapping = dataSet.GetBinaryFeatures().GetMapping().Transform([&](const TSlice& features) -> ui64 {
            return (1 << maxDepth) * foldCount * features.Size() * 2;
        });

        Histograms.Reset(histMapping);

        const ui64 blockCount = 32;
        auto bestSplitMapping = dataSet.GetBinaryFeatures().GetMapping().Transform([&](const TSlice& features) -> ui64 {
            return std::min(NHelpers::CeilDivide(features.Size(), 128), blockCount);
        });

        BestScores.Reset(bestSplitMapping);
    }

    TScoreHelper& SubmitCompute(const TOptimizationSubsets& newSubsets,
                                const TMirrorBuffer<ui32>& docs) {
        Y_ASSERT(DataSet);
        ++CurrentBit;
        if (static_cast<ui32>(CurrentBit) != newSubsets.CurrentDepth || CurrentBit == 0) {
            BuildFromScratch = true;
            CurrentBit = newSubsets.CurrentDepth;
        }
        if (BuildFromScratch) {
            FillBuffer(Histograms, 0.0f, Stream);
        }

        if (DataSet->GetFeatureCount()) {
            auto& profiler = NCudaLib::GetProfiler();
            auto guard = profiler.Profile(TStringBuilder() << "Compute histograms for features #" << DataSet->GetHostFeatures().size() << " depth " << CurrentBit);
            ComputeHistogram2<TGpuDataSet>(*DataSet,
                                           newSubsets.GatheredTarget.WeightedTarget,
                                           newSubsets.GatheredTarget.Weights,
                                           docs,
                                           newSubsets.Partitions,
                                           static_cast<ui32>(1 << CurrentBit),
                                           FoldCount,
                                           Histograms,
                                           BuildFromScratch,
                                           static_cast<ui32>(Stream.GetId()));

            BuildFromScratch = false;
            Computing = true;
        }
        return *this;
    }

    yvector<float> ReadHistograms() const {
        yvector<float> dst;
        TCudaBuffer<float, TFeaturesMapping> gatheredHistogramsByLeaves;

        auto currentStripe = DataSet->GetBinaryFeatures().GetMapping().Transform([&](const TSlice& features) -> ui64 {
            return (1 << CurrentBit) * FoldCount * features.Size() * 2;
        });
        gatheredHistogramsByLeaves.Reset(currentStripe);

        if (DataSet->GetFeatureCount()) {
            GatherHistogramByLeaves(Histograms,
                                    DataSet->GetBinFeatureCount(),
                                    2,
                                    static_cast<ui32>(1 << (CurrentBit)),
                                    FoldCount,
                                    gatheredHistogramsByLeaves,
                                    Stream.GetId());
        }
        gatheredHistogramsByLeaves.Read(dst);
        return dst;
    }

    TScoreHelper& ComputeOptimalSplit(const TMirrorBuffer<const TPartitionStatistics>& partStats,
                                      double scoreStdDev = 0,
                                      ui64 seed = 0) {
        auto& profiler = NCudaLib::GetProfiler();
        if (DataSet->GetFeatureCount()) {
            {
                auto guard = profiler.Profile(TStringBuilder() << "Find optimal split #" << DataSet->GetBinaryFeatures().GetObjectsSlice().Size());

                FindOptimalSplit(DataSet->GetBinaryFeatures(),
                                 Histograms,
                                 partStats,
                                 FoldCount,
                                 BestScores,
                                 ScoreFunction,
                                 L2,
                                 Normalize,
                                 scoreStdDev,
                                 seed,
                                 Stream);
            }
        }
        return *this;
    }

    TBestSplitProperties ReadAndRemapOptimalSplit() {
        if (DataSet->GetFeatureCount()) {
            auto split = BestSplit(BestScores);
            auto feature = DataSet->GetFeatureByLocalId(split.FeatureId);
            return {DataSet->GetFeatureId(feature.Index), split.BinId, split.Score};
        } else {
            return {static_cast<ui32>(-1), 0, std::numeric_limits<float>::infinity()};
        }
    }

    TCudaBuffer<float, TFeaturesMapping>& GetHistograms() {
        EnsureHistCompute();
        return Histograms;
    }

private:
    void EnsureHistCompute() {
        if (Computing) {
            Stream.Synchronize();
            Computing = false;
        }
    }

private:
    const TGpuDataSet* DataSet = nullptr;
    NCudaLib::TCudaManager::TComputationStream Stream;

    ui32 FoldCount;
    ui32 MaxDepth;
    int CurrentBit = -1;
    bool BuildFromScratch = true;
    bool Computing = false;
    EScoreFunction ScoreFunction;
    double L2 = 1.0;
    bool Normalize = false;
    ui64 RandomSeed = 0;
    TCudaBuffer<float, TFeaturesMapping> Histograms;
    //TODO: do it on slave
    TCudaBuffer<TBestSplitProperties, TFeaturesMapping> BestScores;
};
