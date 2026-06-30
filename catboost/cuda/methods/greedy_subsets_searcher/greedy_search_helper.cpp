#include "greedy_search_helper.h"
#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/compute_scores.cuh>
#include <catboost/cuda/methods/update_feature_weights.h>
#include <catboost/cuda/targets/weak_objective.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>

namespace NKernelHost {
    class TComputeOptimalSplitsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCBinFeature> BinaryFeatures;
        TCudaBufferPtr<const float> FeatureWeights;
        TCudaBufferPtr<const float> Histograms;
        TCudaBufferPtr<const double> PartStats;
        TCudaBufferPtr<const ui32> PartIds;
        TCudaBufferPtr<const ui32> RestPartIds;
        ui32 NumScoreBlocks;
        TCudaBufferPtr<TBestSplitProperties> Result;
        bool MultiClassOptimization;
        ui32 ArgmaxBlockCount;
        EScoreFunction ScoreFunction;
        double L2;
        bool Normalize;
        double ScoreStdDev;
        ui64 Seed;

    public:
        TComputeOptimalSplitsKernel() = default;

        TComputeOptimalSplitsKernel(TCudaBufferPtr<const TCBinFeature> binaryFeatures,
                                    TCudaBufferPtr<const float> featureWeights,
                                    TCudaBufferPtr<const float> histograms,
                                    TCudaBufferPtr<const double> partStats,
                                    TCudaBufferPtr<const ui32> partIds,
                                    TCudaBufferPtr<const ui32> restPartIds,
                                    ui32 numScoreBlocks,
                                    TCudaBufferPtr<TBestSplitProperties> result,
                                    bool multiclassLastApproxDimOptimization,
                                    ui32 argmaxBlockCount,
                                    EScoreFunction scoreFunction,
                                    double l2,
                                    bool normalize,
                                    double scoreStdDev,
                                    ui64 seed)
            : BinaryFeatures(binaryFeatures)
            , FeatureWeights(featureWeights)
            , Histograms(histograms)
            , PartStats(partStats)
            , PartIds(partIds)
            , RestPartIds(restPartIds)
            , NumScoreBlocks(numScoreBlocks)
            , Result(result)
            , MultiClassOptimization(multiclassLastApproxDimOptimization)
            , ArgmaxBlockCount(argmaxBlockCount)
            , ScoreFunction(scoreFunction)
            , L2(l2)
            , Normalize(normalize)
            , ScoreStdDev(scoreStdDev)
            , Seed(seed)
        {
        }

        Y_SAVELOAD_DEFINE(BinaryFeatures, FeatureWeights, Histograms, PartStats, PartIds, RestPartIds, NumScoreBlocks, Result, ArgmaxBlockCount,
                          ScoreFunction, L2, Normalize, ScoreStdDev, Seed, MultiClassOptimization);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(PartIds.Size() % NumScoreBlocks == 0);
            const ui32 partBlockSize = PartIds.Size() / NumScoreBlocks;
            CB_ENSURE(partBlockSize, PartIds.Size() << " " << NumScoreBlocks);

            NKernel::ComputeOptimalSplits(BinaryFeatures.Get(), BinaryFeatures.Size(), FeatureWeights.Get(), FeatureWeights.Size(), Histograms.Get(),
                                          PartStats.Get(), PartStats.ObjectSize(), PartIds.Get(), partBlockSize,
                                          NumScoreBlocks, RestPartIds.Get(), RestPartIds.Size(), Result.Get(), ArgmaxBlockCount, ScoreFunction, MultiClassOptimization, L2, Normalize,
                                          ScoreStdDev, Seed, stream.GetStream());
        }
    };

    class TComputeOptimalSplitsLeafwiseKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCBinFeature> BinaryFeatures;
        TCudaBufferPtr<const float> FeatureWeights;
        TCudaBufferPtr<const float> Histograms;
        TCudaBufferPtr<const double> PartStats;
        TCudaBufferPtr<const ui32> PartIds;
        TCudaBufferPtr<TBestSplitProperties> Result;
        bool MultiClassOptimization;
        ui32 ArgmaxBlockCount;
        EScoreFunction ScoreFunction;
        double L2;
        bool Normalize;
        double ScoreStdDev;
        ui64 Seed;

    public:
        TComputeOptimalSplitsLeafwiseKernel() = default;

        TComputeOptimalSplitsLeafwiseKernel(TCudaBufferPtr<const TCBinFeature> binaryFeatures,
                                            TCudaBufferPtr<const float> featureWeights,
                                            TCudaBufferPtr<const float> histograms,
                                            TCudaBufferPtr<const double> partStats,
                                            TCudaBufferPtr<const ui32> partIds,
                                            TCudaBufferPtr<TBestSplitProperties> result,
                                            bool multiclassLastApproxDimOptimization,
                                            ui32 argmaxBlockCount,
                                            EScoreFunction scoreFunction,
                                            double l2,
                                            bool normalize,
                                            double scoreStdDev,
                                            ui64 seed)
            : BinaryFeatures(binaryFeatures)
            , FeatureWeights(featureWeights)
            , Histograms(histograms)
            , PartStats(partStats)
            , PartIds(partIds)
            , Result(result)
            , MultiClassOptimization(multiclassLastApproxDimOptimization)
            , ArgmaxBlockCount(argmaxBlockCount)
            , ScoreFunction(scoreFunction)
            , L2(l2)
            , Normalize(normalize)
            , ScoreStdDev(scoreStdDev)
            , Seed(seed)
        {
        }

        Y_SAVELOAD_DEFINE(BinaryFeatures, FeatureWeights, Histograms, PartStats, PartIds, Result, ArgmaxBlockCount,
                          ScoreFunction, L2, Normalize, ScoreStdDev, Seed, MultiClassOptimization);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeOptimalSplitsRegion(BinaryFeatures.Get(), BinaryFeatures.Size(),
                                                FeatureWeights.Get(), FeatureWeights.Size(),
                                                Histograms.Get(),
                                                PartStats.Get(), PartStats.ObjectSize(), PartIds.Get(), PartIds.Size(),
                                                Result.Get(), ArgmaxBlockCount, ScoreFunction, MultiClassOptimization, L2, Normalize,
                                                ScoreStdDev, Seed, stream.GetStream());
        }
    };

    class TComputeOptimalSplitLeafwiseKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCBinFeature> BinaryFeatures;
        TCudaBufferPtr<const float> FeatureWeights;
        TCudaBufferPtr<const float> Histograms;
        TCudaBufferPtr<const double> PartStats;
        ui32 FirstPartId;
        ui32 MaybeSecondPartId;
        TCudaBufferPtr<TBestSplitProperties> Result;
        bool MultiClassOptimization;
        ui32 ArgmaxBlockCount;
        EScoreFunction ScoreFunction;
        double L2;
        bool Normalize;
        double ScoreStdDev;
        ui64 Seed;

    public:
        TComputeOptimalSplitLeafwiseKernel() = default;

        TComputeOptimalSplitLeafwiseKernel(TCudaBufferPtr<const TCBinFeature> binaryFeatures,
                                           TCudaBufferPtr<const float> featureWeights,
                                           TCudaBufferPtr<const float> histograms,
                                           TCudaBufferPtr<const double> partStats,
                                           ui32 firstPartId,
                                           ui32 maybeSecondPartId,
                                           TCudaBufferPtr<TBestSplitProperties> result,
                                           bool multiclassLastApproxDimOptimization,
                                           ui32 argmaxBlockCount,
                                           EScoreFunction scoreFunction,
                                           double l2,
                                           bool normalize,
                                           double scoreStdDev,
                                           ui64 seed)
            : BinaryFeatures(binaryFeatures)
            , FeatureWeights(featureWeights)
            , Histograms(histograms)
            , PartStats(partStats)
            , FirstPartId(firstPartId)
            , MaybeSecondPartId(maybeSecondPartId)
            , Result(result)
            , MultiClassOptimization(multiclassLastApproxDimOptimization)
            , ArgmaxBlockCount(argmaxBlockCount)
            , ScoreFunction(scoreFunction)
            , L2(l2)
            , Normalize(normalize)
            , ScoreStdDev(scoreStdDev)
            , Seed(seed)
        {
        }

        Y_SAVELOAD_DEFINE(BinaryFeatures, FeatureWeights, Histograms, PartStats, FirstPartId, MaybeSecondPartId, Result, ArgmaxBlockCount,
                          ScoreFunction, L2, Normalize, ScoreStdDev, Seed, MultiClassOptimization);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeOptimalSplit(BinaryFeatures.Get(), BinaryFeatures.Size(),
                                         FeatureWeights.Get(), FeatureWeights.Size(),
                                         Histograms.Get(),
                                         PartStats.Get(), PartStats.ObjectSize(), FirstPartId, MaybeSecondPartId,
                                         Result.Get(), ArgmaxBlockCount, ScoreFunction, MultiClassOptimization, L2, Normalize,
                                         ScoreStdDev, Seed, stream.GetStream());
        }
    };

    class TComputeCurrentTreeScoreKernel: public TStatelessKernel {
    private:
        TCudaHostBufferPtr<const double> PartStats;
        TCudaHostBufferPtr<const ui32> PartIds;
        bool MultiClassOptimization;
        EScoreFunction ScoreFunction;
        double L2;
        bool Normalize;
        double ScoreStdDev;
        ui64 Seed;
        TCudaHostBufferPtr<double> Result;

    public:
        TComputeCurrentTreeScoreKernel() = default;

        TComputeCurrentTreeScoreKernel(TCudaHostBufferPtr<const double> partStats,
                                       TCudaHostBufferPtr<const ui32> partIds,
                                       bool multiclassLastApproxDimOptimization,
                                       EScoreFunction scoreFunction,
                                       double l2,
                                       bool normalize,
                                       double scoreStdDev,
                                       ui64 seed,
                                       TCudaHostBufferPtr<double> result)
            : PartStats(partStats)
            , PartIds(partIds)
            , MultiClassOptimization(multiclassLastApproxDimOptimization)
            , ScoreFunction(scoreFunction)
            , L2(l2)
            , Normalize(normalize)
            , ScoreStdDev(scoreStdDev)
            , Seed(seed)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(PartStats, PartIds, MultiClassOptimization, ScoreFunction, L2, Normalize, ScoreStdDev, Seed, Result);

        void Run(const TCudaStream& stream) const {
            stream.Synchronize();
            NKernel::ComputeTreeScore(PartStats.Get(), PartStats.ObjectSize(), PartIds.Get(), PartIds.Size(),
                                      ScoreFunction, MultiClassOptimization, L2, Normalize,
                                      ScoreStdDev, Seed,
                                      Result.Get(),
                                      stream.GetStream());
        }
    };

    class TComputeTargetVarianceKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Stats;
        TCudaBufferPtr<double> AggregatedStats;
        bool IsMulticlass;

    public:
        TComputeTargetVarianceKernel() = default;

        TComputeTargetVarianceKernel(TCudaBufferPtr<const float> stats, TCudaBufferPtr<double> aggregatedStats, bool isMulticlass)
            : Stats(stats)
            , AggregatedStats(aggregatedStats)
            , IsMulticlass(isMulticlass)
        {
        }

        Y_SAVELOAD_DEFINE(Stats, AggregatedStats, IsMulticlass);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeTargetVariance(Stats.Get(), static_cast<ui32>(Stats.Size()), static_cast<ui32>(Stats.GetColumnCount()), Stats.AlignedColumnSize(), IsMulticlass, AggregatedStats.Get(), stream.GetStream());
        }
    };

}

namespace NCudaLib {
    REGISTER_KERNEL(0xA1BCA11, NKernelHost::TComputeOptimalSplitsKernel);
    REGISTER_KERNEL(0xA1BCA12, NKernelHost::TComputeTargetVarianceKernel);
    REGISTER_KERNEL(0xA1BCA13, NKernelHost::TComputeOptimalSplitLeafwiseKernel);
    REGISTER_KERNEL(0xA1BCA14, NKernelHost::TComputeOptimalSplitsLeafwiseKernel);
    REGISTER_KERNEL(0xA1BCA15, NKernelHost::TComputeCurrentTreeScoreKernel);

}

namespace NCatboostCuda {
    static TOptimizationTarget ComputeTarget(EScoreFunction scoreFunction,
                                             const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                             const IWeakObjective& objective) {
        const bool isSecondOrderScoreFunction = IsSecondOrderScoreFunction(scoreFunction);
        TOptimizationTarget target;
        objective.StochasticDer(bootstrapConfig,
                                isSecondOrderScoreFunction,
                                &target);
        return target;
    }

    TMaybe<ui32> FindBestLeafToSplit(const TPointsSubsets& subsets) {
        double bestScore = std::numeric_limits<double>::infinity();
        TMaybe<ui32> bestLeaf;

        for (size_t i = 0; i < subsets.Leaves.size(); ++i) {
            if (subsets.Leaves[i].BestSplit.Defined()) {
                if (subsets.Leaves[i].BestSplit.Score < bestScore) {
                    bestScore = subsets.Leaves[i].BestSplit.Score;
                    bestLeaf = i;
                }
            }
        }
        return bestLeaf;
    };

    inline ui32 FindMaxDepth(const TVector<TLeaf>& leaves) {
        ui32 depth = 0;
        for (auto& leaf : leaves) {
            depth = Max<ui32>(leaf.Path.GetDepth(), depth);
        }
        return depth;
    }

    void TGreedySearchHelper::SelectLeavesToSplit(const TPointsSubsets& subsets,
                                                  TVector<ui32>* leavesToSplit) {
        if (Options.Policy == EGrowPolicy::Lossguide) {
            TMaybe<ui32> leafToSplit = FindBestLeafToSplit(subsets);

            for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
                if (leafToSplit.Defined() && leaf == *leafToSplit) {
                    leavesToSplit->push_back(leaf);
                }
            }
        } else if (Options.Policy == EGrowPolicy::Region) {
            if (subsets.Leaves.size() > 1) {
                //split one of last leaf always
                ui32 maxDepth = FindMaxDepth(subsets.Leaves);
                TVector<ui32> maxDepthLeaves;

                leavesToSplit->clear();

                for (ui32 i = 0; i < subsets.Leaves.size(); ++i) {
                    auto& leaf = subsets.Leaves[i];
                    if (leaf.Path.GetDepth() == maxDepth) {
                        maxDepthLeaves.push_back(i);
                    }
                }

                Sort(maxDepthLeaves.begin(), maxDepthLeaves.end(), [&](ui32 left, ui32 right) -> bool {
                    return subsets.Leaves[left].BestSplit.Score < subsets.Leaves[right].BestSplit.Score;
                });

                CB_ENSURE(maxDepthLeaves.size() == 2);
                const ui32 leafToSplitId = maxDepthLeaves[0];
                const auto& leaf = subsets.Leaves[leafToSplitId];
                if (leaf.BestSplit.Defined() && leaf.BestSplit.Score < 0) {
                    leavesToSplit->push_back(leafToSplitId);
                }
            } else {
                leavesToSplit->push_back(0);
            }
        } else {
            CB_ENSURE(Options.Policy == EGrowPolicy::SymmetricTree || Options.Policy == EGrowPolicy::Depthwise);

            for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
                if (subsets.Leaves[leaf].BestSplit.Defined() && subsets.Leaves[leaf].BestSplit.Score < 0) {
                    leavesToSplit->push_back(leaf);
                }
            }
        }
    }

    static inline double ComputeTargetStdDev(const TOptimizationTarget& target) {
        using TKernel = NKernelHost::TComputeTargetVarianceKernel;
        auto l2Stats = TStripeBuffer<double>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(3));
        LaunchKernels<TKernel>(l2Stats.NonEmptyDevices(), 0, target.StatsToAggregate, l2Stats, target.MultiLogitOptimization);
        auto l2StatsCpu = ReadReduce(l2Stats);
        //        double sum = l2StatsCpu[0];
        double sum2 = l2StatsCpu[1];
        double weight = l2StatsCpu[2];
        return sqrt(sum2 / (weight + 1e-100));
    }

    TPointsSubsets TGreedySearchHelper::CreateInitialSubsets(const IWeakObjective& objective) {
        auto target = ComputeTarget(Options.ScoreFunction,
                                    Options.BootstrapOptions,
                                    objective);
        if (Options.RandomStrength) {
            ScoreStdDev = Options.RandomStrength * ComputeTargetStdDev(target);
        } else {
            ScoreStdDev = 0;
        }
        return SplitPropsHelper.CreateInitialSubsets(std::move(target),
                                                     Options.MaxLeaves,
                                                     Options.FeatureWeights);
    }

    bool TGreedySearchHelper::HaveFixedSplits(ui32 depth) const {
        return Options.FixedBinarySplits.size() > depth; // allow empty leaves after fixed splits
    }

    void TGreedySearchHelper::ComputeOptimalSplits(TPointsSubsets* subsets) {
        SplitPropsHelper.BuildNecessaryHistograms(subsets);

        TVector<ui32> leavesToVisit;
        SelectLeavesToVisit(*subsets, &leavesToVisit);
        if (leavesToVisit.empty()) {
            return;
        }

        const auto depth = FindMaxDepth(subsets->Leaves);
        if (HaveFixedSplits(depth)) {
            TBestSplitProperties bestSplit;
            bestSplit.BinId = 0;
            bestSplit.FeatureId = Options.FixedBinarySplits[depth];
            bestSplit.Score = -std::numeric_limits<float>::infinity();
            bestSplit.Gain = -std::numeric_limits<float>::infinity();
            for (auto leafId : leavesToVisit) {
                subsets->Leaves[leafId].UpdateBestSplit(bestSplit);
            }
            return;
        }

        ui32 numScoreBlocks = 1;
        switch (Options.Policy) {
            case EGrowPolicy::SymmetricTree: {
                numScoreBlocks = 1;
                break;
            }
            case EGrowPolicy::Region:
            case EGrowPolicy::Lossguide:
            case EGrowPolicy::Depthwise: {
                numScoreBlocks = static_cast<ui32>(leavesToVisit.size());
                break;
            }
            default: {
                CB_ENSURE(false, "should be implemented");
            }
        }
        const ui32 binFeatureCountPerDevice = static_cast<const ui32>(NHelpers::CeilDivide(subsets->BinFeatures.GetObjectsSlice().Size(),
                                                                                           NCudaLib::GetCudaManager().GetDeviceCount()));

        const ui32 argmaxBlockCount = Min<ui32>(NHelpers::CeilDivide(binFeatureCountPerDevice, 256), 64);
        auto bestProps = TStripeBuffer<TBestSplitProperties>::Create(NCudaLib::TStripeMapping::RepeatOnAllDevices(argmaxBlockCount * numScoreBlocks));

        //TODO(noxoomo): it's could be slow for lossguide learning
        TMirrorBuffer<double> reducedStats;
        AllReduceThroughMaster(subsets->CurrentPartStats(), reducedStats);

        if (Options.Policy == EGrowPolicy::SymmetricTree) {
            UpdateFeatureWeightsForBestSplits(FeaturesManager, Options.ModelSizeReg, subsets->FeatureWeights);
            TMirrorBuffer<ui32> restLeafIds;
            auto leafIds = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leavesToVisit.size()));
            leafIds.Write(leavesToVisit);

            using TKernel = NKernelHost::TComputeOptimalSplitsKernel;
            LaunchKernels<TKernel>(bestProps.NonEmptyDevices(),
                                   0,
                                   subsets->BinFeatures,
                                   subsets->FeatureWeights,
                                   subsets->Histograms,
                                   reducedStats,
                                   leafIds,
                                   restLeafIds,
                                   numScoreBlocks,
                                   bestProps,
                                   subsets->Target.MultiLogitOptimization,
                                   argmaxBlockCount,
                                   Options.ScoreFunction,
                                   Options.L2Reg,
                                   false,
                                   ScoreStdDev,
                                   Random.NextUniformL());
        } else if (Options.Policy == EGrowPolicy::Depthwise || HaveFixedSplits(depth)) {
            auto leafIds = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leavesToVisit.size()));
            leafIds.Write(leavesToVisit);

            using TKernel = NKernelHost::TComputeOptimalSplitsLeafwiseKernel;
            LaunchKernels<TKernel>(bestProps.NonEmptyDevices(),
                                   0,
                                   subsets->BinFeatures,
                                   subsets->FeatureWeights,
                                   subsets->Histograms,
                                   reducedStats,
                                   leafIds,
                                   bestProps,
                                   subsets->Target.MultiLogitOptimization,
                                   argmaxBlockCount,
                                   Options.ScoreFunction,
                                   Options.L2Reg,
                                   false,
                                   ScoreStdDev,
                                   Random.NextUniformL());

        } else {
            CB_ENSURE(leavesToVisit.size() <= 2, leavesToVisit.size());

            using TKernel = NKernelHost::TComputeOptimalSplitLeafwiseKernel;
            LaunchKernels<TKernel>(bestProps.NonEmptyDevices(),
                                   0,
                                   subsets->BinFeatures,
                                   subsets->FeatureWeights,
                                   subsets->Histograms,
                                   reducedStats,
                                   leavesToVisit[0],
                                   leavesToVisit.size() == 2 ? leavesToVisit[1] : leavesToVisit[0],
                                   bestProps,
                                   subsets->Target.MultiLogitOptimization,
                                   argmaxBlockCount,
                                   Options.ScoreFunction,
                                   Options.L2Reg,
                                   false,
                                   ScoreStdDev,
                                   Random.NextUniformL());
        }

        TVector<TBestSplitProperties> bestSplits(numScoreBlocks);

        {
            TVector<TBestSplitProperties> propsCpu;
            bestProps.Read(propsCpu);
            const ui32 devCount = static_cast<const ui32>(NCudaLib::GetCudaManager().GetDeviceCount());

            for (ui32 dev = 0; dev < devCount; ++dev) {
                const ui32 devOffset = argmaxBlockCount * numScoreBlocks * dev;

                for (ui32 scoreBlockId = 0; scoreBlockId < numScoreBlocks; ++scoreBlockId) {
                    TBestSplitProperties* blockProps = propsCpu.data() + devOffset + scoreBlockId * argmaxBlockCount;
                    for (ui32 i = 0; i < argmaxBlockCount; ++i) {
                        if (blockProps[i] < bestSplits[scoreBlockId]) {
                            bestSplits[scoreBlockId] = blockProps[i];
                        }
                    }
                }
            }
        }

        if (IsObliviousSplit()) {
            CB_ENSURE(
                bestSplits.size() == 1 && bestSplits[0].FeatureId != static_cast<ui32>(-1),
                "All splits have infinite score. "
                "Probably, numerical overflow occurs in loss function and/or split score calculation. "
                "Try increasing l2_leaf_reg, and/or decreasing learning_rate, etc.");
            for (const auto& leafId : leavesToVisit) {
                subsets->Leaves[leafId].UpdateBestSplit(bestSplits[0]);
            }
            if (FeaturesManager.IsCtr(bestSplits[0].FeatureId)) {
                FeaturesManager.AddUsedCtr(bestSplits[0].FeatureId);
            }
        } else {
            CB_ENSURE(bestSplits.size() == leavesToVisit.size(),
                      "Error: for non-oblivious splits we should select best splits for each leaf");

            for (size_t i = 0; i < leavesToVisit.size(); ++i) {
                const ui32 leafId = leavesToVisit[i];
                subsets->Leaves[leafId].UpdateBestSplit(bestSplits[i]);
            }
        }
    }

    void PrintScoreLossguide(const TBinarizedFeaturesManager& featuresManager,
                             const TLeaf& leaf,
                             ui32 iterations) {
        auto splitStr = TStringBuilder() << leaf.BestSplit.FeatureId << " / " << leaf.BestSplit.BinId << " (" << SplitConditionToString(featuresManager, ToSplit(featuresManager, leaf.BestSplit)) << ")";
        TStringBuilder path;
        for (ui32 i = 0; i < leaf.Path.GetDepth(); ++i) {
            const auto& split = leaf.Path.Splits[i];
            path << split.FeatureId << " / " << split.BinIdx << " (" << SplitConditionToString(featuresManager, split, leaf.Path.Directions[i]) << ") -> ";
        }
        path << splitStr;

        TStringBuilder logEntry;
        logEntry
            << "Split on iteration # " << iterations << ": " << leaf.BestSplit.FeatureId << " / " << leaf.BestSplit.BinId << " with score " << leaf.BestSplit.Score << Endl;
        logEntry << "Leaf path: " << path;
        CATBOOST_INFO_LOG << logEntry << Endl;
    }

    bool TGreedySearchHelper::SplitLeaves(TPointsSubsets* subsetsPtr,
                                          TVector<TLeafPath>* resultLeaves,
                                          TVector<double>* resultsLeafWeights,
                                          TVector<TVector<float>>* resultValues) {
        auto& subsets = *subsetsPtr;

        TVector<ui32> leavesToSplit;

        const auto depth = FindMaxDepth(subsets.Leaves);
        if (HaveFixedSplits(depth + 1)) {
            const auto& leaves = subsets.Leaves;
            leavesToSplit.reserve(leaves.size());
            for (auto leaf : xrange(leaves.size())) {
                if (leaves[leaf].BestSplit.Defined() && leaves[leaf].BestSplit.Score < 0) {
                    leavesToSplit.push_back(leaf);
                }
            }
        } else {
            SelectLeavesToSplit(subsets,
                                &leavesToSplit);
        }

        if (!leavesToSplit.empty()) {
            if (IsObliviousSplit() || Options.Policy == EGrowPolicy::Region) {
                auto& bestSplit = subsets.Leaves[leavesToSplit.back()].BestSplit;
                TBinarySplit split = ToSplit(FeaturesManager, subsets.Leaves[leavesToSplit.back()].BestSplit);
                const ui32 depth = subsets.Leaves.back().Path.GetDepth();
                PrintBestScore(FeaturesManager, split, bestSplit.Score, depth);
            } else {
                ui32 iteration = subsets.Leaves.size();

                if (Options.Policy == EGrowPolicy::Depthwise || HaveFixedSplits(depth + 1)) {
                    iteration = FindMaxDepth(subsets.Leaves);
                }

                for (ui32 leafId : leavesToSplit) {
                    const auto& leaf = subsets.Leaves[leafId];
                    PrintScoreLossguide(FeaturesManager, leaf, iteration);
                }
            }
            TVector<ui32> leftIds;
            TVector<ui32> rightIds;
            SplitPropsHelper.MakeSplit(leavesToSplit, subsetsPtr, &leftIds, &rightIds);
            MarkTerminal(leftIds, subsetsPtr);
            MarkTerminal(rightIds, subsetsPtr);
        } else {
            for (ui32 i = 0; i < subsets.Leaves.size(); ++i) {
                subsets.Leaves[i].IsTerminal = true;
            }
        }

        if (ShouldTerminate(subsets)) {
            const ui32 numStats = static_cast<const ui32>(subsetsPtr->PartitionStats.GetMapping().SingleObjectSize());
            CB_ENSURE(numStats, "Size of stats should be > 0");
            const ui32 numLeaves = static_cast<const ui32>(subsets.Leaves.size());
            auto currentPartStats = NCudaLib::ParallelStripeView(subsetsPtr->PartitionStats, TSlice(0, numLeaves));

            TVector<double> stats = ReadReduce(currentPartStats);

            resultValues->clear();

            resultValues->resize(numLeaves, TVector<float>(numStats - 1));
            resultLeaves->resize(numLeaves);
            resultsLeafWeights->resize(numLeaves);

            for (size_t leafId = 0; leafId < numLeaves; ++leafId) {
                double w = stats[leafId * numStats];
                (*resultsLeafWeights)[leafId] = w;

                double totalSum = 0;
                for (size_t approxId = 0; approxId < (numStats - 1); ++approxId) {
                    (*resultValues)[leafId][approxId] = static_cast<float>(w > 1e-20 ? stats[leafId * numStats + 1 + approxId] / (w + Options.L2Reg)
                                                                                     : 0.0);

                    totalSum += (*resultValues)[leafId][approxId];
                }
                if (subsets.Target.MultiLogitOptimization) {
                    for (size_t approxId = 0; approxId < (numStats - 1); ++approxId) {
                        (*resultValues)[leafId][approxId] += totalSum;
                    }
                }
                (*resultLeaves)[leafId] = subsets.Leaves[leafId].Path;
            }
            return false;
        }
        return true;
    }

    void TGreedySearchHelper::MarkTerminal(const TVector<ui32>& ids, TPointsSubsets* subsets) {
        for (ui32 i : ids) {
            subsets->Leaves[i].IsTerminal = IsTerminalLeaf(*subsets, i);
        }
    }

    bool TGreedySearchHelper::AreAllTerminal(const TPointsSubsets& subsets, const TVector<ui32>& leaves) {
        for (ui32 leaf : leaves) {
            if (!subsets.Leaves[leaf].IsTerminal) {
                return false;
            }
        }
        return true;
    }

    bool TGreedySearchHelper::ShouldTerminate(const TPointsSubsets& subsets) {
        const ui32 leafCount = static_cast<ui32>(subsets.Leaves.size());

        if (leafCount >= Options.MaxLeaves) {
            return true;
        }

        TVector<ui32> allLeaves(leafCount);
        Iota(allLeaves.begin(), allLeaves.end(), 0);

        return AreAllTerminal(subsets, allLeaves);
    }

    bool TGreedySearchHelper::IsTerminalLeaf(const TPointsSubsets& subsets, ui32 leafId) {
        auto& leaf = subsets.Leaves.at(leafId);
        const bool checkLeafSize = Options.Policy != EGrowPolicy::SymmetricTree;
        const bool flag = (checkLeafSize && leaf.Size <= Options.MinLeafSize) || leaf.Path.GetDepth() >= Options.MaxDepth;
        return flag;
    }

    void TGreedySearchHelper::SelectLeavesToVisit(const TPointsSubsets& subsets,
                                                  TVector<ui32>* leavesToVisit) {
        leavesToVisit->clear();
        leavesToVisit->reserve(subsets.Leaves.size());

        for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
            if (!subsets.Leaves[leaf].IsTerminal) {
                if (subsets.Leaves[leaf].BestSplit.Defined()) {
                    continue;
                }
                leavesToVisit->push_back(leaf);
            }
        }
    }
}
