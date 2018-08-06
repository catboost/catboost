#include "greedy_search_helper.h"
#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/compute_scores.cuh>
#include <catboost/cuda/targets/weak_objective.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>

namespace NKernelHost {

    class TComputeOptimalSplitsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCBinFeature> BinaryFeatures;
        TCudaBufferPtr<const float> Histograms;
        TCudaBufferPtr<const double> PartStats;
        TCudaBufferPtr<ui32> PartIds;
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
                                    TCudaBufferPtr<const float> histograms,
                                    TCudaBufferPtr<const double> partStats,
                                    TCudaBufferPtr<ui32> partIds,
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
                  , Histograms(histograms)
                  , PartStats(partStats)
                  , PartIds(partIds)
                  , NumScoreBlocks(numScoreBlocks)
                  , Result(result)
                  , MultiClassOptimization(multiclassLastApproxDimOptimization)
                  , ArgmaxBlockCount(argmaxBlockCount)
                  , ScoreFunction(scoreFunction)
                  , L2(l2)
                  , Normalize(normalize)
                  , ScoreStdDev(scoreStdDev)
                  , Seed(seed) {
        }

        Y_SAVELOAD_DEFINE(BinaryFeatures, Histograms, PartStats, PartIds, NumScoreBlocks, Result, ArgmaxBlockCount,
                          ScoreFunction, L2, Normalize, ScoreStdDev, Seed, MultiClassOptimization);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(PartIds.Size() % NumScoreBlocks == 0);
            const ui32 partBlockSize = PartIds.Size() / NumScoreBlocks;
            CB_ENSURE(partBlockSize, PartIds.Size() << " " << NumScoreBlocks);

            NKernel::ComputeOptimalSplits(BinaryFeatures.Get(), BinaryFeatures.Size(), Histograms.Get(),
                                          PartStats.Get(), PartStats.ObjectSize(), PartIds.Get(), partBlockSize,
                                          NumScoreBlocks, Result.Get(), ArgmaxBlockCount, ScoreFunction, MultiClassOptimization, L2, Normalize,
                                          ScoreStdDev, Seed, stream.GetStream());
        }
    };
}

namespace NCudaLib {
    REGISTER_KERNEL(0xA1BCA11, NKernelHost::TComputeOptimalSplitsKernel);
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

    void TGreedySearchHelper::SelectLeavesToSplit(const TPointsSubsets& subsets,
                                                  TVector<ui32>* leavesToSkip,
                                                  TVector<ui32>* leavesToSplit) {

        if (Options.Policy == EGrowingPolicy::Leafwise) {
            TMaybe<ui32> leafToSplit = FindBestLeafToSplit(subsets);

            for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
                if (leafToSplit.Defined() && leaf == *leafToSplit) {
                    leavesToSplit->push_back(leaf);
                } else {
                    leavesToSkip->push_back(leaf);
                }
            }
        } else if (Options.Policy == EGrowingPolicy::Region) {
            //split last leaf always
            leavesToSkip->resize(subsets.Leaves.size() - 1);
            Iota(leavesToSkip->begin(), leavesToSkip->end(), 0);

            if (subsets.Leaves.back().BestSplit.Defined()) {
                leavesToSplit->push_back(subsets.Leaves.size() - 1);
            } else {
                leavesToSkip->push_back(subsets.Leaves.size() - 1);
            }
        } else {
            CB_ENSURE(Options.Policy == EGrowingPolicy::ObliviousTree || Options.Policy == EGrowingPolicy::Levelwise);

            for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
                if (subsets.Leaves[leaf].BestSplit.Defined()) {
                    leavesToSplit->push_back(leaf);
                } else {
                    leavesToSkip->push_back(leaf);
                }
            }
        }
    }

    TPointsSubsets TGreedySearchHelper::CreateInitialSubsets(const IWeakObjective& objective) {
        return SplitPropsHelper.CreateInitialSubsets(ComputeTarget(Options.ScoreFunction,
                                                                   Options.BootstrapOptions,
                                                                   objective),
                                                     Options.MaxLeaves);
    }

    void TGreedySearchHelper::ComputeOptimalSplits(TPointsSubsets* subsets) {
        SplitPropsHelper.BuildNecessaryHistograms(subsets);
        TVector<ui32> leavesToVisit;
        SelectLeavesToVisit(*subsets, &leavesToVisit);

        const ui32 numScoreBlocks = IsObliviousSplit() ? 1 : leavesToVisit.size();
        const ui32 binFeatureCountPerDevice = NHelpers::CeilDivide(subsets->BinFeatures.GetObjectsSlice().Size(),
                                                                   NCudaLib::GetCudaManager().GetDeviceCount());

        const ui32 argmaxBlockCount = NHelpers::CeilDivide(binFeatureCountPerDevice, 128);
        auto bestProps = TStripeBuffer<TBestSplitProperties>::Create(
                NCudaLib::TStripeMapping::RepeatOnAllDevices(argmaxBlockCount * numScoreBlocks));

        auto leafIds = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leavesToVisit.size()));
        leafIds.Write(leavesToVisit);

        const ui32 numStats = subsets->Leaves.size() * subsets->GetStatCount();
        auto reducedStats = TMirrorBuffer<double>::Create(NCudaLib::TMirrorMapping(numStats));
        AllReduceThroughMaster(subsets->CurrentPartStats(), reducedStats);

        {
            using TKernel = NKernelHost::TComputeOptimalSplitsKernel;
            /*TODO(noxoomo): normalization and randoms */
            LaunchKernels<TKernel>(bestProps.NonEmptyDevices(),
                                   0,
                                   subsets->BinFeatures,
                                   subsets->Histograms,
                                   reducedStats,
                                   leafIds,
                                   numScoreBlocks,
                                   bestProps,
                                   subsets->Target.MultiLogitOptimization,
                                   argmaxBlockCount,
                                   Options.ScoreFunction,
                                   Options.L2Reg,
                                   false,
                                   0.0,
                                   0u);
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
            CB_ENSURE(bestSplits.size() == 1);
            TBinarySplit split = ToSplit(FeaturesManager, bestSplits[0]);
            const ui32 depth = subsets->Leaves.back().Path.GetDepth();
            PrintBestScore(FeaturesManager, split, bestSplits[0].Score, depth);
            for (const auto& leafId : leavesToVisit) {
                subsets->Leaves[leafId].BestSplit = bestSplits[0];
            }
        } else {
            CB_ENSURE(bestSplits.size() == leavesToVisit.size(),
                      "Error: for non-oblivious splits we should select best splits for each leaf");
            for (size_t i = 0; i < leavesToVisit.size(); ++i) {
                const ui32 leafId = leavesToVisit[i];
                subsets->Leaves[leafId].BestSplit = bestSplits[i];
            }
        }
    }

    bool TGreedySearchHelper::SplitLeaves(TPointsSubsets* subsetsPtr,
                                          TVector<TLeafPath>* resultLeaves,
                                          TVector<double>* resultsLeafWeights,
                                          TVector<TVector<float>>* resultValues) {
        auto& subsets = *subsetsPtr;

        TVector<ui32> leavesToSplit;
        TVector<ui32> leavesToSkip;
        SelectLeavesToSplit(subsets,
                            &leavesToSkip,
                            &leavesToSplit);

        if (leavesToSplit.size()) {
            SplitPropsHelper.MakeSplit(leavesToSplit, subsetsPtr);
            MarkTerminal(subsetsPtr);
        } else {
            for (ui32 i = 0; i < subsets.Leaves.size(); ++i) {
                subsets.Leaves[i].IsTerminal = true;
            }
        }

        if (ShouldTerminate(subsets)) {
            const ui32 numStats = static_cast<const ui32>(subsetsPtr->PartitionStats.GetMapping().SingleObjectSize());
            Y_VERIFY(numStats);
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
                    (*resultValues)[leafId][approxId] = static_cast<float>(w > 1e-20 ?
                                                                           stats[leafId * numStats + 1 + approxId] / (w + Options.L2Reg)
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

    void TGreedySearchHelper::MarkTerminal(TPointsSubsets* subsets) {
        for (ui32 i = 0; i < subsets->Leaves.size(); ++i) {
            subsets->Leaves[i].IsTerminal = IsTerminalLeaf(*subsets, i);
        }
    }

    bool TGreedySearchHelper::AreAllTerminal(const TPointsSubsets& subsets, const TVector<ui32>& leaves) {
        for (ui32 leaf : leaves) {
            if (!IsTerminalLeaf(subsets, leaf)) {
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
        const bool flag= leaf.Size < Options.MinLeafSize || leaf.Path.GetDepth() >= Options.MaxDepth;
        return flag;
    }

    void TGreedySearchHelper::SelectLeavesToVisit(const TPointsSubsets& subsets,
                                                  TVector<ui32>* leavesToVisit) {
        leavesToVisit->clear();

        for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
            if (!IsTerminalLeaf(subsets, leaf) && !subsets.Leaves[leaf].BestSplit.Defined()) {
                leavesToVisit->push_back(leaf);
            }
        }
    }
}
