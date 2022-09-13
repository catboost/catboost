#include "oblivious_tree_structure_searcher.h"
#include "pointwise_scores_calcer.h"
#include "random_score_helper.h"
#include "tree_ctrs.h"
#include "tree_ctr_datasets_visitor.h"
#include "update_feature_weights.h"

#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/cuda/methods/langevin_utils.h>

#include <catboost/libs/helpers/math_utils.h>

namespace NCatboostCuda {
    template <class TFunc, class TTask>
    static inline void ForeachOptimizationPartTask(TVector<TTask>& tasks,
                                                   TFunc&& func) {
        ui32 cursor = 0;
        RunInStreams(tasks.size(), Min<ui32>(tasks.size(), 8), [&](ui32 taskId, ui32 streamId) {
            auto& task = tasks[taskId];
            auto learnSlice = TSlice(cursor, cursor + task.LearnTarget->GetTarget().GetIndices().GetObjectsSlice().Size());
            cursor = learnSlice.Right;
            auto testSlice = TSlice(cursor, cursor + task.TestTarget->GetTarget().GetIndices().GetObjectsSlice().Size());
            cursor = testSlice.Right;
            func(learnSlice, testSlice, task, streamId);
        });
    }

    TOptimizationSubsets<NCudaLib::TMirrorMapping> TFeatureParallelObliviousTreeSearcher::CreateSubsets(ui32 maxDepth, TL2Target<NCudaLib::TMirrorMapping>& src) {
        TOptimizationSubsets<NCudaLib::TMirrorMapping> subsets;
        auto initParts = SingleTaskTarget == nullptr ? WriteFoldBasedInitialBins(subsets.Bins)
                                                     : WriteSingleTaskInitialBins(subsets.Bins);
        subsets.Indices = TMirrorBuffer<ui32>::CopyMapping(subsets.Bins);

        subsets.CurrentDepth = 0;
        subsets.FoldCount = initParts.size();
        subsets.FoldBits = NCB::IntLog2(subsets.FoldCount);
        MakeSequence(subsets.Indices);
        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions = TMirrorBuffer<TDataPartition>::Create(NCudaLib::TMirrorMapping(maxPartCount));
        subsets.PartitionStats = TMirrorBuffer<TPartitionStatistics>::Create(NCudaLib::TMirrorMapping(maxPartCount));
        UpdateSubsetsStats(src, &subsets);
        return subsets;
    }

    TObliviousTreeStructure TFeatureParallelObliviousTreeSearcher::Fit() {
        CB_ENSURE(FoldBasedTasks.size() || SingleTaskTarget);

        TMirrorBuffer<ui32> docBins = TMirrorBuffer<ui32>::CopyMapping(DataSet.GetIndices());

        TTreeUpdater treeUpdater(ScopedCache,
                                 FeaturesManager,
                                 CtrTargets,
                                 DataSet,
                                 docBins);

        TL2Target<NCudaLib::TMirrorMapping> target = ComputeWeakTarget();
        //TODO: two bootstrap type: docs and gathered target
        {
            auto slices = MakeTaskSlices();
            auto weights = Bootstrap.BootstrappedWeights(GetRandom(), &target.Weights);
            //TODO(noxoomo): remove tiny overhead from bootstrap learn also
            if (TreeConfig.ObservationsToBootstrap == EObservationsToBootstrap::TestOnly) {
                //make learn weights equal to 1
                for (ui32 i = 0, j = 0; i < FoldBasedTasks.size(); ++i, j += 2) {
                    const auto& learnSlice = slices[j];
                    auto learnWeights = weights.SliceView(learnSlice);
                    FillBuffer(learnWeights, 1.0f);
                }
            }
            MultiplyVector(target.Weights, weights);
            MultiplyVector(target.WeightedTarget, weights);
        }

        auto subsets = CreateSubsets(TreeConfig.MaxDepth,
                                     target);

        auto observationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);
        TMirrorBuffer<ui32> directObservationIndices;
        if (DataSet.HasPermutationDependentFeatures()) {
            directObservationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);
        }
        const ui32 foldCount = subsets.FoldCount;

        //score helpers will do all their job in own stream, so don't forget device-sync for the
        using TScoreCaclerPtr = THolder<TScoresCalcerOnCompressedDataSet<>>;
        TScoreCaclerPtr featuresScoreCalcer;
        TScoreCaclerPtr simpleCtrScoreCalcer;

        if (DataSet.HasFeatures()) {
            featuresScoreCalcer = MakeHolder<TScoresCalcerOnCompressedDataSet<>>(DataSet.GetFeatures(),
                                                                         TreeConfig,
                                                                         foldCount,
                                                                         true);
        }
        if (DataSet.HasPermutationDependentFeatures()) {
            simpleCtrScoreCalcer = MakeHolder<TScoresCalcerOnCompressedDataSet<>>(DataSet.GetPermutationFeatures(),
                                                                          TreeConfig,
                                                                          foldCount,
                                                                          true);
        }

        TObliviousTreeStructure result;
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        THolder<TTreeCtrDataSetsHelper> ctrDataSetsHelperPtr;

        TMirrorBuffer<float> catFeatureWeights;

        const auto featureCount = FeaturesManager.GetFeatureCount();
        const auto& featureWeightsCpu = ExpandFeatureWeights(TreeConfig.FeaturePenalties.Get(), featureCount);
        TMirrorBuffer<float> featureWeights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(featureCount));
        featureWeights.Write(featureWeightsCpu);
        double scoreBeforeSplit = 0.0;

        for (ui32 depth = 0; depth < TreeConfig.MaxDepth; ++depth) {
            //warning: don't change order of commands. current pipeline ensures maximum stream-parallelism until read
            //best score stage
            const auto& partitionsStats = subsets.PartitionStats;
            //gather doc-ids by leaves
            {
                auto guard = profiler.Profile("Make and gather observation indices");
                TMirrorBuffer<ui32> docIndices;
                MakeDocIndices(docIndices);
                Gather(observationIndices, docIndices, subsets.Indices);
            }
            if (simpleCtrScoreCalcer) {
                auto guard = profiler.Profile("Make and gather direct observation indices");

                TMirrorBuffer<ui32> directDocIndices;
                MakeDirectDocIndicesIndices(directDocIndices);
                Gather(directObservationIndices, directDocIndices, subsets.Indices);
            }
            TBinarySplit bestSplit;

            ui32 maxUniqueValues = 1;
            if (FeaturesManager.IsTreeCtrsEnabled()) {
                if (ctrDataSetsHelperPtr == nullptr) {
                    using TCtrHelperType = TTreeCtrDataSetsHelper;
                    ctrDataSetsHelperPtr = MakeHolder<TCtrHelperType>(DataSet,
                                                                      FeaturesManager,
                                                                      TreeConfig.MaxDepth,
                                                                      foldCount,
                                                                      *treeUpdater.CreateEmptyTensorTracker());
                }
                maxUniqueValues = ctrDataSetsHelperPtr->GetMaxUniqueValues();
            }

            UpdateFeatureWeightsForBestSplits(FeaturesManager, TreeConfig.ModelSizeReg, catFeatureWeights, maxUniqueValues);

            auto& manager = NCudaLib::GetCudaManager();

            manager.WaitComplete();
            {
                auto guard = profiler.Profile(TStringBuilder() << "Compute best splits " << depth);
                {
                    if (featuresScoreCalcer) {
                        featuresScoreCalcer->SubmitCompute(subsets,
                                                           observationIndices);
                    }

                    if (simpleCtrScoreCalcer) {
                        simpleCtrScoreCalcer->SubmitCompute(subsets,
                                                            directObservationIndices);
                    }
                }
                {
                    if (featuresScoreCalcer) {
                        featuresScoreCalcer->ComputeOptimalSplit(partitionsStats.AsConstBuf(),
                                                                 catFeatureWeights.AsConstBuf(),
                                                                 featureWeights.AsConstBuf(),
                                                                 scoreBeforeSplit,
                                                                 ScoreStdDev,
                                                                 GetRandom().NextUniformL());
                    }
                    if (simpleCtrScoreCalcer) {
                        simpleCtrScoreCalcer->ComputeOptimalSplit(partitionsStats.AsConstBuf(),
                                                                  catFeatureWeights.AsConstBuf(),
                                                                  featureWeights.AsConstBuf(),
                                                                  scoreBeforeSplit,
                                                                  ScoreStdDev,
                                                                  GetRandom().NextUniformL());
                    }
                }
            }
            manager.WaitComplete();

            TBestSplitProperties bestSplitProp = {static_cast<ui32>(-1),
                                                  0,
                                                  std::numeric_limits<float>::infinity(),
                                                  std::numeric_limits<float>::infinity()};

            if (featuresScoreCalcer) {
                bestSplitProp = TakeBest(bestSplitProp, featuresScoreCalcer->ReadOptimalSplit());
            }

            if (simpleCtrScoreCalcer) {
                bestSplitProp = TakeBest(bestSplitProp, simpleCtrScoreCalcer->ReadOptimalSplit());
            }

            TSingleBuffer<const ui64> treeCtrSplitBits;
            bool isTreeCtrSplit = false;

            if (FeaturesManager.IsTreeCtrsEnabled()) {

                auto& ctrDataSetsHelper = *ctrDataSetsHelperPtr;

                if (ctrDataSetsHelper.GetUsedPermutations().size()) {
                    TTreeCtrDataSetVisitor ctrDataSetVisitor(FeaturesManager,
                                                             foldCount,
                                                             TreeConfig,
                                                             subsets);

                    ctrDataSetVisitor.SetBestGain(bestSplitProp.Gain)
                        .SetScoreStdDevAndSeed(ScoreStdDev,
                                               GetRandom().NextUniformL());
                    TMirrorBuffer<ui32> inverseIndices;

                    for (auto permutation : ctrDataSetsHelper.GetUsedPermutations()) {
                        const auto& indices = ctrDataSetsHelper.GetPermutationIndices(permutation);
                        inverseIndices.Reset(indices.GetMapping());
                        //observations indices with store index of document inf ctrDataSet
                        {
                            //reuse buffers. var names aren't what they mean
                            InversePermutation(indices, inverseIndices);
                            TMirrorBuffer<ui32> tmp = TMirrorBuffer<ui32>::CopyMapping(observationIndices);
                            MakeIndicesFromInversePermutation(inverseIndices, tmp);
                            directObservationIndices.Reset(subsets.Indices.GetMapping());
                            Gather(directObservationIndices, tmp, subsets.Indices);
                        }

                        std::function<void(const TTreeCtrDataSet&)> treeCtrDataSetScoreCalcer = [&](
                            const TTreeCtrDataSet& ctrDataSet
                        ) {
                            ctrDataSetVisitor.Accept(ctrDataSet,
                                                     partitionsStats.AsConstBuf(),
                                                     inverseIndices,
                                                     directObservationIndices,
                                                     featureWeights.AsConstBuf(),
                                                     scoreBeforeSplit,
                                                     maxUniqueValues,
                                                     TreeConfig.ModelSizeReg);
                        };

                        ctrDataSetsHelper.VisitPermutationDataSets(permutation,
                                                                   treeCtrDataSetScoreCalcer);
                    }

                    if (ctrDataSetVisitor.HasSplit()) {
                        bestSplitProp = ctrDataSetVisitor.CreateBestSplitProperties();
                        treeCtrSplitBits = ctrDataSetVisitor.GetBestSplitBits();
                        isTreeCtrSplit = true;
                    }
                }
            }

            scoreBeforeSplit = bestSplitProp.Score;

            CB_ENSURE(bestSplitProp.FeatureId != static_cast<ui32>(-1),
                      TStringBuilder() << "Error: something went wrong, best split is NaN with score"
                                       << bestSplitProp.Score);

            bestSplit = ToSplit(FeaturesManager, bestSplitProp);
            PrintBestScore(FeaturesManager, bestSplit, bestSplitProp.Score, depth);

            if (result.HasSplit(bestSplit)) {
                break;
            }

            {
                auto guard = profiler.Profile(TStringBuilder() << "Compute new bins");
                if (isTreeCtrSplit) {
                    CB_ENSURE(treeCtrSplitBits.GetObjectsSlice().Size());
                    treeUpdater.AddSplit(bestSplit, treeCtrSplitBits);
                } else {
                    treeUpdater.AddSplit(bestSplit);
                }
            }

            if ((depth + 1) != TreeConfig.MaxDepth) {
                {
                    auto guard = profiler.Profile(TStringBuilder() << "Update subsets");
                    TSubsetsHelper<NCudaLib::TMirrorMapping>::Split(target,
                                                                    docBins,
                                                                    observationIndices,
                                                                    &subsets);
                }
                if (ctrDataSetsHelperPtr) {
                    ctrDataSetsHelperPtr->AddSplit(bestSplit,
                                                   docBins);
                }
            }

            result.Splits.push_back(bestSplit);
            if (isTreeCtrSplit) {
                FeaturesManager.AddUsedCtr(bestSplitProp.FeatureId);
            }
        }

        CacheBinsForModel(ScopedCache,
                          FeaturesManager,
                          DataSet,
                          result,
                          std::move(docBins));
        return result;
    }

    TVector<TSlice> TFeatureParallelObliviousTreeSearcher::MakeTaskSlices() {
        TVector<TSlice> slices;
        ui32 cursor = 0;
        for (auto& task : FoldBasedTasks) {
            auto learnSlice = task.LearnTarget->GetTarget().GetIndices().GetObjectsSlice();
            slices.push_back(TSlice(cursor, cursor + learnSlice.Size()));
            cursor += learnSlice.Size();

            auto testSlice = task.TestTarget->GetTarget().GetIndices().GetObjectsSlice();
            slices.push_back(TSlice(cursor, cursor + testSlice.Size()));
            cursor += testSlice.Size();
        }
        return slices;
    }

    ui64 TFeatureParallelObliviousTreeSearcher::GetTotalIndicesSize() const {
        if (SingleTaskTarget != nullptr) {
            return SingleTaskTarget->GetTarget().GetIndices().GetObjectsSlice().Size();
        } else {
            ui32 cursor = 0;
            for (auto& task : FoldBasedTasks) {
                auto learnSlice = task.LearnTarget->GetTarget().GetIndices().GetObjectsSlice();
                cursor += learnSlice.Size();
                auto testSlice = task.TestTarget->GetTarget().GetIndices().GetObjectsSlice();
                cursor += testSlice.Size();
            }
            return cursor;
        }
    }

    TVector<TDataPartition> TFeatureParallelObliviousTreeSearcher::WriteFoldBasedInitialBins(TMirrorBuffer<ui32>& bins) {
        bins.Reset(NCudaLib::TMirrorMapping(GetTotalIndicesSize()));

        TVector<TDataPartition> parts;

        ui32 currentBin = 0;
        ui32 cursor = 0;
        ForeachOptimizationPartTask(FoldBasedTasks,
                                    [&](const TSlice& learnSlice,
                                        const TSlice& testSlice,
                                        const TOptimizationTask& task,
                                        ui32 streamId) {
                                        Y_UNUSED(task);
                                        auto learnBins = bins.SliceView(learnSlice);
                                        auto testBins = bins.SliceView(testSlice);

                                        FillBuffer(learnBins, currentBin, streamId);
                                        FillBuffer(testBins, currentBin + 1, streamId);

                                        parts.push_back({cursor, (ui32)learnBins.GetObjectsSlice().Size()});
                                        cursor += learnBins.GetObjectsSlice().Size();
                                        parts.push_back({cursor, (ui32)testBins.GetObjectsSlice().Size()});
                                        cursor += testBins.GetObjectsSlice().Size();
                                        currentBin += 2;
                                    });
        return parts;
    }

    TVector<TDataPartition> TFeatureParallelObliviousTreeSearcher::WriteSingleTaskInitialBins(TMirrorBuffer<ui32>& bins) {
        CB_ENSURE(SingleTaskTarget);
        bins.Reset(NCudaLib::TMirrorMapping(SingleTaskTarget->GetTarget().GetIndices().GetMapping()));
        TDataPartition part;
        part.Size = SingleTaskTarget->GetTarget().GetIndices().GetObjectsSlice().Size();
        part.Offset = 0;
        TVector<TDataPartition> parts = {part};
        FillBuffer(bins, 0u);
        return parts;
    }

    TL2Target<NCudaLib::TMirrorMapping> TFeatureParallelObliviousTreeSearcher::ComputeWeakTarget() {
        auto& profiler = NCudaLib::GetProfiler();
        auto guard = profiler.Profile("Build tree search target (gradient)");
        TL2Target<NCudaLib::TMirrorMapping> target;
        auto slices = MakeTaskSlices();
        //TODO: (noxoomo) check and enable device-side sync

        bool isNewtonApproximation = IsSecondOrderScoreFunction(TreeConfig.ScoreFunction);

        if (FoldBasedTasks.size()) {
            CB_ENSURE(SingleTaskTarget == nullptr);
            NCudaLib::GetCudaManager().WaitComplete();

            double sum2 = 0;
            double count = 0;

            TVector<TComputationStream> streams;
            const ui32 streamCount = Min<ui32>(FoldBasedTasks.size(), 8);
            for (ui32 i = 0; i < streamCount; ++i) {
                streams.push_back(NCudaLib::GetCudaManager().RequestStream());
            }

            target.WeightedTarget.Reset(NCudaLib::TMirrorMapping(slices.back().Right));
            target.Weights.Reset(NCudaLib::TMirrorMapping(slices.back().Right));

            for (ui32 i = 0, j = 0; i < FoldBasedTasks.size(); ++i, j += 2) {
                auto& task = FoldBasedTasks[i];
                const auto& learnSlice = slices[j];
                const auto& testSlice = slices[j + 1];

                auto learnTarget = target.WeightedTarget.SliceView(learnSlice);
                auto testTarget = target.WeightedTarget.SliceView(testSlice);

                auto learnWeights = target.Weights.SliceView(learnSlice);
                auto testWeights = target.Weights.SliceView(testSlice);

                if (!isNewtonApproximation) {
                    task.LearnTarget->GradientAtZero(learnTarget, learnWeights,
                                                     streams[(2 * i) % streamCount].GetId());
                    task.TestTarget->GradientAtZero(testTarget, testWeights,
                                                    streams[(2 * i + 1) % streamCount].GetId());
                } else {
                    task.LearnTarget->NewtonAtZero(learnTarget, learnWeights,
                                                   streams[(2 * i) % streamCount].GetId());
                    task.TestTarget->NewtonAtZero(testTarget, testWeights,
                                                  streams[(2 * i + 1) % streamCount].GetId());
                }

                if (BoostingOptions.Langevin) {
                    auto &trainSeeds = Random.GetGpuSeeds<NCudaLib::TMirrorMapping>();
                    AddLangevinNoise(trainSeeds,
                                     &learnTarget,
                                     BoostingOptions.DiffusionTemperature,
                                     BoostingOptions.LearningRate);

                    auto &testSeeds = Random.GetGpuSeeds<NCudaLib::TMirrorMapping>();
                    AddLangevinNoise(testSeeds,
                                     &testTarget,
                                     BoostingOptions.DiffusionTemperature,
                                     BoostingOptions.LearningRate);
                }
            }

            if (ModelLengthMultiplier) {
                for (ui32 i = 0, j = 0; i < FoldBasedTasks.size(); ++i, j += 2) {
                    const auto& testSlice = slices[j + 1];
                    auto testTarget = target.WeightedTarget.SliceView(testSlice);
                    auto testWeights = target.Weights.SliceView(testSlice);
                    auto streamId = streams[(2 * i + 1) % streamCount].GetId();
                    DivideVector(testTarget, testWeights, streamId);
                    sum2 += DotProduct(testTarget, testTarget, &testWeights, streamId);
                    MultiplyVector(testTarget, testWeights, streamId);

                    count += testSlice.Size();
                }
                ScoreStdDev = ModelLengthMultiplier * sqrt(sum2 / (count + 1e-100)) * TreeConfig.RandomStrength;
            }
            NCudaLib::GetCudaManager().WaitComplete();
        } else {
            CB_ENSURE(SingleTaskTarget != nullptr);
            target.WeightedTarget.Reset(SingleTaskTarget->GetTarget().GetSamplesMapping());
            target.Weights.Reset(SingleTaskTarget->GetTarget().GetSamplesMapping());
            if (isNewtonApproximation) {
                SingleTaskTarget->NewtonAtZero(target.WeightedTarget, target.Weights);
            } else {
                SingleTaskTarget->GradientAtZero(target.WeightedTarget, target.Weights);
            }
            ScoreStdDev = ComputeScoreStdDev(ModelLengthMultiplier, TreeConfig.RandomStrength, target);
        }
        return target;
    }

    void TFeatureParallelObliviousTreeSearcher::MakeDocIndicesForSingleTask(TMirrorBuffer<ui32>& indices, ui32 stream) {
        CB_ENSURE(SingleTaskTarget != nullptr);
        const auto& targetIndices = SingleTaskTarget->GetTarget().GetIndices();
        indices.Reset(NCudaLib::TMirrorMapping(targetIndices.GetMapping()));
        indices.Copy(targetIndices, stream);
    }

    void TFeatureParallelObliviousTreeSearcher::MakeDocIndices(TMirrorBuffer<ui32>& indices) {
        if (SingleTaskTarget != nullptr) {
            MakeDocIndicesForSingleTask(indices);
        } else {
            indices.Reset(NCudaLib::TMirrorMapping(GetTotalIndicesSize()));

            ForeachOptimizationPartTask(FoldBasedTasks,
                                        [&](const TSlice& learnSlice,
                                            const TSlice& testSlice,
                                            const TOptimizationTask& task,
                                            ui32 stream) {
                                            indices
                                                .SliceView(learnSlice)
                                                .Copy(task.LearnTarget->GetTarget().GetIndices(),
                                                      stream);

                                            indices
                                                .SliceView(testSlice)
                                                .Copy(task.TestTarget->GetTarget().GetIndices(),
                                                      stream);
                                        });
        }
    }

    void TFeatureParallelObliviousTreeSearcher::MakeIndicesFromInversePermutation(const TMirrorBuffer<ui32>& inversePermutation, TMirrorBuffer<ui32>& indices) {
        if (SingleTaskTarget != nullptr) {
            MakeIndicesFromInversePermutationSingleTask(inversePermutation,
                                                        indices);
        } else {
            indices.Reset(NCudaLib::TMirrorMapping(GetTotalIndicesSize()));

            ForeachOptimizationPartTask(FoldBasedTasks,
                                        [&](const TSlice& learnSlice,
                                            const TSlice& testSlice,
                                            const TOptimizationTask& task,
                                            ui32 stream) {
                                            auto learnIndices = indices.SliceView(learnSlice);
                                            auto testIndices = indices.SliceView(testSlice);

                                            Gather(learnIndices,
                                                   inversePermutation,
                                                   task.LearnTarget->GetTarget().GetIndices(),
                                                   stream);

                                            Gather(testIndices,
                                                   inversePermutation,
                                                   task.TestTarget->GetTarget().GetIndices(),
                                                   stream);
                                        });
        }
    }

}
