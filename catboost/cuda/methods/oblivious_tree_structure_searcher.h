#pragma once

#include "histograms_helper.h"
#include "bootstrap.h"
#include "helpers.h"
#include "tree_ctrs.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_base.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/libs/options/oblivious_tree_options.h>

namespace NCatboostCuda
{
    template<class TGridPolicy,
            class TLayoutPolicy>
    inline THolder<TScoreHelper<TGridPolicy, TLayoutPolicy>>
    CreateScoreHelper(const TGpuBinarizedDataSet<TGridPolicy, TLayoutPolicy>& dataSet,
                      ui32 foldCount,
                      const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                      bool requestStream = false)
    {
        using TFeatureScoresHelper = TScoreHelper<TGridPolicy, TLayoutPolicy>;
        return MakeHolder<TFeatureScoresHelper>(dataSet,
                                                foldCount,
                                                treeConfig.MaxDepth,
                                                treeConfig.ScoreFunction,
                                                treeConfig.L2Reg,
                                                treeConfig.FoldSizeLossNormalization,
                                                requestStream);
    };


    template<class TLayoutPolicy = TCatBoostPoolLayout>
    class TGpuFeaturesScoreCalcer
    {
    public:
        TGpuFeaturesScoreCalcer(const TGpuFeatures<>& features,
                                const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                                ui32 foldCount,
                                bool requestStream)
                : Features(features)
                  , TreeConfig(treeConfig)
                  , FoldCount(foldCount)
        {
            if (Features.GetBinaryFeatures().NotEmpty())
            {
                BinaryFeatureHelper = CreateScoreHelper(Features.GetBinaryFeatures(), foldCount, TreeConfig,
                                                        requestStream);
            }
            if (Features.GetHalfByteFeatures().NotEmpty())
            {
                HalfByteFeatureHelper = CreateScoreHelper(Features.GetHalfByteFeatures(), foldCount, TreeConfig,
                                                          requestStream);
            }
            if (Features.GetFeatures().NotEmpty())
            {
                ByteFeatureHelper = CreateScoreHelper(Features.GetFeatures(), foldCount, TreeConfig, requestStream);
            }
        }

        bool HasByteFeatureHelper() const
        {
            return ByteFeatureHelper != nullptr;
        }

        bool HasBinaryFeatureHelper() const
        {
            return BinaryFeatureHelper != nullptr;
        }

        bool HasHalfByteFeatureHelper() const
        {
            return HalfByteFeatureHelper != nullptr;
        }


        const TScoreHelper<TByteFeatureGridPolicy, TLayoutPolicy>& GetByteFeatureHelper() const
        {
            return *ByteFeatureHelper;
        }

        const TScoreHelper<TBinaryFeatureGridPolicy, TLayoutPolicy>& GetBinaryFeatureHelper() const
        {
            return *BinaryFeatureHelper;
        }

        const TScoreHelper<THalfByteFeatureGridPolicy, TLayoutPolicy>& GetHalfByteFeatureHelper() const
        {
            return *HalfByteFeatureHelper;
        }

        TGpuFeaturesScoreCalcer& SubmitCompute(const TOptimizationSubsets& newSubsets,
                                               const TMirrorBuffer<ui32>& docs)
        {
            if (BinaryFeatureHelper)
            {
                BinaryFeatureHelper->SubmitCompute(newSubsets, docs);
            }
            if (HalfByteFeatureHelper)
            {
                HalfByteFeatureHelper->SubmitCompute(newSubsets, docs);
            }
            if (ByteFeatureHelper)
            {
                ByteFeatureHelper->SubmitCompute(newSubsets, docs);
            }
            return *this;

        }

        TGpuFeaturesScoreCalcer& ComputeOptimalSplit(const TMirrorBuffer<const TPartitionStatistics>& partStats,
                                                     double scoreStdDev = 0,
                                                     ui64 seed = 0)
        {
            TRandom rand(seed);
            if (BinaryFeatureHelper)
            {
                BinaryFeatureHelper->ComputeOptimalSplit(partStats, scoreStdDev, rand.NextUniformL());
            }
            if (HalfByteFeatureHelper)
            {
                HalfByteFeatureHelper->ComputeOptimalSplit(partStats, scoreStdDev, rand.NextUniformL());
            }
            if (ByteFeatureHelper)
            {
                ByteFeatureHelper->ComputeOptimalSplit(partStats, scoreStdDev, rand.NextUniformL());
            }
            return *this;
        }

        TBestSplitProperties ReadAndRemapOptimalSplit()
        {
            TBestSplitProperties best = {static_cast<ui32>(-1), 0, std::numeric_limits<float>::infinity()};
            if (BinaryFeatureHelper)
            {
                best = TakeBest(BinaryFeatureHelper->ReadAndRemapOptimalSplit(), best);
            }
            if (HalfByteFeatureHelper)
            {
                best = TakeBest(HalfByteFeatureHelper->ReadAndRemapOptimalSplit(), best);
            }
            if (ByteFeatureHelper)
            {
                best = TakeBest(ByteFeatureHelper->ReadAndRemapOptimalSplit(), best);
            }
            return best;
        }

    private:
        const TGpuFeatures<>& Features;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        ui32 FoldCount;


        THolder<TScoreHelper<TBinaryFeatureGridPolicy, TLayoutPolicy>> BinaryFeatureHelper;
        THolder<TScoreHelper<THalfByteFeatureGridPolicy, TLayoutPolicy>> HalfByteFeatureHelper;
        THolder<TScoreHelper<TByteFeatureGridPolicy, TLayoutPolicy>> ByteFeatureHelper;
    };

    class TTreeCtrDataSetVisitor
    {
    public:
        using TFeatureScoresHelper = TScoreHelper<THalfByteFeatureGridPolicy, TSingleDevPoolLayout>;

        TTreeCtrDataSetVisitor(TBinarizedFeaturesManager& featuresManager,
                               const ui32 foldCount,
                               const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                               const TOptimizationSubsets& subsets)
                : FeaturesManager(featuresManager)
                  , FoldCount(foldCount)
                  , TreeConfig(treeConfig)
                  , Subsets(subsets)
                  , BestScore(std::numeric_limits<double>::infinity())
                  , BestBin(-1)
                  , BestDevice(-1)
                  , BestBorders(NCudaLib::GetCudaManager().GetDeviceCount())
                  , BestSplits(NCudaLib::GetCudaManager().GetDeviceCount())
                  , Seeds(NCudaLib::GetCudaManager().GetDeviceCount(), 0)
        {
        }

        TTreeCtrDataSetVisitor& SetBestScore(double score)
        {
            BestScore = score;
            return *this;
        }

        TTreeCtrDataSetVisitor& SetScoreStdDevAndSeed(double scoreStdDev,
                                                      ui64 seed)
        {
            ScoreStdDev = scoreStdDev;
            TRandom random(seed);
            for (ui32 i = 0; i < Seeds.size(); ++i)
            {
                Seeds[i] = random.NextUniformL();
            }
            return *this;
        }

        bool HasSplit()
        {
            return BestDevice >= 0;
        }

        void Accept(const TTreeCtrDataSet& ctrDataSet,
                    const TMirrorBuffer<const TPartitionStatistics>& partStats,
                    const TMirrorBuffer<ui32>& ctrDataSetInverseIndices,
                    const TMirrorBuffer<ui32>& subsetDocs)
        {
            {
                auto cacheIds = GetCtrsBordersToCacheIds(ctrDataSet.GetCtrs());
                if (cacheIds.size())
                {
                    CacheCtrBorders(ctrDataSet.ReadBorders(cacheIds));
                }
            }

            auto& scoreHelper = *ctrDataSet.GetCacheHolder().Cache(ctrDataSet, 0, [&]() -> THolder<TFeatureScoresHelper>
            {
                return CreateScoreHelper(ctrDataSet.GetDataSet(), FoldCount, TreeConfig);
            });
            const ui64 taskSeed = Seeds[ctrDataSet.GetDeviceId()] + ctrDataSet.GetBaseTensor().GetHash();

            scoreHelper.SubmitCompute(Subsets,
                                      subsetDocs);
            scoreHelper.ComputeOptimalSplit(partStats,
                                            ScoreStdDev,
                                            taskSeed);

            UpdateBestSplit(ctrDataSet,
                            ctrDataSetInverseIndices,
                            scoreHelper.ReadAndRemapOptimalSplit());
        }

        TBestSplitProperties CreateBestSplitProperties()
        {
            EnsureHasBestProps();

            if (!FeaturesManager.IsKnown(BestCtr))
            {
                TVector<float> borders(BestBorders[BestDevice].begin(), BestBorders[BestDevice].end());
                FeaturesManager.AddCtr(BestCtr, std::move(borders));
            }

            TBestSplitProperties splitProperties;
            splitProperties.FeatureId = FeaturesManager.GetId(BestCtr);
            splitProperties.BinId = BestBin;
            splitProperties.Score = static_cast<float>(BestScore);
            Y_ASSERT(splitProperties.BinId < FeaturesManager.GetBorders(splitProperties.FeatureId).size());
            return splitProperties;
        }

        TSingleBuffer<const ui64> GetBestSplitBits() const
        {
            EnsureHasBestProps();
            return BestSplits[BestDevice].ConstCopyView();
        }

    private:
        void EnsureHasBestProps() const
        {
            CB_ENSURE(BestDevice >= 0, "Error: no good split in visitor found");
            CB_ENSURE(BestBin <= 255);
        }

        void CacheCtrBorders(const TMap<TCtr, TVector<float>>& bordersMap)
        {
            for (auto& entry : bordersMap)
            {
                if (!FeaturesManager.IsKnown(entry.first))
                {
                    TVector<float> borders(entry.second.begin(), entry.second.end());
                    {
                        TGuard<TAdaptiveLock> guard(Lock);
                        Y_ASSERT(
                                !FeaturesManager.IsKnown(entry.first)); //we can't add one ctr from 2 different threads.
                        FeaturesManager.AddCtr(entry.first, std::move(borders));
                    }
                }
            }
        }

        TVector<ui32> GetCtrsBordersToCacheIds(const TVector<TCtr>& ctrs)
        {
            TVector<ui32> result;
            for (ui32 i = 0; i < ctrs.size(); ++i)
            {
                const auto& ctr = ctrs.at(i);
                if (IsNeedToCacheBorders(ctr))
                {
                    result.push_back(i);
                }
            }
            return result;
        }

        bool IsNeedToCacheBorders(const TCtr& ctr)
        {
            return ctr.FeatureTensor.GetSplits().size() == 0 &&
                   ctr.FeatureTensor.GetCatFeatures().size() < TreeConfig.MaxCtrComplexityForBordersCaching;
        }

        void UpdateBestSplit(const TTreeCtrDataSet& dataSet,
                             const TMirrorBuffer<ui32>& inverseIndices,
                             const TBestSplitProperties& bestSplitProperties)
        {
            const ui32 dev = dataSet.GetDataSet().GetCompressedIndex().GetMapping().GetDeviceId();

            { //we don't need complex logic here. this should be pretty fast
                TGuard<TAdaptiveLock> guard(Lock);
                if (bestSplitProperties.Score < BestScore)
                {
                    BestScore = bestSplitProperties.Score;
                    BestBin = bestSplitProperties.BinId;
                    BestDevice = dev;
                    BestCtr = dataSet.GetCtrs()[bestSplitProperties.FeatureId];
                } else
                {
                    return;
                }
            }

            {
                const ui32 featureId = bestSplitProperties.FeatureId;
                const ui32 binId = bestSplitProperties.BinId;

                const auto& ctr = dataSet.GetCtrs()[featureId];
                const ui32 compressedSize = CompressedSize<ui64>(
                        static_cast<ui32>(inverseIndices.GetObjectsSlice().Size()),
                        2);
                BestSplits[dev].Reset(NCudaLib::TSingleMapping(dev, compressedSize));
                const auto devInverseIndices = inverseIndices.ConstDeviceView(dev);
                auto& binarizedDataSet = dataSet.GetDataSet();

                CreateCompressedSplit(binarizedDataSet,
                                      binarizedDataSet.GetHostFeatures()[featureId],
                                      binId,
                                      BestSplits[dev],
                                      &devInverseIndices);

                if (!FeaturesManager.IsKnown(ctr))
                {
                    BestBorders[dev] = dataSet.ReadBorders(featureId);
                    CB_ENSURE(binId < BestBorders[dev].size(),
                              TStringBuilder() << BestBorders[dev].size() << " " << featureId << " " << binId);
                }
            }
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const ui32 FoldCount;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        const TOptimizationSubsets& Subsets;

        TAdaptiveLock Lock;

        double BestScore;
        double ScoreStdDev = 0;
        ui32 BestBin;
        int BestDevice;
        TCtr BestCtr;

        TVector<TVector<float>> BestBorders;
        TVector<TSingleBuffer<ui64>> BestSplits;
        TVector<ui64> Seeds;
    };

    template<class TTarget,
            class TDataSet>
    class TObliviousTreeStructureSearcher
    {
    public:
        using TVec = typename TTarget::TVec;

        TObliviousTreeStructureSearcher(TScopedCacheHolder& cache,
                                        TBinarizedFeaturesManager& featuresManager,
                                        const TDataSet& dataSet,
                                        TBootstrap<NCudaLib::TMirrorMapping>& bootstrap,
                                        const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions)
                : ScopedCache(cache)
                  , FeaturesManager(featuresManager)
                  , DataSet(dataSet)
                  , CtrTargets(dataSet.GetCtrTargets())
                  , Bootstrap(bootstrap)
                  , TreeConfig(learnerOptions)
        {
        }

        TObliviousTreeStructureSearcher& AddTask(TTarget&& learnTarget,
                                                 TTarget&& testTarget)
        {
            CB_ENSURE(SingleTaskTarget == nullptr, "We can't mix learn/test splits and full estimation");
            FoldBasedTasks.push_back(std::move(TOptimizationTask(std::move(learnTarget),
                                                                 std::move(testTarget))));
            return *this;
        }

        TObliviousTreeStructureSearcher& SetTarget(TTarget&& target)
        {
            CB_ENSURE(SingleTaskTarget == nullptr, "Target already was set");
            CB_ENSURE(FoldBasedTasks.size() == 0, "Can't mix foldBased and singleTask targets");
            SingleTaskTarget.Reset(new TTarget(std::move(target)));
            return *this;
        }

        TObliviousTreeStructureSearcher& SetRandomStrength(double strength)
        {
            RandomStrengthMultipiler = strength;
            return *this;
        }

        TObliviousTreeStructure Fit()
        {
            CB_ENSURE(FoldBasedTasks.size() || SingleTaskTarget);

            TMirrorBuffer<ui32> docBins = TMirrorBuffer<ui32>::CopyMapping(DataSet.GetIndices());

            TTreeUpdater<TDataSet> treeUpdater(ScopedCache,
                                               FeaturesManager,
                                               CtrTargets,
                                               DataSet,
                                               docBins);

            TL2Target target = BuildTreeSearchTarget();

            TOptimizationSubsets subsets = CreateSubsets(TreeConfig.MaxDepth, target);

            auto observationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);
            TMirrorBuffer<ui32> directObservationIndices;
            if (DataSet.GetPermutationFeatures().NotEmpty())
            {
                directObservationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);
            }
            const ui32 foldCount = subsets.FoldCount;

            //score helpers will do all their job in own stream, so don't forget device-sync for the
            TGpuFeaturesScoreCalcer<> featuresScoreCalcer(DataSet.GetFeatures(), TreeConfig, foldCount, true);
            TGpuFeaturesScoreCalcer<> simpleCtrScoreCalcer(DataSet.GetPermutationFeatures(), TreeConfig, foldCount,
                                                           true);


            TObliviousTreeStructure result;
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

            THolder<TTreeCtrDataSetsHelper<TDataSet::GetCatFeaturesStoragePtrType()>> ctrDataSetsHelperPtr;

            for (ui32 depth = 0; depth < TreeConfig.MaxDepth; ++depth)
            {
                //warning: don't change order of commands. current pipeline ensures maximum stream-parallelism until read
                //best score stage
                auto partitionsStats = subsets.ComputePartitionStats();
                //gather doc-ids by leaves
                {
                    auto guard = profiler.Profile("Make and gather observation indices");
                    TMirrorBuffer<ui32> docIndices;
                    MakeDocIndices(docIndices);
                    Gather(observationIndices, docIndices, subsets.Indices);
                }
                if (DataSet.GetPermutationFeatures().NotEmpty())
                {
                    auto guard = profiler.Profile("Make and gather direct observation indices");

                    TMirrorBuffer<ui32> directDocIndices;
                    MakeDirectDocIndicesIndices(directDocIndices);
                    Gather(directObservationIndices, directDocIndices, subsets.Indices);
                }
                TBinarySplit bestSplit;

                auto& manager = NCudaLib::GetCudaManager();

                manager.WaitComplete();
                {
                    auto guard = profiler.Profile(TStringBuilder() << "Compute best splits " << depth);
                    {
                        featuresScoreCalcer.SubmitCompute(subsets, observationIndices);
                        simpleCtrScoreCalcer.SubmitCompute(subsets, directObservationIndices);
                    }
                    {
                        featuresScoreCalcer.ComputeOptimalSplit(partitionsStats, ScoreStdDev,
                                                                GetRandom().NextUniformL());
                        simpleCtrScoreCalcer.ComputeOptimalSplit(partitionsStats, ScoreStdDev,
                                                                 GetRandom().NextUniformL());
                    }

                    manager.WaitComplete();
                }

                auto bestSplitProp = TakeBest(featuresScoreCalcer.ReadAndRemapOptimalSplit(),
                                              simpleCtrScoreCalcer.ReadAndRemapOptimalSplit());

                TSingleBuffer<const ui64> treeCtrSplitBits;
                bool isTreeCtrSplit = false;

                if (FeaturesManager.IsTreeCtrsEnabled())
                {
                    if (ctrDataSetsHelperPtr == nullptr)
                    {
                        using TCtrHelperType = TTreeCtrDataSetsHelper<TDataSet::GetCatFeaturesStoragePtrType()>;
                        ctrDataSetsHelperPtr = MakeHolder<TCtrHelperType>(DataSet,
                                                                          FeaturesManager,
                                                                          TreeConfig.MaxDepth,
                                                                          foldCount,
                                                                          *treeUpdater.CreateEmptyTensorTracker());
                    }

                    auto& ctrDataSetsHelper = *ctrDataSetsHelperPtr;

                    if (ctrDataSetsHelper.GetUsedPermutations().size())
                    {
                        TTreeCtrDataSetVisitor ctrDataSetVisitor(FeaturesManager,
                                                                 foldCount,
                                                                 TreeConfig,
                                                                 subsets);

                        ctrDataSetVisitor.SetBestScore(bestSplitProp.Score)
                                .SetScoreStdDevAndSeed(ScoreStdDev, GetRandom().NextUniformL());
                        TMirrorBuffer<ui32> inverseIndices;

                        for (auto permutation : ctrDataSetsHelper.GetUsedPermutations())
                        {
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

                            auto treeCtrDataSetScoreCalcer = [&](const TTreeCtrDataSet& ctrDataSet)
                            {
                                ctrDataSetVisitor.Accept(ctrDataSet,
                                                         partitionsStats,
                                                         inverseIndices,
                                                         directObservationIndices);
                            };

                            ctrDataSetsHelper.VisitPermutationDataSets(permutation,
                                                                       treeCtrDataSetScoreCalcer);
                        }

                        if (ctrDataSetVisitor.HasSplit())
                        {
                            bestSplitProp = ctrDataSetVisitor.CreateBestSplitProperties();
                            treeCtrSplitBits = ctrDataSetVisitor.GetBestSplitBits();
                            isTreeCtrSplit = true;
                        }
                    }
                }

                CB_ENSURE(bestSplitProp.FeatureId != static_cast<ui32>(-1),
                          TStringBuilder() << "Error: something went wrong, best split is NaN with score"
                                           << bestSplitProp.Score);

                bestSplit.FeatureId = bestSplitProp.FeatureId;
                bestSplit.BinIdx = bestSplitProp.BinId;

                TString splitTypeMessage;

                if (FeaturesManager.IsCat(bestSplit.FeatureId))
                {
                    bestSplit.SplitType = EBinSplitType::TakeBin;
                    splitTypeMessage = "TakeBin";
                } else
                {
                    bestSplit.SplitType = EBinSplitType::TakeGreater;
                    splitTypeMessage = TStringBuilder() << ">" << FeaturesManager.GetBorders(
                            bestSplit.FeatureId)[bestSplit.BinIdx];
                }

                MATRIXNET_INFO_LOG
                << "Best split for depth " << depth << ": " << bestSplit.FeatureId << " / " << bestSplit.BinIdx << " ("
                << splitTypeMessage << ")"
                << " with score " << bestSplitProp.Score;
                if (FeaturesManager.IsCtr(bestSplit.FeatureId))
                {
                    MATRIXNET_INFO_LOG
                    << " tensor : " << FeaturesManager.GetCtr(bestSplit.FeatureId).FeatureTensor << "  (ctr type "
                    << FeaturesManager.GetCtr(bestSplit.FeatureId).Configuration.Type << ")";
                }
                MATRIXNET_INFO_LOG << Endl;

                if (result.HasSplit(bestSplit))
                {
                    break;
                }

                {
                    auto guard = profiler.Profile(TStringBuilder() << "Compute new bins");
                    if (isTreeCtrSplit)
                    {
                        CB_ENSURE(treeCtrSplitBits.GetObjectsSlice().Size());
                        treeUpdater.AddSplit(bestSplit, treeCtrSplitBits);
                    } else
                    {
                        treeUpdater.AddSplit(bestSplit);
                    }
                }

                if ((depth + 1) != TreeConfig.MaxDepth)
                {
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Update subsets");
                        subsets.Split(docBins,
                                      observationIndices);
                    }
                    if (ctrDataSetsHelperPtr)
                    {
                        ctrDataSetsHelperPtr->AddSplit(bestSplit,
                                                       docBins);
                    }
                }

                result.Splits.push_back(bestSplit);
            }

            CacheBinsForModel(ScopedCache,
                              DataSet,
                              result,
                              std::move(docBins));
            return result;
        }

    private:
        struct TOptimizationTask: public TMoveOnly
        {
            TTarget LearnTarget;
            TTarget TestTarget;

            TOptimizationTask(TTarget&& learn,
                              TTarget&& test)
                    : LearnTarget(std::move(learn))
                      , TestTarget(std::move(test))
            {
            }
        };

        TOptimizationSubsets CreateSubsets(ui32 maxDepth,
                                           TL2Target& src)
        {
            TOptimizationSubsets subsets;
            auto initParts =
                    SingleTaskTarget == nullptr ? WriteFoldBasedInitialBins(subsets.Bins) : WriteSingleTaskInitialBins(
                            subsets.Bins);
            subsets.Indices = TMirrorBuffer<ui32>::CopyMapping(subsets.Bins);

            subsets.CurrentDepth = 0;
            subsets.FoldCount = initParts.size();
            subsets.FoldBits = IntLog2(subsets.FoldCount);
            MakeSequence(subsets.Indices);
            ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
            subsets.Partitions = TMirrorBuffer<TDataPartition>::Create(NCudaLib::TMirrorMapping(maxPartCount));
            subsets.Src = &src;
            subsets.Update();
            return subsets;
        }

        //with first zero bit is estimation part, with first 1 bit is evaluation part
        //we store task in first bits of bin
        TVector<TSlice> MakeTaskSlices()
        {
            TVector<TSlice> slices;
            ui32 cursor = 0;
            for (auto& task : FoldBasedTasks)
            {
                auto learnSlice = task.LearnTarget.GetIndices().GetObjectsSlice();
                slices.push_back(TSlice(cursor, cursor + learnSlice.Size()));
                cursor += learnSlice.Size();

                auto testSlice = task.TestTarget.GetIndices().GetObjectsSlice();
                slices.push_back(TSlice(cursor, cursor + testSlice.Size()));
                cursor += testSlice.Size();
            }
            return slices;
        }

        ui64 GetTotalIndicesSize() const
        {
            if (SingleTaskTarget != nullptr)
            {
                return SingleTaskTarget->GetIndices().GetObjectsSlice().Size();
            } else
            {
                ui32 cursor = 0;
                for (auto& task : FoldBasedTasks)
                {
                    auto learnSlice = task.LearnTarget.GetIndices().GetObjectsSlice();
                    cursor += learnSlice.Size();
                    auto testSlice = task.TestTarget.GetIndices().GetObjectsSlice();
                    cursor += testSlice.Size();
                }
                return cursor;
            }
        }

        template<class TFunc>
        inline void ForeachOptimizationPartTask(TFunc&& func)
        {
            ui32 cursor = 0;
            RunInStreams(FoldBasedTasks.size(), Min<ui32>(FoldBasedTasks.size(), 8), [&](ui32 taskId, ui32 streamId)
            {
                auto& task = FoldBasedTasks[taskId];
                auto learnSlice = TSlice(cursor, cursor + task.LearnTarget.GetIndices().GetObjectsSlice().Size());
                cursor = learnSlice.Right;
                auto testSlice = TSlice(cursor, cursor + task.TestTarget.GetIndices().GetObjectsSlice().Size());
                cursor = testSlice.Right;
                func(learnSlice, testSlice, task, streamId);
            });
        }

        TVector<TDataPartition> WriteFoldBasedInitialBins(TMirrorBuffer<ui32>& bins)
        {
            bins.Reset(NCudaLib::TMirrorMapping(GetTotalIndicesSize()));

            TVector<TDataPartition> parts;

            ui32 currentBin = 0;
            ui32 cursor = 0;
            ForeachOptimizationPartTask([&](const TSlice& learnSlice,
                                            const TSlice& testSlice,
                                            const TOptimizationTask& task,
                                            ui32 streamId
            )
                                        {
                                            Y_UNUSED(task);
                                            auto learnBins = bins.SliceView(learnSlice);
                                            auto testBins = bins.SliceView(testSlice);

                                            FillBuffer(learnBins, currentBin, streamId);
                                            FillBuffer(testBins, currentBin + 1, streamId);

                                            parts.push_back({cursor, (ui32) learnBins.GetObjectsSlice().Size()});
                                            cursor += learnBins.GetObjectsSlice().Size();
                                            parts.push_back({cursor, (ui32) testBins.GetObjectsSlice().Size()});
                                            cursor += testBins.GetObjectsSlice().Size();
                                            currentBin += 2;
                                        });
            return parts;
        }

        TVector<TDataPartition> WriteSingleTaskInitialBins(TMirrorBuffer<ui32>& bins)
        {
            CB_ENSURE(SingleTaskTarget);
            bins.Reset(NCudaLib::TMirrorMapping(SingleTaskTarget->GetIndices().GetMapping()));
            TDataPartition part;
            part.Size = SingleTaskTarget->GetIndices().GetObjectsSlice().Size();
            part.Offset = 0;
            TVector<TDataPartition> parts = {part};
            FillBuffer(bins, 0u);
            return parts;
        }

        void MakeDocIndicesForSingleTask(TMirrorBuffer<ui32>& indices,
                                         ui32 stream = 0)
        {
            CB_ENSURE(SingleTaskTarget != nullptr);
            const auto& targetIndices = SingleTaskTarget->GetIndices();
            indices.Reset(NCudaLib::TMirrorMapping(targetIndices.GetMapping()));
            indices.Copy(targetIndices, stream);
        }

        //if features should be accessed by order[i]
        void MakeDocIndices(TMirrorBuffer<ui32>& indices)
        {
            if (SingleTaskTarget != nullptr)
            {
                MakeDocIndicesForSingleTask(indices);
            } else
            {
                indices.Reset(NCudaLib::TMirrorMapping(GetTotalIndicesSize()));

                ForeachOptimizationPartTask([&](const TSlice& learnSlice,
                                                const TSlice& testSlice,
                                                const TOptimizationTask& task,
                                                ui32 stream
                )
                                            {
                                                indices
                                                        .SliceView(learnSlice)
                                                        .Copy(task.LearnTarget.GetIndices(),
                                                              stream);

                                                indices
                                                        .SliceView(testSlice)
                                                        .Copy(task.TestTarget.GetIndices(),
                                                              stream);
                                            });
            }
        }

        //if features should be accessed by i
        void MakeDirectDocIndicesIndices(TMirrorBuffer<ui32>& indices)
        {
            MakeIndicesFromInversePermutation(DataSet.GetInverseIndices(), indices);
        }

        void MakeIndicesFromInversePermutationSingleTask(const TMirrorBuffer<ui32>& inversePermutation,
                                                         TMirrorBuffer<ui32>& indices)
        {
            CB_ENSURE(SingleTaskTarget != nullptr);
            const auto& targetIndices = SingleTaskTarget->GetIndices();
            indices.Reset(NCudaLib::TMirrorMapping(targetIndices.GetMapping()));
            Gather(indices,
                   inversePermutation,
                   targetIndices);
        }

        void MakeIndicesFromInversePermutation(const TMirrorBuffer<ui32>& inversePermutation,
                                               TMirrorBuffer<ui32>& indices)
        {
            if (SingleTaskTarget != nullptr)
            {
                MakeIndicesFromInversePermutationSingleTask(inversePermutation, indices);
            } else
            {
                indices.Reset(NCudaLib::TMirrorMapping(GetTotalIndicesSize()));

                ForeachOptimizationPartTask([&](const TSlice& learnSlice,
                                                const TSlice& testSlice,
                                                const TOptimizationTask& task,
                                                ui32 stream
                )
                                            {
                                                auto learnIndices = indices.SliceView(learnSlice);
                                                auto testIndices = indices.SliceView(testSlice);

                                                Gather(learnIndices,
                                                       inversePermutation,
                                                       task.LearnTarget.GetIndices(),
                                                       stream);

                                                Gather(testIndices,
                                                       inversePermutation,
                                                       task.TestTarget.GetIndices(),
                                                       stream);
                                            });
            }
        }

        TL2Target BuildTreeSearchTarget()
        {
            auto& profiler = NCudaLib::GetProfiler();
            auto guard = profiler.Profile("Build tree search target (gradient)");
            TL2Target target;
            auto slices = MakeTaskSlices();
            //TODO: (noxoomo) check and enable device-side sync
            //        NCudaLib::GetCudaManager().DefaultStream().Synchronize();
            if (FoldBasedTasks.size())
            {
                CB_ENSURE(SingleTaskTarget == nullptr);
                NCudaLib::GetCudaManager().WaitComplete();

                double sum2 = 0;
                double count = 0;

                TVector<TComputationStream> streams;
                const ui32 streamCount = Min<ui32>(FoldBasedTasks.size(), 8);
                for (ui32 i = 0; i < streamCount; ++i)
                {
                    streams.push_back(NCudaLib::GetCudaManager().RequestStream());
                }

                target.WeightedTarget.Reset(NCudaLib::TMirrorMapping(slices.back().Right));
                target.Weights.Reset(NCudaLib::TMirrorMapping(slices.back().Right));

                for (ui32 i = 0, j = 0; i < FoldBasedTasks.size(); ++i, j += 2)
                {
                    auto& task = FoldBasedTasks[i];
                    const auto& learnSlice = slices[j];
                    const auto& testSlice = slices[j + 1];

                    auto learnTarget = target.WeightedTarget.SliceView(learnSlice);
                    auto testTarget = target.WeightedTarget.SliceView(testSlice);

                    auto learnWeights = target.Weights.SliceView(learnSlice);
                    auto testWeights = target.Weights.SliceView(testSlice);

                    task.LearnTarget.GradientAtZero(learnTarget, streams[(2 * i) % streamCount].GetId());
                    task.TestTarget.GradientAtZero(testTarget, streams[(2 * i + 1) % streamCount].GetId());

                    learnWeights.Copy(task.LearnTarget.GetWeights(), streams[(2 * i) % streamCount].GetId());
                    testWeights.Copy(task.TestTarget.GetWeights(), streams[(2 * i + 1) % streamCount].GetId());
                }

                if (RandomStrengthMultipiler)
                {
                    for (ui32 i = 0, j = 0; i < FoldBasedTasks.size(); ++i, j += 2)
                    {
                        const auto& testSlice = slices[j + 1];
                        auto testTarget = target.WeightedTarget.SliceView(testSlice);
                        sum2 += DotProduct(testTarget, testTarget, (decltype(&testTarget)) nullptr,
                                           streams[(2 * i + 1) % streamCount].GetId());
                        count += testSlice.Size();
                    }
                    ScoreStdDev = RandomStrengthMultipiler * sqrt(sum2 / (count + 1e-100)) * TreeConfig.RandomStrength;
                }
                NCudaLib::GetCudaManager().WaitComplete();
            } else
            {
                CB_ENSURE(SingleTaskTarget != nullptr);
                target.WeightedTarget.Reset(SingleTaskTarget->GetTarget().GetMapping());
                target.Weights.Reset(SingleTaskTarget->GetTarget().GetMapping());
                SingleTaskTarget->GradientAtZero(target.WeightedTarget);
                target.Weights.Copy(SingleTaskTarget->GetWeights());
                if (RandomStrengthMultipiler)
                {
                    const double sum2 = DotProduct(target.WeightedTarget, target.WeightedTarget,
                                                   (decltype(&target.WeightedTarget)) nullptr);
                    const double count = target.WeightedTarget.GetObjectsSlice().Size();
                    ScoreStdDev = RandomStrengthMultipiler * sqrt(sum2 / (count + 1e-100));
                }
            }

            //TODO: two bootstrap type: docs and gathered target
            {
                auto weights = Bootstrap.BootstrapedWeights(target.Weights.GetMapping());
                //TODO(noxoomo): remove tiny overhead from bootstrap learn also
                if (TreeConfig.ObservationsToBootstrap == EObservationsToBootstrap::TestOnly)
                {
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

            return target;
        }

        TRandom& GetRandom()
        {
            return SingleTaskTarget == nullptr ? FoldBasedTasks[0].LearnTarget.GetRandom()
                                               : SingleTaskTarget->GetRandom();
        }

    private:
        TScopedCacheHolder& ScopedCache;
        //our learn algorithm could generate new features, so no const
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataSet& DataSet;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;

        TBootstrap<NCudaLib::TMirrorMapping>& Bootstrap;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        double RandomStrengthMultipiler = 0.0;
        double ScoreStdDev = 0.0;

        //should one or another, no mixing
        TVector<TOptimizationTask> FoldBasedTasks;
        THolder<TTarget> SingleTaskTarget;
    };
}

