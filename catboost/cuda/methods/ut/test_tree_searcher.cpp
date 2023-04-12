#include <catboost/cuda/ut_helpers/test_utils.h>
#include <library/cpp/testing/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/methods/histograms_helper.h>
#include <catboost/cuda/methods/oblivious_tree_structure_searcher.h>
#include <catboost/cuda/methods/pointwise_scores_calcer.h>
#include <catboost/cuda/methods/update_feature_weights.h>

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/quantization/grid_creator.h>

using namespace std;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(TPointwiseHistogramTest) {
    template <class TLayout>
    void inline CalcRefSums(EFeaturesGroupingPolicy policy,
                            const typename TSharedCompressedIndex<TLayout>::TCompressedDataSet* dataSet,
                            const TCudaBuffer<float, typename TLayout::TSamplesMapping>& tgts,
                            const TCudaBuffer<float, typename TLayout::TSamplesMapping>& wts,
                            const TCudaBuffer<ui32, typename TLayout::TSamplesMapping>& indices,
                            const TCudaBuffer<const TDataPartition, typename TLayout::TSamplesMapping>& partitioning,
                            ui32 depth,
                            ui32 foldCount,
                            TVector<float>* refSums,
                            TVector<float>* refWeights) {
        ui32 numLeaves = 1 << depth;
        ui32 bitsPerFold = NCB::IntLog2(foldCount);
        ui32 foldsStripe = 1 << bitsPerFold;

        refSums->clear();
        refWeights->clear();

        refSums->resize(dataSet->GetBinFeatures(policy).size() * numLeaves * foldCount);
        refWeights->resize(dataSet->GetBinFeatures(policy).size() * numLeaves * foldCount);

        TVector<float>& binSums = *refSums;
        TVector<float>& binWeights = *refWeights;

        const auto& cpuGrid = dataSet->GetCpuGrid(policy);
        const auto foldOffset = cpuGrid.ComputeFoldOffsets();

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            TSlice docSlice = tgts.GetMapping().DeviceSlice(dev);

            TVector<ui32> compressedIndex;
            dataSet->GetCompressedIndex().DeviceView(dev).Read(compressedIndex);
            const ui32 docsOnDevice = docSlice.Size();

            TVector<TDataPartition> parts;
            partitioning.DeviceView(dev).Read(parts);

            for (ui32 f = 0; f < cpuGrid.FeatureIds.size(); ++f) {
                ui32 featureId = cpuGrid.FeatureIds[f];
                if (dataSet->GetTCFeature(featureId).IsEmpty(dev)) {
                    continue;
                }
                if (docsOnDevice == 0) {
                    continue;
                }

                TVector<ui32> inds;
                TVector<float> targets;
                TVector<float> weights;

                tgts.DeviceView(dev).Read(targets);
                wts.DeviceView(dev).Read(weights);
                indices.DeviceView(dev).Read(inds);

                TCFeature feature = dataSet->GetTCFeature(featureId).At(dev);
                const ui32* cindexPtr = &compressedIndex[feature.Offset];

                const ui32 numParts = numLeaves * foldCount;

                for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                    for (ui32 fold = 0; fold < foldCount; ++fold) {
                        const auto& part = parts[leaf * foldsStripe + fold];

                        TVector<double> featureBinSums(feature.Folds + 1, 0.0f);
                        TVector<double> featureBinWeights(feature.Folds + 1, 0.0f);

                        for (ui32 i = 0; i < part.Size; i++) {
                            ui32 idx = inds[part.Offset + i];
                            UNIT_ASSERT(idx < docsOnDevice);
                            UNIT_ASSERT((feature.Offset + idx) < compressedIndex.size());
                            ui32 ci = cindexPtr[idx];

                            if (((ci >> feature.Shift) & feature.Mask) > featureBinSums.size()) {
                                Cout << "Feature  " << f << " " << dev << " " << Endl;
                                Cout << "Feature offset  " << feature.Offset << Endl;
                                Cout << "Bug " << part.Offset << " " << i << Endl;
                                Cout << feature.Folds << Endl;
                                Cout << feature.FirstFoldIndex << Endl;
                                Cout << ((ci) >> feature.Shift)
                                     << " " << ((ci >> feature.Shift) & feature.Mask) << " " << ci
                                     << " " << feature.Mask << " " << feature.Shift << Endl;
                                UNIT_ASSERT(false);
                            }

                            featureBinSums[(ci >> feature.Shift) & feature.Mask] += targets[part.Offset + i];
                            featureBinWeights[(ci >> feature.Shift) & feature.Mask] += weights[part.Offset + i];
                        }

                        if (!feature.OneHotFeature) {
                            for (ui32 i = 1; i < feature.Folds; i++) {
                                featureBinSums[i] += featureBinSums[i - 1];
                                featureBinWeights[i] += featureBinWeights[i - 1];
                            }
                        }

                        for (ui32 i = 0; i < feature.Folds; i++) {
                            const ui32 offset = (foldOffset.at(featureId) + i) * numParts + leaf * foldCount + fold;
                            binSums[offset] += featureBinSums[i];
                            binWeights[offset] += featureBinWeights[i];
                        }
                    }

                    for (ui32 fold = foldCount; fold < foldsStripe; ++fold) {
                        const auto& part = parts[leaf * foldsStripe + fold];
                        UNIT_ASSERT_VALUES_EQUAL(part.Size, 0);
                    }
                }
            }
        }
    }

    template <class TLayout>
    inline void CheckResults(EFeaturesGroupingPolicy policy,
                             const typename TSharedCompressedIndex<TLayout>::TCompressedDataSet* dataSet,
                             const TCudaBuffer<float, typename TLayout::TSamplesMapping>& approx,
                             const TCudaBuffer<float, typename TLayout::TSamplesMapping>& wts,
                             const TCudaBuffer<ui32, typename TLayout::TSamplesMapping>& indices,
                             const TCudaBuffer<const TDataPartition, typename TLayout::TSamplesMapping>& partitioning,
                             ui32 depth, ui32 foldCount,
                             const TVector<float>& props) {
        TVector<float> refSums, refWts;
        CalcRefSums<TLayout>(policy, dataSet, approx, wts, indices, partitioning, depth, foldCount, &refSums, &refWts);

        const ui32 leavesCount = static_cast<const ui32>(1 << depth);
        const ui32 partCount = leavesCount * foldCount;

        auto& binaryFeatures = dataSet->GetBinFeatures(policy);

        //        ui32 currentDevice = 0;
        //        ui32 deviceOffset = 0;

        const auto& cpuGrid = dataSet->GetCpuGrid(policy);
        const auto foldOffsets = cpuGrid.ComputeFoldOffsets();

        for (ui32 i = 0; i < binaryFeatures.size(); i++) {
            auto& binFeature = binaryFeatures[i];

            const ui32 firstFold = foldOffsets.at(binFeature.FeatureId);

            for (ui32 leaf = 0; leaf < leavesCount; leaf++) {
                for (ui32 fold = 0; fold < foldCount; ++fold) {
                    {
                        double x = 0;
                        for (ui32 offset = ((firstFold + binFeature.BinId) * partCount + leaf * foldCount + fold) * 2; offset < props.size(); offset += 2 * refWts.size()) {
                            x += props[offset];
                        }
                        double refX = refWts[i * partCount + leaf * foldCount + fold];
                        //                        if (std::abs(x - refX) > 1e-6) {
                        //                            ui32 bitsPerFold = NCB::IntLog2(foldCount);
                        //                            ui32 foldsStripe = 1 << bitsPerFold;
                        //
                        ////                            TVector<TDataPartition> parts;
                        ////                            partitioning.Read(parts);
                        ////                            Cout << deviceOffset << "  " << depth << Endl;
                        ////                            Cout << parts[leaf * foldsStripe + fold].Offset << " " << parts[leaf * foldsStripe + fold].Size << Endl;
                        ////                            Cout << i << " " << leaf << " " << fold << " " << x << " " << refX << Endl;
                        ////                            Cout << i * partCount + leaf * foldCount + fold << " " << leavesCount << " " << foldCount << Endl;
                        ////                            Cout << feature.Offset << " " << feature.Folds << " " << feature.Index << " " << feature.FirstFoldIndex << Endl;
                        //
                        ////                            for (ui32 i = 0; i < Min<ui32>(40, refWts.size()); ++i) {
                        ////                                Cout << refWts[i] << " ";
                        ////                            }
                        //                            Cout << Endl;
                        //                        }
                        UNIT_ASSERT_DOUBLES_EQUAL_C(x, refX, 1e-6, leaf << " " << fold << " " << i << " " << binFeature.FeatureId << " " << binFeature.BinId << " " << policy);
                    }
                    {
                        double x = 0;
                        for (ui32 offset = ((firstFold + binFeature.BinId) * partCount + leaf * foldCount + fold) * 2 + 1; offset < props.size(); offset += 2 * refWts.size()) {
                            x += props[offset];
                        }
                        double refX = refSums[i * partCount + leaf * foldCount + fold];

                        UNIT_ASSERT_DOUBLES_EQUAL(x, refX, 1e-6);
                    }
                }
            }
        }
    }

    TOptimizationSubsets<NCudaLib::TMirrorMapping> CreateSubsets(ui32 maxDepth,
                                                                 TL2Target<NCudaLib::TMirrorMapping> & src,
                                                                 ui32 foldCount,
                                                                 TVector<ui32> & bins) {
        TOptimizationSubsets<NCudaLib::TMirrorMapping> subsets;
        subsets.Bins = TMirrorBuffer<ui32>::CopyMapping(src.WeightedTarget);
        subsets.Bins.Write(bins);

        subsets.Indices = TMirrorBuffer<ui32>::CopyMapping(subsets.Bins);
        MakeSequence(subsets.Indices);

        subsets.CurrentDepth = 0;
        subsets.FoldCount = foldCount;
        subsets.FoldBits = NCB::IntLog2(subsets.FoldCount);

        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions = TMirrorBuffer<TDataPartition>::Create(NCudaLib::TMirrorMapping(maxPartCount));
        UpdateSubsetsStats(src, &subsets);

        {
            TVector<TDataPartition> initParts;
            auto currentParts = TSubsetsHelper<NCudaLib::TMirrorMapping>::CurrentPartsView(subsets);
            currentParts.Read(initParts);
            ui32 cursor = 0;
            for (ui32 i = 0; i < initParts.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(cursor, initParts[i].Offset);
                for (ui32 j = 0; j < initParts[i].Size; ++j) {
                    UNIT_ASSERT_VALUES_EQUAL(i, bins[cursor++]);
                }
            }
        }
        return subsets;
    }

    template <class TMapping>
    TL2Target<TMapping> CreateTestTarget(TMapping mapping, ui32 size) {
        TRandom rand(100500);

        TL2Target<TMapping> target;

        target.Weights.Reset(mapping);
        target.WeightedTarget.Reset(mapping);

        {
            TVector<float> relev(size);
            TVector<float> weights(size);

            // fill with trash to avoid rounding comparison
            for (ui32 i = 0; i < size; i++) {
                auto rel = (rand.NextUniformL() % 5);
                if (rel == 0)
                    relev[i] = 0.125f;
                else if (rel == 1)
                    relev[i] = 0.25f;
                else if (rel == 2)
                    relev[i] = 0.5f;
                else if (rel == 3)
                    relev[i] = 0.75f;

                weights[i] = 1.0; //1.0 / (1 + (rand.NextUniformL() % 2));
                relev[i] *= weights[i];
            }
            target.Weights.Write(weights);
            target.WeightedTarget.Write(relev);
        }
        return target;
    }

    template <class TMapping>
    void CheckStats(const TOptimizationSubsets<TMapping>& subsets,
                    const TVector<float>& gatheredTarget,
                    const TVector<float>& gatheredWeights,
                    const TCudaBuffer<TPartitionStatistics, TMapping>& partStats) {
        auto currentParts = TSubsetsHelper<TMapping>::CurrentPartsView(subsets);

        for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
            TVector<TPartitionStatistics> cpuStat;
            partStats.DeviceView(dev).Read(cpuStat);

            TVector<TDataPartition> cpuParts;
            auto devSlice = subsets.Indices.GetMapping().DeviceSlice(dev);
            auto devParts = currentParts.DeviceView(dev);
            devParts.Read(cpuParts);
            //            currentParts.Read(cpuParts);

            {
                for (ui32 partId = 0; partId < cpuParts.size(); ++partId) {
                    auto& part = cpuParts[partId];

                    double sum = 0;
                    double weight = 0;

                    for (ui32 i = 0; i < part.Size; ++i) {
                        sum += gatheredTarget[devSlice.Left + part.Offset + i];
                        weight += gatheredWeights[devSlice.Left + part.Offset + i];
                    }

                    UNIT_ASSERT_DOUBLES_EQUAL_C(cpuStat[partId].Weight, weight, 1e-5, "PartCount " << cpuParts.size() << " device " << dev);
                    UNIT_ASSERT_DOUBLES_EQUAL(cpuStat[partId].Sum, sum, 1e-5);
                }
            }
        }
    }

    template <class TLayout = TFeatureParallelLayout>
    void CheckResultsForCompressedDataSet(const typename TSharedCompressedIndex<TLayout>::TCompressedDataSet& features,
                                          TScoresCalcerOnCompressedDataSet<TLayout>& calcer,
                                          const TCudaBuffer<const TDataPartition, typename TLayout::TSamplesMapping>& partitionStats,
                                          const TCudaBuffer<float, typename TLayout::TSamplesMapping>& weightedTarget,
                                          const TCudaBuffer<float, typename TLayout::TSamplesMapping>& weights,
                                          const TCudaBuffer<ui32, typename TLayout::TSamplesMapping>& indices,
                                          ui32 depth,
                                          ui32 foldCount) {
        for (auto policy : GetEnumAllValues<NCatboostCuda::EFeaturesGroupingPolicy>()) {
            if (calcer.HasHelperForPolicy(policy)) {
                auto& scoreHelper = calcer.GetHelperForPolicy(policy);
                auto histogram = scoreHelper.ReadHistograms();
                CheckResults<TLayout>(policy, &features,
                                      weightedTarget,
                                      weights,
                                      indices,
                                      partitionStats,
                                      depth,
                                      foldCount,
                                      histogram);
            }
        }
    }

    void TestPointwiseHistogramForFeatureParallelDataSet(const TFeatureParallelDataSet& dataSet,
                                                         const TBinarizedFeaturesManager& featuresManager) {
        TRandom rand(10);
        const ui32 dsSize = dataSet.GetIndices().GetObjectsSlice().Size();

        const ui32 foldCount = 28;
        const ui32 maxDepth = 10;
        const float modelSizeReg = 0.5f;
        TVector<ui32> foldSizes;
        ui32 totalSize = 0;

        TVector<ui32> learnIndices;
        TVector<ui32> learnIndicesDirect;
        TVector<ui32> foldBins;
        {
            TVector<ui32> indicesCpu;
            dataSet.GetIndices().Read(indicesCpu);

            for (ui32 i = 0; i < foldCount; ++i) {
                foldSizes.push_back(dsSize / 100 + rand.NextUniformL() % dsSize);
                const ui32 foldSize = foldSizes.back();
                for (ui32 id = 0; id < foldSize; ++id) {
                    const ui32 idx = rand.NextUniformL() % indicesCpu.size();
                    learnIndices.push_back(indicesCpu[idx]);
                    learnIndicesDirect.push_back(idx);
                }
                totalSize += foldSize;
                foldBins.resize(totalSize, i);
            }
        }

        ui32 targetSize = totalSize;
        TL2Target<NCudaLib::TMirrorMapping> target = CreateTestTarget(dataSet.GetIndices().GetMapping(),
                                                                      targetSize);

        auto indices = TMirrorBuffer<ui32>::CopyMapping(target.WeightedTarget);
        indices.Write(learnIndices);
        TOptimizationSubsets<NCudaLib::TMirrorMapping> subsets = CreateSubsets(maxDepth, target, foldCount, foldBins);

        TScopedCacheHolder cache;

        TMirrorBuffer<ui32> docBins = TMirrorBuffer<ui32>::CopyMapping(dataSet.GetIndices());

        TTreeUpdater treeUpdater(cache,
                                 featuresManager,
                                 dataSet.GetCtrTargets(),
                                 dataSet,
                                 docBins);

        TVector<float> gatheredTarget;
        TVector<float> gatheredWeights;

        subsets.WeightedTarget.Read(gatheredTarget);
        subsets.Weights.Read(gatheredWeights);

        auto& partitionStats = subsets.PartitionStats;
        CheckStats(subsets, gatheredTarget, gatheredWeights, partitionStats);

        auto observationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);
        auto directObservationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);

        NCatboostOptions::TObliviousTreeLearnerOptions treeConfig(ETaskType::GPU);
        treeConfig.MaxDepth = maxDepth;

        THolder<TScoresCalcerOnCompressedDataSet<>> featuresScoreCalcer;
        if (dataSet.HasFeatures()) {
            featuresScoreCalcer = MakeHolder<TScoresCalcerOnCompressedDataSet<>>(dataSet.GetFeatures(),
                                                                         treeConfig,
                                                                         foldCount,
                                                                         true);
        }

        THolder<TScoresCalcerOnCompressedDataSet<>> simpleCtrScoreCalcer;
        if (dataSet.HasPermutationDependentFeatures()) {
            simpleCtrScoreCalcer = MakeHolder<TScoresCalcerOnCompressedDataSet<>>(dataSet.GetPermutationFeatures(),
                                                                          treeConfig,
                                                                          foldCount,
                                                                          true);
        }

        TObliviousTreeStructure result;

        const auto featureCount = featuresManager.GetFeatureCount();
        const auto& featureWeightsCpu = ExpandFeatureWeights(treeConfig.FeaturePenalties.Get(), featureCount);
        TMirrorBuffer<float> featureWeights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(featureCount));
        featureWeights.Write(featureWeightsCpu);
        double scoreBeforeSplit = 0.0;

        TMirrorBuffer<float> catFeatureWeights;

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            //warning: don't change order of commands. current pipeline ensures maximum stream-parallelism until read
            //best score stage
            {
                subsets.WeightedTarget.Read(gatheredTarget);
                subsets.Weights.Read(gatheredWeights);
                CheckStats(subsets, gatheredTarget, gatheredWeights, partitionStats);
            }

            //gather doc-ids by leaves
            {
                auto docIndices = TMirrorBuffer<ui32>::CopyMapping(target.WeightedTarget);
                docIndices.Write(learnIndices);
                Gather(observationIndices, docIndices, subsets.Indices);
            }
            {
                auto docIndices = TMirrorBuffer<ui32>::CopyMapping(target.WeightedTarget);
                docIndices.Write(learnIndicesDirect);
                Gather(directObservationIndices, docIndices, subsets.Indices);
            }

            auto& manager = NCudaLib::GetCudaManager();

            manager.WaitComplete();

            {
                if (featuresScoreCalcer) {
                    featuresScoreCalcer->SubmitCompute(subsets, observationIndices);
                }
                if (simpleCtrScoreCalcer) {
                    simpleCtrScoreCalcer->SubmitCompute(subsets, directObservationIndices);
                }
            }
            manager.WaitComplete();
            auto currentParts = TSubsetsHelper<NCudaLib::TMirrorMapping>::CurrentPartsView(subsets);

            UpdateFeatureWeightsForBestSplits(featuresManager, modelSizeReg, catFeatureWeights);

            if (featuresScoreCalcer) {
                CheckResultsForCompressedDataSet(dataSet.GetFeatures(),
                                                 *featuresScoreCalcer,
                                                 currentParts.AsConstBuf(),
                                                 subsets.WeightedTarget,
                                                 subsets.Weights,
                                                 observationIndices,
                                                 depth,
                                                 foldCount);
            }

            if (simpleCtrScoreCalcer) {
                CheckResultsForCompressedDataSet(dataSet.GetPermutationFeatures(),
                                                 *simpleCtrScoreCalcer,
                                                 currentParts.AsConstBuf(),
                                                 subsets.WeightedTarget,
                                                 subsets.Weights,
                                                 directObservationIndices,
                                                 depth,
                                                 foldCount);
            }

            {
                if (featuresScoreCalcer) {
                    featuresScoreCalcer->ComputeOptimalSplit(
                        partitionStats.AsConstBuf(),
                        catFeatureWeights.AsConstBuf(),
                        featureWeights.AsConstBuf(),
                        scoreBeforeSplit);
                }
                if (simpleCtrScoreCalcer) {
                    simpleCtrScoreCalcer->ComputeOptimalSplit(
                        partitionStats.AsConstBuf(),
                        catFeatureWeights.AsConstBuf(),
                        featureWeights.AsConstBuf(),
                        scoreBeforeSplit);
                }
            }

            TBinarySplit bestSplit;
            {
                //                const TGpuBinarizedDataSet<TByteFeatureGridPolicy>& byteFeatures
                auto featureIds = dataSet.GetFeatures().GetFeatures();
                //                TVector<TCFeature> features = byteFeatures.GetHostFeatures();
                auto localIdx = rand.NextUniformL() % featureIds.size();
                bestSplit.FeatureId = featureIds[localIdx];
                bestSplit.BinIdx = featuresManager.GetBinCount(bestSplit.FeatureId) / 2;
                bestSplit.SplitType = featuresManager.IsCat(localIdx) ? EBinSplitType::TakeBin : EBinSplitType::TakeGreater;

                treeUpdater.AddSplit(bestSplit);

                if (featuresManager.IsCtr(bestSplit.FeatureId)) {
                    featuresManager.AddUsedCtr(bestSplit.FeatureId);
                }

                TSubsetsHelper<NCudaLib::TMirrorMapping>::Split(target,
                                                                docBins,
                                                                observationIndices,
                                                                &subsets);
            }
        }
    }

    //
    void TestPointwiseHistogramForDocParallelDataSet(const TDocParallelDataSet& dataSet,
                                                     const TBinarizedFeaturesManager& featuresManager) {
        TRandom rand(10);
        auto samplesMapping = dataSet.GetSamplesMapping();

        TStripeBuffer<ui32> indices;
        indices.Reset(dataSet.GetSamplesMapping());
        MakeSequence(indices);

        float modelSizeReg = 0.5;
        const ui32 maxDepth = 10;
        TVector<ui32> foldSizes;

        TVector<ui32> learnIndices;
        {
            indices.Read(learnIndices);
        }

        ui32 targetSize = samplesMapping.GetObjectsSlice().Size();
        TL2Target<NCudaLib::TStripeMapping> target = CreateTestTarget(samplesMapping,
                                                                      targetSize);

        using THelper = TSubsetsHelper<NCudaLib::TStripeMapping>;
        auto subsets = THelper::CreateSubsets(maxDepth,
                                              target);

        TVector<float> gatheredTarget;
        TVector<float> gatheredWeights;

        subsets.WeightedTarget.Read(gatheredTarget);
        subsets.Weights.Read(gatheredWeights);

        auto& partitionStats = subsets.PartitionStats;
        CheckStats(subsets, gatheredTarget, gatheredWeights, partitionStats);

        auto docs = TStripeBuffer<ui32>::CopyMapping(subsets.Indices);

        NCatboostOptions::TObliviousTreeLearnerOptions treeConfig(ETaskType::GPU);
        treeConfig.MaxDepth = maxDepth;

        using TScoreCalcer = TScoresCalcerOnCompressedDataSet<TDocParallelLayout>;
        THolder<TScoreCalcer> featuresScoreCalcer;
        THolder<TScoreCalcer> simpleCtrScoreCalcer;

        if (dataSet.HasFeatures()) {
            featuresScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetFeatures(),
                                                   treeConfig,
                                                   1,
                                                   true);
        }

        if (dataSet.HasPermutationDependentFeatures()) {
            simpleCtrScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetPermutationFeatures(),
                                                    treeConfig,
                                                    1,
                                                    true);
        }

        TObliviousTreeStructure result;

        const auto featureCount = featuresManager.GetFeatureCount();
        const auto& featureWeightsCpu = ExpandFeatureWeights(treeConfig.FeaturePenalties.Get(), featureCount);
        TMirrorBuffer<float> featureWeights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(featureCount));
        featureWeights.Write(featureWeightsCpu);
        double scoreBeforeSplit = 0.0;

        TMirrorBuffer<float> catFeatureWeights;

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            //warning: don't change order of commands. current pipeline ensures maximum stream-parallelism until read
            //best score stage
            {
                subsets.WeightedTarget.Read(gatheredTarget);
                subsets.Weights.Read(gatheredWeights);
                CheckStats(subsets, gatheredTarget, gatheredWeights, partitionStats);
            }

            //gather doc-ids by leaves
            {
                Gather(docs, indices, subsets.Indices);
            }
            auto& manager = NCudaLib::GetCudaManager();

            manager.WaitComplete();

            {
                if (featuresScoreCalcer) {
                    featuresScoreCalcer->SubmitCompute(subsets,
                                                       docs);
                }
                if (simpleCtrScoreCalcer) {
                    simpleCtrScoreCalcer->SubmitCompute(subsets,
                                                        docs);
                }
            }
            manager.WaitComplete();
            auto currentParts = THelper::CurrentPartsView(subsets);

            TMirrorBuffer<TPartitionStatistics> reducedPartsStats;
            NCudaLib::AllReduceThroughMaster(subsets.PartitionStats, reducedPartsStats);


            UpdateFeatureWeightsForBestSplits(featuresManager, modelSizeReg, catFeatureWeights);

            if (featuresScoreCalcer) {
                CheckResultsForCompressedDataSet(dataSet.GetFeatures(),
                                                 *featuresScoreCalcer,
                                                 currentParts.AsConstBuf(),
                                                 subsets.WeightedTarget,
                                                 subsets.Weights,
                                                 docs,
                                                 depth,
                                                 1);
            }

            if (simpleCtrScoreCalcer) {
                CheckResultsForCompressedDataSet(dataSet.GetPermutationFeatures(),
                                                 *simpleCtrScoreCalcer,
                                                 currentParts.AsConstBuf(),
                                                 subsets.WeightedTarget,
                                                 subsets.Weights,
                                                 docs,
                                                 depth,
                                                 1);
            }

            {
                if (featuresScoreCalcer) {
                    featuresScoreCalcer->ComputeOptimalSplit(
                        reducedPartsStats.AsConstBuf(),
                        catFeatureWeights.AsConstBuf(),
                        featureWeights.AsConstBuf(),
                        scoreBeforeSplit);
                }
                if (simpleCtrScoreCalcer) {
                    simpleCtrScoreCalcer->ComputeOptimalSplit(
                        reducedPartsStats.AsConstBuf(),
                        catFeatureWeights.AsConstBuf(),
                        featureWeights.AsConstBuf(),
                        scoreBeforeSplit);
                }
            }

            TBinarySplit bestSplit;
            {
                auto featureIds = dataSet.GetFeatures().GetFeatures();
                auto localIdx = rand.NextUniformL() % featureIds.size();
                bestSplit.FeatureId = featureIds[localIdx];
                bestSplit.BinIdx = featuresManager.GetBinCount(bestSplit.FeatureId) / 2;
                bestSplit.SplitType = featuresManager.IsCat(localIdx) ? EBinSplitType::TakeBin : EBinSplitType::TakeGreater;
            }
            {
                if (featuresManager.IsCtr(bestSplit.FeatureId)) {
                    featuresManager.AddUsedCtr(bestSplit.FeatureId);
                }
            }
            {
                TSubsetsHelper<NCudaLib::TStripeMapping>::Split(target,
                                                                dataSet.GetCompressedIndex().GetStorage(),
                                                                docs,
                                                                dataSet.GetTCFeature(bestSplit.FeatureId),
                                                                bestSplit.BinIdx,
                                                                &subsets);
            }
        }
    }

    void TestPointwiseHist(ui32 binarization,
                           ui32 oneHotLimit,
                           ui32 permutationCount) {
        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum,
                                                                 binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = oneHotLimit;

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> featuresManager;
        NCB::TFeatureEstimators estimators;

        LoadTrainingData(NCB::TPathWithScheme("dsv://test-pool.txt"),
                         NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
                         floatBinarization,
                         catFeatureParams,
                         estimators,
                         &dataProvider,
                         &featuresManager);

        NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
        {
            NCatboostOptions::TBinarizationOptions bucketsBinarization(EBorderSelectionType::GreedyLogSum, binarization);
            NCatboostOptions::TBinarizationOptions freqBinarization(EBorderSelectionType::GreedyLogSum, binarization);

            TVector<TVector<float>> prior = {{0.5, 1.0}};
            NCatboostOptions::TCtrDescription bucketsCtr(ECtrType::Buckets, prior, bucketsBinarization);
            NCatboostOptions::TCtrDescription freqCtr(ECtrType::FeatureFreq, prior, freqBinarization);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }


        TFeatureParallelDataSetHoldersBuilder dataSetsHolderBuilder(*featuresManager,
                                                                    *dataProvider,
                                                                    estimators,
                                                                    nullptr,
                                                                    1);

        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount, &NPar::LocalExecutor());
        for (ui32 i = 0; i < dataSet.PermutationsCount(); ++i) {
            TestPointwiseHistogramForFeatureParallelDataSet(dataSet.GetDataSetForPermutation(i),
                                                            *featuresManager);
        }
    }

    void TestPointwiseHistDocParallel(ui32 binarization,
                                      ui32 oneHotLimit,
                                      ui32 permutationCount) {
        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum,
                                                                 binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = oneHotLimit;

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> featuresManager;
        NCB::TFeatureEstimators estimators;

        LoadTrainingData(NCB::TPathWithScheme("dsv://test-pool.txt"),
                         NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
                         floatBinarization,
                         catFeatureParams,
                         estimators,
                         &dataProvider,
                         &featuresManager);

        NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
        {
            NCatboostOptions::TBinarizationOptions bucketsBinarization(EBorderSelectionType::GreedyLogSum, binarization);
            NCatboostOptions::TBinarizationOptions freqBinarization(EBorderSelectionType::GreedyLogSum, binarization);

            TVector<TVector<float>> prior = {{0.5, 1.0}};
            NCatboostOptions::TCtrDescription bucketsCtr(ECtrType::Buckets, prior, bucketsBinarization);
            NCatboostOptions::TCtrDescription freqCtr(ECtrType::FeatureFreq, prior, freqBinarization);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }

        TDocParallelDataSetBuilder dataSetsHolderBuilder(*featuresManager,
                                                         *dataProvider,
                                                         estimators,
                                                         nullptr);

        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount, &NPar::LocalExecutor());

        for (ui32 i = 0; i < dataSet.PermutationsCount(); ++i) {
            TestPointwiseHistogramForDocParallelDataSet(dataSet.GetDataSetForPermutation(i),
                                                        *featuresManager);
        }
    }

    void RunTests(ui32 seed, ui32 oneHotLimit, bool featureParallel = true) {
        TRandom random(seed);
        TBinarizedPool pool;

        auto stopCudaManagerGuard = StartCudaManager();
        {
            for (ui32 bin : {4, 16, 32, 64, 128, 255}) {
                {
                    Cout << "Test bin count #" << bin << Endl;
                    const ui32 numCatFeatures = 7;
                    GenerateTestPool(pool, bin, numCatFeatures, random.NextUniformL());

                    SavePoolToFile(pool, "test-pool.txt");
                    SavePoolCDToFile("test-pool.txt.cd", numCatFeatures);

                    if (featureParallel) {
                        TestPointwiseHist(bin, oneHotLimit, 4);
                    } else {
                        TestPointwiseHistDocParallel(bin, oneHotLimit, 4);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestPointwiseTreeSearcherFeatureParallelWithoutOneHot) {
        RunTests(0, 0);
    }

    Y_UNIT_TEST(TestPointwiseTreeSearcherFeatureParallelWithOneHot) {
        RunTests(0, 6);
    }

    Y_UNIT_TEST(TestPointwiseTreeSearcherDocParallelWithoutOneHot) {
        RunTests(0, 0, false);
    }

    Y_UNIT_TEST(TestPointwiseTreeSearcherDocParallelWithOneHot) {
        RunTests(0, 6, false);
    }
    //
}
