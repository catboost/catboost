#include <catboost/cuda/ut_helpers/test_utils.h>
#include <library/unittest/registar.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <catboost/cuda/data/binarization_config.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/gpu_data/fold_based_dataset_builder.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/methods/histograms_helper.h>
#include <catboost/cuda/methods/oblivious_tree_structure_searcher.h>

using namespace std;
using namespace NCatboostCuda;

SIMPLE_UNIT_TEST_SUITE(TPointwiseHistogramTest) {
    template <class TGridPolicy>
    void inline CalcRefSums(const TGpuBinarizedDataSet<TGridPolicy>* dataSet,
                            const TMirrorBuffer<float>& approx,
                            const TMirrorBuffer<float>& wts,
                            const TMirrorBuffer<ui32>& indices,
                            const TMirrorBuffer<const TDataPartition>& partitioning,
                            ui32 depth, ui32 foldCount,
                            TVector<float>* sums1,
                            TVector<float>* sums2) {
        TVector<float> targets, weights;
        TVector<ui32> inds;

        TVector<TDataPartition> parts;
        partitioning.Read(parts);

        TVector<ui32> cindex;
        ui32 numLeaves = 1 << depth;
        ui32 bitsPerFold = IntLog2(foldCount);
        ui32 foldsStripe = 1 << bitsPerFold;

        approx.Read(targets);
        wts.Read(weights);
        indices.Read(inds);
        partitioning.Read(parts);

        sums1->resize(dataSet->GetHostBinaryFeatures().size() * numLeaves * foldCount);
        sums2->resize(dataSet->GetHostBinaryFeatures().size() * numLeaves * foldCount);

        TVector<float>& binSums = *sums1;
        TVector<float>& binWeights = *sums2;

        const auto& featuresMapping = dataSet->GetGrid().GetMapping();
        const auto& grid = dataSet->GetHostFeatures();

        ui32 binFeatureIdx = 0;

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            TSlice featuresSlice = featuresMapping.DeviceSlice(dev);

            TVector<ui32> compressedIndex;
            dataSet->GetCompressedIndex().DeviceView(dev).Read(compressedIndex);
            const ui32 dsSize = dataSet->GetDataSetSize().At(dev);

            for (ui32 f = featuresSlice.Left; f < featuresSlice.Right; ++f) {
                const TCFeature& feature = grid[f];
                const ui32* cindexPtr = &compressedIndex[feature.Offset * dsSize];

                const ui32 numParts = numLeaves * foldCount;

                for (ui32 leaf = 0; leaf < numLeaves; leaf++) {
                    for (ui32 fold = 0; fold < foldCount; ++fold) {
                        const auto& part = parts[leaf * foldsStripe + fold];

                        TVector<double> s1(feature.Folds + 1, 0.0f);
                        TVector<double> s2(feature.Folds + 1, 0.0f);

                        for (ui32 i = 0; i < part.Size; i++) {
                            ui32 idx = inds[part.Offset + i];
                            UNIT_ASSERT(idx < dsSize);
                            UNIT_ASSERT((feature.Offset * dsSize + idx) < compressedIndex.size());
                            ui32 ci = cindexPtr[idx];

                            if (((ci >> feature.Shift) & feature.Mask) > s1.size()) {
                                Cout << "Feature  " << f << " " << dev << " " << dataSet->FeatureCount() << " " << Endl;
                                Cout << "Feature offset  " << feature.Offset << Endl;
                                Cout << "Bug " << part.Offset << " " << i << Endl;
                                Cout << feature.Folds << Endl;
                                Cout << feature.Index << " " << feature.FirstFoldIndex << Endl;
                                Cout << ((ci) >> feature.Shift)
                                     << " " << ((ci >> feature.Shift) & feature.Mask) << " " << ci
                                     << " " << feature.Mask << " " << feature.Shift << Endl;
                                UNIT_ASSERT(false);
                            }

                            s1[(ci >> feature.Shift) & feature.Mask] += targets[part.Offset + i];
                            s2[(ci >> feature.Shift) & feature.Mask] += weights[part.Offset + i];
                        }

                        if (!feature.OneHotFeature) {
                            for (ui32 i = 1; i < feature.Folds; i++) {
                                s1[i] += s1[i - 1];
                                s2[i] += s2[i - 1];
                            }
                        }

                        for (ui32 i = 0; i < feature.Folds; i++) {
                            const ui32 offset = (binFeatureIdx + i) * numParts + leaf * foldCount + fold;
                            binSums[offset] = s1[i];
                            binWeights[offset] = s2[i];
                        }
                    }

                    for (ui32 fold = foldCount; fold < foldsStripe; ++fold) {
                        const auto& part = parts[leaf * foldsStripe + fold];
                        UNIT_ASSERT_VALUES_EQUAL(part.Size, 0);
                    }
                }
                binFeatureIdx += feature.Folds;
            }
        }
    }

    template <class TGridPolicy>
    inline void CheckResults(const TGpuBinarizedDataSet<TGridPolicy>* dataSet,
                             const TMirrorBuffer<float>& approx,
                             const TMirrorBuffer<float>& wts,
                             const TMirrorBuffer<ui32>& indices,
                             const TMirrorBuffer<const TDataPartition>& partitioning,
                             ui32 depth, ui32 foldCount,
                             const TVector<float>& props) {
        TVector<float> refSums, refWts;
        CalcRefSums<TGridPolicy>(dataSet, approx, wts, indices, partitioning, depth, foldCount, &refSums, &refWts);

        const ui32 leavesCount = static_cast<const ui32>(1 << depth);
        const ui32 partCount = leavesCount * foldCount;

        auto& binaryFeatures = dataSet->GetHostBinaryFeatures();

        ui32 currentDevice = 0;
        ui32 deviceOffset = 0;
        for (ui32 i = 0; i < binaryFeatures.size(); i++) {
            auto& binFeature = binaryFeatures[i];
            auto& feature = dataSet->GetHostFeatures()[binFeature.FeatureId];

            if (!dataSet->GetBinaryFeatures().GetMapping().DeviceSlice(currentDevice).Contains(i)) {
                deviceOffset = dataSet->GetBinaryFeatures().GetMapping().DeviceSlice(currentDevice).Right * partCount * 2;
                ++currentDevice;
            }

            for (ui32 leaf = 0; leaf < leavesCount; leaf++) {
                for (ui32 fold = 0; fold < foldCount; ++fold) {
                    {
                        double x = props[deviceOffset + ((feature.FirstFoldIndex + binFeature.BinId) * partCount + leaf * foldCount + fold) * 2];
                        double refX = refWts[i * partCount + leaf * foldCount + fold];
                        if (std::abs(x - refX) > 1e-6) {
                            ui32 bitsPerFold = IntLog2(foldCount);
                            ui32 foldsStripe = 1 << bitsPerFold;

                            TVector<TDataPartition> parts;
                            partitioning.Read(parts);
                            Cout << deviceOffset << "  " << depth << Endl;
                            Cout << parts[leaf * foldsStripe + fold].Offset << " " << parts[leaf * foldsStripe + fold].Size << Endl;
                            Cout << i << " " << leaf << " " << fold << " " << x << " " << refX << Endl;
                            Cout << i * partCount + leaf * foldCount + fold << " " << leavesCount << " " << foldCount << Endl;
                            Cout << feature.Offset << " " << feature.Folds << " " << feature.Index << " " << feature.FirstFoldIndex << Endl;

                            for (ui32 i = 0; i < Min<ui32>(40, refWts.size()); ++i) {
                                Cout << refWts[i] << " ";
                            }
                            Cout << Endl;
                        }

                        UNIT_ASSERT_DOUBLES_EQUAL(x, refX, 1e-6);
                    }
                    {
                        double x = props[deviceOffset + ((feature.FirstFoldIndex + binFeature.BinId) * partCount + leaf * foldCount + fold) * 2 + 1];
                        double refX = refSums[i * partCount + leaf * foldCount + fold];

                        UNIT_ASSERT_DOUBLES_EQUAL(x, refX, 1e-6);
                    }
                }
            }
        }
    }

    TOptimizationSubsets CreateSubsets(ui32 maxDepth,
                                       TL2Target & src,
                                       ui32 foldCount,
                                       TVector<ui32> & bins) {
        TOptimizationSubsets subsets;
        subsets.Bins = TMirrorBuffer<ui32>::CopyMapping(src.WeightedTarget);
        subsets.Bins.Write(bins);

        subsets.Indices = TMirrorBuffer<ui32>::CopyMapping(subsets.Bins);
        MakeSequence(subsets.Indices);

        subsets.CurrentDepth = 0;
        subsets.FoldCount = foldCount;
        subsets.FoldBits = IntLog2(subsets.FoldCount);

        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions = TMirrorBuffer<TDataPartition>::Create(NCudaLib::TMirrorMapping(maxPartCount));
        subsets.Src = &src;
        subsets.Update();

        {
            TVector<TDataPartition> initParts;
            subsets.CurrentPartsView().Read(initParts);
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

    TL2Target CreateTestTarget(ui32 size) {
        TRandom rand(100500);

        TL2Target target;
        auto mapping = NCudaLib::TMirrorMapping(size);

        target.Weights = TMirrorBuffer<float>::Create(mapping);
        target.WeightedTarget = TMirrorBuffer<float>::Create(mapping);

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

    void CheckStats(const TOptimizationSubsets& subsets,
                    const TVector<float>& gatheredTarget,
                    const TVector<float>& gatheredWeights,
                    const TMirrorBuffer<const TPartitionStatistics>& partStats) {
        TVector<TPartitionStatistics> cpuStat;
        partStats.Read(cpuStat);

        TVector<TDataPartition> cpuParts;
        subsets.CurrentPartsView().Read(cpuParts);

        {
            for (ui32 partId = 0; partId < cpuParts.size(); ++partId) {
                auto& part = cpuParts[partId];

                double sum = 0;
                double weight = 0;

                for (ui32 i = 0; i < part.Size; ++i) {
                    sum += gatheredTarget[part.Offset + i];
                    weight += gatheredWeights[part.Offset + i];
                }

                UNIT_ASSERT_DOUBLES_EQUAL(cpuStat[partId].Sum, sum, 1e-5);
                UNIT_ASSERT_DOUBLES_EQUAL(cpuStat[partId].Weight, weight, 1e-5);
            }
        }
    }

    void CheckResultsForGpuFeatures(const TGpuFeatures<>& features,
                                    TGpuFeaturesScoreCalcer<>& calcer,
                                    const TMirrorBuffer<const TDataPartition>& partitionStats,
                                    const TMirrorBuffer<float>& weightedTarget,
                                    const TMirrorBuffer<float>& weights,
                                    const TMirrorBuffer<ui32>& indices,
                                    ui32 depth,
                                    ui32 foldCount) {

        if (calcer.HasBinaryFeatureHelper())
        {
            auto& scoreHelper = calcer.GetBinaryFeatureHelper();
            auto histogram  = scoreHelper.ReadHistograms();

            CheckResults(&features.GetBinaryFeatures(),
                         weightedTarget, weights,
                         indices,
                         partitionStats,
                         depth,
                         foldCount,
                         histogram);
        }
        if (calcer.HasByteFeatureHelper())
        {
            auto& scoreHelper = calcer.GetByteFeatureHelper();
            auto histogram  = scoreHelper.ReadHistograms();

            CheckResults(&features.GetFeatures(),
                         weightedTarget, weights,
                         indices,
                         partitionStats,
                         depth,
                         foldCount,
                         histogram);
        }
        if (calcer.HasHalfByteFeatureHelper())
        {
            auto& scoreHelper = calcer.GetHalfByteFeatureHelper();
            auto histogram  = scoreHelper.ReadHistograms();

            CheckResults(&features.GetHalfByteFeatures(),
                         weightedTarget, weights,
                         indices,
                         partitionStats,
                         depth,
                         foldCount,
                         histogram);
        }

    }

    void TestPointwiseHistogramForDataSet(const TDataSet<>& dataSet,
                                          const TBinarizedFeaturesManager& featuresManager) {
        TRandom rand(10);
        const ui32 dsSize = dataSet.GetIndices().GetObjectsSlice().Size();

        const ui32 foldCount = 28;
        const ui32 maxDepth = 10;
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
        TL2Target target = CreateTestTarget(targetSize);

        auto indices = TMirrorBuffer<ui32>::CopyMapping(target.WeightedTarget);
        indices.Write(learnIndices);
        TOptimizationSubsets subsets = CreateSubsets(maxDepth, target, foldCount, foldBins);

        TScopedCacheHolder cache;

        TMirrorBuffer<ui32> docBins = TMirrorBuffer<ui32>::CopyMapping(dataSet.GetIndices());

        TTreeUpdater<TDataSet<>> treeUpdater(cache,
                                             featuresManager,
                                             dataSet.GetCtrTargets(),
                                             dataSet,
                                             docBins);

        TVector<float> gatheredTarget;
        TVector<float> gatheredWeights;

        subsets.GatheredTarget.WeightedTarget.Read(gatheredTarget);
        subsets.GatheredTarget.Weights.Read(gatheredWeights);

        auto partitionStats = subsets.ComputePartitionStats();
        CheckStats(subsets, gatheredTarget, gatheredWeights, partitionStats);

        auto observationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);
        auto directObservationIndices = TMirrorBuffer<ui32>::CopyMapping(subsets.Indices);

        TObliviousTreeLearnerOptions treeConfig;
        treeConfig.SetMaxDepth(maxDepth);

        TGpuFeaturesScoreCalcer<> featuresScoreCalcer(dataSet.GetFeatures(), treeConfig, foldCount, true);
        TGpuFeaturesScoreCalcer<> simpleCtrScoreCalcer(dataSet.GetPermutationFeatures(), treeConfig, foldCount, true);



        TObliviousTreeStructure result;

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            //warning: don't change order of commands. current pipeline ensures maximum stream-parallelism until read
            //best score stage
            partitionStats = subsets.ComputePartitionStats();
            {
                subsets.GatheredTarget.WeightedTarget.Read(gatheredTarget);
                subsets.GatheredTarget.Weights.Read(gatheredWeights);
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
                featuresScoreCalcer.SubmitCompute(subsets, observationIndices);
                simpleCtrScoreCalcer.SubmitCompute(subsets, directObservationIndices);
            }
            manager.WaitComplete();

            CheckResultsForGpuFeatures(dataSet.GetFeatures(),
                                       featuresScoreCalcer,
                                       subsets.CurrentPartsView(),
                                       subsets.GatheredTarget.WeightedTarget,
                                       subsets.GatheredTarget.Weights,
                                       observationIndices,
                                       depth,
                                       foldCount);

            CheckResultsForGpuFeatures(dataSet.GetPermutationFeatures(),
                                       simpleCtrScoreCalcer,
                                       subsets.CurrentPartsView(),
                                       subsets.GatheredTarget.WeightedTarget,
                                       subsets.GatheredTarget.Weights,
                                       directObservationIndices,
                                       depth,
                                       foldCount);

            {
                featuresScoreCalcer.ComputeOptimalSplit(partitionStats);
                simpleCtrScoreCalcer.ComputeOptimalSplit(partitionStats);
            }

            TBinarySplit bestSplit;
            {
//                const TGpuBinarizedDataSet<TByteFeatureGridPolicy>& byteFeatures
                auto featureIds = dataSet.GetFeatures().ComputeAllFeatureIds();
//                TVector<TCFeature> features = byteFeatures.GetHostFeatures();
                auto localIdx = rand.NextUniformL() % featureIds.size();
                bestSplit.FeatureId = featureIds[localIdx];
                bestSplit.BinIdx =  featuresManager.GetBinCount(bestSplit.FeatureId) / 2;
                bestSplit.SplitType = featuresManager.IsCat(localIdx) ? EBinSplitType::TakeBin : EBinSplitType::TakeGreater;

                treeUpdater.AddSplit(bestSplit);

                subsets.Split(docBins,
                              observationIndices);
            }
        }
    }

    void TestPointwiseHist(ui32 binarization,
                           ui32 oneHotLimit,
                           ui32 permutationCount) {
        TBinarizationConfiguration binarizationConfiguration;
        binarizationConfiguration.DefaultFloatBinarization.Discretization = binarization;
        TFeatureManagerOptions featureManagerOptions(binarizationConfiguration, oneHotLimit);
        TBinarizedFeaturesManager featuresManager(featureManagerOptions);

        TDataProvider dataProvider;
        TOnCpuGridBuilderFactory gridBuilderFactory;
        TDataProviderBuilder dataProviderBuilder(featuresManager,
                                                 dataProvider);

        ReadPool("test-pool.txt.cd",
                 "test-pool.txt",
                 "",
                 16,
                 true,
                 dataProviderBuilder.SetShuffleFlag(false));

        {
            TVector<float> prior = {0.5};
            featuresManager.EnableCtrType(ECtrType::Buckets, prior);
            featuresManager.EnableCtrType(ECtrType::FeatureFreq, prior);
        }

        TDataSetHoldersBuilder<> dataSetsHolderBuilder(featuresManager,
                                                       dataProvider,
                                                       nullptr,
                                                       false);

        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount);
        for (ui32 i = 0; i < dataSet.PermutationsCount(); ++i) {
            TestPointwiseHistogramForDataSet(dataSet.GetDataSetForPermutation(i), featuresManager);
        }
    }

    void RunTest(ui32 seed, ui32 oneHotLimit) {
        TRandom random(seed);
        TBinarizedPool pool;

        StartCudaManager();
        {
            for (ui32 bin : {4, 16, 32, 64, 128, 255}) {
                {
                    Cout << "Test bin count #" << bin << Endl;
                    const ui32 numCatFeatures = 7;
                    GenerateTestPool(pool, bin, numCatFeatures, random.NextUniformL());

                    SavePoolToFile(pool, "test-pool.txt");
                    SavePoolCDToFile("test-pool.txt.cd", numCatFeatures);

                    TestPointwiseHist(bin, oneHotLimit, 8);
                }
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestPointwiseTreeSearcherWithoutOneHot) {
        RunTest(0, 0);
    }

    SIMPLE_UNIT_TEST(TestPointwiseTreeSearcherWitOneHot) {
        RunTest(0, 6);
    }
    //
}
