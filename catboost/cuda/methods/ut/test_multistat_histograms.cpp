#include <library/cpp/testing/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/methods/greedy_subsets_searcher/split_properties_helper.h>
#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/cuda/methods/pointwise_optimization_subsets.h>

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/private/libs/quantization/grid_creator.h>

#include <util/generic/hash.h>
#include <util/system/info.h>

using namespace std;
using namespace NCatboostCuda;

template <>
inline TString Printable(TCBinFeature val) {
    return TStringBuilder() << val.FeatureId << "/" << val.BinId;
}

Y_UNIT_TEST_SUITE(TPointwiseMultiStatHistogramTest) {
    inline TVector<TVector<ui32>>& GetCompressedIndexCpu(const TDocParallelDataSet& dataSet) {
        TString key = "cindex";
        return dataSet.Cache(key, [&]() -> TVector<TVector<ui32>> {
            const ui32 devCount = static_cast<const ui32>(NCudaLib::GetCudaManager().GetDeviceCount());
            TVector<TVector<ui32>> cindexOnDevs(devCount);
            for (ui32 dev = 0; dev < devCount; ++dev) {
                dataSet.GetCompressedIndex().GetStorage().DeviceView(dev).Read(cindexOnDevs[dev]);
            }
            return cindexOnDevs;
        });
    }

    inline bool SplitValue(const ui32* cindex,
                           const ui32 i,
                           const TCFeature feature,
                           const ui32 fold) {
        ui32 bin = (cindex[feature.Offset + i] >> feature.Shift) & feature.Mask;
        if (feature.OneHotFeature) {
            return bin == fold;
        } else {
            return bin > fold;
        }
    }

    using TLayout = TDocParallelLayout;

    void inline CalcRefSums(EFeaturesGroupingPolicy policy,
                            const TVector<TVector<ui32>>& cindexCpu,
                            const typename TSharedCompressedIndex<TLayout>::TCompressedDataSet& dataSet,
                            const TCudaBuffer<float, typename TLayout::TSamplesMapping>& stats,
                            const TCudaBuffer<ui32, typename TLayout::TSamplesMapping>& indices,
                            const TCudaBuffer<const TDataPartition, typename TLayout::TSamplesMapping>& partitioning,
                            ui32 numLeaves,
                            THashMap<ui32, TVector<TVector<float>>>* refStats) {
        if (!dataSet.HasFeaturesForPolicy(policy)) {
            return;
        }
        auto& binFeatures = dataSet.GetBinFeatures(policy);
        const ui32 statCount = stats.GetColumnCount();

        TVector<float> binSums(binFeatures.size() * numLeaves * statCount);

        const auto& cpuGrid = dataSet.GetCpuGrid(policy);

        TAdaptiveLock refStatLock;

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            TSlice docSlice = stats.GetMapping().DeviceSlice(dev);
            const TVector<ui32>& compressedIndex = cindexCpu[dev];

            const ui32 docsOnDevice = docSlice.Size();

            TVector<TDataPartition> parts;
            partitioning.DeviceView(dev).Read(parts);

            TVector<ui32> inds;
            TVector<float> statsCpu;

            stats.DeviceView(dev).Read(statsCpu);
            indices.DeviceView(dev).Read(inds);

            NPar::ParallelFor(0, cpuGrid.FeatureIds.size(), [&](ui32 f) {
                ui32 featureId = cpuGrid.FeatureIds[f];
                if (dataSet.GetTCFeature(featureId).IsEmpty(dev)) {
                    return;
                }
                if (docsOnDevice == 0) {
                    return;
                }

                TCFeature feature = dataSet.GetTCFeature(featureId).At(dev);
                const ui32* cindexPtr = &compressedIndex[feature.Offset];

                TVector<TVector<float>> featureHist(feature.Folds + 1);
                for (ui32 fold = 0; fold <= feature.Folds; ++fold) {
                    featureHist[fold].resize(statCount * numLeaves);
                }

                for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                    const auto& part = parts[leaf];

                    for (ui32 i = 0; i < part.Size; i++) {
                        ui32 idx = inds[part.Offset + i];
                        UNIT_ASSERT(idx < docsOnDevice);
                        UNIT_ASSERT((feature.Offset + idx) < compressedIndex.size());
                        ui32 ci = cindexPtr[idx];

                        int bin = (ci >> feature.Shift) & feature.Mask;
                        for (ui32 stat = 0; stat < statCount; ++stat) {
                            CB_ENSURE(part.Offset + stat * inds.size() + i < statsCpu.size(), part.Offset << " " << part.Size << " " << stat << " " << inds.size() << " " << i << " " << statsCpu.size());
                            Y_ASSERT((ui32)bin < featureHist.size());
                            Y_ASSERT(leaf * statCount + stat < featureHist[bin].size());
                            featureHist[bin][leaf * statCount + stat] += statsCpu[part.Offset + stat * inds.size() + i];
                        }
                    }

                    if (!feature.OneHotFeature) {
                        for (ui32 stat = 0; stat < statCount; ++stat) {
                            for (ui32 i = 1; i <= feature.Folds; i++) {
                                featureHist[i][leaf * statCount + stat] += featureHist[i - 1][leaf * statCount + stat];
                            }
                        }
                    }
                }

                with_lock (refStatLock) {
                    if (!refStats->contains(featureId)) {
                        (*refStats)[featureId] = featureHist;
                    } else {
                        auto& dst = (*refStats)[featureId];
                        for (ui32 i = 0; i <= feature.Folds; ++i) {
                            for (ui32 j = 0; j < statCount * numLeaves; ++j) {
                                dst[i][j] += featureHist[i][j];
                            }
                        }
                    }
                }
            });
        }
    }

    void CheckHistograms(const TDocParallelDataSet& dataSet,
                         const TPointsSubsets& subsets) {
        THashMap<ui32, TVector<TVector<float>>> refStats;

        const auto& cindexCpu = GetCompressedIndexCpu(dataSet);

        for (auto policy : GetEnumAllValues<NCatboostCuda::EFeaturesGroupingPolicy>()) {
            if (dataSet.HasFeatures()) {
                CalcRefSums(policy,
                            cindexCpu,
                            dataSet.GetFeatures(),
                            subsets.Target.StatsToAggregate,
                            subsets.Target.Indices,
                            subsets.Partitions.AsConstBuf(),
                            subsets.Leaves.size(),
                            &refStats);
            }

            if (dataSet.HasPermutationDependentFeatures()) {
                CalcRefSums(policy,
                            cindexCpu,
                            dataSet.GetPermutationFeatures(),
                            subsets.Target.StatsToAggregate,
                            subsets.Target.Indices,
                            subsets.Partitions.AsConstBuf(),
                            subsets.Leaves.size(),
                            &refStats);
            }
        }

        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();

        for (ui32 dev = 0; dev < devCount; ++dev) {
            TVector<float> resultsGpu;
            subsets.Histograms.DeviceView(dev).Read(resultsGpu);
            TVector<TCBinFeature> binFeatures;
            subsets.BinFeatures.DeviceView(dev).Read(binFeatures);

            const ui32 numStats = subsets.Target.StatsToAggregate.GetColumnCount();
            const ui32 numLeaves = subsets.Leaves.size();

            //            leafId * binFeatureCount * statCount + statId * binFeatureCount + binFeatureId;
            const auto binFeatureCountOnDevice = binFeatures.size();
            for (ui32 i = 0; i < binFeatureCountOnDevice; ++i) {
                ui32 fid = binFeatures[i].FeatureId;
                ui32 binId = binFeatures[i].BinId;

                for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                    if (subsets.Leaves[leaf].HistogramsType != EHistogramsType::CurrentPath) {
                        UNIT_ASSERT(subsets.Leaves[leaf].IsTerminal);
                        continue;
                    }

                    for (ui32 statId = 0; statId < numStats; ++statId) {
                        //on GPU we store stats for all binFeatures in sequence, so we could load them in compute scores in coallased way
                        const ui64 idx = leaf * numStats * binFeatureCountOnDevice +
                                         statId * binFeatureCountOnDevice +
                                         i;

                        const float computedOnGpu = resultsGpu[idx];
                        UNIT_ASSERT(refStats.contains(fid));
                        UNIT_ASSERT(refStats[fid].size() > binId);
                        UNIT_ASSERT(refStats[fid][binId].size() == numStats * numLeaves);
                        const float computedOnCpu = refStats[fid][binId][leaf * numStats + statId];
                        if (std::abs(computedOnGpu - computedOnCpu) > 1e-5) {
                            DumpToFile(subsets.Histograms, "histograms");
                        }
                        UNIT_ASSERT_DOUBLES_EQUAL_C(computedOnGpu, computedOnCpu, 1e-5, i << ": " << fid << " " << binId << " " << statId << " " << leaf << " " << numLeaves);
                    }
                }
            }
        }
    }

    TOptimizationTarget CreateTestTarget(const TDocParallelDataSet& dataSet,
                                         ui32 numStats,
                                         double) {
        TRandom rand(100500);

        TOptimizationTarget target;

        target.StatsToAggregate.Reset(dataSet.GetSamplesMapping(),
                                      numStats);
        target.Indices.Reset(dataSet.GetSamplesMapping());
        MakeSequence(target.Indices);
        const ui32 indicesCount = target.Indices.GetObjectsSlice().Size();
        Cout << "Indices count " << indicesCount << Endl;

        {
            TVector<float> stats(numStats * target.Indices.GetObjectsSlice().Size());

            // fill with trash to avoid rounding comparison
            for (ui32 i = 0; i < stats.size(); i++) {
                //            for (ui32 stat = 0; stat < numStats; stat++) {
                //                for (ui32 i = 0; i < indicesCount; ++i) {
                //                    stats[stat * indicesCount + i] = 1.0 / (1 << stat);
                auto rel = (rand.NextUniformL() % 5);
                if (rel == 0) {
                    stats[i] = 0.125f;
                } else if (rel == 1) {
                    stats[i] = 0.25f;
                } else if (rel == 2) {
                    stats[i] = 0.5f;
                } else if (rel == 3) {
                    stats[i] = 0.75f;
                }
                //                }
            }
            target.StatsToAggregate.Write(stats);
        }
        return target;
    }

    void SplitPointsCpu(const TDocParallelDataSet& dataSet,
                        const TVector<ui32>& leaves,
                        const TPointsSubsets& subsets,
                        TVector<TVector<TDataPartition>>* newParts,
                        TVector<TVector<ui32>>* newIndices,
                        TVector<TVector<float>>* newStats) {
        auto currentParts = subsets.CurrentParts();
        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        const auto& cindex = GetCompressedIndexCpu(dataSet);

        newParts->resize(devCount);
        newIndices->resize(devCount);
        newStats->resize(devCount);

        const ui32 numStats = subsets.Target.StatsToAggregate.GetColumnCount();

        for (ui32 dev = 0; dev < devCount; ++dev) {
            TVector<TDataPartition> devParts;

            TVector<ui32> devIndices;
            TVector<ui32> permutation;
            subsets.Target.Indices.DeviceView(dev).Read(devIndices);

            permutation.resize(devIndices.size());
            Iota(permutation.begin(), permutation.end(), 0);

            TVector<float> devAggregateStats;
            currentParts.DeviceView(dev).Read(devParts);

            subsets.Target.StatsToAggregate.DeviceView(dev).Read(devAggregateStats);

            for (ui32 leafId : leaves) {
                auto& leaf = subsets.Leaves[leafId];
                UNIT_ASSERT(leaf.BestSplit.Defined());

                TCFeature feature = dataSet.GetTCFeature(leaf.BestSplit.FeatureId).At(dev);
                ui32 bin = leaf.BestSplit.BinId;
                const ui32 offset = devParts[leafId].Offset;

                TDataPartition newPart;
                newPart.Offset = devParts[leafId].Offset;

                const ui32 partSize = devParts[leafId].Size;

                auto& oldPart = devParts[leafId];
                oldPart.Size = 0;

                TVector<ui32> leftIndices;
                TVector<ui32> rightIndices;

                for (ui32 i = 0; i < partSize; ++i) {
                    const bool isRight = SplitValue(cindex[dev].data(),
                                                    devIndices[offset + i],
                                                    feature,
                                                    bin);
                    if (isRight) {
                        newPart.Size++;
                        rightIndices.push_back(permutation[offset + i]);
                    } else {
                        oldPart.Size++;
                        newPart.Offset++;
                        leftIndices.push_back(permutation[offset + i]);
                    }
                }

                CB_ENSURE(oldPart.Size + newPart.Size == partSize);
                CB_ENSURE(newPart.Offset == oldPart.Offset + oldPart.Size);

                for (ui32 i = 0; i < oldPart.Size; ++i) {
                    permutation[oldPart.Offset + i] = leftIndices[i];
                }
                for (ui32 i = 0; i < newPart.Size; ++i) {
                    permutation[newPart.Offset + i] = rightIndices[i];
                }
                devParts.push_back(newPart);
            }

            auto& permutedIndices = (*newIndices)[dev];
            auto& permutedStats = (*newStats)[dev];

            permutedIndices.resize(devIndices.size());
            permutedStats.resize(devAggregateStats.size());

            for (ui32 i = 0; i < permutation.size(); ++i) {
                permutedIndices[i] = devIndices[permutation[i]];

                for (ui32 statId = 0; statId < numStats; ++statId) {
                    permutedStats[statId * permutation.size() + i] = devAggregateStats[statId * permutation.size() + permutation[i]];
                }
            }
            (*newParts)[dev] = devParts;
            (*newStats)[dev] = permutedStats;
        }
    }

    void CheckPartStats(const TPointsSubsets& subsets) {
        auto currentParts = subsets.CurrentParts();
        auto currentPartsHost = subsets.CurrentPartsCpu();

        {
            TVector<TDataPartition> parts;
            currentParts.Read(parts);

            {
                TVector<TDataPartition> parts2;
                currentPartsHost.Read(parts2);
                for (ui32 i = 0; i < parts.size(); ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(parts[i].Size, parts2[i].Size);
                    UNIT_ASSERT_VALUES_EQUAL(parts[i].Offset, parts2[i].Offset);
                }
            }
        }

        const ui32 numLeaves = subsets.Leaves.size();
        const ui32 numStats = static_cast<ui32>(subsets.Target.StatsToAggregate.GetColumnCount());

        TVector<ui32> leafSizes(numLeaves);

        for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
            auto devIndices = subsets.Target.Indices.DeviceView(dev);
            auto devParts = currentParts.DeviceView(dev);
            TVector<TDataPartition> cpuParts;
            devParts.Read(cpuParts);
            TVector<ui32> indices;
            devIndices.Read(indices);

            TVector<double> devStats;
            subsets.PartitionStats
                .DeviceView(dev)
                .SliceView(TSlice(0, subsets.Leaves.size()))
                .Read(devStats);

            UNIT_ASSERT_VALUES_EQUAL(devStats.size(), subsets.Leaves.size() * numStats);

            TVector<float> devStatsToAggregate;
            subsets.Target.StatsToAggregate.DeviceView(dev).Read(devStatsToAggregate);

            for (ui32 partId = 0; partId < cpuParts.size(); ++partId) {
                auto& part = cpuParts[partId];
                leafSizes[partId] += part.Size;
                TVector<double> stats(numStats);

                for (ui32 i = 0; i < part.Size; ++i) {
                    for (ui32 statId = 0; statId < numStats; ++statId) {
                        stats[statId] += devStatsToAggregate[part.Offset + i + statId * indices.size()];
                    }
                }
                for (ui32 statId = 0; statId < numStats; ++statId) {
                    UNIT_ASSERT_DOUBLES_EQUAL_C(stats[statId], devStats[statId + partId * numStats], 1e-5,
                                                "PartCount " << cpuParts.size() << " device " << dev << " " << partId << " " << part.Offset << " / " << part.Size);
                }
            }
        }

        UNIT_ASSERT_VALUES_EQUAL(subsets.Leaves.size(), leafSizes.size());

        for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
            UNIT_ASSERT_VALUES_EQUAL(subsets.Leaves[leaf].Size, leafSizes[leaf]);
        }
    }

    template <class T>
    void AssertVecEqual(const TVector<T>& left, const TVector<T>& right) {
        UNIT_ASSERT_VALUES_EQUAL(left.size(), right.size());
        for (ui32 i = 0; i < left.size(); ++i) {
            Y_ASSERT(left[i] == right[i]);
            UNIT_ASSERT_VALUES_EQUAL(left[i], right[i]);
        }
    }

    void CheckAndMakeSplit(const TVector<ui32>& leaves,
                           TSplitPropertiesHelper& splitPropertiesHelper,
                           TPointsSubsets& subsets) {
        CheckPartStats(subsets);

        TVector<TVector<TDataPartition>> newParts;
        TVector<TVector<ui32>> indicesAfterSpltiCpu;
        TVector<TVector<float>> statsAfterSplitCpu;

        SplitPointsCpu(splitPropertiesHelper.GetDataSet(),
                       leaves,
                       subsets,
                       &newParts,
                       &indicesAfterSpltiCpu,
                       &statsAfterSplitCpu);

        splitPropertiesHelper.MakeSplit(leaves,
                                        &subsets);

        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();

        for (ui32 dev = 0; dev < devCount; ++dev) {
            TVector<ui32> indicesAfterSplit;
            TVector<float> statsAfterSplit;
            subsets.Target.Indices.DeviceView(dev).Read(indicesAfterSplit);
            subsets.Target.StatsToAggregate.DeviceView(dev).Read(statsAfterSplit);

            AssertVecEqual(indicesAfterSpltiCpu[dev], indicesAfterSplit);
            AssertVecEqual(statsAfterSplit, statsAfterSplitCpu[dev]);

            TVector<TDataPartition> partCpu;
            subsets.Partitions.DeviceView(dev).Read(partCpu);
            partCpu.resize(subsets.Leaves.size());
            for (ui32 i = 0; i < partCpu.size(); ++i) {
                UNIT_ASSERT_EQUAL_C(partCpu[i].Offset, newParts[dev][i].Offset, i << " " << partCpu[i].Offset << " " << partCpu[i].Size << " " << newParts[dev][i].Offset << " " << newParts[dev][i].Size);
                UNIT_ASSERT_EQUAL_C(partCpu[i].Size, newParts[dev][i].Size, i << " " << partCpu[i].Offset << " " << partCpu[i].Size << " " << newParts[dev][i].Offset << " " << newParts[dev][i].Size);
            }
        }

        CheckPartStats(subsets);
    }

    //
    void RunComputeTest(const TDocParallelDataSet& dataSet,
                        const ui32 numStats,
                        const ui32 maxLeaves,
                        double sampleRate,
                        const TComputeByBlocksConfig& byBlocksConfig,
                        const TBinarizedFeaturesManager& featuresManager) {
        TRandom rand(10);

        TComputeSplitPropertiesByBlocksHelper computeSplitPropertiesByBlocksHelper(dataSet,
                                                                                   byBlocksConfig);

        TSplitPropertiesHelper splitPropertiesHelper(dataSet,
                                                     featuresManager,
                                                     computeSplitPropertiesByBlocksHelper);

        TVector<float> featureWeights(featuresManager.GetFeatureCount(), 1.0f);
        auto subsets = splitPropertiesHelper.CreateInitialSubsets(
            CreateTestTarget(dataSet, numStats, sampleRate),
            maxLeaves,
            featureWeights);

        while (subsets.Leaves.size() < maxLeaves) {
            CATBOOST_DEBUG_LOG << "Leaves count #" << subsets.Leaves.size() << Endl;

            splitPropertiesHelper.BuildNecessaryHistograms(&subsets);
            CheckHistograms(dataSet, subsets);

            TVector<ui32> leavesToSplit;
            for (ui32 i = 0; i < subsets.Leaves.size(); ++i) {
                if (subsets.Leaves[i].Size) {
                    leavesToSplit.push_back(i);
                }
            }
            UNIT_ASSERT(leavesToSplit.size() > 0);

            Shuffle(leavesToSplit.begin(), leavesToSplit.end(), rand);
            leavesToSplit.resize(Max<ui32>(leavesToSplit.size() * 0.5, 1));
            ui32 maxLeavesToSplit = leavesToSplit.size() + subsets.Leaves.size() > maxLeaves ? maxLeaves - subsets.Leaves.size() : leavesToSplit.size();
            leavesToSplit.resize(maxLeavesToSplit);
            //            leavesToSplit.resize(1);

            for (ui32 leafId : leavesToSplit) {
                TBinarySplit bestSplit;
                auto featureIds = dataSet.GetFeatures().GetFeatures();
                auto localIdx = rand.NextUniformL() % featureIds.size();
                bestSplit.FeatureId = featureIds[localIdx];
                bestSplit.BinIdx = (featuresManager.GetBinCount(bestSplit.FeatureId) - 1) / 2;
                bestSplit.SplitType = featuresManager.IsCat(localIdx) ? EBinSplitType::TakeBin
                                                                      : EBinSplitType::TakeGreater;

                subsets.Leaves[leafId].BestSplit.Score = static_cast<float>(-rand.NextUniform());
                subsets.Leaves[leafId].BestSplit.FeatureId = bestSplit.FeatureId;
                subsets.Leaves[leafId].BestSplit.BinId = bestSplit.BinIdx;
            }

            CheckAndMakeSplit(leavesToSplit,
                              splitPropertiesHelper,
                              subsets);
        }
    }

    void TestSplitPropsHelper(ui32 binarization,
                              ui32 oneHotLimit,
                              ui32 permutationCount,
                              ui32 numStats) {
        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum,
                                                                 binarization - 1);
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
            NCatboostOptions::TBinarizationOptions bucketsBinarization(EBorderSelectionType::GreedyLogSum,
                                                                       binarization - 1);
            NCatboostOptions::TBinarizationOptions freqBinarization(EBorderSelectionType::GreedyLogSum,
                                                                    binarization - 1);

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
            TComputeByBlocksConfig config;

            config.LoadPolicyAfterSplit = ELoadFromCompressedIndexPolicy::GatherBins;
            RunComputeTest(dataSet.GetDataSetForPermutation(i),
                           numStats,
                           127,
                           config.SampleRate,
                           config,
                           *featuresManager);

            config.LoadPolicyAfterSplit = ELoadFromCompressedIndexPolicy::LoadByIndexBins;
            RunComputeTest(dataSet.GetDataSetForPermutation(i),
                           numStats,
                           35,
                           config.SampleRate,
                           config,
                           *featuresManager);
        }
    }

    void RunTests(ui32 seed,
                  ui32 oneHotLimit,
                  ui32 numStats,
                  ui32 numSamples) {
        TRandom random(seed);
        TBinarizedPool pool;

        auto& localExecutor = NPar::LocalExecutor();
        const int cpuCount = NSystemInfo::CachedNumberOfCpus();
        if (localExecutor.GetThreadCount() < cpuCount) {
            const int threadsToRun = cpuCount - localExecutor.GetThreadCount() - 1;
            localExecutor.RunAdditionalThreads(threadsToRun);
        }

        auto stopCudaManagerGuard = StartCudaManager();
        {
            //            for (ui32 bin : {2, 15, 32, 64, 128, 255}) {
            //            for (ui32 bin : {2, 15, 32, 64, 128, 255}) {
            for (ui32 bin : {32, 64, 128, 255}) {
                {
                    Cout << "Test bin count #" << bin << Endl;
                    const ui32 numCatFeatures = 7 + random.NextUniformL() % 5;
                    GenerateTestPool(pool, bin, numCatFeatures, random.NextUniformL(), numSamples);

                    SavePoolToFile(pool, "test-pool.txt");
                    SavePoolCDToFile("test-pool.txt.cd", numCatFeatures);

                    TestSplitPropsHelper(bin, oneHotLimit, 2, numStats);
                }
            }
        }
    }

    Y_UNIT_TEST(TestSplitPropsHelperWithoutOneHot1) {
        RunTests(0, 0, 1, 45527);
    }

    Y_UNIT_TEST(TestSplitPropsHelperWithoutOneHot2) {
        RunTests(0, 0, 2, 25527);
    }

    Y_UNIT_TEST(TestSplitPropsHelperWithoutOneHot17) {
        RunTests(0, 0, 17, 5527);
    }

    Y_UNIT_TEST(TestSplitPropsHelperWithOneHot3) {
        RunTests(0, 18, 3, 53521);
    }

    Y_UNIT_TEST(FatSplitPropsTest) {
        RunTests(0, 18, 7, 2000000);
    }

    Y_UNIT_TEST(TestSplitPropsHelperWithOneHot13) {
        RunTests(0, 17, 13, 700);
    }
    //
}
