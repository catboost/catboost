#include <catboost/cuda/ut_helpers/test_utils.h>

#include <catboost/cuda/gpu_data/compressed_index_builder.h>
#include <catboost/cuda/ctrs/ut/calc_ctr_cpu.h>
#include <catboost/cuda/data/permutation.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>

#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/quantization/grid_creator.h>
#include <catboost/private/libs/quantization/utils.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(BinBuilderTest) {
    struct TTreeCtrSplit {
        TVector<ui32> Bins;
        ui32 UniqueCount = 0;
    };

    struct TCpuTreeCtrHelper {
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider& DataProvider;

        TVector<ui8> BinarizedTarget;
        TVector<ui32> Indices;
        ui32 NumClasses;

        TCpuTreeCtrHelper(const TBinarizedFeaturesManager& featuresManager,
                          const NCB::TTrainingDataProvider& dataProvider,
                          const TDataPermutation& permutation)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
        {
            BinarizedTarget = NCB::BinarizeLine<ui8>(*dataProvider.TargetData->GetOneDimensionalTarget(), ENanMode::Forbidden, featuresManager.GetTargetBorders());
            NumClasses = 0;
            {
                std::array<bool, 255> seen;
                for (ui32 i = 0; i < 255; ++i) {
                    seen[i] = false;
                }
                for (auto val : BinarizedTarget) {
                    seen[val] = true;
                }
                for (ui32 i = 0; i < 255; ++i) {
                    NumClasses += seen[i];
                }
            }
            permutation.FillOrder(Indices);
        }

        inline TTreeCtrSplit BuildTreeCtrSplitCpu(const TFeatureTensor& featureTensor) {
            TTreeCtrSplit ctrSplit;
            TMap<ui64, ui32> uniqueBins;

            const size_t sampleCount = DataProvider.GetObjectCount();
            TVector<ui64> keys(sampleCount, 0);
            ui32 shift = 0;

            for (auto split : featureTensor.GetSplits()) {
                auto updateKeys = [&](const auto& splitBins) {
                    for (ui32 i = 0; i < sampleCount; ++i) {
                        keys[i] |= split.SplitType == EBinSplitType::TakeBin
                                       ? static_cast<ui64>(split.BinIdx == splitBins[i]) << shift
                                       : static_cast<ui64>(splitBins[i] > split.BinIdx) << shift;
                    }
                };

                if (FeaturesManager.IsFloat(split.FeatureId)) {
                    const auto floatFeatureIdx = DataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Float>(split.FeatureId);
                    auto& valuesHolder = **(DataProvider.ObjectsData->GetFloatFeature(*floatFeatureIdx));
                    updateKeys(valuesHolder.ExtractValues<ui16>(&NPar::LocalExecutor()));
                } else if (FeaturesManager.IsCat(split.FeatureId)) {
                    const auto catFeatureIdx = DataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(split.FeatureId);
                    auto& valuesHolder = **(DataProvider.ObjectsData->GetCatFeature(*catFeatureIdx));
                    updateKeys(valuesHolder.ExtractValues<ui32>(&NPar::LocalExecutor()));
                }
                ++shift;
            }

            for (auto featureId : featureTensor.GetCatFeatures()) {
                ui32 binarization = 0;
                CB_ENSURE(FeaturesManager.IsCat(featureId));

                const auto catFeatureIdx = DataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(featureId);
                auto& valuesHolder = **(DataProvider.ObjectsData->GetCatFeature(*catFeatureIdx));
                TVector<ui32> splitBins = valuesHolder.ExtractValues<ui32>(&NPar::LocalExecutor());
                binarization = DataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(catFeatureIdx).OnAll;
                const ui32 bits = NCB::IntLog2(binarization);
                for (ui32 i = 0; i < sampleCount; ++i) {
                    keys[i] |= static_cast<ui64>(splitBins[i]) << shift;
                }
                shift += bits;
                CB_ENSURE(shift < 64);
            }
            ctrSplit.Bins.resize(keys.size());

            for (ui32 i = 0; i < keys.size(); ++i) {
                if (uniqueBins.count(keys[i]) == 0) {
                    uniqueBins[keys[i]] = static_cast<unsigned int>(uniqueBins.size());
                }
                ctrSplit.Bins[i] = uniqueBins[keys[i]];
            }
            ctrSplit.UniqueCount = uniqueBins.size();
            return ctrSplit;
        }

        void Split(const TBinarySplit& split,
                   ui32 depth,
                   TVector<ui32>& bins) {
            if (FeaturesManager.IsFloat(split.FeatureId)) {
                const auto floatFeatureIdx = DataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Float>(split.FeatureId);
                auto& valuesHolder = **(DataProvider.ObjectsData->GetFloatFeature(*floatFeatureIdx));
                auto featureBins = valuesHolder.ExtractValues<ui16>(&NPar::LocalExecutor());
                for (ui32 i = 0; i < bins.size(); ++i) {
                    bins[i] |= (featureBins[i] > split.BinIdx) << depth;
                }
            } else if (FeaturesManager.IsCat(split.FeatureId)) {
                const auto catFeatureIdx = DataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(split.FeatureId);
                auto& valuesHolder = **(DataProvider.ObjectsData->GetCatFeature(*catFeatureIdx));
                auto featureBins = valuesHolder.ExtractValues<ui32>(&NPar::LocalExecutor());
                for (ui32 i = 0; i < bins.size(); ++i) {
                    bins[i] |= (featureBins[i] == split.BinIdx) << depth;
                }
            } else {
                const auto& ctr = FeaturesManager.GetCtr(split.FeatureId);
                auto treeSplit = BuildTreeCtrSplitCpu(ctr.FeatureTensor);

                const auto& borders = FeaturesManager.GetBorders(split.FeatureId);

                TCpuTargetClassCtrCalcer calcer(treeSplit.UniqueCount,
                                                treeSplit.Bins,
                                                GetWeights(*DataProvider.TargetData),
                                                ctr.Configuration.Prior[0], ctr.Configuration.Prior[1]);

                TVector<ui32> featureBins;
                if (ctr.Configuration.Type == ECtrType::FeatureFreq) {
                    auto freqCtr = calcer.ComputeFreqCtr();
                    featureBins = NCB::BinarizeLine<ui32>(freqCtr, ENanMode::Forbidden, borders);
                } else if (ctr.Configuration.Type == ECtrType::Buckets) {
                    auto floatCtr = calcer.Calc(Indices, TConstArrayRef<ui32>(Indices), BinarizedTarget, NumClasses);
                    TVector<float> values;
                    for (ui32 i = 0; i < treeSplit.Bins.size(); ++i) {
                        values.push_back(floatCtr[i][ctr.Configuration.ParamId]);
                    }
                    featureBins = NCB::BinarizeLine<ui32>(values,
                                                          ENanMode::Forbidden,
                                                          borders);
                } else {
                    ythrow yexception() << "Test for ctr type " << ctr.Configuration.Type
                                        << " isn't supported currently " << Endl;
                }

                for (ui32 i = 0; i < bins.size(); ++i) {
                    bins[i] |= (featureBins[i] > split.BinIdx) << depth;
                }
            }
        }
    };

    void CheckBins(const TFeatureParallelDataSet& dataSet,
                   const TBinarizedFeaturesManager& featuresManager,
                   const TBinarySplit& split, ui32 depth,
                   const TCudaBuffer<ui32, NCudaLib::TMirrorMapping>& bins,
                   TVector<ui32>& currentBins) {
        auto& dataProvider = dataSet.GetDataProvider();
        auto& permutation = dataSet.GetCtrsEstimationPermutation();

        TCpuTreeCtrHelper helper(featuresManager, dataProvider, permutation);
        helper.Split(split, depth, currentBins);
        TVector<ui32> gpuBins;
        bins.Read(gpuBins);
        for (ui32 i = 0; i < currentBins.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(currentBins[i], gpuBins[i]);
        }
    }

    void TestTreeBuilder(ui32 binarization,
                         ui32 permutationCount,
                         ui32 seed = 0) {
        TRandom random(seed);
        TBinarizedPool pool;

        const ui32 numCatFeatures = 7;
        GenerateTestPool(pool, binarization, numCatFeatures);

        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd", numCatFeatures);

        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum, binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = 6;
        {
            TVector<TVector<float>> prior = {{0.5, 1.0}};
            NCatboostOptions::TCtrDescription bucketsCtr(ECtrType::Buckets, prior);
            NCatboostOptions::TCtrDescription freqCtr(ECtrType::FeatureFreq, prior);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }

        NCB::TOnCpuGridBuilderFactory gridBuilderFactory;

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> featuresManager;
        NCB::TFeatureEstimators estimators;
        LoadTrainingData(NCB::TPathWithScheme("dsv://test-pool.txt"),
                         NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
                         NCatboostOptions::TBinarizationOptions(),
                         NCatboostOptions::TCatFeatureParams(ETaskType::GPU),
                         estimators,
                         &dataProvider,
                         &featuresManager);

        TFeatureParallelDataSetHoldersBuilder dataSetsHolderBuilder(*featuresManager,
                                                                    *dataProvider,
                                                                    estimators
                                                                    );

        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount, &NPar::LocalExecutor());

        TVector<ui32> oneHotIds;
        TVector<ui32> catIds;
        TVector<ui32> binaryIds;
        TVector<ui32> floatIds;
        for (ui32 id : featuresManager->GetCatFeatureIds()) {
            if (featuresManager->UseForOneHotEncoding(id)) {
                oneHotIds.push_back(id);
            } else if (featuresManager->UseForCtr(id)) {
                catIds.push_back(id);
            }
        }
        for (ui32 id : featuresManager->GetFloatFeatureIds()) {
            if (featuresManager->GetBinCount(id) == 2) {
                binaryIds.push_back(id);
            } else {
                floatIds.push_back(id);
            }
        }

        TVector<ui32> simpleCtrIds = featuresManager->GetAllSimpleCtrs();

        TVector<TBinarySplit> splits;
        TBinarySplit firstSplit = TBinarySplit(floatIds[random.NextUniformL() % floatIds.size()], 2, EBinSplitType::TakeGreater);
        TBinarySplit secondSplit = TBinarySplit(binaryIds[random.NextUniformL() % binaryIds.size()], 0, EBinSplitType::TakeGreater);
        splits.push_back(firstSplit);
        splits.push_back(secondSplit);

        TCtr treeCtr;
        for (auto split : splits) {
            treeCtr.FeatureTensor.AddBinarySplit(split);
        }
        treeCtr.FeatureTensor.AddCatFeature(catIds[random.NextUniformL() % catIds.size()]);
        treeCtr.Configuration.Type = random.NextUniformL() % 2 == 0 ? ECtrType::Buckets : ECtrType::FeatureFreq;
        {
            if (treeCtr.Configuration.Type == ECtrType::Buckets) {
                treeCtr.Configuration.Prior.resize(featuresManager->GetTargetBorders().size() + 1, 0.5f);
            } else {
                treeCtr.Configuration.Prior = {0.5f, 1};
            }
        }
        {
            TVector<float> naiveBorders;
            for (ui32 i = 0; i < 8; ++i) {
                naiveBorders.push_back(i * 1.0 / 8);
            }
            featuresManager->AddCtr(treeCtr, std::move(naiveBorders));
        }

        TBinarySplit thirdSplit = TBinarySplit(featuresManager->GetId(treeCtr), 4, EBinSplitType::TakeGreater);
        splits.push_back(thirdSplit);

        ui32 simpleCtr = simpleCtrIds[random.NextUniformL() % simpleCtrIds.size()];
        TBinarySplit forthSplit = TBinarySplit(simpleCtr, 4, EBinSplitType::TakeGreater);
        splits.push_back(forthSplit);

        TBinarySplit fifthSplit = TBinarySplit(oneHotIds[random.NextUniformL() % oneHotIds.size()], 2, EBinSplitType::TakeBin);
        splits.push_back(fifthSplit);

        TScopedCacheHolder cacheHolder;

        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        {
            for (ui32 permutation = 0; permutation < permutationCount; ++permutation) {
                const TFeatureParallelDataSet& ds = dataSet.GetDataSetForPermutation(0);
                TMirrorBuffer<ui32> bins = TMirrorBuffer<ui32>::CopyMapping(ds.GetIndices());

                {
                    TTreeUpdater treeBuilder(cacheHolder, *featuresManager,
                                             dataSet.GetCtrTargets(), ds,
                                             bins);
                    TVector<ui32> currentBinsCpu;
                    currentBinsCpu.resize(ds.GetIndices().GetObjectsSlice().Size(), 0);

                    for (ui32 i = 0; i < splits.size(); ++i) {
                        {
                            NCudaLib::GetCudaManager().WaitComplete();
                            auto guard = profiler.Profile(TStringBuilder() << "build with permutation-independent cache for depth " << i);
                            treeBuilder.AddSplit(splits[i]);
                            NCudaLib::GetCudaManager().WaitComplete();
                        }

                        CheckBins(ds, *featuresManager,
                                  splits[i], i,
                                  bins, currentBinsCpu);
                    }
                }
            }
        }

        {
            for (ui32 permutation = 0; permutation < permutationCount; ++permutation) {
                TScopedCacheHolder cacheHolder;
                const TFeatureParallelDataSet& ds = dataSet.GetDataSetForPermutation(0);
                TMirrorBuffer<ui32> bins = TMirrorBuffer<ui32>::CopyMapping(ds.GetIndices());
                {
                    TTreeUpdater treeBuilder(cacheHolder,
                                             *featuresManager,
                                             dataSet.GetCtrTargets(),
                                             ds,
                                             bins);

                    TVector<ui32> currentBinsCpu;
                    currentBinsCpu.resize(ds.GetIndices().GetObjectsSlice().Size(), 0);

                    for (ui32 i = 0; i < splits.size(); ++i) {
                        {
                            NCudaLib::GetCudaManager().WaitComplete();
                            auto guard = profiler.Profile(TStringBuilder() << "build without permutation-independent cache for depth " << i);
                            treeBuilder.AddSplit(splits[i]);
                            NCudaLib::GetCudaManager().WaitComplete();
                        }
                        CheckBins(ds, *featuresManager,
                                  splits[i], i,
                                  bins, currentBinsCpu);
                    }
                }
            }
        }
    }

    void RunCompressedSplitFloatTest() {
        TRandom random(0);
        ui32 size = 115322;
        auto vec = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(size));
        TVector<float> ref;
        float border = 0.3;
        TVector<ui32> refBits;
        for (ui32 i = 0; i < size; ++i) {
            ref.push_back(random.NextUniform());
            refBits.push_back(ref.back() > border);
        }
        vec.Write(ref);
        auto compressedBits = TMirrorBuffer<ui64>::Create(NCudaLib::TMirrorMapping(CompressedSize<ui64>(size, 2)));
        auto decompressedBits = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(size));
        CreateCompressedSplitFloat(vec, border, compressedBits);
        Decompress(compressedBits, decompressedBits, 2);
        TVector<ui32> bins;
        decompressedBits.Read(bins);
        for (ui32 i = 0; i < size; ++i) {
            UNIT_ASSERT_EQUAL(bins[i], refBits[i]);
        }
    }

    Y_UNIT_TEST(TreeBuilderTest4) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestTreeBuilder(32, 4);
        }
    }

    Y_UNIT_TEST(TestCompressedSplitFloat) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            RunCompressedSplitFloatTest();
        }
    }

    Y_UNIT_TEST(TreeBuilderTest32) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestTreeBuilder(32, 32);
        }
    }
}
