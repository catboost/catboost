#include <util/random/shuffle.h>
#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/cuda/gpu_data/compressed_index_builder.h>
#include <catboost/cuda/ctrs/ut/calc_ctr_cpu.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/feature_layout_feature_parallel.h>
#include <catboost/cuda/gpu_data/feature_layout_doc_parallel.h>
#include <catboost/cuda/gpu_data/dataset_helpers.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/cuda/data/permutation.h>
#include <catboost/cuda/data/permutation.h>

#include <catboost/private/libs/quantization/grid_creator.h>
#include <catboost/private/libs/quantization/utils.h>

#include <library/cpp/containers/2d_array/2d_array.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace std;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(BinarizationsTests) {
    template <class TCompressedDataSet>
    void CheckDataSet(const TCompressedDataSet& dataSet,
                      const TBinarizedFeaturesManager& featuresManager,
                      const TDataPermutation& ctrsPermutation,
                      const NCB::TTrainingDataProvider& dataProvider,
                      const TDataPermutation* onGpuPermutation = nullptr) {
        auto binarizedTarget = NCB::BinarizeLine<ui8>(*dataProvider.TargetData->GetOneDimensionalTarget(),
                                                      ENanMode::Forbidden,
                                                      featuresManager.GetTargetBorders());
        ui32 numClasses = 0;
        {
            std::array<bool, 255> seen;
            for (ui32 i = 0; i < 255; ++i) {
                seen[i] = false;
            }
            for (auto val : binarizedTarget) {
                seen[val] = true;
            }
            for (ui32 i = 0; i < 255; ++i) {
                numClasses += seen[i];
            }
        }
        TVector<ui32> ctrToDirectIndices;
        TVector<ui32> ctrsEstimationPermutation;
        ctrsPermutation.FillInversePermutation(ctrToDirectIndices);
        ctrsPermutation.FillOrder(ctrsEstimationPermutation);

        TVector<ui32> gatherBinIndices;
        if (onGpuPermutation != nullptr) {
            onGpuPermutation->FillOrder(gatherBinIndices);
        } else {
            gatherBinIndices.resize(ctrToDirectIndices.size());
            std::iota(gatherBinIndices.begin(), gatherBinIndices.end(), 0);
        }

        TMap<ui32, TArray2D<float>> ctrsCache;

        auto features = dataSet.GetFeatures();

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            //            TSlice featuresSlice = featuresMapping.DeviceSlice(dev);

            TVector<ui32> compressedIndex;
            dataSet.GetCompressedIndex().DeviceView(dev).Read(compressedIndex);

            for (ui32 f = 0; f < dataSet.GetFeatureCount(); ++f) {
                auto featureId = features[f];
                auto cudaFeature = dataSet.GetTCFeature(featureId);
                if (cudaFeature.IsEmpty(dev)) {
                    continue;
                }
                auto feature = cudaFeature.At(dev);
                //                const ui32 featureId = dataSet.GetFeatureId(f);
                TSlice docsSlice = dataSet.GetSamplesMapping().DeviceSlice(dev);

                TVector<ui32> bins;

                ui32 binarization = 0;

                if (featuresManager.IsFloat(featureId)) {
                    const auto floatFeatureIdx = dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Float>(featureId);
                    auto& valuesHolder = **(dataProvider.ObjectsData->GetFloatFeature(*floatFeatureIdx));
                    auto binsArray = valuesHolder.template ExtractValues<ui16>(&NPar::LocalExecutor());
                    bins.assign(binsArray.begin(), binsArray.end());
                    binarization = dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetBinCount(floatFeatureIdx);
                } else if (featuresManager.IsCat(featureId)) {
                    const auto catFeatureIdx = dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(featureId);
                    auto& valuesHolder = **(dataProvider.ObjectsData->GetCatFeature(*catFeatureIdx));
                    auto binsArray = valuesHolder.template ExtractValues<ui32>(&NPar::LocalExecutor());
                    bins.assign(binsArray.begin(), binsArray.end());
                    binarization = dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(catFeatureIdx).OnAll;
                } else {
                    CB_ENSURE(featuresManager.IsCtr(featureId));
                    const auto& ctr = featuresManager.GetCtr(featureId);
                    CB_ENSURE(ctr.IsSimple());

                    const ui32 catFeatureFlatIdx = ctr.FeatureTensor.GetCatFeatures()[0];
                    const auto catFeatureIdx = dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(catFeatureFlatIdx);
                    auto& valuesHolder = **(dataProvider.ObjectsData->GetCatFeature(*catFeatureIdx));
                    auto binsArray = valuesHolder.ExtractValues<ui32>(&NPar::LocalExecutor());
                    TVector<ui32> catFeatureBins(binsArray.begin(), binsArray.end());
                    const auto& borders = featuresManager.GetBorders(featureId);
                    binarization = borders.size() + 1;

                    TCpuTargetClassCtrCalcer calcer(dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(catFeatureIdx).OnAll,
                                                    catFeatureBins,
                                                    GetWeights(*dataProvider.TargetData),
                                                    ctr.Configuration.Prior[0],
                                                    ctr.Configuration.Prior[1]);

                    TVector<ui32> ctrOrderedBins;
                    if (ctr.Configuration.Type == ECtrType::FeatureFreq) {
                        auto freqCtr = calcer.ComputeFreqCtr(&ctrsEstimationPermutation);
                        ctrOrderedBins = NCB::BinarizeLine<ui32>(freqCtr, ENanMode::Forbidden, borders);
                    } else if (ctr.Configuration.Type == ECtrType::Buckets) {
                        if (!ctrsCache.contains(featureId)) {
                            ctrsCache[featureId] = calcer.Calc(ctrsEstimationPermutation,
                                                               TConstArrayRef<ui32>(ctrsEstimationPermutation),
                                                               binarizedTarget,
                                                               numClasses);
                        }
                        TVector<float> values;
                        for (size_t i = 0; i < catFeatureBins.size(); ++i) {
                            values.push_back(ctrsCache[featureId][i][ctr.Configuration.ParamId]);
                        }
                        ctrOrderedBins = NCB::BinarizeLine<ui32>(MakeArrayRef(values.data(), binarizedTarget.size()),
                                                                 ENanMode::Forbidden,
                                                                 borders);

                    } else {
                        ythrow yexception() << "Test for ctr type " << ctr.Configuration.Type << " isn't supported currently " << Endl;
                    }

                    bins.resize(ctrOrderedBins.size());
                    for (ui32 i = 0; i < ctrOrderedBins.size(); ++i) {
                        bins[i] = ctrOrderedBins[ctrToDirectIndices[i]];
                        //                       ctrOrderedBins[i] = bins[ctrsEstimationPermutation[i]]
                    }
                }

                //                TCFeature feature = dataSet.GetFeatureByGlobalId(featureId);
                UNIT_ASSERT_C((binarization - 1) <= feature.Folds, "Feature #" << featureId << " " << binarization << " / " << feature.Folds);
                UNIT_ASSERT_C(feature.Mask != 0, "Feature mask should not be zero");

                for (ui32 i = 0; i < docsSlice.Size(); ++i) {
                    const ui32 readIdx = gatherBinIndices[i + docsSlice.Left];
                    UNIT_ASSERT_VALUES_EQUAL_C(bins[readIdx],
                                               (compressedIndex[feature.Offset + i] >> feature.Shift) & feature.Mask,
                                               i << " " << docsSlice << " "
                                                 << " offset " << feature.Offset << " shift " << feature.Shift << " mask " << feature.Mask << " fid " << featureId << " (" << f << ")");
                }
            }
        }
    }

    template <class TDataSet>
    void CheckPermutationDataSet(const TDataSet& dataSet,
                                 const TBinarizedFeaturesManager& featuresManager,
                                 const TDataPermutation& ctrsPermutation,
                                 const NCB::TTrainingDataProvider& dataProvider,
                                 const bool onlyPermutationDependent = false,
                                 const TDataPermutation* onGpuPermutation = nullptr) {
        if (dataSet.HasFeatures() && !onlyPermutationDependent) {
            CheckDataSet(dataSet.GetFeatures(),
                         featuresManager,
                         ctrsPermutation,
                         dataProvider,
                         onGpuPermutation);
        }

        if (dataSet.HasPermutationDependentFeatures()) {
            CheckDataSet(dataSet.GetPermutationFeatures(),
                         featuresManager,
                         ctrsPermutation,
                         dataProvider,
                         onGpuPermutation);
        }
    }

    template <class TLayout = TFeatureParallelLayout>
    void TestCompressedIndexBuilder(ui32 binarization,
                                    ui32 permutationId) {
        TBinarizedPool pool;

        GenerateTestPool(pool, binarization);
        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd");

        NCatboostOptions::TBinarizationOptions binarizationOptions(EBorderSelectionType::GreedyLogSum,
                                                                   binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.OneHotMaxSize = Min<ui32>(6, binarization);

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> binarizedFeaturesManager;
        NCB::TFeatureEstimators estimators;

        LoadTrainingData(NCB::TPathWithScheme("dsv://test-pool.txt"),
                         NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
                         binarizationOptions,
                         catFeatureParams,
                         estimators,
                         &dataProvider,
                         &binarizedFeaturesManager);

        UNIT_ASSERT_VALUES_EQUAL(
            pool.NumFeatures + 1,
            dataProvider->MetaInfo.FeaturesLayout->GetExternalFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.NumSamples, dataProvider->GetObjectCount());

        auto docsMapping = TCudaFeaturesLayoutHelper<TLayout>::CreateDocLayout(pool.NumSamples);

        TDataPermutation permutation = GetPermutation(*dataProvider,
                                                      permutationId);
        TVector<ui32> order;
        permutation.FillOrder(order);

        using TCompressedIndex = TSharedCompressedIndex<TLayout>;
        TCompressedIndex compressedIndex;
        //        TFoldBa
        TSet<ui32> referenceFeatures;
        {
            TSharedCompressedIndexBuilder<TLayout> builder(compressedIndex, &NPar::LocalExecutor());
            TBinarizationInfoProvider binarizationInfoProvider(*binarizedFeaturesManager,
                                                               dataProvider.Get());
            TDataSetDescription desc;
            desc.Name = "UnitTest";

            TVector<ui32> features;
            for (ui32 feature : binarizedFeaturesManager->GetFloatFeatureIds()) {
                features.push_back(feature);
            }
            for (ui32 catFeature : binarizedFeaturesManager->GetCatFeatureIds()) {
                if (binarizedFeaturesManager->GetBinCount(catFeature) <= catFeatureParams.OneHotMaxSize) {
                    features.push_back(catFeature);
                }
            }
            Shuffle(features.begin(), features.end());
            referenceFeatures = TSet<ui32>(features.begin(), features.end());
            ui32 id = builder.AddDataSet(
                binarizationInfoProvider,
                desc,
                docsMapping,
                features,
                TDatasetPermutationOrderAndSubsetIndexing::ConstructShared(
                    dataProvider->ObjectsData->GetFeaturesArraySubsetIndexing(),
                    std::move(order)
                )
            );
            builder.PrepareToWrite();
            UNIT_ASSERT_EQUAL(id, 0u);

            TFloatAndOneHotFeaturesWriter<TLayout> writer(*binarizedFeaturesManager,
                                                          builder,
                                                          *dataProvider,
                                                          id,
                                                          /*skipExclusiveFeatureBundles*/false);
            writer.Write(features);
            builder.Finish();
        }
        UNIT_ASSERT_EQUAL(compressedIndex.DataSetCount(), 1u);
        auto& ds = compressedIndex.GetDataSet(0);
        UNIT_ASSERT_EQUAL(ds.GetFeatureCount(), referenceFeatures.size());

        TSet<ui32> dataSetFeatures = TSet<ui32>(ds.GetFeatures().begin(),
                                                ds.GetFeatures().end());
        UNIT_ASSERT_EQUAL(dataSetFeatures, referenceFeatures);
        CheckDataSet(ds,
                     *binarizedFeaturesManager,
                     permutation,
                     *dataProvider,
                     &permutation);
    }

    template <class TMapping>
    void CheckCtrTargets(const TCtrTargets<TMapping>& targets,
                         const TVector<ui32>& binarizedTargetRef,
                         const NCB::TTrainingDataProvider& dataProvider) {
        auto dataProviderTargets = *dataProvider.TargetData->GetOneDimensionalTarget();
        auto dataProviderWeights = GetWeights(*dataProvider.TargetData);

        TVector<float> targetsCpu;
        targets.WeightedTarget.Read(targetsCpu);
        for (ui32 i = 0; i < dataProviderTargets.size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL(
                dataProviderTargets[i] * (dataProviderWeights.empty() ? 1.0f : dataProviderWeights[i]),
                targetsCpu[i],
                1e-9);
        }

        TVector<ui8> binTargetsCpu;
        targets.BinarizedTarget.Read(binTargetsCpu);
        for (ui32 i = 0; i < dataProviderTargets.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(binTargetsCpu[i], binarizedTargetRef[i]);
        }

        TVector<float> weightsCpu;
        targets.Weights.Read(weightsCpu);
        for (ui32 i = 0; i < dataProviderTargets.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(weightsCpu[i], dataProviderWeights.empty() ? 1.0f : dataProviderWeights[i]);
        }
    }

    void CheckIndices(const NCB::TTrainingDataProvider& dataProvider,
                      const TFeatureParallelDataSetsHolder& dataSet) {
        for (ui32 i = 0; i < dataSet.PermutationsCount(); ++i) {
            auto permutation = GetPermutation(dataProvider, i);
            TVector<ui32> order;
            permutation.FillOrder(order);
            TVector<ui32> inverseOrder;
            permutation.FillInversePermutation(inverseOrder);

            auto& ds = dataSet.GetDataSetForPermutation(i);
            {
                TVector<ui32> tmp;
                ds.GetIndices().Read(tmp);
                for (ui32 i = 0; i < order.size(); ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(order[i], tmp[i]);
                }
                ds.GetInverseIndices().Read(tmp);
                for (ui32 i = 0; i < order.size(); ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(inverseOrder[i], tmp[i]);
                }
            }
        }
    }

    void TestDatasetHolderBuilder(ui32 binarization,
                                  ui32 permutationCount,
                                  ui32 bucketsCtrBinarization = 32,
                                  ui32 freqCtrBinarization = 15) {
        TBinarizedPool pool;

        GenerateTestPool(pool, binarization);

        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd");

        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum, binarization);
        NCatboostOptions::TBinarizationOptions bucketsBinarization(EBorderSelectionType::GreedyLogSum, bucketsCtrBinarization);
        NCatboostOptions::TBinarizationOptions freqBinarization(EBorderSelectionType::GreedyLogSum, freqCtrBinarization);

        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = 0;
        {
            TVector<TVector<float>> prior = {{0.5, 1.0}};
            NCatboostOptions::TCtrDescription bucketsCtr(ECtrType::Buckets, prior, bucketsBinarization);
            NCatboostOptions::TCtrDescription freqCtr(ECtrType::FeatureFreq, prior, freqBinarization);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }

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

        const auto dataProviderTarget = *dataProvider->TargetData->GetOneDimensionalTarget();

        {
            featuresManager->SetTargetBorders(NCB::TBordersBuilder(gridBuilderFactory,
                                                                   dataProviderTarget)(floatBinarization));

            const auto& targetBorders = featuresManager->GetTargetBorders();
            UNIT_ASSERT_VALUES_EQUAL(targetBorders.size(), 4);
        }

        UNIT_ASSERT_VALUES_EQUAL(pool.NumFeatures + 1,
                                 dataProvider->MetaInfo.FeaturesLayout->GetExternalFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.NumSamples, dataProvider->GetObjectCount());

        TFeatureParallelDataSetHoldersBuilder dataSetsHolderBuilder(*featuresManager,
                                                                    *dataProvider,
                                                                    estimators
                                                                    );
        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount, &NPar::LocalExecutor());

        {
            auto binarizedTargetRef = NCB::BinarizeLine<ui32>(dataProviderTarget,
                                                              ENanMode::Forbidden,
                                                              featuresManager->GetTargetBorders());
            CheckCtrTargets(dataSet.GetCtrTargets(),
                            binarizedTargetRef,
                            *dataProvider);
        }
        {
            CheckIndices(*dataProvider, dataSet);
        }

        for (ui32 permutation = 0; permutation < dataSet.PermutationsCount(); ++permutation) {
            //
            const auto& dataSetForPermutation = dataSet.GetDataSetForPermutation(permutation);
            if (permutationCount > 1) {
                CB_ENSURE(dataSetForPermutation.HasPermutationDependentFeatures());
            }
            CB_ENSURE(dataSetForPermutation.HasFeatures());

            CheckPermutationDataSet(dataSetForPermutation,
                                    *featuresManager,
                                    dataSet.GetPermutation(permutation),
                                    *dataProvider,
                                    permutation > 0,
                                    &dataSet.GetPermutation(permutation));
        }
    }

    //
    void TestDocParallelDataSetBuilder(ui32 binarization,
                                       ui32 permutationCount,
                                       ui32 bucketsCtrBinarization = 32,
                                       ui32 freqCtrBinarization = 15) {
        TBinarizedPool pool;

        GenerateTestPool(pool, binarization);

        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd");

        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum, binarization);
        NCatboostOptions::TBinarizationOptions bucketsBinarization(EBorderSelectionType::GreedyLogSum, bucketsCtrBinarization);
        NCatboostOptions::TBinarizationOptions freqBinarization(EBorderSelectionType::GreedyLogSum, freqCtrBinarization);

        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = 0;
        {
            TVector<TVector<float>> prior = {{0.5, 1.0}};
            NCatboostOptions::TCtrDescription bucketsCtr(ECtrType::Buckets, prior, bucketsBinarization);
            NCatboostOptions::TCtrDescription freqCtr(ECtrType::FeatureFreq, prior, freqBinarization);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> featuresManager;

        NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
        NCB::TFeatureEstimators estimators;


        LoadTrainingData(NCB::TPathWithScheme("dsv://test-pool.txt"),
                         NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
                         floatBinarization,
                         catFeatureParams,
                         estimators,
                         &dataProvider,
                         &featuresManager);

        {
            featuresManager->SetTargetBorders(NCB::TBordersBuilder(gridBuilderFactory,
                                                                   *dataProvider->TargetData->GetOneDimensionalTarget())(floatBinarization));

            const auto& targetBorders = featuresManager->GetTargetBorders();
            UNIT_ASSERT_VALUES_EQUAL(targetBorders.size(), 4);
        }

        UNIT_ASSERT_VALUES_EQUAL(
            pool.NumFeatures + 1,
            dataProvider->MetaInfo.FeaturesLayout->GetExternalFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.NumSamples, dataProvider->GetObjectCount());

        TDocParallelDataSetBuilder dataSetsHolderBuilder(*featuresManager,
                                                         *dataProvider,
                                                         estimators);

        TDocParallelDataSetsHolder dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount, &NPar::LocalExecutor());
        const TDataPermutation& loadBalancingPermutation = dataSet.GetLoadBalancingPermutation();

        for (ui32 permutation = 0; permutation < dataSet.PermutationsCount(); ++permutation) {
            const auto& dataSetForPermutation = dataSet.GetDataSetForPermutation(permutation);

            if (permutationCount > 1) {
                CB_ENSURE(dataSetForPermutation.HasPermutationDependentFeatures());
            }
            CB_ENSURE(dataSetForPermutation.HasFeatures());

            CheckPermutationDataSet(dataSetForPermutation,
                                    *featuresManager,
                                    dataSetForPermutation.GetCtrsEstimationPermutation(),
                                    *dataProvider,
                                    permutation > 0,
                                    &loadBalancingPermutation);
        }
    }

    //
    Y_UNIT_TEST(TestCreateCompressedIndexBinary) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder(2, 0);
        }
    }
    //
    Y_UNIT_TEST(TestCreateCompressedHalfByte) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder(15, 0);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndex) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder(32, 0);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndexWithPermutation) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder(32, 1);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndexDocParallel) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder<TDocParallelLayout>(32, 1);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndexHalfByteWithPermutation) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder(15, 1);
        }
    }

    Y_UNIT_TEST(TestCreateDataSetsHolder) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 1);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndex32_4) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndex32_4_1_8) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4, 1, 8);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndex32_4_1_64) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4, 1, 64);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndex32_4_15_64) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4, 15, 64);
        }
    }

    Y_UNIT_TEST(TestCreateCompressedIndex128) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestCompressedIndexBuilder(128, 0);
        }
    }

    Y_UNIT_TEST(TestCreateDocParallelDataSetBuilderOnePermutation) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDocParallelDataSetBuilder(32, 1, 15, 64);
        }
    }

    Y_UNIT_TEST(TestCreateDocParallelDataSet32_4_32_64) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDocParallelDataSetBuilder(32, 4, 32, 64);
        }
    }

    Y_UNIT_TEST(TestCreateDocParallelDataSet15_4_1_64) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDocParallelDataSetBuilder(15, 4, 1, 64);
        }
    }

    Y_UNIT_TEST(TestCreateDocParallelDataSet15_4_32_2) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            TestDocParallelDataSetBuilder(15, 4, 32, 2);
        }
    }
}
