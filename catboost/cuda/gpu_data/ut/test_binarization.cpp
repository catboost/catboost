#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/gpu_data/binarized_dataset_builder.h>
#include <catboost/cuda/ctrs/ut/calc_ctr_cpu.h>
#include <catboost/cuda/data/permutation.h>
#include <library/unittest/registar.h>
#include <catboost/cuda/gpu_data/fold_based_dataset_builder.h>

using namespace std;
using namespace NCatboostCuda;

SIMPLE_UNIT_TEST_SUITE(BinarizationsTests) {

    template <class TGridPolicy>
    void CheckDataSet(const TGpuBinarizedDataSet<TGridPolicy>& dataSet,
                      const TBinarizedFeaturesManager& featuresManager,
                      const TDataPermutation& permutation,
                      const TDataProvider& dataProvider) {
        const auto& featuresMapping = dataSet.GetGrid().GetMapping();

        auto binarizedTarget = BinarizeLine<ui8>(~dataProvider.GetTargets(), +dataProvider.GetTargets(), featuresManager.GetTargetBorders());
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
        TVector<ui32> indices;
        permutation.FillOrder(indices);

        TMap<ui32, TArray2D<float>> ctrsCache;

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            TSlice featuresSlice = featuresMapping.DeviceSlice(dev);

            TVector<ui32> compressedIndex;
            dataSet.GetCompressedIndex().DeviceView(dev).Read(compressedIndex);


            for (ui32 f = featuresSlice.Left; f < featuresSlice.Right; ++f) {
                const ui32 featureId = dataSet.GetFeatureId(f);

                TVector<ui32> bins;
                ui32 binarization = 0;
                if (featuresManager.IsFloat(featureId)) {
                    auto& valuesHolder = dynamic_cast<const TBinarizedFloatValuesHolder&>(dataProvider.GetFeatureById(featuresManager.GetDataProviderId(featureId)));
                    bins = valuesHolder.ExtractValues();
                    binarization = valuesHolder.Discretization() + 1;
                } else if (featuresManager.IsCat(featureId)) {
                    auto& valuesHolder = dynamic_cast<const TCatFeatureValuesHolder&>(dataProvider.GetFeatureById(featuresManager.GetDataProviderId(featureId)));
                    bins = valuesHolder.ExtractValues();
                    binarization = valuesHolder.GetUniqueValues();
                } else {
                    CB_ENSURE(featuresManager.IsCtr(featureId));
                    const auto& ctr = featuresManager.GetCtr(featureId);
                    CB_ENSURE(ctr.IsSimple());

                    const ui32 catFeatureId = ctr.FeatureTensor.GetCatFeatures()[0];
                    auto& valuesHolder = dynamic_cast<const TCatFeatureValuesHolder&>(dataProvider.GetFeatureById(featuresManager.GetDataProviderId(catFeatureId)));
                    auto catFeatureBins = valuesHolder.ExtractValues();
                    const auto& borders = featuresManager.GetBorders(featureId);
                    binarization = borders.size() + 1;

                    TCpuTargetClassCtrCalcer calcer(valuesHolder.GetUniqueValues(),
                                                    catFeatureBins,
                                                    dataProvider.GetWeights(),
                                                    ctr.Configuration.Prior[0], ctr.Configuration.Prior[1]);

                    if (ctr.Configuration.Type == ECtrType::FeatureFreq) {
                        auto freqCtr = calcer.ComputeFreqCtr();
                        bins = BinarizeLine<ui32>(~freqCtr, +freqCtr, borders);
                    } else if (ctr.Configuration.Type == ECtrType::Buckets) {
                        if (!ctrsCache.has(catFeatureId)) {
                            ctrsCache[catFeatureId] = calcer.Calc(indices,
                                                                  binarizedTarget,
                                                                  numClasses);
                        }
                        TVector<float> values;
                        for (ui32 i = 0; i < catFeatureBins.size(); ++i) {
                            values.push_back(ctrsCache[catFeatureId][i][ctr.Configuration.ParamId]);
                        }
                        bins = BinarizeLine<ui32>(~values,
                                                  binarizedTarget.size(),
                                                  borders);

                    } else {
                        ythrow yexception() << "Test for ctr type " << ctr.Configuration.Type << " isn't supported currently " << Endl;
                    }
                }

                TCFeature feature = dataSet.GetFeatureByGlobalId(featureId);
                UNIT_ASSERT_VALUES_EQUAL(binarization - 1, feature.Folds);
                UNIT_ASSERT_VALUES_EQUAL(feature.Index, f);

                for (ui32 i = 0; i < bins.size(); ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(bins[i], (compressedIndex[feature.Offset * bins.size() + i] >> feature.Shift) & feature.Mask);
                }

            }
        }
    }

    void CheckDataSets(const TGpuFeatures<>& gpuFeatures,
                       const TBinarizedFeaturesManager& featuresManager,
                       const TDataPermutation& permutation,
                       const TDataProvider& dataProvider) {

        CheckDataSet(gpuFeatures.GetFeatures(),
                     featuresManager,
                     permutation,
                     dataProvider);

        CheckDataSet(gpuFeatures.GetHalfByteFeatures(),
                     featuresManager,
                     permutation,
                     dataProvider);

        CheckDataSet(gpuFeatures.GetBinaryFeatures(),
                     featuresManager,
                     permutation,
                     dataProvider);
    }

    template <class TGridPolicy = TByteFeatureGridPolicy>
    void TestGpuDatasetBuilder(ui32 binarization,
                               ui32 permutationId) {
        TBinarizedPool pool;

        GenerateTestPool(pool, binarization);
        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd");

        NCatboostOptions::TBinarizationOptions binarizationOptions(EBorderSelectionType::GreedyLogSum, binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.OneHotMaxSize = 6;
        TBinarizedFeaturesManager binarizedFeaturesManager(catFeatureParams, binarizationOptions);

        TDataProvider dataProvider;
        TDataProviderBuilder dataProviderBuilder(binarizedFeaturesManager, dataProvider);

        ReadPool("test-pool.txt.cd",
                 "test-pool.txt",
                 "",
                 16,
                 true,
                 dataProviderBuilder.SetShuffleFlag(false));

        UNIT_ASSERT_VALUES_EQUAL(pool.NumFeatures + 1, dataProvider.GetEffectiveFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.NumSamples, dataProvider.GetSampleCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.Queries.size(), dataProvider.GetQueries().size());

        auto docsMapping = NCudaLib::TMirrorMapping(pool.NumSamples);
        auto featuresMapping = NCudaLib::TStripeMapping::SplitBetweenDevices(pool.NumFeatures);

        TDataPermutation permutation = GetPermutation(dataProvider, permutationId);
        TVector<ui32> order;
        permutation.FillOrder(order);

        using TDataSet = TGpuBinarizedDataSet<TGridPolicy>;
        TDataSet binarizedDataSet;
        {
            TGpuBinarizedDataSetBuilder<TGridPolicy> builder(featuresMapping,
                                                             docsMapping,
                                                             &order);

            TVector<ui32> features(pool.NumFeatures);
            std::iota(features.begin(), features.end(), 1);
            builder.SetFeatureIds(features)
                .UseForOneHotIds(binarizedFeaturesManager.GetOneHotIds(features));
            std::random_shuffle(features.begin(), features.end());

            {
                for (auto f : features) {
                    auto& valuesHolder = dynamic_cast<const TBinarizedFloatValuesHolder&>(dataProvider.GetFeatureById(f));
                    auto bins = valuesHolder.ExtractValues();

                    builder.Write(f,
                                  valuesHolder.GetBorders().size() + 1,
                                  bins);
                }
            }
            binarizedDataSet = builder.Finish();
        }

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            TSlice featuresSlice = featuresMapping.DeviceSlice(dev);

            TVector<ui32> compressedIndex;
            binarizedDataSet.GetCompressedIndex().DeviceView(dev).Read(compressedIndex);

            for (ui32 f = featuresSlice.Left; f < featuresSlice.Right; ++f) {
                const ui32 featureId = f + 1;
                auto& valuesHolder = dynamic_cast<const TBinarizedFloatValuesHolder&>(dataProvider.GetFeatureById(featureId));
                auto bins = valuesHolder.ExtractValues();
                TCFeature feature = binarizedDataSet.GetFeatureByGlobalId(featureId);
                UNIT_ASSERT_VALUES_EQUAL(valuesHolder.Discretization(), feature.Folds);
                UNIT_ASSERT_VALUES_EQUAL(feature.Index, f);

                for (ui32 i = 0; i < bins.size(); ++i) {
                    UNIT_ASSERT_VALUES_EQUAL(bins[order[i]], (compressedIndex[feature.Offset * bins.size() + i] >> feature.Shift) & feature.Mask);
                }
            }
        }
    }

    template <class TMapping>
    void CheckCtrTargets(const TCtrTargets<TMapping>& targets,
                         const TVector<ui32>& binarizedTargetRef,
                         const TDataProvider& dataProvider) {
        TVector<float> targetsCpu;
        targets.WeightedTarget.Read(targetsCpu);
        for (ui32 i = 0; i < dataProvider.GetTargets().size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL(dataProvider.GetTargets()[i] * dataProvider.GetWeights()[i], targetsCpu[i], 1e-9);
        }

        TVector<ui8> binTargetsCpu;
        targets.BinarizedTarget.Read(binTargetsCpu);
        for (ui32 i = 0; i < dataProvider.GetTargets().size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(binTargetsCpu[i], binarizedTargetRef[i]);
        }

        TVector<float> weightsCpu;
        targets.Weights.Read(weightsCpu);
        for (ui32 i = 0; i < dataProvider.GetTargets().size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(weightsCpu[i], dataProvider.GetWeights()[i]);
        }
    }

    void CheckIndices(const TDataProvider& dataProvider,
                      const TDataSetsHolder<>& dataSet) {
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
                                  ui32 bucketsCtrBinarization=32,
                                  ui32 freqCtrBinarization=15) {
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
            NCatboostOptions::TCtrDescription bucketsCtr(ETaskType::GPU, ECtrType::Buckets, prior, bucketsBinarization);
            NCatboostOptions::TCtrDescription freqCtr(ETaskType::GPU, ECtrType::FeatureFreq, prior, freqBinarization);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }
        TBinarizedFeaturesManager featuresManager(catFeatureParams, floatBinarization);


        TDataProvider dataProvider;
        TOnCpuGridBuilderFactory gridBuilderFactory;
        TDataProviderBuilder dataProviderBuilder(featuresManager, dataProvider);

        ReadPool("test-pool.txt.cd",
                 "test-pool.txt",
                 "",
                 16,
                 true,
                 dataProviderBuilder.SetShuffleFlag(false));

        {
            featuresManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                             dataProvider.GetTargets())(floatBinarization));

            const auto& targetBorders = featuresManager.GetTargetBorders();
            UNIT_ASSERT_VALUES_EQUAL(targetBorders.size(), 4);
        }


        UNIT_ASSERT_VALUES_EQUAL(pool.NumFeatures + 1, dataProvider.GetEffectiveFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.NumSamples, dataProvider.GetSampleCount());
        UNIT_ASSERT_VALUES_EQUAL(pool.Queries.size(), dataProvider.GetQueries().size());

        TDataSetHoldersBuilder<> dataSetsHolderBuilder(featuresManager,
                                                       dataProvider);
        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount);

        {
            auto binarizedTargetRef = BinarizeLine<ui32>(~dataProvider.GetTargets(),
                                                         dataProvider.GetTargets().size(),
                                                         featuresManager.GetTargetBorders());
            CheckCtrTargets(dataSet.GetCtrTargets(), binarizedTargetRef, dataProvider);
        }
        {
            CheckIndices(dataProvider, dataSet);
        }

        {
            auto& gpuFeatures = dataSet.GetPermutationIndependentFeatures();
            CheckDataSets(gpuFeatures, featuresManager, dataSet.GetPermutation(0), dataProvider);

        }


        for (ui32 permutation = 0; permutation < dataSet.PermutationsCount(); ++permutation) {
            //
            CheckDataSets(dataSet.GetDataSetForPermutation(permutation).GetPermutationFeatures(),
                         featuresManager,
                         dataSet.GetPermutation(permutation),
                         dataProvider);
        }
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndex) {
        StartCudaManager();
        {
            TestGpuDatasetBuilder(32, 0);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndexWithPermutation) {
        StartCudaManager();
        {
            TestGpuDatasetBuilder(32, 1);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndexHalfByteWithPermutation) {
        StartCudaManager();
        {
            TestGpuDatasetBuilder<THalfByteFeatureGridPolicy>(15, 1);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateDataSetsHolder) {
        StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 1);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndex32_4) {
        StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4);
        }
        StopCudaManager();
    }


    SIMPLE_UNIT_TEST(TestCreateCompressedIndex32_4_1_8) {
        StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4, 1, 8);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndex32_4_1_64) {
        StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4, 1, 64);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndex32_4_15_64) {
        StartCudaManager();
        {
            TestDatasetHolderBuilder(32, 4, 15, 64);
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestCreateCompressedIndex128) {
        StartCudaManager();
        {
            TestGpuDatasetBuilder(128, 0);
        }
        StopCudaManager();
    }
}
