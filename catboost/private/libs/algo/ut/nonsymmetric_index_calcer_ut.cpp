#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/fold.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/private/libs/algo/split.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/unittest/registar.h>

#include <util/folder/dirut.h>
#include <util/generic/xrange.h>

using namespace NCB;

Y_UNIT_TEST_SUITE(NonSymmetricIndexCalcerTest) {
    /*
     * Creates simple TTrainingForCPUDataProviderPtr for test purposes from quantized features and target
     * Float features will have flat indices from 0 to quantizedFloatFeatures.size()
     * Cat features will have flat indices from quantizedFloatFeatures.size() to quantizedFloatFeatures.size() + quantizedCatFeatures.size()
     */
    TTrainingForCPUDataProviderPtr CreateTrainingForCpuDataProviderFromQuantizedData(
        const TVector<TVector<ui8>>& quantizedFloatFeatures,
        const TVector<TVector<ui8>>& quantizedCatFeatures,
        const TVector<float>& target) {

        TDataProviderPtr dataProviderPtr = CreateDataProvider<IQuantizedFeaturesDataVisitor>(
            [&] (IQuantizedFeaturesDataVisitor* visitor) {
                const ui32 floatFeatureCount = quantizedFloatFeatures.size();
                const ui32 oheFeatureCount = quantizedCatFeatures.size();
                const ui32 objectCount = target.size();

                TDataMetaInfo metaInfo;

                metaInfo.TargetType = ERawTargetType::Float;
                metaInfo.TargetCount = 1;
                TVector<ui32> catFeatureIndices(oheFeatureCount);
                for (auto featureIdx : xrange(oheFeatureCount)) {
                    catFeatureIndices[featureIdx] = floatFeatureCount + featureIdx;
                }
                metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                    floatFeatureCount + oheFeatureCount,
                    catFeatureIndices,
                    TVector<TString>{}
                );

                TPoolQuantizationSchema schema;
                for (auto featureIdx : xrange(floatFeatureCount)) {
                    schema.FeatureIndices.push_back(featureIdx);
                    ui8 borderCount = *MaxElement(quantizedFloatFeatures[featureIdx].begin(), quantizedFloatFeatures[featureIdx].end());
                    schema.Borders.emplace_back();
                    schema.Borders[featureIdx].yresize(borderCount);
                    // actual border values does not matter here
                    Iota(schema.Borders[featureIdx].begin(), schema.Borders[featureIdx].end(), 0.0f);
                    schema.NanModes.push_back(ENanMode::Forbidden);
                }
                for (auto featureIdx : xrange(oheFeatureCount)) {
                    schema.CatFeatureIndices.push_back(floatFeatureCount + featureIdx);
                    ui8 bucketCount = *MaxElement(quantizedCatFeatures[featureIdx].begin(), quantizedCatFeatures[featureIdx].end()) + 1;
                    schema.FeaturesPerfectHash.emplace_back();
                    for (auto bucketIdx : xrange(bucketCount)) {
                        schema.FeaturesPerfectHash[featureIdx][bucketIdx] = {bucketIdx, 1};
                    }
                    schema.NanModes.push_back(ENanMode::Forbidden);
                }

                visitor->Start(metaInfo, objectCount, EObjectsOrder::Undefined, {}, schema);

                for (auto featureIdx : xrange(floatFeatureCount)) {
                    auto holder = TMaybeOwningArrayHolder<const ui8>::CreateNonOwning(quantizedFloatFeatures[featureIdx]);
                    visitor->AddFloatFeaturePart(featureIdx, 0, 8, holder);
                }

                for (auto featureIdx : xrange(oheFeatureCount)) {
                    auto holder = TMaybeOwningArrayHolder<const ui8>::CreateNonOwning(quantizedCatFeatures[featureIdx]);
                    visitor->AddCatFeaturePart(floatFeatureCount + featureIdx, 0, 8, holder);
                }

                visitor->AddTargetPart(0, {target.data(), target.size() * sizeof(float)});

                visitor->Finish();
            }
        );

        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
        TLabelConverter labelConverter;
        NPar::TLocalExecutor localExecutor;
        TRestorableFastRng64 rand(0);
        TMaybe<float> targetBorder = catBoostOptions.DataProcessingOptions->TargetBorder;
        TTrainingForCPUDataProvider trainData = GetTrainingData(
            std::move(dataProviderPtr),
            true,
            "learn",
            Nothing(),
            true,
            false,
            GetSystemTempDir(),
            nullptr,
            &catBoostOptions,
            &labelConverter,
            &targetBorder,
            &localExecutor,
            &rand).Get()->Cast<TQuantizedForCPUObjectsDataProvider>();

        return MakeIntrusive<TTrainingForCPUDataProvider>(trainData);
    }

    Y_UNIT_TEST(EmptyTree) {
        TNonSymmetricTreeStructure tree;

        TVector<TVector<ui8>> quantizedFloatFeatures = {
            {0, 1, 2, 2},
            {2, 1, 0, 0},
            {0, 2, 1, 0}
        };
        TVector<float> target = {1.0, 2.0, 1.5, 0.5};

        const TVector<TIndexType> expectedIndices = {0, 0, 0, 0};

        TTrainingForCPUDataProviderPtr trainDataProviderPtr = CreateTrainingForCpuDataProviderFromQuantizedData(
            quantizedFloatFeatures,
            {},
            target
        );

        TFold fold;
        fold.LearnPermutationFeaturesSubset = TFeaturesArraySubsetIndexing(TFullSubset<ui32>((ui32)target.size()));
        NPar::TLocalExecutor localExecutor;
        auto indices = BuildIndices(
            fold,
            tree,
            trainDataProviderPtr,
            {}, // test data
            &localExecutor);

        UNIT_ASSERT_VALUES_EQUAL(expectedIndices, indices);
    }

    Y_UNIT_TEST(ThreeSplitsTree) {
        TNonSymmetricTreeStructure tree;
        {
            TSplit split;
            split.FeatureIdx = 0;
            split.BinBorder = 1;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 0);
        }
        {
            TSplit split;
            split.FeatureIdx = 1;
            split.BinBorder = 0;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 0);
        }
        {
            TSplit split;
            split.FeatureIdx = 2;
            split.BinBorder = 2;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 1);
        }

        TVector<TVector<ui8>> quantizedFloatFeatures = {
            {0, 1, 2, 2, 1, 2},
            {2, 1, 0, 0, 0, 1},
            {0, 2, 1, 0, 1, 3}
        };
        TVector<float> target = {1.0, 2.0, 1.5, 0.5, 1.0, 1.4};
        const TVector<TIndexType> expectedIndices = {2, 2, 1, 1, 0, 3};

        TTrainingForCPUDataProviderPtr trainDataProviderPtr = CreateTrainingForCpuDataProviderFromQuantizedData(
            quantizedFloatFeatures,
            {},
            target
        );

        NPar::TLocalExecutor localExecutor;

        TFold fold;
        fold.LearnPermutationFeaturesSubset = TFeaturesArraySubsetIndexing(TFullSubset<ui32>((ui32)target.size()));
        auto indices = BuildIndices(
            fold,
            tree,
            trainDataProviderPtr,
            {}, // test data
            &localExecutor);

        UNIT_ASSERT_VALUES_EQUAL(expectedIndices, indices);
    }

    Y_UNIT_TEST(WithTestData) {
        TNonSymmetricTreeStructure tree;
        {
            TSplit split;
            split.FeatureIdx = 0;
            split.BinBorder = 1;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 0);
        }
        {
            TSplit split;
            split.FeatureIdx = 1;
            split.BinBorder = 0;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 0);
        }
        {
            TSplit split;
            split.FeatureIdx = 2;
            split.BinBorder = 2;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 1);
        }

        TVector<TVector<ui8>> learnQuantizedFloatFeatures = {
            {0, 1, 2, 2, 1, 2},
            {2, 1, 0, 0, 0, 1},
            {0, 2, 1, 0, 1, 3}
        };
        TVector<float> learnTarget = {1.0, 2.0, 1.5, 0.5, 1.0, 1.4};
        TTrainingForCPUDataProviderPtr trainDataProviderPtr = CreateTrainingForCpuDataProviderFromQuantizedData(
            learnQuantizedFloatFeatures,
            {},
            learnTarget
        );

        TVector<TVector<ui8>> testQuantizedFloatFeatures1 = {
            {0, 2, 1},
            {2, 0, 0},
            {0, 1, 1,}
        };
        TVector<float> testTarget1 = {1.0, 1.5, 1.0,};
        TTrainingForCPUDataProviderPtr testDataProviderPtr1 = CreateTrainingForCpuDataProviderFromQuantizedData(
            testQuantizedFloatFeatures1,
            {},
            testTarget1
        );

        TVector<TVector<ui8>> testQuantizedFloatFeatures2 = {
            {1, 2, 2},
            {1, 0, 1},
            {2, 0, 3}
        };
        TVector<float> testTarget2 = {2.0, 0.5, 1.4};
        TTrainingForCPUDataProviderPtr testDataProviderPtr2 = CreateTrainingForCpuDataProviderFromQuantizedData(
            testQuantizedFloatFeatures2,
            {},
            testTarget2
        );

        const TVector<TIndexType> expectedIndices = {2, 2, 1, 1, 0, 3, 2, 1, 0, 2, 1, 3};

        NPar::TLocalExecutor localExecutor;

        TFold fold;
        fold.LearnPermutationFeaturesSubset = TFeaturesArraySubsetIndexing(TFullSubset<ui32>((ui32)learnTarget.size()));
        auto indices = BuildIndices(
            fold,
            tree,
            trainDataProviderPtr,
            {testDataProviderPtr1, testDataProviderPtr2}, // test data
            &localExecutor);

        UNIT_ASSERT_VALUES_EQUAL(expectedIndices, indices);
    }

    Y_UNIT_TEST(WithOneHotSplits) {
        TNonSymmetricTreeStructure tree;
        {
            TSplit split;
            split.FeatureIdx = 0;
            split.BinBorder = 0;
            split.Type = ESplitType::OneHotFeature;
            tree.AddSplit(split, 0);
        }
        {
            TSplit split;
            split.FeatureIdx = 1;
            split.BinBorder = 1;
            split.Type = ESplitType::OneHotFeature;
            tree.AddSplit(split, 0);
        }
        {
            TSplit split;
            split.FeatureIdx = 0;
            split.BinBorder = 1;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 1);
        }
        {
            TSplit split;
            split.FeatureIdx = 1;
            split.BinBorder = 1;
            split.Type = ESplitType::FloatFeature;
            tree.AddSplit(split, 1);
        }

        TVector<TVector<ui8>> quantizedFloatFeatures = {
            {0, 1, 2, 2, 1, 2},
            {2, 1, 0, 0, 0, 1}
        };
        TVector<TVector<ui8>> quantizedCatFeatures = {
            {0, 2, 1, 0, 0, 3},
            {1, 3, 1, 2, 0, 0}
        };
        TVector<float> target = {1.0, 2.0, 1.5, 0.5, 1.0, 1.4};
        const TVector<TIndexType> expectedIndices = {4, 0, 2, 3, 1, 0};

        TTrainingForCPUDataProviderPtr trainDataProviderPtr = CreateTrainingForCpuDataProviderFromQuantizedData(
            quantizedFloatFeatures,
            quantizedCatFeatures,
            target
        );

        NPar::TLocalExecutor localExecutor;

        TFold fold;
        fold.LearnPermutationFeaturesSubset = TFeaturesArraySubsetIndexing(TFullSubset<ui32>((ui32)target.size()));
        auto indices = BuildIndices(
            fold,
            tree,
            trainDataProviderPtr,
            {}, // test data
            &localExecutor);

        UNIT_ASSERT_VALUES_EQUAL(expectedIndices, indices);
    }
}
