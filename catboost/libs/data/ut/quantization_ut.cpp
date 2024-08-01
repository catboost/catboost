#include <catboost/libs/data/quantization.h>

#include <catboost/libs/data/data_provider.h>

#include <catboost/libs/data/ut/lib/for_data_provider.h>
#include <catboost/libs/data/ut/lib/for_objects.h>

#include <util/generic/xrange.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(Quantization) {
    struct TTestCase {
        TRawBuilderData SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        TExpectedQuantizedData ExpectedData;
    };

    void PackBinaryFeatures(
        TVector<TVector<TBinaryFeaturesPack>>&& packs,
        TExpectedQuantizedObjectsData* data
    ) {
        const ui32 bitsPerKey = sizeof(TBinaryFeaturesPack) * CHAR_BIT;
        for (auto packIdx : xrange(packs.size())) {
            const ui32 objectCount = packs[packIdx].size();
            data->PackedBinaryFeaturesData.SrcData[packIdx]
                = MakeHolder<TBinaryPacksArrayHolder>(
                    packIdx,
                    TCompressedArray(
                        objectCount,
                        bitsPerKey,
                        CompressVector<ui64>(packs[packIdx], bitsPerKey)
                    ),
                    data->FullSubsetIndexing.Get()
                );
        }
    }

    void Test(std::function<TTestCase(bool)>&& generateTestCase) {
        for (auto quantizationOptions : {
                TQuantizationOptions{true, false},
                TQuantizationOptions{false, true},
                TQuantizationOptions{true, true}
             })
        {
            quantizationOptions.MaxSubsetSizeForBuildBordersAlgorithms = 7;

            TVector<bool> bundleExclusiveFeaturesVariants = {false};

            TVector<bool> packBinaryFeaturesVariants = {false, true};

            for (auto packBinaryFeatures : packBinaryFeaturesVariants) {
                quantizationOptions.PackBinaryFeaturesForCpu = packBinaryFeatures;

                for (auto bundleExclusiveFeatures : bundleExclusiveFeaturesVariants) {
                    quantizationOptions.BundleExclusiveFeatures = bundleExclusiveFeatures;

                    for (auto clearSrcData : {false, true}) {
                        TTestCase testCase = generateTestCase(packBinaryFeatures);

                        TRestorableFastRng64 rand(0);

                        NPar::TLocalExecutor localExecutor;
                        //localExecutor.RunAdditionalThreads(3);

                        TRawDataProviderPtr rawDataProvider = MakeDataProvider<TRawObjectsDataProvider>(
                            Nothing(),
                            std::move(testCase.SrcData),
                            false,
                            /*forceUnitAutoPairWeights*/ false,
                            &localExecutor
                        );

                        TDataProviderPtr quantizedDataProvider = Quantize(
                            quantizationOptions,
                            clearSrcData ? std::move(rawDataProvider) : rawDataProvider,
                            testCase.QuantizedFeaturesInfo,
                            &rand,
                            &localExecutor)->CastMoveTo<TObjectsDataProvider>();

                        Compare<TQuantizedObjectsDataProvider>(
                                std::move(quantizedDataProvider),
                                testCase.ExpectedData
                        );
                    }
                }
            }
        }
    }


    Y_UNIT_TEST(TestFloatFeatures) {
        auto generateTestCase = [](bool packBinaryFeatures) {
            TTestCase testCase;
            TRawBuilderData srcData;

            constexpr auto quiet_NaN = std::numeric_limits<float>::quiet_NaN();

            TVector<TVector<float>> floatFeatures = {
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f}, // 0
                {0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f}, // 1, binary
                {0.0f, 0.11f, 0.82f, 0.93f, 0.15f, 0.18f, 2.2f, 3.1f, 0.21f}, // 2
                {0.88f, 0.0f, 0.12f, quiet_NaN, 0.45f, 0.19f, quiet_NaN, 0.82f, 0.11f}, // 3
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 4, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f}, // 5, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 6, binary
                {0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f}, // 7, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f}, // 8, binary
                {0.0f, 0.11f, 0.82f, 0.93f, 0.15f, 0.18f, 2.2f, 3.1f, 0.21f}, // 9
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 10, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f}, // 11, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 12, binary
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f}, // 13
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 14, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f}, // 15, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 16, binary
                {0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f}, // 17, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f}, // 18, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f}, // 19, binary
                {0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f}, // 20, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f}, // 21, binary
            };

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns.push_back(TColumn{EColumn::Label, ""});

            TVector<TString> featureId;

            for (auto featureIdx : xrange(floatFeatures.size())) {
                dataColumnsMetaInfo.Columns.push_back(TColumn{EColumn::Num, ""});
                featureId.push_back("f" + ToString(featureIdx));
            }

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, Nothing(), &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "1", "0", "1", "0", "1", "0", "0"}};
            srcData.TargetData.Target.assign(rawTarget.begin(), rawTarget.end());
            srcData.TargetData.SetTrivialWeights(9);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(9)
            );

            ui32 featureIdx = 0;

            InitFeatures(
                floatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                &featureIdx,
                &srcData.ObjectsData.FloatFeatures
            );


            NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::GreedyLogSum,
                4,
                ENanMode::Min
            );

            testCase.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *metaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );


            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.FloatFeatures = {
                TVector<ui8>{1, 2, 0, 1, 4, 3, 4, 4, 3}, // 0
                TVector<ui8>{0, 1, 0, 1, 0, 0, 0, 1, 1}, // 1, binary
                TVector<ui8>{0, 1, 3, 4, 1, 2, 4, 4, 2}, // 2
                TVector<ui8>{4, 1, 2, 0, 4, 3, 0, 4, 2}, // 3
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 4, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0}, // 5, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 6, binary
                TVector<ui8>{0, 1, 0, 1, 0, 0, 0, 1, 1}, // 7, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0}, // 8, binary
                TVector<ui8>{0, 1, 3, 4, 1, 2, 4, 4, 2}, // 9
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 10, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0}, // 11, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 12, binary
                TVector<ui8>{1, 2, 0, 1, 4, 3, 4, 4, 3}, // 13
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 14, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0}, // 15, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 16, binary
                TVector<ui8>{0, 1, 0, 1, 0, 0, 0, 1, 1}, // 17, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0}, // 18, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1}, // 19, binary
                TVector<ui8>{0, 1, 0, 1, 0, 0, 0, 1, 1}, // 20, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0}, // 21, binary
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 0;

                TVector<TVector<float>> borders = {
                    {0.055f, 0.225f, 0.5f, 0.785f}, // 0
                    {0.5f}, // 1, binary
                    {0.055f, 0.165f, 0.5f, 0.875f}, // 2
                    {
                        std::numeric_limits<float>::lowest(),
                        0.06f,
                        0.155f,
                        0.32f
                    }, // 3
                    {1.f}, // 4, binary
                    {0.15f}, // 5, binary
                    {1.f}, // 6, binary
                    {0.5f}, // 7, binary
                    {0.15f}, // 8, binary
                    {0.055f, 0.165f, 0.5f, 0.875f}, // 9
                    {1.f}, // 10, binary
                    {0.15f}, // 11, binary
                    {1.f}, // 12, binary
                    {0.055f, 0.225f, 0.5f, 0.785f}, // 13
                    {1.f}, // 14, binary
                    {0.15f}, // 15, binary
                    {1.f}, // 16, binary
                    {0.5f}, // 17, binary
                    {0.15f}, // 18, binary
                    {1.f}, // 19, binary
                    {0.5f}, // 20, binary
                    {0.15f}, // 21, binary
                };
            TVector<ENanMode> nanModes(floatFeatures.size(), ENanMode::Forbidden);
            nanModes[3] = ENanMode::Min;

            for (auto i : xrange(floatFeatures.size())) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            expectedData.Objects.FullSubsetIndexing = MakeHolder<TFeaturesArraySubsetIndexing>(
                TFullSubset<ui32>(9)
            );

            expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>()
            );
            expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *expectedData.MetaInfo.FeaturesLayout,
                *expectedData.Objects.QuantizedFeaturesInfo,
                expectedData.Objects.ExclusiveFeatureBundlesData,
                !packBinaryFeatures
            );
            expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>()
            );

            if (packBinaryFeatures) {
                TVector<TVector<TBinaryFeaturesPack>> packs = {
                    TVector<TBinaryFeaturesPack>{
                        0b00000000,
                        0b10110101,
                        0b11101110,
                        0b00010001,
                        0b10100100,
                        0b01001010,
                        0b11101110,
                        0b10110101,
                        0b01011011
                    },
                    TVector<TBinaryFeaturesPack>{
                        0b00000000,
                        0b10110100,
                        0b01101111,
                        0b10010000,
                        0b00100100,
                        0b01001011,
                        0b01101111,
                        0b10110100,
                        0b11011011
                    },
                    TVector<TBinaryFeaturesPack>{0, 1, 1, 0, 1, 0, 1, 1, 0}
                };

                PackBinaryFeatures(std::move(packs), &expectedData.Objects);
            }

            expectedData.ObjectsGrouping = TObjectsGrouping(9);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestFloatFeaturesWithCalcBordersOverSubset) {
        auto generateTestCase = [](bool packBinaryFeatures) {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"f0", "f1"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, Nothing(), &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"}};
            srcData.TargetData.Target.assign(rawTarget.begin(), rawTarget.end());
            srcData.TargetData.SetTrivialWeights(13);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(13)
            );

            constexpr auto quiet_NaN = std::numeric_limits<float>::quiet_NaN();

            TVector<TVector<float>> floatFeatures = {
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f, 0.31f, 0.0f, 0.21f, 2.0f},
                {0.88f, 0.0f, 0.12f, quiet_NaN, 0.45f, 0.19f, quiet_NaN, 0.82f, 0.11f, 0.31f, 0.31f, 0.22f, 0.67f}
            };

            ui32 featureIdx = 0;

            InitFeatures(
                floatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                &featureIdx,
                &srcData.ObjectsData.FloatFeatures
            );


            NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::MinEntropy,
                4,
                ENanMode::Min
            );

            testCase.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *metaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );


            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.FloatFeatures = {
                TVector<ui8>{1, 4, 0, 1, 4, 4, 4, 4, 4, 3, 0, 2, 4},
                TVector<ui8>{4, 1, 1, 0, 4, 2, 0, 4, 1, 4, 4, 3, 4}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 0;

                TVector<TVector<float>> borders = {
                    {0.055f, 0.16f, 0.26f, 0.32f},
                    {
                        std::numeric_limits<float>::lowest(),
                        0.155f,
                        0.205f,
                        0.265f
                    }
                };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Min};

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>()
            );
            expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *expectedData.MetaInfo.FeaturesLayout,
                *expectedData.Objects.QuantizedFeaturesInfo,
                expectedData.Objects.ExclusiveFeatureBundlesData,
                !packBinaryFeatures
            );
            expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>()
            );

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestFloatFeaturesWithNanModeMax) {
        auto generateTestCase = [](bool packBinaryFeatures) {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"f0", "f1"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, Nothing(), &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "1", "0", "1", "0", "1", "0", "0"}};
            srcData.TargetData.Target.assign(rawTarget.begin(), rawTarget.end());
            srcData.TargetData.SetTrivialWeights(9);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(9)
            );

            constexpr auto quiet_NaN = std::numeric_limits<float>::quiet_NaN();

            TVector<TVector<float>> floatFeatures = {
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f},
                {0.88f, 0.0f, 0.12f, quiet_NaN, 0.45f, 0.19f, quiet_NaN, 0.82f, 0.11f}
            };

            ui32 featureIdx = 0;

            InitFeatures(
                floatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                &featureIdx,
                &srcData.ObjectsData.FloatFeatures
            );


            NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::GreedyLogSum,
                4,
                ENanMode::Max
            );

            testCase.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *metaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );


            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.FloatFeatures = {
                TVector<ui8>{1, 2, 0, 1, 4, 3, 4, 4, 3},
                TVector<ui8>{3, 0, 1, 4, 3, 2, 4, 3, 1}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 0;

                TVector<TVector<float>> borders = {
                    {0.055f, 0.225f, 0.5f, 0.785f},
                    {
                        0.06f,
                        0.155f,
                        0.32f,
                        std::numeric_limits<float>::max()
                    }
                };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Max};

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>()
            );
            expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *expectedData.MetaInfo.FeaturesLayout,
                *expectedData.Objects.QuantizedFeaturesInfo,
                expectedData.Objects.ExclusiveFeatureBundlesData,
                !packBinaryFeatures
            );
            expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>()
            );

            expectedData.ObjectsGrouping = TObjectsGrouping(9);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestCatFeatures) {
        auto generateTestCase = [](bool packBinaryFeatures) {
            TTestCase testCase;
            TRawBuilderData srcData;

            TVector<TVector<ui32>> hashedCatFeatures = {
                {12, 25, 10, 8, 165, 12, 1, 0, 112, 23, 12, 8, 25}, // 0
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 1, binary
                {256, 45, 9, 110, 50, 10, 9, 256, 9, 110, 257, 90, 0}, // 2
                {78, 2, 78, 2, 78, 78, 2, 2, 78, 2, 2, 78, 2}, // 3, binary
                {1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0}, // 4, binary
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 5, binary
                {78, 2, 78, 2, 78, 78, 2, 2, 78, 2, 2, 78, 2}, // 6, binary
                {92, 0, 92, 0, 0, 92, 92, 0, 0, 0, 0, 0, 92}, // 7, binary
                {1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0}, // 8, binary
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 9, binary
                {78, 2, 78, 2, 78, 78, 2, 2, 78, 2, 2, 78, 2}, // 10, binary
                {92, 0, 92, 0, 0, 92, 92, 0, 0, 0, 0, 0, 92}, // 11, binary
                {1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0} // 12, binary
            };

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns.push_back(TColumn{EColumn::Label, ""});

            TVector<TString> featureId;

            for (auto featureIdx : xrange(hashedCatFeatures.size())) {
                dataColumnsMetaInfo.Columns.push_back(TColumn{EColumn::Categ, ""});
                featureId.push_back("c" + ToString(featureIdx));
            }

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, Nothing(), &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"}};
            srcData.TargetData.Target.assign(rawTarget.begin(), rawTarget.end());
            srcData.TargetData.SetTrivialWeights(13);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(13)
            );

            ui32 featureIdx = 0;

            InitFeatures(
                hashedCatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                &featureIdx,
                &srcData.ObjectsData.CatFeatures
            );

            TVector<THashMap<ui32, TString>> catFeaturesHashToString(hashedCatFeatures.size());
            for (auto catFeatureIdx : xrange(hashedCatFeatures.size())) {
                for (auto hashedCatValue : hashedCatFeatures[catFeatureIdx]) {
                    catFeaturesHashToString[catFeatureIdx][hashedCatValue] = ToString(hashedCatValue);
                }
            }

            srcData.CommonObjectsData.CatFeaturesHashToString
                = MakeAtomicShared<TVector<THashMap<ui32, TString>>>(catFeaturesHashToString);


            NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::GreedyLogSum,
                4,
                ENanMode::Min
            );

            testCase.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *metaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );


            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.CatFeatures = {
                TVector<ui32>{0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 3, 1}, // 0
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 1, binary
                TVector<ui32>{0, 1, 2, 3, 4, 5, 2, 0, 2, 3, 6, 7, 8}, // 2
                TVector<ui32>{0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1}, // 3, binary
                TVector<ui32>{0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1}, // 4, binary
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 5, binary
                TVector<ui32>{0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1}, // 6, binary
                TVector<ui32>{0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0}, // 7, binary
                TVector<ui32>{0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1}, // 8, binary
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 9, binary
                TVector<ui32>{0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1}, // 10, binary
                TVector<ui32>{0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0}, // 11, binary
                TVector<ui32>{0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1}, // 12, binary
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 9;

            TVector<TCatFeaturePerfectHash> expectedPerfectHash = {
                {
                    Nothing(),
                    {
                        {12, {0, 3}},
                        {25, {1, 2}},
                        {10, {2, 1}},
                        {8, {3, 2}},
                        {165, {4, 1}},
                        {1, {5, 1}},
                        {0, {6, 1}},
                        {112, {7, 1}},
                        {23, {8, 1}}
                    }
                }, // 0
                {Nothing(), {{0, {0, 6}}, {1, {1, 7}}}}, // 1, binary
                {
                    Nothing(),
                    {
                        {256, {0, 2}},
                        {45, {1, 1}},
                        {9, {2, 3}},
                        {110, {3, 2}},
                        {50, {4, 1}},
                        {10, {5, 1}},
                        {257, {6, 1}},
                        {90, {7, 1}},
                        {0, {8, 1}}
                    }
                }, // 2
                {Nothing(), {{78, {0, 6}}, {2, {1, 7}}}}, // 3, binary
                {Nothing(), {{1, {0, 6}}, {0, {1, 7}}}}, // 4, binary
                {Nothing(), {{0, {0, 6}}, {1, {1, 7}}}}, // 5, binary
                {Nothing(), {{78, {0, 6}}, {2, {1, 7}}}}, // 6, binary
                {Nothing(), {{92, {0, 5}}, {0, {1, 8}}}}, // 7, binary
                {Nothing(), {{1, {0, 6}}, {0, {1, 7}}}}, // 8, binary
                {Nothing(), {{0, {0, 6}}, {1, {1, 7}}}}, // 9, binary
                {Nothing(), {{78, {0, 6}}, {2, {1, 7}}}}, // 10, binary
                {Nothing(), {{92, {0, 5}}, {0, {1, 8}}}}, // 11, binary
                {Nothing(), {{1, {0, 6}}, {0, {1, 7}}}} // 12, binary
            };

            for (auto i : xrange(hashedCatFeatures.size())) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    std::move(expectedPerfectHash[i])
                );
            }

            expectedData.Objects.FullSubsetIndexing = MakeHolder<TFeaturesArraySubsetIndexing>(
                TFullSubset<ui32>(13)
            );

            expectedData.Objects.CatFeatureUniqueValuesCounts = {
                {9,9}, // 0
                {2,2}, // 1, binary
                {9,9}, // 2
                {2,2}, // 3, binary
                {2,2}, // 4, binary
                {2,2}, // 5, binary
                {2,2}, // 6, binary
                {2,2}, // 7, binary
                {2,2}, // 8, binary
                {2,2}, // 9, binary
                {2,2}, // 10, binary
                {2,2}, // 11, binary
                {2,2}  // 12, binary
            };

            expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>()
            );
            expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *expectedData.MetaInfo.FeaturesLayout,
                *expectedData.Objects.QuantizedFeaturesInfo,
                expectedData.Objects.ExclusiveFeatureBundlesData,
                !packBinaryFeatures
            );
            expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>()
            );
            if (packBinaryFeatures) {
                TVector<TVector<TBinaryFeaturesPack>> packs = {
                    TVector<TBinaryFeaturesPack>{
                        0b00000000,
                        0b00110010,
                        0b11001101,
                        0b11111111,
                        0b01100100,
                        0b10001001,
                        0b00010010,
                        0b11111111,
                        0b01100100,
                        0b10111011,
                        0b11111111,
                        0b00100000,
                        0b11011111
                    },
                    TVector<TBinaryFeaturesPack>{
                        0b000,
                        0b011,
                        0b100,
                        0b111,
                        0b110,
                        0b000,
                        0b001,
                        0b111,
                        0b110,
                        0b011,
                        0b111,
                        0b010,
                        0b101
                    }
                };

                PackBinaryFeatures(std::move(packs), &expectedData.Objects);
            }

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestFloatAndCatFeatures) {
        auto generateTestCase = [](bool packBinaryFeatures) {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""}, // 0
                {EColumn::Categ, ""}, // 1
                {EColumn::Num, ""}, // 2
                {EColumn::Categ, ""}, // 3
                {EColumn::Categ, ""}, // 4
                {EColumn::Num, ""}, // 5
                {EColumn::Num, ""}, // 6
                {EColumn::Num, ""}, // 7
                {EColumn::Categ, ""}, // 8
                {EColumn::Categ, ""}, // 9
                {EColumn::Categ, ""}, // 10
                {EColumn::Categ, ""}, // 11
                {EColumn::Num, ""}, // 12
                {EColumn::Num, ""}, // 13
                {EColumn::Num, ""}, // 14
                {EColumn::Num, ""}, // 15
                {EColumn::Categ, ""}, // 16
                {EColumn::Num, ""}, // 17
                {EColumn::Num, ""}, // 18
                {EColumn::Num, ""}, // 19
                {EColumn::Num, ""}, // 20
                {EColumn::Num, ""} // 21
            };

            TVector<TString> featureId = {
                "f0", // 0
                "c0", // 1
                "f1", // 2
                "c1", // 3
                "c2", // 4
                "f2", // 5
                "f3", // 6
                "f4", // 7
                "c3", // 8
                "c4", // 9
                "c5", // 10
                "c6", // 11
                "f5", // 12
                "f6", // 13
                "f7", // 14
                "f8", // 15
                "c7", // 16
                "f9", // 17
                "f10", // 18
                "f11", // 19
                "f12", // 20
                "f13", // 21
            };

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, Nothing(), &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"}};
            srcData.TargetData.Target.assign(rawTarget.begin(), rawTarget.end());
            srcData.TargetData.SetTrivialWeights(13);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(13)
            );

            constexpr auto quiet_NaN = std::numeric_limits<float>::quiet_NaN();

            TVector<TVector<float>> floatFeatures = {
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f, 0.31f, 0.0f, 0.21f, 2.0f}, // 0
                {0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f}, // 1, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f}, // 2, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f}, // 3, binary
                {0.88f, 0.0f, 0.12f, quiet_NaN, 0.45f, 0.19f, quiet_NaN, 0.82f, 0.11f, 0.31f, 0.31f, 0.22f, 0.67f}, // 4
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f}, // 5, binary
                {0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f}, // 6, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f}, // 7, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f}, // 8, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f}, // 9, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f}, // 10, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f}, // 11, binary
                {0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.2f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f}, // 12, binary
                {0.f, 0.f, 2.f, 0.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f, 0.f, 2.f, 2.f}, // 13, binary
            };

            InitFeatures(
                floatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                TConstArrayRef<ui32>{0, 2, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20, 21},
                &srcData.ObjectsData.FloatFeatures
            );

            TVector<TVector<ui32>> hashedCatFeatures = {
                {12, 25, 10, 8, 165, 12, 1, 0, 112, 23, 12, 8, 25}, // 0
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 1, binary
                {256, 45, 9, 110, 50, 10, 9, 256, 9, 110, 257, 90, 0}, // 2
                {78, 2, 78, 2, 78, 78, 2, 2, 78, 2, 2, 78, 2}, // 3, binary
                {1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0}, // 4, binary
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 5, binary
                {78, 2, 78, 2, 78, 78, 2, 2, 78, 2, 2, 78, 2}, // 6, binary
                {92, 0, 92, 0, 0, 92, 92, 0, 0, 0, 0, 0, 92}, // 7, binary
            };

            InitFeatures(
                hashedCatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                TConstArrayRef<ui32>{1, 3, 4, 8, 9, 10, 11, 16},
                &srcData.ObjectsData.CatFeatures
            );

            TVector<THashMap<ui32, TString>> catFeaturesHashToString(hashedCatFeatures.size());
            for (auto catFeatureIdx : xrange(hashedCatFeatures.size())) {
                for (auto hashedCatValue : hashedCatFeatures[catFeatureIdx]) {
                    catFeaturesHashToString[catFeatureIdx][hashedCatValue] = ToString(hashedCatValue);
                }
            }

            srcData.CommonObjectsData.CatFeaturesHashToString
                = MakeAtomicShared<TVector<THashMap<ui32, TString>>>(catFeaturesHashToString);


            NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::GreedyLogSum,
                4,
                ENanMode::Min
            );

            testCase.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *metaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );

            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.FloatFeatures = {
                TVector<ui8>{1, 4, 0, 1, 4, 4, 4, 4, 4, 3, 0, 2, 4}, // 0, f0
                TVector<ui8>{0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1}, // 2, f1, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1}, // 5, f2, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0}, // 6, f3, binary
                TVector<ui8>{4, 1, 2, 0, 4, 2, 0, 4, 2, 4, 4, 3, 4}, // 7, f4
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1}, // 12, f5, binary
                TVector<ui8>{0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1}, // 13, f6, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0}, // 14, f7, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1}, // 15, f8, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0}, // 17, f9, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1}, // 18, f10, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1}, // 19, f11, binary
                TVector<ui8>{0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0}, // 20, f12, binary
                TVector<ui8>{0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1}  // 21, f13, binary
            };
            expectedData.Objects.CatFeatures = {
                TVector<ui32>{0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 3, 1}, // 1, c0
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 3, c1, binary
                TVector<ui32>{0, 1, 2, 3, 4, 5, 2, 0, 2, 3, 6, 7, 8}, // 4, c2
                TVector<ui32>{0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1}, // 8, c3, binary
                TVector<ui32>{0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1}, // 9, c4, binary
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}, // 10, c5, binary
                TVector<ui32>{0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1}, // 11, c6, binary
                TVector<ui32>{0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0}  // 16, c7, binary
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );

            TVector<TVector<float>> borders = {
                {0.055f, 0.16f, 0.26f, 0.32f}, // 0
                {0.5f}, // 1, binary
                {1.f}, // 2, binary
                {0.15f}, // 3, binary
                {
                    std::numeric_limits<float>::lowest(),
                    0.06f,
                    0.205f,
                    0.265f
                }, // 4
                {1.f}, // 5, binary
                {0.5f}, // 6, binary
                {0.15f}, // 7, binary
                {1.f}, // 8, binary
                {0.15f}, // 9, binary
                {1.f}, // 10, binary
                {1.f}, // 11, binary
                {0.15f}, // 12, binary
                {1.f} // 13, binary
            };
            TVector<ENanMode> nanModes(floatFeatures.size(), ENanMode::Forbidden);
            nanModes[4] = ENanMode::Min;

            for (auto i : xrange(floatFeatures.size())) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            TVector<TCatFeaturePerfectHash> expectedPerfectHash = {
                {
                    Nothing(),
                    {
                        {12, {0, 3}},
                        {25, {1, 2}},
                        {10, {2, 1}},
                        {8, {3, 2}},
                        {165, {4, 1}},
                        {1, {5, 1}},
                        {0, {6, 1}},
                        {112, {7, 1}},
                        {23, {8, 1}}
                    }
                }, // 0
                {Nothing(), {{0, {0, 6}}, {1, {1, 7}}}}, // 1, binary
                {
                    Nothing(),
                    {
                        {256, {0, 2}},
                        {45, {1, 1}},
                        {9, {2, 3}},
                        {110, {3, 2}},
                        {50, {4, 1}},
                        {10, {5, 1}},
                        {257, {6, 1}},
                        {90, {7, 1}},
                        {0, {8, 1}}
                    }
                }, // 2
                {Nothing(), {{78, {0, 6}}, {2, {1, 7}}}}, // 3, binary
                {Nothing(), {{1, {0, 6}}, {0, {1, 7}}}}, // 4, binary
                {Nothing(), {{0, {0, 6}}, {1, {1, 7}}}}, // 5, binary
                {Nothing(), {{78, {0, 6}}, {2, {1, 7}}}}, // 6, binary
                {Nothing(), {{92, {0, 5}}, {0, {1, 8}}}} // 7, binary
            };

            for (auto i : xrange(hashedCatFeatures.size())) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    std::move(expectedPerfectHash[i])
                );
            }
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 9;

            expectedData.Objects.FullSubsetIndexing = MakeHolder<TFeaturesArraySubsetIndexing>(
                TFullSubset<ui32>(13)
            );

            expectedData.Objects.CatFeatureUniqueValuesCounts = {
                {9,9}, // 0
                {2,2}, // 1, binary
                {9,9}, // 2
                {2,2}, // 3, binary
                {2,2}, // 4, binary
                {2,2}, // 5, binary
                {2,2}, // 6, binary
                {2,2}  // 7, binary
            };

            expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>()
            );
            expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *expectedData.MetaInfo.FeaturesLayout,
                *expectedData.Objects.QuantizedFeaturesInfo,
                expectedData.Objects.ExclusiveFeatureBundlesData,
                !packBinaryFeatures
            );
            expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>()
            );
            if (packBinaryFeatures) {
                TVector<TVector<TBinaryFeaturesPack>> packs = {
                    TVector<TBinaryFeaturesPack>{ // bits: f9 f8 f7 f6 f5 f3 f2 f1
                        0b00000000,
                        0b10110101,
                        0b11101110,
                        0b00010001,
                        0b10100100,
                        0b01001010,
                        0b11101110,
                        0b10110101,
                        0b01011011,
                        0b11101110,
                        0b00000000,
                        0b01011011,
                        0b01011011
                    },
                    TVector<TBinaryFeaturesPack>{ // bits: c5 c4 c3 c1 f13 f12 f11 f10
                        0b00000000,
                        0b00100100,
                        0b11011111,
                        0b11110000,
                        0b01000100,
                        0b10011011,
                        0b00101111,
                        0b11110100,
                        0b01001011,
                        0b10111111,
                        0b11110000,
                        0b00001011,
                        0b11111011
                    },
                    TVector<TBinaryFeaturesPack>{ // bits: c7 c6
                        0b00,
                        0b11,
                        0b00,
                        0b11,
                        0b10,
                        0b00,
                        0b01,
                        0b11,
                        0b10,
                        0b11,
                        0b11,
                        0b10,
                        0b01
                    },
                };

                PackBinaryFeatures(std::move(packs), &expectedData.Objects);
            }

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;

            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestUpdateFloatAndCatFeatures) {
        auto generateTestCase = [](bool packBinaryFeatures) {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Categ, ""},
                {EColumn::Num, ""},
                {EColumn::Categ, ""},
                {EColumn::Categ, ""}
            };

            TVector<TString> featureId = {"f0", "c0", "f1", "c1", "c2"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, Nothing(), &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"}};
            srcData.TargetData.Target.assign(rawTarget.begin(), rawTarget.end());
            srcData.TargetData.SetTrivialWeights(13);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(13)
            );

            constexpr auto quiet_NaN = std::numeric_limits<float>::quiet_NaN();

            TVector<TVector<float>> floatFeatures = {
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f, 0.31f, 0.0f, 0.21f, 2.0f},
                {0.88f, 0.0f, 0.12f, quiet_NaN, 0.45f, 0.19f, quiet_NaN, 0.82f, 0.11f, 0.31f, 0.31f, 0.22f, 0.67f}
            };

            InitFeatures(
                floatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                TConstArrayRef<ui32>{0, 2},
                &srcData.ObjectsData.FloatFeatures
            );

            TVector<TVector<ui32>> hashedCatFeatures = {
                {12, 25, 10, 8, 165, 12, 1, 0, 112, 23, 12, 8, 25},
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
                {90, 256, 45, 9, 110, 50, 10, 9, 256, 9, 110, 257, 0}
            };

            InitFeatures(
                hashedCatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                TConstArrayRef<ui32>{1, 3, 4},
                &srcData.ObjectsData.CatFeatures
            );

            TVector<THashMap<ui32, TString>> catFeaturesHashToString(3);
            for (auto catFeatureIdx : xrange(hashedCatFeatures.size())) {
                for (auto hashedCatValue : hashedCatFeatures[catFeatureIdx]) {
                    catFeaturesHashToString[catFeatureIdx][hashedCatValue] = ToString(hashedCatValue);
                }
            }

            srcData.CommonObjectsData.CatFeaturesHashToString
                = MakeAtomicShared<TVector<THashMap<ui32, TString>>>(catFeaturesHashToString);


            NCatboostOptions::TBinarizationOptions binarizationOptions(
                EBorderSelectionType::GreedyLogSum,
                4,
                ENanMode::Min
            );

            testCase.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *metaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );

            TVector<TVector<float>> borders = {
                {0.1149999946f, 0.3199999928f, 0.7849999666f, 1.600000024f},
                {
                    std::numeric_limits<float>::lowest(),
                    0.1149999946f,
                    0.2649999857f,
                    0.3799999952f,
                    0.5600000024f
                }
            };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Min};

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                testCase.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, TVector<float>(borders[i]));
                testCase.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            TVector<TCatFeaturePerfectHash> srcPerfectHash = {
                {
                    Nothing(),
                    {{12, {0, 4}}, {25, {1, 3}}, {10, {2, 1}}, {8, {3, 3}}, {165, {4, 2}}, {0, {5, 1}}}
                },
                {Nothing(), {{0, {0, 6}}, {1, {1, 8}}}},
                {
                    Nothing(),
                    {
                        {256, {0, 2}},
                        {45, {1, 1}},
                        {9, {2, 3}},
                        {110, {3, 4}},
                        {50, {4, 1}},
                        {10, {5, 2}},
                        {257, {6, 1}}
                    }
                }
            };

            for (auto i : xrange(3)) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                testCase.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    TCatFeaturePerfectHash(srcPerfectHash[i])
                );
            }


            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.FloatFeatures = {
                TVector<ui8>{1, 2, 0, 0, 3, 2, 3, 4, 2, 1, 0, 1, 4},
                TVector<ui8>{5, 1, 2, 0, 4, 2, 0, 5, 1, 3, 3, 2, 5}
            };
            expectedData.Objects.CatFeatures = {
                TVector<ui32>{0, 1, 2, 3, 4, 0, 6, 5, 7, 8, 0, 3, 1},
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
                TVector<ui32>{7, 0, 1, 2, 3, 4, 5, 2, 0, 2, 3, 6, 8}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
                    floatFeatureIdx,
                    TVector<float>(borders[i])
                );
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            TVector<TCatFeaturePerfectHash> expectedPerfectHash = {
                {
                    Nothing(),
                    {
                        {12, {0, 4 + 3}},
                        {25, {1, 3 + 2}},
                        {10, {2, 1 + 1}},
                        {8, {3, 3 + 2}},
                        {165, {4, 2 + 1}},
                        {0, {5, 1 + 1}},
                        {1, {6, 0 + 1}},
                        {112, {7, 0 + 1}},
                        {23, {8, 0 + 1}}
                    }
                },
                {Nothing(), {{0, {0, 6 + 6}}, {1, {1, 8 + 7}}}},
                {
                    Nothing(),
                    {
                        {256, {0, 2 + 2}},
                        {45, {1, 1 + 1}},
                        {9, {2, 3 + 3}},
                        {110, {3, 4 + 2}},
                        {50, {4, 1 + 1}},
                        {10, {5, 2 + 1}},
                        {257, {6, 1 + 1}},
                        {90, {7, 0 + 1}},
                        {0, {8, 0 + 1}}
                    }
                }
            };

            for (auto i : xrange(3)) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    TCatFeaturePerfectHash(srcPerfectHash[i])
                );
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    std::move(expectedPerfectHash[i])
                );
            }
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 7;

            expectedData.Objects.FullSubsetIndexing = MakeHolder<TFeaturesArraySubsetIndexing>(
                TFullSubset<ui32>(13)
            );

            expectedData.Objects.CatFeatureUniqueValuesCounts = {{6, 9}, {2, 2}, {7, 9}};

            expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TExclusiveFeaturesBundle>()
            );
            expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                *expectedData.MetaInfo.FeaturesLayout,
                *expectedData.Objects.QuantizedFeaturesInfo,
                expectedData.Objects.ExclusiveFeatureBundlesData,
                !packBinaryFeatures
            );
            expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                *expectedData.MetaInfo.FeaturesLayout,
                TVector<TFeaturesGroup>()
            );
            if (packBinaryFeatures) {
                TVector<TVector<TBinaryFeaturesPack>> packs = {
                    TVector<TBinaryFeaturesPack>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1}
                };

                PackBinaryFeatures(std::move(packs), &expectedData.Objects);
            }

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
   }
}
