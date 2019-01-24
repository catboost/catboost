#include <catboost/libs/data_new/quantization.h>

#include <catboost/libs/data_new/data_provider.h>

#include <catboost/libs/data_new/ut/lib/for_data_provider.h>
#include <catboost/libs/data_new/ut/lib/for_objects.h>


#include <library/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(Quantization) {
    struct TTestCase {
        TRawBuilderData SrcData;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        TExpectedQuantizedData ExpectedData;
    };

    void Test(std::function<TTestCase()>&& generateTestCase) {
        for (auto quantizationOptions : {
                TQuantizationOptions{true, false},
                TQuantizationOptions{false, true},
                TQuantizationOptions{true, true}
             })
        {
            quantizationOptions.MaxSubsetSizeForSlowBuildBordersAlgorithms = 7;

            for (auto clearSrcData : {false, true}) {
                TTestCase testCase = generateTestCase();

                TRestorableFastRng64 rand(0);

                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(3);

                TRawDataProviderPtr rawDataProvider = MakeDataProvider<TRawObjectsDataProvider>(
                    Nothing(),
                    std::move(testCase.SrcData),
                    false,
                    &localExecutor
                );

                TDataProviderPtr quantizedDataProvider = Quantize(
                    quantizationOptions,
                    clearSrcData ? std::move(rawDataProvider) : rawDataProvider,
                    testCase.QuantizedFeaturesInfo,
                    &rand,
                    &localExecutor)->CastMoveTo<TObjectsDataProvider>();

                if (quantizationOptions.CpuCompatibleFormat) {
                    Compare<TQuantizedForCPUObjectsDataProvider>(
                        std::move(quantizedDataProvider),
                        testCase.ExpectedData
                    );
                } else {
                    Compare<TQuantizedObjectsDataProvider>(
                        std::move(quantizedDataProvider),
                        testCase.ExpectedData
                    );
                }
            }
        }
    }


    Y_UNIT_TEST(TestFloatFeatures) {
        auto generateTestCase = []() {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"f0", "f1", "f2"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), false, false, &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.Target = {"0", "1", "1", "0", "1", "0", "1", "0", "0"};
            srcData.TargetData.SetTrivialWeights(9);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(9)
            );

            constexpr auto quiet_NaN = std::numeric_limits<float>::quiet_NaN();

            TVector<TVector<float>> floatFeatures = {
                {0.12f, 0.33f, 0.0f, 0.11f, 0.9f, 0.67f, 1.2f, 2.1f, 0.56f},
                {0.0f, 0.11f, 0.82f, 0.93f, 0.15f, 0.18f, 2.2f, 3.1f, 0.21f},
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
                TVector<ui8>{1, 1, 0, 0, 3, 2, 4, 4, 2},
                TVector<ui8>{0, 0, 2, 3, 1, 1, 4, 4, 2},
                TVector<ui8>{4, 1, 2, 0, 3, 3, 0, 4, 2}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 0;

                TVector<TVector<float>> borders = {
                    {0.1149999946f, 0.4449999928f, 0.7849999666f, 1.049999952f},
                    {0.1299999952f, 0.1949999928f, 0.875f, 1.565000057f},
                    {
                        std::numeric_limits<float>::lowest(),
                        0.0549999997f,
                        0.1550000012f,
                        0.6349999905f
                    }
                };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Forbidden, ENanMode::Min};

            for (auto i : xrange(3)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
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
        auto generateTestCase = []() {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"f0", "f1"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), false, false, &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.Target = {"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"};
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
                TVector<ui8>{1, 2, 0, 0, 4, 4, 4, 4, 3, 2, 0, 1, 4},
                TVector<ui8>{4, 1, 1, 0, 3, 2, 0, 4, 1, 3, 3, 2, 4}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 0;

                TVector<TVector<float>> borders = {
                    {0.1149999946f, 0.2250000089f, 0.4449999928f, 0.6150000095f},
                    {
                        std::numeric_limits<float>::lowest(),
                        0.150000006f,
                        0.25f,
                        0.4900000095f
                    }
                };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Min};

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestFloatFeaturesWithNanModeMax) {
        auto generateTestCase = []() {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"f0", "f1"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), false, false, &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.Target = {"0", "1", "1", "0", "1", "0", "1", "0", "0"};
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
                TVector<ui8>{1, 1, 0, 0, 3, 2, 4, 4, 2},
                TVector<ui8>{3, 0, 1, 4, 2, 2, 4, 3, 1}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 0;

                TVector<TVector<float>> borders = {
                    {0.1149999946f, 0.4449999928f, 0.7849999666f, 1.049999952f},
                    {
                        0.0549999997f,
                        0.1550000012f,
                        0.6349999905f,
                        std::numeric_limits<float>::max()
                    }
                };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Max};

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            expectedData.ObjectsGrouping = TObjectsGrouping(9);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestCatFeatures) {
        auto generateTestCase = []() {
            TTestCase testCase;
            TRawBuilderData srcData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Categ, ""},
                {EColumn::Categ, ""},
                {EColumn::Categ, ""}
            };

            TVector<TString> featureId = {"c0", "c1", "c2"};

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), false, false, &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.Target = {"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"};
            srcData.TargetData.SetTrivialWeights(13);

            srcData.CommonObjectsData.FeaturesLayout = srcData.MetaInfo.FeaturesLayout;
            srcData.CommonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(13)
            );

            TVector<TVector<ui32>> hashedCatFeatures = {
                {12, 25, 10, 8, 165, 12, 1, 0, 112, 23, 12, 8, 25},
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
                {256, 45, 9, 110, 50, 10, 9, 256, 9, 110, 257, 90, 0}
            };

            ui32 featureIdx = 0;

            InitFeatures(
                hashedCatFeatures,
                *srcData.CommonObjectsData.SubsetIndexing,
                &featureIdx,
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


            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.CatFeatures = {
                TVector<ui32>{0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 3, 1},
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
                TVector<ui32>{0, 1, 2, 3, 4, 5, 2, 0, 2, 3, 6, 7, 8}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 9;

            TVector<TMap<ui32, ui32>> expectedPerfectHash = {
                {{12, 0}, {25, 1}, {10, 2}, {8, 3}, {165, 4}, {1, 5}, {0, 6}, {112, 7}, {23, 8}},
                {{0, 0}, {1, 1}},
                {{256, 0}, {45, 1}, {9, 2}, {110, 3}, {50, 4}, {10, 5}, {257, 6}, {90, 7}, {0, 8}}
            };

            for (auto i : xrange(3)) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    std::move(expectedPerfectHash[i])
                );
            }

            expectedData.Objects.CatFeatureUniqueValuesCounts = {{9,9}, {2,2}, {9,9}};

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestFloatAndCatFeatures) {
        auto generateTestCase = []() {
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

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), false, false, &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.Target = {"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"};
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
                {12, 25, 10, 8, 165, 12, 0, 1, 112, 23, 12, 8, 25},
                {0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
                {256, 45, 9, 110, 50, 10, 9, 256, 9, 110, 257, 90, 0}
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

            TExpectedQuantizedData expectedData;
            expectedData.MetaInfo = metaInfo;
            expectedData.Objects.FloatFeatures = {
                TVector<ui8>{1, 2, 0, 0, 3, 2, 3, 4, 2, 1, 0, 1, 4},
                TVector<ui8>{4, 1, 2, 0, 3, 2, 0, 4, 1, 3, 3, 2, 4}
            };
            expectedData.Objects.CatFeatures = {
                TVector<ui32>{0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 3, 1},
                TVector<ui32>{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
                TVector<ui32>{0, 1, 2, 3, 4, 5, 2, 0, 2, 3, 6, 7, 8}
            };

            expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *expectedData.MetaInfo.FeaturesLayout,
                TConstArrayRef<ui32>(),
                binarizationOptions
            );

            TVector<TVector<float>> borders = {
                {0.1149999946f, 0.3199999928f, 0.7849999666f, 1.600000024f},
                {
                    std::numeric_limits<float>::lowest(),
                    0.1149999946f,
                    0.2649999857f,
                    0.5600000024f
                }
            };
            TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Min};

            for (auto i : xrange(2)) {
                auto floatFeatureIdx = TFloatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
                expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
            }

            TVector<TMap<ui32, ui32>> expectedPerfectHash = {
                {{12, 0}, {25, 1}, {10, 2}, {8, 3}, {165, 4}, {0, 5}, {1, 6}, {112, 7}, {23, 8}},
                {{0, 0}, {1, 1}},
                {{256, 0}, {45, 1}, {9, 2}, {110, 3}, {50, 4}, {10, 5}, {257, 6}, {90, 7}, {0, 8}}
            };

            for (auto i : xrange(3)) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    std::move(expectedPerfectHash[i])
                );
            }
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 9;

            expectedData.Objects.CatFeatureUniqueValuesCounts = {{9,9}, {2,2}, {9,9}};

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;

            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
    }

    Y_UNIT_TEST(TestUpdateFloatAndCatFeatures) {
        auto generateTestCase = []() {
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

            TDataMetaInfo metaInfo(std::move(dataColumnsMetaInfo), false, false, &featureId);

            srcData.MetaInfo = metaInfo;

            srcData.TargetData.Target = {"0", "1", "1", "0", "1", "0", "1", "0", "0", "1", "0", "0", "0"};
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

            TVector<TMap<ui32, ui32>> srcPerfectHash = {
                {{12, 0}, {25, 1}, {10, 2}, {8, 3}, {165, 4}, {0, 5}},
                {{0, 0}, {1, 1}},
                {{256, 0}, {45, 1}, {9, 2}, {110, 3}, {50, 4}, {10, 5}, {257, 6}}
            };

            for (auto i : xrange(3)) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                testCase.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    TMap<ui32, ui32>(srcPerfectHash[i])
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

            TVector<TMap<ui32, ui32>> expectedPerfectHash = {
                {{12, 0}, {25, 1}, {10, 2}, {8, 3}, {165, 4}, {0, 5}, {1, 6}, {112, 7}, {23, 8}},
                {{0, 0}, {1, 1}},
                {{256, 0}, {45, 1}, {9, 2}, {110, 3}, {50, 4}, {10, 5}, {257, 6}, {90, 7}, {0, 8}}
            };

            for (auto i : xrange(3)) {
                auto catFeatureIdx = TCatFeatureIdx(i);
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    TMap<ui32, ui32>(srcPerfectHash[i])
                );
                expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                    catFeatureIdx,
                    std::move(expectedPerfectHash[i])
                );
            }
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 7;

            expectedData.Objects.CatFeatureUniqueValuesCounts = {{6, 9}, {2, 2}, {7, 9}};

            expectedData.ObjectsGrouping = TObjectsGrouping(13);
            expectedData.Target = srcData.TargetData;


            testCase.SrcData = std::move(srcData);
            testCase.ExpectedData = std::move(expectedData);

            return testCase;
        };

        Test(std::move(generateTestCase));
   }
}
