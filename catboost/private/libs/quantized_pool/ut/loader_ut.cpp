
#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/ut/lib/for_data_provider.h>
#include <catboost/libs/data/ut/lib/for_loader.h>
#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/private/libs/quantized_pool/pool.h>
#include <catboost/private/libs/quantized_pool/serialization.h>
#include <catboost/private/libs/quantization_schema/schema.h>
#include <catboost/private/libs/quantization_schema/serialization.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/xrange.h>
#include <util/memory/blob.h>
#include <util/random/random.h>
#include <util/stream/file.h>
#include <util/string/printf.h>
#include <util/system/tempfile.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(LoadDataFromQuantized) {

    struct TTestCase {
        NCB::TSrcData SrcData;
        TExpectedQuantizedData ExpectedData;
    };

    struct TReadDatasetMainParams {
        TPathWithScheme PoolPath;
        TPathWithScheme PairsFilePath; // can be uninited
        TPathWithScheme GraphFilePath; // can be uninited
        TPathWithScheme GroupWeightsFilePath; // can be uninited
        TPathWithScheme BaselineFilePath; // can be uninited
        TVector<NJson::TJsonValue> ClassLabels;
    };


    void SaveQuantizedPool(
        const NCB::TSrcData& srcData,
        TPathWithScheme* dstPath,
        TVector<THolder<TTempFile>>* srcDataFiles
    ) {
        auto tmpFileName = MakeTempName();
        NCB::SaveQuantizedPool(srcData, tmpFileName);
        *dstPath = TPathWithScheme("quantized://" + tmpFileName);
        srcDataFiles->emplace_back(MakeHolder<TTempFile>(tmpFileName));
    }


    void SaveSrcData(
        const NCB::TSrcData& srcData,
        TReadDatasetMainParams* readDatasetMainParams,
        TVector<THolder<TTempFile>>* srcDataFiles
    ) {
        SaveQuantizedPool(srcData, &(readDatasetMainParams->PoolPath), srcDataFiles);
        SaveDataToTempFile(srcData.PairsFileData, &(readDatasetMainParams->PairsFilePath), srcDataFiles);
        readDatasetMainParams->PairsFilePath.Scheme = srcData.PairsFilePathScheme;
        SaveDataToTempFile(
            srcData.GroupWeightsFileData,
            &(readDatasetMainParams->GroupWeightsFilePath),
            srcDataFiles
        );
        SaveDataToTempFile(
            srcData.BaselineFileData,
            &(readDatasetMainParams->BaselineFilePath),
            srcDataFiles
        );
    }

    template <class T>
    THolder<TSrcColumnBase> MakeFeaturesColumn(EColumn type, TVector<TVector<T>>&& data) {
        return THolder<TSrcColumn<T>>(new TSrcColumn<T>(type, std::move(data)));
    }


    void Test(const TTestCase& testCase) {
        TReadDatasetMainParams readDatasetMainParams;

        // TODO(akhropov): temporarily use THolder until TTempFile move semantic are fixed
        TVector<THolder<TTempFile>> srcDataFiles;

        SaveSrcData(testCase.SrcData, &readDatasetMainParams, &srcDataFiles);

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(3);

        TDataProviderPtr dataProvider = ReadDataset(
            /*taskType*/Nothing(),
            readDatasetMainParams.PoolPath,
            readDatasetMainParams.PairsFilePath, // can be uninited
            readDatasetMainParams.GraphFilePath, // can be uninited
            readDatasetMainParams.GroupWeightsFilePath, // can be uninited
            /*timestampsFilePath*/TPathWithScheme(),
            readDatasetMainParams.BaselineFilePath, // can be uninited
            /*featureNamesPath*/TPathWithScheme(),
            /*poolMetaInfoPath*/TPathWithScheme(),
            NCatboostOptions::TColumnarPoolFormatParams(),
            testCase.SrcData.IgnoredFeatures,
            testCase.SrcData.ObjectsOrder,
            TDatasetSubset::MakeColumns(),
            /*loadSampleIds*/ false,
            /*forceUnitAutoPairWeights*/ true,
            &readDatasetMainParams.ClassLabels,
            &localExecutor
        );

        Compare<TQuantizedObjectsDataProvider>(std::move(dataProvider), testCase.ExpectedData);
    }


    Y_UNIT_TEST(ReadDatasetSimpleFloatFeatures) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 5;
        srcData.LocalIndexToColumnIndex = {0, 1, 2};
        srcData.PoolQuantizationSchema.FloatFeatureIndices = {0, 1};
        srcData.PoolQuantizationSchema.Borders = {{0.1f, 0.2f, 0.3f}, {0.25f, 0.5f, 0.75f}};
        srcData.PoolQuantizationSchema.NanModes = {ENanMode::Forbidden, ENanMode::Min};
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{1, 3}, {0, 1, 2}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{2, 3}, {0, 3, 1}}));

        srcData.Target = TSrcColumn<float>{EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}}};

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Num, ""},
            {EColumn::Num, ""},
            {EColumn::Label, ""}
        };

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), Nothing());
        expectedData.Objects.FloatFeatures = {
            TVector<ui8>{1, 3, 0, 1, 2},
            TVector<ui8>{2, 3, 0, 3, 1}
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 3)
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(0), {0.1f, 0.2f, 0.3f});
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(1), {0.25f, 0.5f, 0.75f});
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(1), ENanMode::Min);
        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );

        expectedData.ObjectsGrouping = TObjectsGrouping(5);

        expectedData.Target.TargetType = ERawTargetType::Float;

        TVector<TVector<TString>> rawTarget{{"0.12", "0", "0.45", "0.1", "0.22"}};
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.SetTrivialWeights(5);

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }

    Y_UNIT_TEST(ReadDatasetSimpleCatFeatures) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 5;
        srcData.LocalIndexToColumnIndex = {0, 1, 2, 3};
        srcData.PoolQuantizationSchema.CatFeatureIndices = {0, 1, 2};
        {
            TMap<ui32, TValueWithCount> perfectHash;
            perfectHash.emplace(0x00, TValueWithCount{0, 1});
            perfectHash.emplace(0xF1, TValueWithCount{1, 2});
            perfectHash.emplace(0xFF, TValueWithCount{2, 7});
            perfectHash.emplace(0xAC, TValueWithCount{3, 2});

            srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                std::move(perfectHash)
            );
        }
        {
            TMap<ui32, TValueWithCount> perfectHash;
            perfectHash.emplace(0x23, TValueWithCount{0, 2});
            perfectHash.emplace(0x13, TValueWithCount{1, 1});
            perfectHash.emplace(0xBA, TValueWithCount{2, 1});
            perfectHash.emplace(0x00, TValueWithCount{3, 4});
            perfectHash.emplace(0x91, TValueWithCount{4, 6});

            srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                std::move(perfectHash)
            );
        }
        {
            TMap<ui32, TValueWithCount> perfectHash;
            perfectHash.emplace(0x01, TValueWithCount{0, 4});
            perfectHash.emplace(0x12, TValueWithCount{1, 1});
            perfectHash.emplace(0x02, TValueWithCount{2, 17});

            srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                std::move(perfectHash)
            );
        }
        srcData.CatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Categ, {{1, 3}, {0, 1, 2}}));
        srcData.CatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Categ, {{2, 4}, {0, 3, 1}}));
        srcData.CatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Categ, {{2, 1}, {1, 0, 0}}));

        srcData.Target = TSrcColumn<float>{EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}}};

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Categ, ""},
            {EColumn::Categ, ""},
            {EColumn::Categ, ""},
            {EColumn::Label, ""}
        };

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), Nothing());
        expectedData.Objects.CatFeatures = {
            TVector<ui32>{1, 3, 0, 1, 2},
            TVector<ui32>{2, 4, 0, 3, 1},
            TVector<ui32>{2, 1, 1, 0, 0}
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 32)
        );
        {
            TCatFeaturePerfectHash perfectHash;
            perfectHash.Map.emplace(0x00, TValueWithCount{0, 1});
            perfectHash.Map.emplace(0xF1, TValueWithCount{1, 2});
            perfectHash.Map.emplace(0xFF, TValueWithCount{2, 7});
            perfectHash.Map.emplace(0xAC, TValueWithCount{3, 2});

            expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                TCatFeatureIdx(0),
                std::move(perfectHash)
            );
        }
        {
            TCatFeaturePerfectHash perfectHash;
            perfectHash.Map.emplace(0x23, TValueWithCount{0, 2});
            perfectHash.Map.emplace(0x13, TValueWithCount{1, 1});
            perfectHash.Map.emplace(0xBA, TValueWithCount{2, 1});
            perfectHash.Map.emplace(0x00, TValueWithCount{3, 4});
            perfectHash.Map.emplace(0x91, TValueWithCount{4, 6});

            expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                TCatFeatureIdx(1),
                std::move(perfectHash)
            );
        }
        {
            TCatFeaturePerfectHash perfectHash;
            perfectHash.Map.emplace(0x01, TValueWithCount{0, 4});
            perfectHash.Map.emplace(0x12, TValueWithCount{1, 1});
            perfectHash.Map.emplace(0x02, TValueWithCount{2, 17});

            expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                TCatFeatureIdx(2),
                std::move(perfectHash)
            );
        }
        expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 5;

        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );
        expectedData.Objects.CatFeatureUniqueValuesCounts = {
            TCatFeatureUniqueValuesCounts{4, 4},
            TCatFeatureUniqueValuesCounts{5, 5},
            TCatFeatureUniqueValuesCounts{3, 3}
        };

        expectedData.ObjectsGrouping = TObjectsGrouping(5);

        expectedData.Target.TargetType = ERawTargetType::Float;

        TVector<TVector<TString>> rawTarget{{"0.12", "0", "0.45", "0.1", "0.22"}};
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.SetTrivialWeights(5);

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }

    Y_UNIT_TEST(ReadDatasetFloatAndCatFeatures) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 5;
        srcData.LocalIndexToColumnIndex = {0, 3, 1, 2, 4, 5};

        srcData.PoolQuantizationSchema.FloatFeatureIndices = {0, 3};
        srcData.PoolQuantizationSchema.Borders = {{0.1f, 0.2f, 0.3f}, {0.25f, 0.5f, 0.75f}};
        srcData.PoolQuantizationSchema.NanModes = {ENanMode::Forbidden, ENanMode::Min};
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{1, 3}, {0, 1, 2}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{2, 3}, {0, 3, 1}}));

        srcData.PoolQuantizationSchema.CatFeatureIndices = {1, 2, 4};
        {
            TMap<ui32, TValueWithCount> perfectHash;
            perfectHash.emplace(0x00, TValueWithCount{0, 1});
            perfectHash.emplace(0xF1, TValueWithCount{1, 2});
            perfectHash.emplace(0xFF, TValueWithCount{2, 7});
            perfectHash.emplace(0xAC, TValueWithCount{3, 2});

            srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                std::move(perfectHash)
            );
        }
        {
            TMap<ui32, TValueWithCount> perfectHash;
            perfectHash.emplace(0x23, TValueWithCount{0, 2});
            perfectHash.emplace(0x13, TValueWithCount{1, 1});
            perfectHash.emplace(0xBA, TValueWithCount{2, 1});
            perfectHash.emplace(0x00, TValueWithCount{3, 4});
            perfectHash.emplace(0x91, TValueWithCount{4, 6});

            srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                std::move(perfectHash)
            );
        }
        {
            TMap<ui32, TValueWithCount> perfectHash;
            perfectHash.emplace(0x01, TValueWithCount{0, 4});
            perfectHash.emplace(0x12, TValueWithCount{1, 1});
            perfectHash.emplace(0x02, TValueWithCount{2, 17});

            srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(
                std::move(perfectHash)
            );
        }
        srcData.CatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Categ, {{1, 3}, {0, 1, 2}}));
        srcData.CatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Categ, {{2, 4}, {0, 3, 1}}));
        srcData.CatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Categ, {{2, 1}, {1, 0, 0}}));

        srcData.Target = TSrcColumn<float>{EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}}};

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Num, ""},
            {EColumn::Categ, ""},
            {EColumn::Categ, ""},
            {EColumn::Num, ""},
            {EColumn::Categ, ""},
            {EColumn::Label, ""}
        };

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), Nothing());
        expectedData.Objects.FloatFeatures = {
            TVector<ui8>{1, 3, 0, 1, 2},
            TVector<ui8>{2, 3, 0, 3, 1}
        };
        expectedData.Objects.CatFeatures = {
            TVector<ui32>{1, 3, 0, 1, 2},
            TVector<ui32>{2, 4, 0, 3, 1},
            TVector<ui32>{2, 1, 1, 0, 0}
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 3)
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(0), {0.1f, 0.2f, 0.3f});
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(1), {0.25f, 0.5f, 0.75f});
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(1), ENanMode::Min);
        {
            TCatFeaturePerfectHash perfectHash;
            perfectHash.Map.emplace(0x00, TValueWithCount{0, 1});
            perfectHash.Map.emplace(0xF1, TValueWithCount{1, 2});
            perfectHash.Map.emplace(0xFF, TValueWithCount{2, 7});
            perfectHash.Map.emplace(0xAC, TValueWithCount{3, 2});

            expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                TCatFeatureIdx(0),
                std::move(perfectHash)
            );
        }
        {
            TCatFeaturePerfectHash perfectHash;
            perfectHash.Map.emplace(0x23, TValueWithCount{0, 2});
            perfectHash.Map.emplace(0x13, TValueWithCount{1, 1});
            perfectHash.Map.emplace(0xBA, TValueWithCount{2, 1});
            perfectHash.Map.emplace(0x00, TValueWithCount{3, 4});
            perfectHash.Map.emplace(0x91, TValueWithCount{4, 6});

            expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                TCatFeatureIdx(1),
                std::move(perfectHash)
            );
        }
        {
            TCatFeaturePerfectHash perfectHash;
            perfectHash.Map.emplace(0x01, TValueWithCount{0, 4});
            perfectHash.Map.emplace(0x12, TValueWithCount{1, 1});
            perfectHash.Map.emplace(0x02, TValueWithCount{2, 17});

            expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                TCatFeatureIdx(2),
                std::move(perfectHash)
            );
        }
        expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = 5;

        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );
        expectedData.Objects.CatFeatureUniqueValuesCounts = {
            TCatFeatureUniqueValuesCounts{4, 4},
            TCatFeatureUniqueValuesCounts{5, 5},
            TCatFeatureUniqueValuesCounts{3, 3}
        };

        expectedData.ObjectsGrouping = TObjectsGrouping(5);

        expectedData.Target.TargetType = ERawTargetType::Float;

        TVector<TVector<TString>> rawTarget{{"0.12", "0", "0.45", "0.1", "0.22"}};
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.SetTrivialWeights(5);

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }

    Y_UNIT_TEST(ReadDatasetGroupData) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 6;
        srcData.LocalIndexToColumnIndex = {1, 2, 3, 4, 5, 0, 6, 7};
        srcData.PoolQuantizationSchema.FloatFeatureIndices = {0, 1, 2};
        srcData.PoolQuantizationSchema.Borders = {
            {0.1f, 0.2f, 0.3f, 0.4f},
            {0.25f, 0.5f, 0.75f, 0.95f},
            {0.2f, 0.5f, 0.55f, 0.82f},
        };
        srcData.PoolQuantizationSchema.NanModes = {
            ENanMode::Forbidden,
            ENanMode::Min,
            ENanMode::Forbidden
        };

        srcData.ColumnNames = {
            "f0",
            "f1",
            "f2",
            "GroupId",
            "SubgroupId",
            "Target",
            "Weight",
            "GroupWeight"
        };

        srcData.GroupIds = TSrcColumn<TGroupId>{EColumn::GroupId, {{2, 2}, {0, 11, 11}, {11}}};
        srcData.SubgroupIds = TSrcColumn<TSubgroupId>{EColumn::SubgroupId, {{1}, {22, 9, 12}, {22, 45}}};

        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{2, 3}, {4, 3, 1}, {0}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}));

        srcData.Target = TSrcColumn<float>{
            EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}, {0.42f}}
        };

        srcData.Weights = TSrcColumn<float>{
            EColumn::Weight,
            {{0.12f, 0.18f}, {1.0f, 0.45f, 1.0f}, {0.9f}}
        };
        srcData.GroupWeights = TSrcColumn<float>{
            EColumn::GroupWeight,
            {{1.0f, 1.0f}, {0.0f, 0.5f, 0.5f}, {0.5f}}
        };
        srcData.ObjectsOrder = EObjectsOrder::Ordered;

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Label, "Target"},
            {EColumn::Num, "f0"},
            {EColumn::Num, "f1"},
            {EColumn::Num, "f2"},
            {EColumn::GroupId, "GroupId"},
            {EColumn::SubgroupId, "SubgroupId"},
            {EColumn::Weight, "Weight"},
            {EColumn::GroupWeight, "GroupWeight"}
        };

        TVector<TString> featureId = {"f0", "f1", "f2"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        expectedData.Objects.Order = EObjectsOrder::Ordered;
        expectedData.Objects.GroupIds = {2, 2, 0, 11, 11, 11};
        expectedData.Objects.SubgroupIds = {1, 22, 9, 12, 22, 45};

        expectedData.Objects.FloatFeatures = {
            TVector<ui8>{1, 3, 0, 1, 2, 4},
            TVector<ui8>{2, 3, 4, 3, 1, 0},
            TVector<ui8>{0, 2, 3, 1, 4, 2}
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 4)
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(0),
            {0.1f, 0.2f, 0.3f, 0.4f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(1),
            {0.25f, 0.5f, 0.75f, 0.95f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(2),
            {0.2f, 0.5f, 0.55f, 0.82f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(1), ENanMode::Min);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(2), ENanMode::Forbidden);
        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );

        expectedData.ObjectsGrouping = TObjectsGrouping(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );

        expectedData.Target.TargetType = ERawTargetType::Float;

        TVector<TVector<TString>> rawTarget{{"0.12", "0", "0.45", "0.1", "0.22", "0.42"}};
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.Weights = TWeights<float>(
            TVector<float>{0.12f, 0.18f, 1.0f, 0.45f, 1.0f, 0.9f}
        );
        expectedData.Target.GroupWeights = TWeights<float>(
            TVector<float>{1.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f}
        );

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }

    Y_UNIT_TEST(ReadDatasetPairsOnly) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 6;
        srcData.LocalIndexToColumnIndex = {0, 1, 2, 3, 4};
        srcData.PoolQuantizationSchema.FloatFeatureIndices = {0, 1, 2};
        srcData.PoolQuantizationSchema.Borders = {
            {0.1f, 0.2f, 0.3f, 0.4f},
            {0.25f, 0.5f, 0.75f, 0.95f},
            {0.2f, 0.5f, 0.55f, 0.82f},
        };
        srcData.PoolQuantizationSchema.NanModes = {
            ENanMode::Forbidden,
            ENanMode::Min,
            ENanMode::Forbidden
        };

        srcData.ColumnNames = {
            "f0",
            "f1",
            "f2",
            "GroupId",
            "SubgroupId"
        };

        srcData.GroupIds = TSrcColumn<TGroupId>{EColumn::GroupId, {{2, 2}, {0, 11, 11}, {11}}};
        srcData.SubgroupIds = TSrcColumn<TSubgroupId>{EColumn::SubgroupId, {{1}, {22, 9, 12}, {22, 45}}};

        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{2, 3}, {4, 3, 1}, {0}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}));

        srcData.PairsFileData = TStringBuf(
            "0\t1\t0.1\n"
            "4\t3\t1.0\n"
            "3\t5\t0.2"
        );

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Num, "f0"},
            {EColumn::Num, "f1"},
            {EColumn::Num, "f2"},
            {EColumn::GroupId, "GroupId"},
            {EColumn::SubgroupId, "SubgroupId"}
        };

        TVector<TString> featureId = {"f0", "f1", "f2"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, true, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        expectedData.Objects.GroupIds = {2, 2, 0, 11, 11, 11};
        expectedData.Objects.SubgroupIds = {1, 22, 9, 12, 22, 45};

        expectedData.Objects.FloatFeatures = {
            TVector<ui8>{1, 3, 0, 1, 2, 4},
            TVector<ui8>{2, 3, 4, 3, 1, 0},
            TVector<ui8>{0, 2, 3, 1, 4, 2}
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 4)
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(0),
            {0.1f, 0.2f, 0.3f, 0.4f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(1),
            {0.25f, 0.5f, 0.75f, 0.95f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(2),
            {0.2f, 0.5f, 0.55f, 0.82f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(1), ENanMode::Min);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(2), ENanMode::Forbidden);
        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );

        expectedData.ObjectsGrouping = TObjectsGrouping(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );

        expectedData.Target.Weights = TWeights<float>(6);
        expectedData.Target.GroupWeights = TWeights<float>(6);
        expectedData.Target.Pairs = TFlatPairsInfo{TPair(0, 1, 0.1f), TPair(4, 3, 1.0f), TPair(3, 5, 0.2f)};

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }

    Y_UNIT_TEST(ReadDatasetSeparateGroupWeights) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 6;
        srcData.LocalIndexToColumnIndex = {1, 2, 3, 4, 0};
        srcData.PoolQuantizationSchema.FloatFeatureIndices = {0, 1, 2};
        srcData.PoolQuantizationSchema.Borders = {
            {0.1f, 0.2f, 0.3f, 0.4f},
            {0.25f, 0.5f, 0.75f, 0.95f},
            {0.2f, 0.5f, 0.55f, 0.82f},
        };
        srcData.PoolQuantizationSchema.NanModes = {
            ENanMode::Forbidden,
            ENanMode::Min,
            ENanMode::Forbidden
        };

        srcData.ColumnNames = {
            "f0",
            "f1",
            "f2",
            "GroupId",
            "Target"
        };

        srcData.GroupIds = TSrcColumn<TGroupId>{
            EColumn::GroupId,
            {
                {CalcGroupIdFor("query0"), CalcGroupIdFor("query0")},
                {CalcGroupIdFor("query1"), CalcGroupIdFor("Query 2"), CalcGroupIdFor("Query 2")},
                {CalcGroupIdFor("Query 2")}
            }
        };

        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{2, 3}, {4, 3, 1}, {0}}));
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}));

        srcData.Target = TSrcColumn<float>{
            EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}, {0.42f}}
        };
        srcData.GroupWeightsFileData = TStringBuf(
            "query0\t1.0\n"
            "query1\t0.0\n"
            "Query 2\t0.5"
        );

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Label, "Target"},
            {EColumn::Num, "f0"},
            {EColumn::Num, "f1"},
            {EColumn::Num, "f2"},
            {EColumn::GroupId, "GroupId"}
        };

        TVector<TString> featureId = {"f0", "f1", "f2"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, true, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        expectedData.Objects.GroupIds = {
            CalcGroupIdFor("query0"),
            CalcGroupIdFor("query0"),
            CalcGroupIdFor("query1"),
            CalcGroupIdFor("Query 2"),
            CalcGroupIdFor("Query 2"),
            CalcGroupIdFor("Query 2")
        };


        expectedData.Objects.FloatFeatures = {
            TVector<ui8>{1, 3, 0, 1, 2, 4},
            TVector<ui8>{2, 3, 4, 3, 1, 0},
            TVector<ui8>{0, 2, 3, 1, 4, 2}
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 4)
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(0), {0.1f, 0.2f, 0.3f, 0.4f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(1),
            {0.25f, 0.5f, 0.75f, 0.95f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(2),
            {0.2f, 0.5f, 0.55f, 0.82f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(1), ENanMode::Min);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(2), ENanMode::Forbidden);
        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );

        expectedData.ObjectsGrouping = TObjectsGrouping(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );


        expectedData.Target.TargetType = ERawTargetType::Float;

        TVector<TVector<TString>> rawTarget{{"0.12", "0", "0.45", "0.1", "0.22", "0.42"}};
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.Weights = TWeights<float>(6);
        expectedData.Target.GroupWeights = TWeights<float>(
            TVector<float>{1.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f}
        );

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }

    Y_UNIT_TEST(ReadDatasetIgnoredFeatures) {
        TTestCase testCase;
        NCB::TSrcData srcData;

        srcData.DocumentCount = 6;
        srcData.LocalIndexToColumnIndex = {1, 2, 3, 4, 5, 0};
        srcData.PoolQuantizationSchema.FloatFeatureIndices = {0, 2};
        srcData.PoolQuantizationSchema.Borders = {
            {0.1f, 0.2f, 0.3f, 0.4f},
            {0.2f, 0.5f, 0.55f, 0.82f}
        };
        srcData.PoolQuantizationSchema.NanModes = {
            ENanMode::Forbidden,
            ENanMode::Forbidden
        };

        srcData.ColumnNames = {
            "f0",
            "f1",
            "f2",
            "f3",
            "GroupId",
            "Target"
        };

        srcData.GroupIds = TSrcColumn<TGroupId>{EColumn::GroupId, {{2, 2}, {0, 11, 11}, {11}}};

        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}));
        srcData.FloatFeatures.push_back(nullptr);
        srcData.FloatFeatures.push_back(MakeFeaturesColumn<ui8>(EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}));
        srcData.FloatFeatures.push_back(nullptr);

        srcData.Target = TSrcColumn<float>{
            EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}, {0.42f}}
        };

        srcData.IgnoredColumnIndices = {2, 4};

        testCase.SrcData = std::move(srcData);


        TExpectedQuantizedData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Label, "Target"},
            {EColumn::Num, "f0"},
            {EColumn::Num, "f1"},
            {EColumn::Num, "f2"},
            {EColumn::Num, "f3"},
            {EColumn::GroupId, "GroupId"}
        };

        TVector<TString> featureId = {"f0", "f1", "f2", "f3"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        auto& featuresLayout = *expectedData.MetaInfo.FeaturesLayout;
        featuresLayout.IgnoreExternalFeature(1);
        featuresLayout.IgnoreExternalFeature(3);

        expectedData.Objects.GroupIds = {2, 2, 0, 11, 11, 11};

        expectedData.Objects.FloatFeatures = {
            TVector<ui8>{1, 3, 0, 1, 2, 4},
            Nothing(),
            TVector<ui8>{0, 2, 3, 1, 4, 2},
            Nothing()
        };
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 4)
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(0),
            {0.1f, 0.2f, 0.3f, 0.4f}
        );
        expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
            TFloatFeatureIdx(2),
            {0.2f, 0.5f, 0.55f, 0.82f}
        );


        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(0), ENanMode::Forbidden);
        expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(2), ENanMode::Forbidden);
        expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TExclusiveFeaturesBundle>()
        );
        expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *expectedData.MetaInfo.FeaturesLayout,
            *expectedData.Objects.QuantizedFeaturesInfo,
            expectedData.Objects.ExclusiveFeatureBundlesData
        );
        expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
            *expectedData.MetaInfo.FeaturesLayout,
            TVector<TFeaturesGroup>()
        );

        expectedData.ObjectsGrouping = TObjectsGrouping(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );

        expectedData.Target.TargetType = ERawTargetType::Float;

        TVector<TVector<TString>> rawTarget{{"0.12", "0", "0.45", "0.1", "0.22", "0.42"}};
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.Weights = TWeights<float>(6);
        expectedData.Target.GroupWeights = TWeights<float>(6);

        testCase.ExpectedData = std::move(expectedData);

        Test(testCase);
    }


    template <class T, class GenFunc>
    TVector<T> GenerateData(ui32 size, GenFunc&& genFunc) {
        TVector<T> result;
        result.yresize(size);
        for (auto i : xrange(size)) {
            result[i] = genFunc(i);
        }
        return result;
    }


    Y_UNIT_TEST(ReadDatasetMidSize) {
        for (bool hasFloatFeatures : {false, true}) {
            for (bool hasCatFeatures : {false, true}) {
                if (!hasFloatFeatures && !hasCatFeatures) {
                    continue;
                }

                // for this test case we set some srcData from expectedData explicitly, because data is big
                TTestCase testCase;

                NCB::TSrcData srcData;
                TExpectedQuantizedData expectedData;

                const ui32 binCount = 5;
                const ui32 floatFeatureCount = hasFloatFeatures ? 300 : 0;
                const ui32 catFeatureCount = hasCatFeatures ? 300 : 0;
                const ui32 featureCount = floatFeatureCount + catFeatureCount;

                srcData.DocumentCount = 100000;

                for (auto featureIdx : xrange(floatFeatureCount)) {
                    srcData.LocalIndexToColumnIndex.push_back(featureIdx + 2);
                    srcData.PoolQuantizationSchema.FloatFeatureIndices.push_back(featureIdx);
                }
                for (auto featureIdx : xrange(floatFeatureCount, featureCount)) {
                    srcData.LocalIndexToColumnIndex.push_back(featureIdx + 2);
                    srcData.PoolQuantizationSchema.CatFeatureIndices.push_back(featureIdx);
                }

                for (auto floatFeatureIdx : xrange(floatFeatureCount)) {
                    Y_UNUSED(floatFeatureIdx);
                    srcData.PoolQuantizationSchema.Borders.push_back({0.1f, 0.2f, 0.3f, 0.4f});
                    srcData.PoolQuantizationSchema.NanModes.push_back(ENanMode::Forbidden);
                }
                for (auto catFeatureIdx : xrange(catFeatureCount)) {
                    Y_UNUSED(catFeatureIdx);

                    TMap<ui32, TValueWithCount> perfectHash;
                    perfectHash.emplace(0x10, TValueWithCount{0, 2});
                    perfectHash.emplace(0x2F, TValueWithCount{1, 1});
                    perfectHash.emplace(0xF3, TValueWithCount{2, 3});
                    perfectHash.emplace(0x23, TValueWithCount{3, 7});
                    perfectHash.emplace(0xEA, TValueWithCount{4, 4});

                    srcData.PoolQuantizationSchema.FeaturesPerfectHash.push_back(perfectHash);
                }

                for (auto floatFeatureIdx : xrange(floatFeatureCount)) {
                    srcData.ColumnNames.push_back("f" + ToString(floatFeatureIdx));

                    expectedData.Objects.FloatFeatures.push_back(
                        GenerateData<ui8>(
                            srcData.DocumentCount,
                            [&](ui32 /*i*/) { return RandomNumber<ui8>(binCount); }
                        )
                    );
                    srcData.FloatFeatures.emplace_back(
                        new TSrcColumn<ui8>(
                            NCB::GenerateSrcColumn<ui8>(
                                std::get<TVector<ui8>>(*expectedData.Objects.FloatFeatures.back()),
                                EColumn::Num
                            )
                        )
                    );
                }
                for (auto catFeatureIdx : xrange(catFeatureCount)) {
                    srcData.ColumnNames.push_back("c" + ToString(catFeatureIdx));

                    TVector<ui8> data = GenerateData<ui8>(
                        srcData.DocumentCount,
                        [&](ui32 /*i*/) { return RandomNumber<ui8>(binCount); }
                    );

                    expectedData.Objects.CatFeatures.push_back(TVector<ui32>(data.begin(), data.end()));
                    srcData.CatFeatures.emplace_back(
                        new TSrcColumn<ui8>(NCB::GenerateSrcColumn<ui8>(data, EColumn::Categ))
                    );
                }

                srcData.ColumnNames.push_back("GroupId");
                srcData.LocalIndexToColumnIndex.push_back(1);

                expectedData.Objects.GroupIds = GenerateData<TGroupId>(
                    srcData.DocumentCount,
                    [] (ui32 i) { return i / 5; }
                );
                srcData.GroupIds = NCB::GenerateSrcColumn<TGroupId>(
                    *expectedData.Objects.GroupIds,
                    EColumn::GroupId
                );

                srcData.ColumnNames.push_back("Target");
                srcData.LocalIndexToColumnIndex.push_back(0);

                TVector<float> target = GenerateData<float>(
                    srcData.DocumentCount,
                    [] (ui32 /*i*/) { return RandomNumber<float>(); }
                );
                srcData.Target = NCB::GenerateSrcColumn<float>(target, EColumn::Label);

                expectedData.Target.TargetType = ERawTargetType::Float;

                size_t sumOfStringSizes = 0;
                TVector<TVector<TString>> rawTarget{
                    GenerateData<TString>(
                        srcData.DocumentCount,
                        [&] (ui32 i) {
                            auto targetString = Sprintf("%.9e", target[i]);
                            sumOfStringSizes += targetString.size();
                            return targetString;
                        }
                    )
                };
                expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());

                TDataColumnsMetaInfo dataColumnsMetaInfo;
                dataColumnsMetaInfo.Columns = {
                    {EColumn::Label, "Target"},
                    {EColumn::GroupId, "GroupId"}
                };

                TVector<TString> featureId;
                for (auto i : xrange(floatFeatureCount)) {
                    featureId.push_back("f" + ToString(i));
                    dataColumnsMetaInfo.Columns.push_back({EColumn::Num, featureId.back()});
                }
                for (auto i : xrange(catFeatureCount)) {
                    featureId.push_back("c" + ToString(i));
                    dataColumnsMetaInfo.Columns.push_back({EColumn::Categ, featureId.back()});
                }

                expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
                expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                    *expectedData.MetaInfo.FeaturesLayout,
                    TConstArrayRef<ui32>(),
                    NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, hasFloatFeatures ? 4 : 32)
                );
                expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn = hasCatFeatures ? 5 : 0;

                for (auto i : xrange(floatFeatureCount)) {
                    auto floatFeatureIdx = TFloatFeatureIdx(i);
                    expectedData.Objects.QuantizedFeaturesInfo->SetBorders(
                        floatFeatureIdx,
                        TVector<float>(srcData.PoolQuantizationSchema.Borders[i])
                    );
                    expectedData.Objects.QuantizedFeaturesInfo->SetNanMode(
                        floatFeatureIdx,
                        srcData.PoolQuantizationSchema.NanModes[i]
                    );
                }
                for (auto i : xrange(catFeatureCount)) {
                    auto catFeatureIdx = TCatFeatureIdx(i);

                    TCatFeaturePerfectHash perfectHash;
                    perfectHash.Map = srcData.PoolQuantizationSchema.FeaturesPerfectHash[i];

                    expectedData.Objects.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
                        catFeatureIdx,
                        std::move(perfectHash)
                    );
                }
                expectedData.Objects.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                    *expectedData.MetaInfo.FeaturesLayout,
                    TVector<TExclusiveFeaturesBundle>()
                );
                expectedData.Objects.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                    *expectedData.MetaInfo.FeaturesLayout,
                    *expectedData.Objects.QuantizedFeaturesInfo,
                    expectedData.Objects.ExclusiveFeatureBundlesData
                );
                expectedData.Objects.FeatureGroupsData = TFeatureGroupsData(
                    *expectedData.MetaInfo.FeaturesLayout,
                    TVector<TFeaturesGroup>()
                );
                if (hasCatFeatures) {
                    TVector<TCatFeatureUniqueValuesCounts> catFeatureUniqueValuesCounts(
                        catFeatureCount,
                        TCatFeatureUniqueValuesCounts{5, 5}
                    );
                    expectedData.Objects.CatFeatureUniqueValuesCounts = catFeatureUniqueValuesCounts;
                }

                TVector<TGroupBounds> groupsBounds;
                for (auto groupIdx : xrange<ui32>(srcData.DocumentCount / 5)) {
                    groupsBounds.push_back(TGroupBounds{groupIdx * 5, (groupIdx + 1) * 5});
                }
                expectedData.ObjectsGrouping = TObjectsGrouping(std::move(groupsBounds));
                expectedData.Target.SetTrivialWeights(srcData.DocumentCount);


                testCase.SrcData = std::move(srcData);
                testCase.ExpectedData = std::move(expectedData);

                Test(testCase);
            }
        }
    }
}
