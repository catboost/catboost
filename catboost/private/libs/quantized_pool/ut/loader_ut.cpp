
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
#include <util/system/mktemp.h>

#include <library/cpp/unittest/registar.h>


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
            readDatasetMainParams.GroupWeightsFilePath, // can be uninited
            /*timestampsFilePath*/TPathWithScheme(),
            readDatasetMainParams.BaselineFilePath, // can be uninited
            /*featureNamesPath*/TPathWithScheme(),
            NCatboostOptions::TColumnarPoolFormatParams(),
            testCase.SrcData.IgnoredFeatures,
            testCase.SrcData.ObjectsOrder,
            TDatasetSubset::MakeColumns(),
            &readDatasetMainParams.ClassLabels,
            &localExecutor
        );

        Compare<TQuantizedForCPUObjectsDataProvider>(std::move(dataProvider), testCase.ExpectedData);
    }


    Y_UNIT_TEST(ReadDataset) {
        TVector<TTestCase> testCases;

        {
            TTestCase simpleTestCase;
            NCB::TSrcData srcData;

            srcData.DocumentCount = 5;
            srcData.LocalIndexToColumnIndex = {0, 1, 2};
            srcData.PoolQuantizationSchema.FeatureIndices = {0, 1};
            srcData.PoolQuantizationSchema.Borders = {{0.1f, 0.2f, 0.3f}, {0.25f, 0.5f, 0.75f}};
            srcData.PoolQuantizationSchema.NanModes = {ENanMode::Forbidden, ENanMode::Min};
            srcData.FloatFeatures = {
                TSrcColumn<ui8>{EColumn::Num, {{1, 3}, {0, 1, 2}}},
                TSrcColumn<ui8>{EColumn::Num, {{2, 3}, {0, 3, 1}}}
            };

            srcData.Target = TSrcColumn<float>{EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}}};

            simpleTestCase.SrcData = std::move(srcData);


            TExpectedQuantizedData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Label, ""}
            };

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, /* additionalBaselineCount */ Nothing(), Nothing());
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

            simpleTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(simpleTestCase));
        }

        {
            TTestCase groupDataTestCase;
            NCB::TSrcData srcData;

            srcData.DocumentCount = 6;
            srcData.LocalIndexToColumnIndex = {1, 2, 3, 4, 5, 0, 6, 7};
            srcData.PoolQuantizationSchema.FeatureIndices = {0, 1, 2};
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
                "GroupId",
                "SubgroupId",
                "f0",
                "f1",
                "f2",
                "Target",
                "Weight",
                "GroupWeight"
            };

            srcData.GroupIds = TSrcColumn<TGroupId>{EColumn::GroupId, {{2, 2}, {0, 11, 11}, {11}}};
            srcData.SubgroupIds = TSrcColumn<TSubgroupId>{EColumn::SubgroupId, {{1}, {22, 9, 12}, {22, 45}}};

            srcData.FloatFeatures = {
                TSrcColumn<ui8>{EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}},
                TSrcColumn<ui8>{EColumn::Num, {{2, 3}, {4, 3, 1}, {0}}},
                TSrcColumn<ui8>{EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}}
            };

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

            groupDataTestCase.SrcData = std::move(srcData);


            TExpectedQuantizedData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, "Target"},
                {EColumn::GroupId, "GroupId"},
                {EColumn::SubgroupId, "SubgroupId"},
                {EColumn::Num, "f0"},
                {EColumn::Num, "f1"},
                {EColumn::Num, "f2"},
                {EColumn::Weight, "Weight"},
                {EColumn::GroupWeight, "GroupWeight"}
            };

            TVector<TString> featureId = {"f0", "f1", "f2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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

            groupDataTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(groupDataTestCase));
        }

        {
            TTestCase pairsOnlyTestCase;
            NCB::TSrcData srcData;

            srcData.DocumentCount = 6;
            srcData.LocalIndexToColumnIndex = {0, 1, 2, 3, 4};
            srcData.PoolQuantizationSchema.FeatureIndices = {0, 1, 2};
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
                "GroupId",
                "SubgroupId",
                "f0",
                "f1",
                "f2"
            };

            srcData.GroupIds = TSrcColumn<TGroupId>{EColumn::GroupId, {{2, 2}, {0, 11, 11}, {11}}};
            srcData.SubgroupIds = TSrcColumn<TSubgroupId>{EColumn::SubgroupId, {{1}, {22, 9, 12}, {22, 45}}};

            srcData.FloatFeatures = {
                TSrcColumn<ui8>{EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}},
                TSrcColumn<ui8>{EColumn::Num, {{2, 3}, {4, 3, 1}, {0}}},
                TSrcColumn<ui8>{EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}}
            };
            srcData.PairsFileData = AsStringBuf(
                "0\t1\t0.1\n"
                "4\t3\t1.0\n"
                "3\t5\t0.2"
            );

            pairsOnlyTestCase.SrcData = std::move(srcData);


            TExpectedQuantizedData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::GroupId, "GroupId"},
                {EColumn::SubgroupId, "SubgroupId"},
                {EColumn::Num, "f0"},
                {EColumn::Num, "f1"},
                {EColumn::Num, "f2"}
            };

            TVector<TString> featureId = {"f0", "f1", "f2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, true, /* additionalBaselineCount */ Nothing(), &featureId);
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
            expectedData.Target.Pairs = {TPair(0, 1, 0.1f), TPair(4, 3, 1.0f), TPair(3, 5, 0.2f)};

            pairsOnlyTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(pairsOnlyTestCase));
        }


        {
            TTestCase separateGroupWeightsTestCase;
            NCB::TSrcData srcData;

            srcData.DocumentCount = 6;
            srcData.LocalIndexToColumnIndex = {1, 2, 3, 4, 0};
            srcData.PoolQuantizationSchema.FeatureIndices = {0, 1, 2};
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
                "GroupId",
                "f0",
                "f1",
                "f2",
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

            srcData.FloatFeatures = {
                TSrcColumn<ui8>{EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}},
                TSrcColumn<ui8>{EColumn::Num, {{2, 3}, {4, 3, 1}, {0}}},
                TSrcColumn<ui8>{EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}}
            };

            srcData.Target = TSrcColumn<float>{
                EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}, {0.42f}}
            };
            srcData.GroupWeightsFileData = AsStringBuf(
                "query0\t1.0\n"
                "query1\t0.0\n"
                "Query 2\t0.5"
            );

            separateGroupWeightsTestCase.SrcData = std::move(srcData);


            TExpectedQuantizedData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, "Target"},
                {EColumn::GroupId, "GroupId"},
                {EColumn::Num, "f0"},
                {EColumn::Num, "f1"},
                {EColumn::Num, "f2"}
            };

            TVector<TString> featureId = {"f0", "f1", "f2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, true, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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

            separateGroupWeightsTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(separateGroupWeightsTestCase));
        }

        {
            TTestCase ignoredFeaturesTestCase;
            NCB::TSrcData srcData;

            srcData.DocumentCount = 6;
            srcData.LocalIndexToColumnIndex = {1, 2, 3, 4, 5, 0};
            srcData.PoolQuantizationSchema.FeatureIndices = {0, 2};
            srcData.PoolQuantizationSchema.Borders = {
                {0.1f, 0.2f, 0.3f, 0.4f},
                {0.2f, 0.5f, 0.55f, 0.82f}
            };
            srcData.PoolQuantizationSchema.NanModes = {
                ENanMode::Forbidden,
                ENanMode::Forbidden
            };

            srcData.ColumnNames = {
                "GroupId",
                "f0",
                "f1",
                "f2",
                "f3",
                "Target"
            };

            srcData.GroupIds = TSrcColumn<TGroupId>{EColumn::GroupId, {{2, 2}, {0, 11, 11}, {11}}};

            srcData.FloatFeatures = {
                TSrcColumn<ui8>{EColumn::Num, {{1, 3}, {0, 1, 2}, {4}}},
                Nothing(),
                TSrcColumn<ui8>{EColumn::Num, {{0, 2, 3, 1}, {4}, {2}}},
                Nothing()
            };

            srcData.Target = TSrcColumn<float>{
                EColumn::Label, {{0.12f, 0.0f}, {0.45f, 0.1f, 0.22f}, {0.42f}}
            };

            srcData.IgnoredColumnIndices = {3, 5};

            ignoredFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedQuantizedData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, "Target"},
                {EColumn::GroupId, "GroupId"},
                {EColumn::Num, "f0"},
                {EColumn::Num, "f1"},
                {EColumn::Num, "f2"},
                {EColumn::Num, "f3"}
            };

            TVector<TString> featureId = {"f0", "f1", "f2", "f3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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

            ignoredFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(ignoredFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            Test(testCase);
        }
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
        TTestCase testCase;

        // for this test case we set some srcData from expectedData explicitly, because data is big

        NCB::TSrcData srcData;
        TExpectedQuantizedData expectedData;

        const ui32 binCount = 5;
        const ui32 featureCount = 300;

        srcData.DocumentCount = 100000;
        srcData.LocalIndexToColumnIndex = {1};
        for (auto featureIdx : xrange(featureCount)) {
            srcData.LocalIndexToColumnIndex.push_back(featureIdx + 2);
            srcData.PoolQuantizationSchema.FeatureIndices.push_back(featureIdx);
        }
        srcData.LocalIndexToColumnIndex.push_back(0);

        for (auto featureIdx : xrange(featureCount)) {
            Y_UNUSED(featureIdx);
            srcData.PoolQuantizationSchema.Borders.push_back({0.1f, 0.2f, 0.3f, 0.4f});
            srcData.PoolQuantizationSchema.NanModes.push_back(ENanMode::Forbidden);
        }

        srcData.ColumnNames = {
            "GroupId"
        };

        expectedData.Objects.GroupIds = GenerateData<TGroupId>(
            srcData.DocumentCount,
            [] (ui32 i) { return i / 5; }
        );
        srcData.GroupIds = NCB::GenerateSrcColumn<TGroupId>(
            *expectedData.Objects.GroupIds,
            EColumn::GroupId
        );

        for (auto featureIdx : xrange(featureCount)) {
            srcData.ColumnNames.push_back("f" + ToString(featureIdx));

            expectedData.Objects.FloatFeatures.push_back(
                GenerateData<ui8>(
                    srcData.DocumentCount,
                    [&](ui32 /*i*/) { return RandomNumber<ui8>(binCount); }
                )
            );
            srcData.FloatFeatures.push_back(
                NCB::GenerateSrcColumn<ui8>(
                    Get<TVector<ui8>>(*expectedData.Objects.FloatFeatures.back()),
                    EColumn::Num
                )
            );
        }
        srcData.ColumnNames.push_back("Target");

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
        for (auto i : xrange(featureCount)) {
            featureId.push_back("f" + ToString(i));
            dataColumnsMetaInfo.Columns.push_back({EColumn::Num, featureId.back()});
        }

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::Float, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        expectedData.Objects.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *expectedData.MetaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 4)
        );

        for (auto i : xrange(featureCount)) {
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
