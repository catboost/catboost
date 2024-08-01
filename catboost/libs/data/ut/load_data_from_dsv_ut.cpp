#include "util.h"

#include <catboost/libs/data/ut/lib/for_data_provider.h>
#include <catboost/libs/data/ut/lib/for_loader.h>

#include <catboost/libs/data/load_data.h>

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects_grouping.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>

#include <library/cpp/testing/unittest/registar.h>

#include <functional>
#include <limits>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(LoadDataFromDsv) {

    Y_UNIT_TEST(ReadDataset) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase simpleTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tTarget\n"sv;
            srcData.DatasetFileData =
                "Target0\tTarget1\tFeat\n"
                "0\t0.1\t0.2\n"
                "1\t0.97\t0.82\n"
                "0\t0.13\t0.22\n"sv;
            srcData.DsvFileHasHeader = true;
            simpleTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Label, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.2f, 0.82f, 0.22f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}, {"0.1", "0.97", "0.13"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            simpleTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(simpleTestCase));
        }

        {
            TReadDatasetTestCase simpleTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\n"
                "0\t0.1\t0.2\n"
                "1\t0.97\t0.82\n"
                "0\t0.13\t0.22\n"sv;
            srcData.DsvFileHasHeader = true;
            simpleTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f},
                TVector<float>{0.2f, 0.82f, 0.22f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            simpleTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(simpleTestCase));
        }

        {
            TReadDatasetTestCase groupDataTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tSubgroupId\n"
                "3\tWeight\n"
                "4\tGroupWeight\n"
                "5\tNum\tf0\n"
                "6\tNum\tf1\n"
                "7\tNum\tf2\n"sv;
            srcData.DatasetFileData =
                "0.12\tquery0\tsite1\t0.12\t1.0\t0.1\t0.2\t0.11\n"
                "0.22\tquery0\tsite22\t0.18\t1.0\t0.97\t0.82\t0.33\n"
                "0.34\tquery1\tSite9\t1.0\t0.0\t0.13\t0.22\t0.23\n"
                "0.42\tQuery 2\tsite12\t0.45\t0.5\t0.14\t0.18\t0.1\n"
                "0.01\tQuery 2\tsite22\t1.0\t0.5\t0.9\t0.67\t0.17\n"
                "0.0\tQuery 2\tSite45\t2.0\t0.5\t0.66\t0.1\t0.31\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Ordered;
            groupDataTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::GroupId, ""},
                {EColumn::SubgroupId, ""},
                {EColumn::Weight, ""},
                {EColumn::GroupWeight, ""},
                {EColumn::Num, "f0"},
                {EColumn::Num, "f1"},
                {EColumn::Num, "f2"},
            };

            TVector<TString> featureId = {"f0", "f1", "f2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Ordered;
            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "query0",
                "query0",
                "query1",
                "Query 2",
                "Query 2",
                "Query 2"
            };
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
                "site1",
                "site22",
                "Site9",
                "site12",
                "site22",
                "Site45"
            };
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
            );
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.42", "0.01", "0.0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(
                TVector<float>{0.12f, 0.18f, 1.0f, 0.45f, 1.0f, 2.0f}
            );
            expectedData.Target.GroupWeights = TWeights<float>(
                TVector<float>{1.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f}
            );

            groupDataTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(groupDataTestCase));
        }

        {
            TReadDatasetTestCase pairsOnlyTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tGroupId\n"
                "1\tSubgroupId\n"
                "2\tNum\tf0\n"
                "3\tNum\tf1\n"
                "4\tNum\tf2\n"sv;
            srcData.DatasetFileData =
                "query0\tsite1\t0.1\t0.2\t0.11\n"
                "query0\tsite22\t0.97\t0.82\t0.33\n"
                "query1\tSite9\t0.13\t0.22\t0.23\n"
                "Query 2\tsite12\t0.14\t0.18\t0.1\n"
                "Query 2\tsite22\t0.9\t0.67\t0.17\n"
                "Query 2\tSite45\t0.66\t0.1\t0.31\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.PairsFileData =
                "0\t1\t0.1\n"
                "4\t3\t1.0\n"
                "3\t5\t0.2"sv;
            pairsOnlyTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::GroupId, ""},
                {EColumn::SubgroupId, ""},
                {EColumn::Num, "f0"},
                {EColumn::Num, "f1"},
                {EColumn::Num, "f2"},
            };

            TVector<TString> featureId = {"f0", "f1", "f2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, true, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "query0",
                "query0",
                "query1",
                "Query 2",
                "Query 2",
                "Query 2"
            };
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
                "site1",
                "site22",
                "Site9",
                "site12",
                "site22",
                "Site45"
            };
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
            );
            expectedData.Target.Weights = TWeights<float>(6);
            expectedData.Target.GroupWeights = TWeights<float>(6);
            expectedData.Target.Pairs = TFlatPairsInfo{
                TPair(0, 1, 0.1f),
                TPair(4, 3, 1.0f),
                TPair(3, 5, 0.2f)
            };

            pairsOnlyTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(pairsOnlyTestCase));
        }

        {
            TReadDatasetTestCase floatAndCatFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tNum\tfloat0\n"
                "3\tCateg\tGender1\n"
                "4\tNum\tfloat2\n"
                "5\tCateg\tCountry3\n"
                "6\tNum\tfloat4\n"sv;
            srcData.DatasetFileData =
                "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n"
                "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n"
                "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n"
                "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n"
                "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n"
                "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::RandomShuffled;
            floatAndCatFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::GroupId, ""},
                {EColumn::Num, "float0"},
                {EColumn::Categ, "Gender1"},
                {EColumn::Num, "float2"},
                {EColumn::Categ, "Country3"},
                {EColumn::Num, "float4"},
            };

            TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3", "float4"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::RandomShuffled;
            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "query0",
                "query0",
                "query1",
                "Query 2",
                "Query 2",
                "Query 2"
            };
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Male", "Male", "Female", "Female"},
                TVector<TStringBuf>{"Germany", "Russia", "USA", "Finland", "USA", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
            );
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.42", "0.01", "0.0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(6);
            expectedData.Target.GroupWeights = TWeights<float>(6);

            floatAndCatFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(floatAndCatFeaturesTestCase));
        }

        {
            TReadDatasetTestCase separateGroupWeightsTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tNum\tfloat0\n"
                "3\tCateg\tGender1\n"
                "4\tNum\tfloat2\n"
                "5\tCateg\tCountry3\n"
                "6\tNum\tfloat4\n"sv;
            srcData.DatasetFileData =
                "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n"
                "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n"
                "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n"
                "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n"
                "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n"
                "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.GroupWeightsFileData =
                "query0\t1.0\n"
                "query1\t0.0\n"
                "Query 2\t0.5"sv;
            separateGroupWeightsTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::GroupId, ""},
                {EColumn::Num, "float0"},
                {EColumn::Categ, "Gender1"},
                {EColumn::Num, "float2"},
                {EColumn::Categ, "Country3"},
                {EColumn::Num, "float4"},
            };

            TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3", "float4"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, true, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "query0",
                "query0",
                "query1",
                "Query 2",
                "Query 2",
                "Query 2"
            };
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Male", "Male", "Female", "Female"},
                TVector<TStringBuf>{"Germany", "Russia", "USA", "Finland", "USA", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
            );
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.42", "0.01", "0.0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(6);
            expectedData.Target.GroupWeights = TWeights<float>(TVector<float>{1.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f});

            separateGroupWeightsTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(separateGroupWeightsTestCase));
        }

        {
            TReadDatasetTestCase ignoredFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tNum\tfloat0\n"
                "3\tCateg\tGender1\n"
                "4\tNum\tfloat2\n"
                "5\tCateg\tCountry3\n"
                "6\tNum\tfloat4\n"sv;
            srcData.DatasetFileData =
                "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n"
                "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n"
                "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n"
                "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n"
                "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n"
                "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.IgnoredFeatures = {1, 4};
            ignoredFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::GroupId, ""},
                {EColumn::Num, "float0"},
                {EColumn::Categ, "Gender1"},
                {EColumn::Num, "float2"},
                {EColumn::Categ, "Country3"},
                {EColumn::Num, "float4"},
            };

            TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3", "float4"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            auto& featuresLayout = *expectedData.MetaInfo.FeaturesLayout;
            featuresLayout.IgnoreExternalFeature(1);
            featuresLayout.IgnoreExternalFeature(4);

            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "query0",
                "query0",
                "query1",
                "Query 2",
                "Query 2",
                "Query 2"
            };
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
                Nothing()
            };
            expectedData.Objects.CatFeatures = {
                Nothing(),
                TVector<TStringBuf>{"Germany", "Russia", "USA", "Finland", "USA", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
            );
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.42", "0.01", "0.0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(6);
            expectedData.Target.GroupWeights = TWeights<float>(6);

            ignoredFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(ignoredFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithTimestamp) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase orderedByTimestampTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tTimestamp"sv;
            srcData.DatasetFileData =
                "Target\tTimestamp\tFeat0\tFeat1\n"
                "0\t10\t0.1\t0.2\n"
                "1\t10\t0.97\t0.82\n"
                "0\t20\t0.13\t0.22\n"sv;
            srcData.DsvFileHasHeader = true;
            orderedByTimestampTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Timestamp, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Ordered;
            expectedData.Objects.Timestamp = {10, 10, 20};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f},
                TVector<float>{0.2f, 0.82f, 0.22f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            orderedByTimestampTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(orderedByTimestampTestCase));
        }

        {
            TReadDatasetTestCase notOrderedByTimestampTestCase1;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tTimestamp"sv;
            srcData.DatasetFileData =
                "Target\tTimestamp\tFeat0\tFeat1\n"
                "0\t20\t0.1\t0.2\n"
                "1\t10\t0.97\t0.82\n"
                "0\t20\t0.13\t0.22\n"sv;
            srcData.DsvFileHasHeader = true;
            notOrderedByTimestampTestCase1.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Timestamp, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;
            expectedData.Objects.Timestamp = {20, 10, 20};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f},
                TVector<float>{0.2f, 0.82f, 0.22f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            notOrderedByTimestampTestCase1.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(notOrderedByTimestampTestCase1));
        }

        {
            TReadDatasetTestCase notOrderedByTimestampTestCase2;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tTimestamp"sv;
            srcData.DatasetFileData =
                "Target\tTimestamp\tFeat0\tFeat1\n"
                "0\t20\t0.1\t0.2\n"
                "1\t20\t0.97\t0.82\n"
                "0\t20\t0.13\t0.22\n"sv;
            srcData.DsvFileHasHeader = true;
            notOrderedByTimestampTestCase2.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Timestamp, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;
            expectedData.Objects.Timestamp = {20, 20, 20};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.13f},
                TVector<float>{0.2f, 0.82f, 0.22f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            notOrderedByTimestampTestCase2.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(notOrderedByTimestampTestCase2));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithMissingValues) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase floatAndCatFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNum\tfloat0\n"
                "2\tCateg\tGender1\n"
                "3\tNum\tfloat2\n"
                "4\tCateg\tCountry3\n"sv;
            srcData.DatasetFileData =
                "0.12\t0.1\tNan\t0.2\tGermany\n"
                "0.22\t\t\tNA\tRussia\n"
                "0.341\tnan\tMale\t0.22\tN/A\n"
                "None\t0.14\tMale\tNULL\tFinland\n"
                "0.01\tna\tFemale\tNaN\tUSA\n"
                "0.0\t0.66\t#NA\t0.1\tNone\n"
                "N/A\tNone\tFemale\t0.12\tNULL\n"
                "0.11\t-1.#QNAN\tN/a\t1.#IND\t1.#IND\n"
                "-\t#N/A N/A\t#N/A N/A\t-\t-\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            floatAndCatFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, "float0"},
                {EColumn::Categ, "Gender1"},
                {EColumn::Num, "float2"},
                {EColumn::Categ, "Country3"},
            };

            TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            auto nanValue = std::numeric_limits<float>::quiet_NaN();

            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, nanValue, nanValue, 0.14f, nanValue, 0.66f, nanValue, nanValue, nanValue},
                TVector<float>{0.2f, nanValue, 0.22f, nanValue, nanValue, 0.1f, 0.12f, nanValue, nanValue}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Nan", "", "Male", "Male", "Female", "#NA", "Female", "N/a", "#N/A N/A"},
                TVector<TStringBuf>{"Germany", "Russia", "N/A", "Finland", "USA", "None", "NULL", "1.#IND", "-"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(9);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.341", "None", "0.01", "0.0", "N/A", "0.11", "-"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(9);
            expectedData.Target.GroupWeights = TWeights<float>(9);

            floatAndCatFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(floatAndCatFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithQuotedValues) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase floatAndCatFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNum\tfloat0\n"
                "2\tCateg\tPlace1\n"
                "3\tNum\tfloat2\n"sv;
            srcData.DatasetFileData =
                "0\t0\t\"Berlin, Germany\"\t0\n"
                "0\t0\t\"\"\"Skolkovo\"\", Russia\"\t0\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            floatAndCatFeaturesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, "float0"},
                {EColumn::Categ, "Place1"},
                {EColumn::Num, "float2"},
            };

            TVector<TString> featureId = {"float0", "Place1", "float2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float>{0, 0},
                TVector<float>{0, 0}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Berlin, Germany", "\"Skolkovo\", Russia"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(2);
            expectedData.Target.GroupWeights = TWeights<float>(2);

            floatAndCatFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(floatAndCatFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithQuotedHeadersAndQuotedValues) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase floatAndCatFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNum\tfloat0\n"
                "2\tCateg\tPlace1\n"
                "3\tNum\tfloat2\n"sv;
            srcData.DatasetFileData =
                "Target\t\"just \"\"float\"\"\"\t\"Place, Country\"\t\"second\tfloat\"\n"
                "0\t0\t\"Berlin, Germany\"\t0\n"
                "0\t0\t\"\"\"Skolkovo\"\", Russia\"\t0\n"sv;
            srcData.DsvFileHasHeader = true;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            floatAndCatFeaturesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, "float0"},
                {EColumn::Categ, "Place1"},
                {EColumn::Num, "float2"},
            };

            TVector<TString> featureId = {"float0", "Place1", "float2"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float>{0, 0},
                TVector<float>{0, 0}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Berlin, Germany", "\"Skolkovo\", Russia"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(2);
            expectedData.Target.GroupWeights = TWeights<float>(2);

            floatAndCatFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(floatAndCatFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithTextColumns) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase oneTextFeatureTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tText\ttext0\n"sv;
            srcData.DatasetFileData =
                "0.12\tWhat\n"
                "0.22\tnoise\n"
                "0.34\tannoys\n"
                "0.23\tan oyster\n"
                "0.99\t\n"
                "0.01\tmost\n"
                "0.02\t?\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            oneTextFeatureTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Text, "text0"},
            };

            TVector<TString> featureId = {"text0"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.TextFeatures = {
                TVector<TStringBuf>{"What", "noise", "annoys", "an oyster", "", "most", "?"},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(7);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.23", "0.99", "0.01", "0.02"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(7);
            expectedData.Target.GroupWeights = TWeights<float>(7);

            oneTextFeatureTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(oneTextFeatureTestCase));
        }
        {
            TReadDatasetTestCase textFloatAndCatFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tText\tName\n"
                "2\tCateg\tCountry\n"
                "3\tNum\tAge\n"
                "4\tText\tFavouriteMusic\n"
                "5\tCateg\tGender\n"sv;
            srcData.DatasetFileData =
                "0.12\tSpiderman\tUSA\t18\tjazz\tMale\n"
                "0.22\tWonderwoman\tEngland\t20\tsoul\tFemale\n"
                "0.34\tBatman\tUSA\t35\tclassical\tMale\n"
                "0.23\tCow\tRussia\t5\tNaN\t-\n"
                "0.99\tFaramir\tGondor\t500\tfolk\tMale\n"
                "0.01\tPotter\t,.?!#$\t5\tblues\tMale\n"
                "0.02\tCollins\tEngland\t50\t-\tMale\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            textFloatAndCatFeaturesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Text, "Name"},
                {EColumn::Categ, "Country"},
                {EColumn::Num, "Age"},
                {EColumn::Text, "FavouriteMusic"},
                {EColumn::Categ, "Gender"},
            };

            TVector<TString> featureId = {
                "Name",
                "Country",
                "Age",
                "FavouriteMusic",
                "Gender"
            };

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float>{18.f, 20.f, 35.f, 5.f, 500.f, 5.f, 50.f},
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"USA", "England", "USA", "Russia", "Gondor", ",.?!#$", "England"},
                TVector<TStringBuf>{"Male", "Female", "Male", "-", "Male", "Male", "Male"},
            };
            expectedData.Objects.TextFeatures = {
                TVector<TStringBuf>{"Spiderman", "Wonderwoman", "Batman", "Cow", "Faramir", "Potter", "Collins"},
                TVector<TStringBuf>{"jazz", "soul", "classical", "NaN", "folk", "blues", "-"},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(7);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.23", "0.99", "0.01", "0.02"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(7);
            expectedData.Target.GroupWeights = TWeights<float>(7);

            textFloatAndCatFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(textFloatAndCatFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithEmbeddingColumns) {
        TVector<TReadDatasetTestCase> testCases;

        auto differentDelimiterTestCases = TVector<std::pair<char, TStringBuf>>{
            {
                ';',
                    "0.12\t0;1;1;0.1\n"
                    "0.22\t0.1;0;0;0.2\n"
                    "0.34\t0;2;0;1.1\n"
                    "0.23\t2.1;1.3;1;0\n"
                    "0.99\t1;0;2.1;0\n"
                    "0.01\t0.3;0;1;0.1\n"
                    "0.02\t0.6;0.8;0.9;1.4\n"sv
            },
            {
                ',',
                    "0.12\t0,1,1,0.1\n"
                    "0.22\t0.1,0,0,0.2\n"
                    "0.34\t0,2,0,1.1\n"
                    "0.23\t2.1,1.3,1,0\n"
                    "0.99\t1,0,2.1,0\n"
                    "0.01\t0.3,0,1,0.1\n"
                    "0.02\t0.6,0.8,0.9,1.4\n"sv
            }
        };

        for (const auto& [numVectorDelimiter, datasetFileData]: differentDelimiterTestCases) {
            TReadDatasetTestCase oneEmbeddingFeatureTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNumVector\tembedding0\n"sv;
            srcData.DatasetFileData = datasetFileData;
            srcData.DsvFileHasHeader = false;
            srcData.NumVectorDelimiter = numVectorDelimiter;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            oneEmbeddingFeatureTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::NumVector, "embedding0"},
            };

            TVector<TString> featureId = {"embedding0"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.EmbeddingFeatures = {
                TVector<TVector<float>>{
                    {0.f,  1.f,  1.f,  0.1f},
                    {0.1f, 0.f,  0.f,  0.2f},
                    {0.f,  2.f,  0.f,  1.1f},
                    {2.1f, 1.3f, 1.f,  0.f},
                    {1.f,  0.f,  2.1f, 0.f},
                    {0.3f, 0.f, 1.f,  0.1f},
                    {0.6f, 0.8f, 0.9f, 1.4f}
                },
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(7);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.23", "0.99", "0.01", "0.02"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(7);
            expectedData.Target.GroupWeights = TWeights<float>(7);

            oneEmbeddingFeatureTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(oneEmbeddingFeatureTestCase));
        }
        {
            TReadDatasetTestCase textFloatAndEmbeddingFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNum\tFeat0\n"
                "2\tNumVector\tEmbedding0\n"
                "3\tNum\tFeat1\n"
                "4\tNumVector\tEmbedding1\n"
                "5\tNum\tFeat2\n"sv;
            srcData.DatasetFileData =
                "0.12\t0\t0.1;0;1\t1\t0.18;0\t10\n"
                "0.22\t1\t0.2;1;0\t1.1\t0.3;0.1\t20\n"
                "0.34\t0\t0.3;0;1\t2.1\t0.2;0\t10\n"
                "0.23\t0\t0.4;0.2;0\t0.9\t0.1;0\t30\n"
                "0.99\t1\t0.5;0.1;1\t2.2\t0.5;1\t42\n"
                "0.01\t1\t0.1;0;0\t1.2\t0.7;0\t22\n"
                "0.02\t0\t0.2;0.3;1\t3.2\t0.8;0.3\t30\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            textFloatAndEmbeddingFeaturesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, "Feat0"},
                {EColumn::NumVector, "Embedding0"},
                {EColumn::Num, "Feat1"},
                {EColumn::NumVector, "Embedding1"},
                {EColumn::Num, "Feat2"}
            };

            TVector<TString> featureId = {
                "Feat0",
                "Embedding0",
                "Feat1",
                "Embedding1",
                "Feat2"
            };

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f},
                TVector<float>{1.f, 1.1f, 2.1f, 0.9f, 2.2f, 1.2f, 3.2f},
                TVector<float>{10.f, 20.f, 10.f, 30.f, 42.f, 22.f, 30.f}
            };
            expectedData.Objects.EmbeddingFeatures = {
                TVector<TVector<float>>{
                    {0.1f, 0.f, 1.f},
                    {0.2f, 1.f, 0.f},
                    {0.3f, 0.f, 1.f},
                    {0.4f, 0.2f, 0.f},
                    {0.5f, 0.1f, 1.f},
                    {0.1f, 0.f, 0.f},
                    {0.2f, 0.3f, 1.f}
                },
                TVector<TVector<float>>{
                    {0.18f, 0.f},
                    {0.3f, 0.1f},
                    {0.2f, 0.f},
                    {0.1f, 0.f},
                    {0.5f, 1.f},
                    {0.7f, 0.f},
                    {0.8f, 0.3f}
                },
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(7);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.23", "0.99", "0.01", "0.02"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(7);
            expectedData.Target.GroupWeights = TWeights<float>(7);

            textFloatAndEmbeddingFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(textFloatAndEmbeddingFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithEmbeddingColumnsWithMissingValues) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase textFloatAndEmbeddingFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNum\tFeat0\n"
                "2\tNumVector\tEmbedding0\n"
                "3\tNum\tFeat1\n"
                "4\tNumVector\tEmbedding1\n"
                "5\tNum\tFeat2\n"sv;
            srcData.DatasetFileData =
                "0.12\t0\t0.1;0;1\t1\t0.18;0\t10\n"
                "0.22\t1\t-;-;-\t1.1\t0.3;\t20\n"
                "0.34\t0\t0.3;0;nan\t2.1\t0.2;0\t10\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            textFloatAndEmbeddingFeaturesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, "Feat0"},
                {EColumn::NumVector, "Embedding0"},
                {EColumn::Num, "Feat1"},
                {EColumn::NumVector, "Embedding1"},
                {EColumn::Num, "Feat2"}
            };

            TVector<TString> featureId = {
                "Feat0",
                "Embedding0",
                "Feat1",
                "Embedding1",
                "Feat2"
            };

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.f, 1.f, 0.f},
                TVector<float>{1.f, 1.1f, 2.1f},
                TVector<float>{10.f, 20.f, 10.f}
            };

            auto nanValue = std::numeric_limits<float>::quiet_NaN();

            expectedData.Objects.EmbeddingFeatures = {
                TVector<TVector<float>>{
                    {0.1f, 0.f, 1.f},
                    {nanValue, nanValue, nanValue},
                    {0.3f, 0.f, nanValue}
                },
                TVector<TVector<float>>{
                    {0.18f, 0.f},
                    {0.3f, nanValue},
                    {0.2f, 0.f}
                },
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            textFloatAndEmbeddingFeaturesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(textFloatAndEmbeddingFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadBrokenDatasetWithEmbeddingColumns) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase inconsistentNumVectorLengthTestCase;
            inconsistentNumVectorLengthTestCase.ExpectedReadError = true;

            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNumVector\tembedding0\n"sv;
            srcData.DatasetFileData =
                "0.12\t0;1;1;0.1\n"
                "0.22\t0.1;0;0;0.6;0.2\n"
                "0.34\t0;2;0;1.1\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            inconsistentNumVectorLengthTestCase.SrcData = std::move(srcData);

            testCases.push_back(std::move(inconsistentNumVectorLengthTestCase));
        }

        {
            TReadDatasetTestCase nonFloadSubFieldTestCase1;
            nonFloadSubFieldTestCase1.ExpectedReadError = true;

            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNumVector\tembedding0\n"sv;
            srcData.DatasetFileData =
                "0.12\t0;1;xx;0.1\n"
                "0.22\t0.1;0;0.3;0.6\n"
                "0.34\t0;2;0;1.1\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            nonFloadSubFieldTestCase1.SrcData = std::move(srcData);

            testCases.push_back(std::move(nonFloadSubFieldTestCase1));
        }

        {
            TReadDatasetTestCase nonFloadSubFieldTestCase2;
            nonFloadSubFieldTestCase2.ExpectedReadError = true;

            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tNumVector\tembedding0\n"sv;
            srcData.DatasetFileData =
                "0.12\t0;1;1;--\n"
                "0.22\t0.1;0;0.3;0.6\n"
                "0.34\t0;2;0;1.1\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            nonFloadSubFieldTestCase2.SrcData = std::move(srcData);

            testCases.push_back(std::move(nonFloadSubFieldTestCase2));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithExternalFeatureNames) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase externalFeatureNamesTestCase;
            TSrcData srcData;
            srcData.CdFileData =
                "0\tTarget\n"
                "1\tText\n"
                "2\tCateg\n"
                "3\tNum\n"
                "4\tText\n"
                "5\tCateg\n"sv;
            srcData.DatasetFileData =
                "0.12\tSpiderman\tUSA\t18\tjazz\tMale\n"
                "0.22\tWonderwoman\tEngland\t20\tsoul\tFemale\n"
                "0.34\tBatman\tUSA\t35\tclassical\tMale\n"
                "0.23\tCow\tRussia\t5\tNaN\t-\n"
                "0.99\tFaramir\tGondor\t500\tfolk\tMale\n"
                "0.01\tPotter\t,.?!#$\t5\tblues\tMale\n"
                "0.02\tCollins\tEngland\t50\t-\tMale\n"sv;
            srcData.FeatureNamesFileData =
                "0\tName\n"
                "1\tCountry\n"
                "2\tAge\n"
                "3\tFavouriteMusic\n"
                "4\tGender\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            externalFeatureNamesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Text, ""},
                {EColumn::Categ, ""},
                {EColumn::Num, ""},
                {EColumn::Text, ""},
                {EColumn::Categ, ""},
            };

            TVector<TString> featureId = {
                "Name",
                "Country",
                "Age",
                "FavouriteMusic",
                "Gender"
            };

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float>{18.f, 20.f, 35.f, 5.f, 500.f, 5.f, 50.f},
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"USA", "England", "USA", "Russia", "Gondor", ",.?!#$", "England"},
                TVector<TStringBuf>{"Male", "Female", "Male", "-", "Male", "Male", "Male"},
            };
            expectedData.Objects.TextFeatures = {
                TVector<TStringBuf>{"Spiderman", "Wonderwoman", "Batman", "Cow", "Faramir", "Potter", "Collins"},
                TVector<TStringBuf>{"jazz", "soul", "classical", "NaN", "folk", "blues", "-"},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(7);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.23", "0.99", "0.01", "0.02"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(7);
            expectedData.Target.GroupWeights = TWeights<float>(7);

            externalFeatureNamesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(externalFeatureNamesTestCase));
        }
        {
            TReadDatasetTestCase cdAndExternalFeatureNamesTestCase;
            TSrcData srcData;
            srcData.CdFileData = "0\tTarget\n"
                "1\tText\tName\n"
                "2\tCateg\tCountry\n"
                "3\tNum\tAge\n"
                "4\tText\tFavouriteMusic\n"
                "5\tCateg\tGender\n"sv;
            srcData.DatasetFileData = "0.12\tSpiderman\tUSA\t18\tjazz\tMale\n"
                "0.22\tWonderwoman\tEngland\t20\tsoul\tFemale\n"
                "0.34\tBatman\tUSA\t35\tclassical\tMale\n"
                "0.23\tCow\tRussia\t5\tNaN\t-\n"
                "0.99\tFaramir\tGondor\t500\tfolk\tMale\n"
                "0.01\tPotter\t,.?!#$\t5\tblues\tMale\n"
                "0.02\tCollins\tEngland\t50\t-\tMale\n"sv;
            srcData.FeatureNamesFileData = "0\tName\n"
                "1\tCountry\n"
                "2\tAge\n"
                "3\tFavouriteMusic\n"
                "4\tGender\n"sv;
            srcData.DsvFileHasHeader = false;
            srcData.ObjectsOrder = EObjectsOrder::Undefined;
            cdAndExternalFeatureNamesTestCase.SrcData = std::move(srcData);

            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Text, "Name"},
                {EColumn::Categ, "Country"},
                {EColumn::Num, "Age"},
                {EColumn::Text, "FavouriteMusic"},
                {EColumn::Categ, "Gender"},
            };

            TVector<TString> featureId = {
                "Name",
                "Country",
                "Age",
                "FavouriteMusic",
                "Gender"
            };

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* baselineColumn */ Nothing(), &featureId);
            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.FloatFeatures = {
                TVector<float> {18.f, 20.f, 35.f, 5.f, 500.f, 5.f, 50.f},
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf> {"USA", "England", "USA", "Russia", "Gondor", ",.?!#$", "England"},
                TVector<TStringBuf> {"Male", "Female", "Male", "-", "Male", "Male", "Male"},
            };
            expectedData.Objects.TextFeatures = {
                TVector<TStringBuf> {"Spiderman", "Wonderwoman", "Batman", "Cow", "Faramir", "Potter", "Collins"},
                TVector<TStringBuf> {"jazz", "soul", "classical", "NaN", "folk", "blues", "-"},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(7);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget { { "0.12", "0.22", "0.34", "0.23", "0.99", "0.01", "0.02" } };
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(7);
            expectedData.Target.GroupWeights = TWeights<float>(7);

            cdAndExternalFeatureNamesTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(cdAndExternalFeatureNamesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithGroupedPairs) {
        TReadDatasetTestCase testCase;
        TSrcData srcData;

        srcData.CdFileData =
            "0\tGroupId\n"
            "1\tSubgroupId\n"
            "2\tNum\tf0\n"
            "3\tNum\tf1\n"
            "4\tNum\tf2\n"sv;
        srcData.DatasetFileData =
            "query0\tsite1\t0.1\t0.2\t0.11\n"
            "query0\tsite22\t0.97\t0.82\t0.33\n"
            "query1\tSite9\t0.13\t0.22\t0.23\n"
            "Query 2\tsite12\t0.14\t0.18\t0.1\n"
            "Query 2\tsite22\t0.9\t0.67\t0.17\n"
            "Query 2\tSite45\t0.66\t0.1\t0.31\n"sv;
        srcData.DsvFileHasHeader = false;
        srcData.PairsFileData =
            "query0\t0\t1\t0.1\n"
            "Query 2\t1\t0\t1.0\n"
            "Query 2\t0\t2\t0.2\n"sv;
        srcData.PairsScheme = "dsv-grouped";
        testCase.SrcData = std::move(srcData);


        TExpectedRawData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::GroupId, ""},
            {EColumn::SubgroupId, ""},
            {EColumn::Num, "f0"},
            {EColumn::Num, "f1"},
            {EColumn::Num, "f2"},
        };

        TVector<TString> featureId = {"f0", "f1", "f2"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, true, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        expectedData.Objects.GroupIds = TVector<TStringBuf>{
            "query0",
            "query0",
            "query1",
            "Query 2",
            "Query 2",
            "Query 2"
        };
        expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
            "site1",
            "site22",
            "Site9",
            "site12",
            "site22",
            "Site45"
        };
        expectedData.Objects.FloatFeatures = {
            TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
            TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
            TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f}
        };

        expectedData.ObjectsGrouping = TObjectsGrouping(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );
        expectedData.Target.Weights = TWeights<float>(6);
        expectedData.Target.GroupWeights = TWeights<float>(6);
        expectedData.Target.Pairs = TGroupedPairsInfo{
            TPairInGroup{0, 0, 1, 0.1f},
            TPairInGroup{2, 1, 0, 1.0f},
            TPairInGroup{2, 0, 2, 0.2f}
        };

        testCase.ExpectedData = std::move(expectedData);

        TestReadDataset(testCase);
    }

    Y_UNIT_TEST(ReadDatasetWithGroupedWithIdxPairs) {
        TReadDatasetTestCase testCase;
        TSrcData srcData;

        srcData.CdFileData =
            "0\tGroupId\n"
            "1\tSubgroupId\n"
            "2\tNum\tf0\n"
            "3\tNum\tf1\n"
            "4\tNum\tf2\n"sv;
        srcData.DatasetFileData =
            "query0\tsite1\t0.1\t0.2\t0.11\n"
            "query0\tsite22\t0.97\t0.82\t0.33\n"
            "query1\tSite9\t0.13\t0.22\t0.23\n"
            "Query 2\tsite12\t0.14\t0.18\t0.1\n"
            "Query 2\tsite22\t0.9\t0.67\t0.17\n"
            "Query 2\tSite45\t0.66\t0.1\t0.31\n"sv;
        srcData.DsvFileHasHeader = false;
        srcData.PairsFileData =
            "0\t0\t1\t0.1\n"
            "2\t1\t0\t1.0\n"
            "2\t0\t2\t0.2\n"sv;
        srcData.PairsScheme = "dsv-grouped-with-idx";
        testCase.SrcData = std::move(srcData);


        TExpectedRawData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::GroupId, ""},
            {EColumn::SubgroupId, ""},
            {EColumn::Num, "f0"},
            {EColumn::Num, "f1"},
            {EColumn::Num, "f2"},
        };

        TVector<TString> featureId = {"f0", "f1", "f2"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, true, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
        expectedData.Objects.GroupIds = TVector<TStringBuf>{
            "query0",
            "query0",
            "query1",
            "Query 2",
            "Query 2",
            "Query 2"
        };
        expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
            "site1",
            "site22",
            "Site9",
            "site12",
            "site22",
            "Site45"
        };
        expectedData.Objects.FloatFeatures = {
            TVector<float>{0.1f, 0.97f, 0.13f, 0.14f, 0.9f, 0.66f},
            TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f},
            TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f}
        };

        expectedData.ObjectsGrouping = TObjectsGrouping(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );
        expectedData.Target.Weights = TWeights<float>(6);
        expectedData.Target.GroupWeights = TWeights<float>(6);
        expectedData.Target.Pairs = TGroupedPairsInfo{
            TPairInGroup{0, 0, 1, 0.1f},
            TPairInGroup{2, 1, 0, 1.0f},
            TPairInGroup{2, 0, 2, 0.2f}
        };

        testCase.ExpectedData = std::move(expectedData);

        TestReadDataset(testCase);
    }
}

Y_UNIT_TEST_SUITE(SampleDataFromDsvByIndices) {
    Y_UNIT_TEST(SampleDataset) {
        TVector<TSampleDatasetTestCase> testCases;

        {
            TSampleDatasetTestCase simpleTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\t0.1\t0.2\t0\t0\n"
                "1\t0.97\t0.82\t0.33\t0.9\n"
                "2\t0.13\t0.22\t0.89\t0.5\n"
                "3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            simpleTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.13f},
                TVector<float>{0.2f, 0.22f},
                TVector<float>{0.0f, 0.89f},
                TVector<float>{0.0f, 0.5f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "2"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(2);
            expectedData.Target.GroupWeights = TWeights<float>(2);

            simpleTestCase.ExpectedData = std::move(expectedData);

            simpleTestCase.OnlyFeaturesData = false;
            simpleTestCase.SubsetIndices = TVector<ui32>{0, 2};

            testCases.push_back(std::move(simpleTestCase));
        }

        {
            TSampleDatasetTestCase simpleAndOnlyFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\t0.1\t0.2\t0\t0\n"
                "1\t0.97\t0.82\t0.33\t0.9\n"
                "2\t0.13\t0.22\t0.89\t0.5\n"
                "3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            simpleAndOnlyFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Auxiliary, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.13f},
                TVector<float>{0.2f, 0.22f},
                TVector<float>{0.0f, 0.89f},
                TVector<float>{0.0f, 0.5f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);
            expectedData.Target.TargetType = ERawTargetType::None;
            expectedData.Target.Weights = TWeights<float>(2);
            expectedData.Target.GroupWeights = TWeights<float>(2);

            simpleAndOnlyFeaturesTestCase.ExpectedData = std::move(expectedData);

            simpleAndOnlyFeaturesTestCase.OnlyFeaturesData = true;
            simpleAndOnlyFeaturesTestCase.SubsetIndices = TVector<ui32>{0, 2};

            testCases.push_back(std::move(simpleAndOnlyFeaturesTestCase));
        }

        {
            TSampleDatasetTestCase reorderedIndicesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\t0.1\t0.2\t0\t0\n"
                "1\t0.97\t0.82\t0.33\t0.9\n"
                "2\t0.13\t0.22\t0.89\t0.5\n"
                "3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            reorderedIndicesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"3", "0", "1"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            reorderedIndicesTestCase.ExpectedData = std::move(expectedData);

            reorderedIndicesTestCase.OnlyFeaturesData = false;
            reorderedIndicesTestCase.SubsetIndices = TVector<ui32>{3, 0, 1};

            testCases.push_back(std::move(reorderedIndicesTestCase));
        }

        {
            TSampleDatasetTestCase reorderedIndicesAndOnlyFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\t0.1\t0.2\t0\t0\n"
                "1\t0.97\t0.82\t0.33\t0.9\n"
                "2\t0.13\t0.22\t0.89\t0.5\n"
                "3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            reorderedIndicesAndOnlyFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Auxiliary, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::None;
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            reorderedIndicesAndOnlyFeaturesTestCase.ExpectedData = std::move(expectedData);

            reorderedIndicesAndOnlyFeaturesTestCase.OnlyFeaturesData = true;
            reorderedIndicesAndOnlyFeaturesTestCase.SubsetIndices = TVector<ui32>{3, 0, 1};

            testCases.push_back(std::move(reorderedIndicesAndOnlyFeaturesTestCase));
        }

        {
            TSampleDatasetTestCase duplicateIndicesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\t0.1\t0.2\t0\t0\n"
                "1\t0.97\t0.82\t0.33\t0.9\n"
                "2\t0.13\t0.22\t0.89\t0.5\n"
                "3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            duplicateIndicesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f, 0.78f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f, 1.0f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f, 2.2f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f, 3.6f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(5);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"3", "0", "1", "3", "1"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(5);
            expectedData.Target.GroupWeights = TWeights<float>(5);

            duplicateIndicesTestCase.ExpectedData = std::move(expectedData);

            duplicateIndicesTestCase.OnlyFeaturesData = false;
            duplicateIndicesTestCase.SubsetIndices = TVector<ui32>{3, 0, 1, 3, 1};

            testCases.push_back(std::move(duplicateIndicesTestCase));
        }

        {
            TSampleDatasetTestCase duplicateIndicesAndOnlyFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget");
            srcData.DatasetFileData =
                "Target\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\t0.1\t0.2\t0\t0\n"
                "1\t0.97\t0.82\t0.33\t0.9\n"
                "2\t0.13\t0.22\t0.89\t0.5\n"
                "3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            duplicateIndicesAndOnlyFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Auxiliary, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f, 0.78f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f, 1.0f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f, 2.2f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f, 3.6f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(5);
            expectedData.Target.TargetType = ERawTargetType::None;
            expectedData.Target.Weights = TWeights<float>(5);
            expectedData.Target.GroupWeights = TWeights<float>(5);

            duplicateIndicesAndOnlyFeaturesTestCase.ExpectedData = std::move(expectedData);

            duplicateIndicesAndOnlyFeaturesTestCase.OnlyFeaturesData = true;
            duplicateIndicesAndOnlyFeaturesTestCase.SubsetIndices = TVector<ui32>{3, 0, 1, 3, 1};

            testCases.push_back(std::move(duplicateIndicesAndOnlyFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestSampleDataset(testCase);
        }
    }
}

Y_UNIT_TEST_SUITE(SampleDataFromDsvBySampleIds) {
    Y_UNIT_TEST(SampleDataset) {
        TVector<TSampleDatasetTestCase> testCases;

        {
            TSampleDatasetTestCase simpleTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget\n1\tSampleId");
            srcData.DatasetFileData =
                "Target\tSampleId\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\td0\t0.1\t0.2\t0\t0\n"
                "1\td1\t0.97\t0.82\t0.33\t0.9\n"
                "2\td2\t0.13\t0.22\t0.89\t0.5\n"
                "3\td3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            simpleTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::SampleId, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, true, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.13f},
                TVector<float>{0.2f, 0.22f},
                TVector<float>{0.0f, 0.89f},
                TVector<float>{0.0f, 0.5f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"0", "2"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(2);
            expectedData.Target.GroupWeights = TWeights<float>(2);

            simpleTestCase.ExpectedData = std::move(expectedData);

            simpleTestCase.OnlyFeaturesData = false;
            simpleTestCase.SubsetSampleIds = TVector<TString>{"d0", "d2"};

            testCases.push_back(std::move(simpleTestCase));
        }

        {
            TSampleDatasetTestCase simpleAndOnlyFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget\n1\tSampleId");
            srcData.DatasetFileData =
                "Target\tSampleId\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\td0\t0.1\t0.2\t0\t0\n"
                "1\td1\t0.97\t0.82\t0.33\t0.9\n"
                "2\td2\t0.13\t0.22\t0.89\t0.5\n"
                "3\td3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            simpleAndOnlyFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Auxiliary, ""},
                {EColumn::SampleId, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, false, false, true, false, /*additionalBaselineCount*/ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.13f},
                TVector<float>{0.2f, 0.22f},
                TVector<float>{0.0f, 0.89f},
                TVector<float>{0.0f, 0.5f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);
            expectedData.Target.TargetType = ERawTargetType::None;
            expectedData.Target.Weights = TWeights<float>(2);
            expectedData.Target.GroupWeights = TWeights<float>(2);

            simpleAndOnlyFeaturesTestCase.ExpectedData = std::move(expectedData);

            simpleAndOnlyFeaturesTestCase.OnlyFeaturesData = true;
            simpleAndOnlyFeaturesTestCase.SubsetSampleIds = TVector<TString>{"d0", "d2"};

            testCases.push_back(std::move(simpleAndOnlyFeaturesTestCase));
        }

        {
            TSampleDatasetTestCase reorderedSampleIdsTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget\n1\tSampleId");
            srcData.DatasetFileData =
                "Target\tSampleId\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\td0\t0.1\t0.2\t0\t0\n"
                "1\td1\t0.97\t0.82\t0.33\t0.9\n"
                "2\td2\t0.13\t0.22\t0.89\t0.5\n"
                "3\td3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            reorderedSampleIdsTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::SampleId, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, true, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"3", "0", "1"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            reorderedSampleIdsTestCase.ExpectedData = std::move(expectedData);

            reorderedSampleIdsTestCase.OnlyFeaturesData = false;
            reorderedSampleIdsTestCase.SubsetSampleIds = TVector<TString>{"d3", "d0", "d1"};

            testCases.push_back(std::move(reorderedSampleIdsTestCase));
        }

        {
            TSampleDatasetTestCase reorderedSampleIdsAndOnlyFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget\n1\tSampleId");
            srcData.DatasetFileData =
                "Target\tSampleId\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\td0\t0.1\t0.2\t0\t0\n"
                "1\td1\t0.97\t0.82\t0.33\t0.9\n"
                "2\td2\t0.13\t0.22\t0.89\t0.5\n"
                "3\td3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            reorderedSampleIdsAndOnlyFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Auxiliary, ""},
                {EColumn::SampleId, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, false, false, true, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::None;
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            reorderedSampleIdsAndOnlyFeaturesTestCase.ExpectedData = std::move(expectedData);

            reorderedSampleIdsAndOnlyFeaturesTestCase.OnlyFeaturesData = true;
            reorderedSampleIdsAndOnlyFeaturesTestCase.SubsetSampleIds = TVector<TString>{"d3", "d0", "d1"};

            testCases.push_back(std::move(reorderedSampleIdsAndOnlyFeaturesTestCase));
        }

        {
            TSampleDatasetTestCase duplicateSampleIdsTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget\n1\tSampleId");
            srcData.DatasetFileData =
                "Target\tSampleId\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\td0\t0.1\t0.2\t0\t0\n"
                "1\td1\t0.97\t0.82\t0.33\t0.9\n"
                "2\td2\t0.13\t0.22\t0.89\t0.5\n"
                "3\td3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            duplicateSampleIdsTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Label, ""},
                {EColumn::SampleId, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, true, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f, 0.78f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f, 1.0f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f, 2.2f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f, 3.6f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(5);
            expectedData.Target.TargetType = ERawTargetType::String;
            TVector<TVector<TString>> rawTarget{{"3", "0", "1", "3", "1"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(5);
            expectedData.Target.GroupWeights = TWeights<float>(5);

            duplicateSampleIdsTestCase.ExpectedData = std::move(expectedData);

            duplicateSampleIdsTestCase.OnlyFeaturesData = false;
            duplicateSampleIdsTestCase.SubsetSampleIds = TVector<TString>{"d3", "d0", "d1", "d3", "d1"};

            testCases.push_back(std::move(duplicateSampleIdsTestCase));
        }

        {
            TSampleDatasetTestCase duplicateSampleIdsAndOnlyFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = TStringBuf("0\tTarget\n1\tSampleId");
            srcData.DatasetFileData =
                "Target\tSampleId\tFeat0\tFeat1\tFeat2\tFeat3\n"
                "0\td0\t0.1\t0.2\t0\t0\n"
                "1\td1\t0.97\t0.82\t0.33\t0.9\n"
                "2\td2\t0.13\t0.22\t0.89\t0.5\n"
                "3\td3\t0.78\t1.0\t2.2\t3.6\n"sv;
            srcData.DsvFileHasHeader = true;
            duplicateSampleIdsAndOnlyFeaturesTestCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            TDataColumnsMetaInfo dataColumnsMetaInfo;
            dataColumnsMetaInfo.Columns = {
                {EColumn::Auxiliary, ""},
                {EColumn::SampleId, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""},
                {EColumn::Num, ""}
            };

            TVector<TString> featureId = {"Feat0", "Feat1", "Feat2", "Feat3"};

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, false, false, true, false, /* additionalBaselineCount */ Nothing(), &featureId);
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.78f, 0.1f, 0.97f, 0.78f, 0.97f},
                TVector<float>{1.0f, 0.2f, 0.82f, 1.0f, 0.82f},
                TVector<float>{2.2f, 0.0f, 0.33f, 2.2f, 0.33f},
                TVector<float>{3.6f, 0.0f, 0.9f, 3.6f, 0.9f},
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(5);
            expectedData.Target.TargetType = ERawTargetType::None;
            expectedData.Target.Weights = TWeights<float>(5);
            expectedData.Target.GroupWeights = TWeights<float>(5);

            duplicateSampleIdsAndOnlyFeaturesTestCase.ExpectedData = std::move(expectedData);

            duplicateSampleIdsAndOnlyFeaturesTestCase.OnlyFeaturesData = true;
            duplicateSampleIdsAndOnlyFeaturesTestCase.SubsetSampleIds = TVector<TString>{"d3", "d0", "d1", "d3", "d1"};

            testCases.push_back(std::move(duplicateSampleIdsAndOnlyFeaturesTestCase));
        }

        for (const auto& testCase : testCases) {
            TestSampleDataset(testCase);
        }
    }
}
