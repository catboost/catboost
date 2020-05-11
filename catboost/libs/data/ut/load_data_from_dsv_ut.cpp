#include "util.h"

#include <catboost/libs/data/ut/lib/for_data_provider.h>
#include <catboost/libs/data/ut/lib/for_loader.h>

#include <catboost/libs/data/load_data.h>

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects_grouping.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>

#include <library/cpp/unittest/registar.h>

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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tTarget\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "Target0\tTarget1\tFeat\n"
                "0\t0.1\t0.2\n"
                "1\t0.97\t0.82\n"
                "0\t0.13\t0.22\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf("0\tTarget");
            srcData.DatasetFileData = AsStringBuf(
                "Target\tFeat0\tFeat1\n"
                "0\t0.1\t0.2\n"
                "1\t0.97\t0.82\n"
                "0\t0.13\t0.22\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tSubgroupId\n"
                "3\tWeight\n"
                "4\tGroupWeight\n"
                "5\tNum\tf0\n"
                "6\tNum\tf1\n"
                "7\tNum\tf2\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tquery0\tsite1\t0.12\t1.0\t0.1\t0.2\t0.11\n"
                "0.22\tquery0\tsite22\t0.18\t1.0\t0.97\t0.82\t0.33\n"
                "0.34\tquery1\tSite9\t1.0\t0.0\t0.13\t0.22\t0.23\n"
                "0.42\tQuery 2\tsite12\t0.45\t0.5\t0.14\t0.18\t0.1\n"
                "0.01\tQuery 2\tsite22\t1.0\t0.5\t0.9\t0.67\t0.17\n"
                "0.0\tQuery 2\tSite45\t2.0\t0.5\t0.66\t0.1\t0.31\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tGroupId\n"
                "1\tSubgroupId\n"
                "2\tNum\tf0\n"
                "3\tNum\tf1\n"
                "4\tNum\tf2\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "query0\tsite1\t0.1\t0.2\t0.11\n"
                "query0\tsite22\t0.97\t0.82\t0.33\n"
                "query1\tSite9\t0.13\t0.22\t0.23\n"
                "Query 2\tsite12\t0.14\t0.18\t0.1\n"
                "Query 2\tsite22\t0.9\t0.67\t0.17\n"
                "Query 2\tSite45\t0.66\t0.1\t0.31\n"
            );
            srcData.DsvFileHasHeader = false;
            srcData.PairsFileData = AsStringBuf(
                "0\t1\t0.1\n"
                "4\t3\t1.0\n"
                "3\t5\t0.2"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::None, false, false, true, /* additionalBaselineCount */ Nothing(), &featureId);
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
            expectedData.Target.Pairs = {TPair(0, 1, 0.1f), TPair(4, 3, 1.0f), TPair(3, 5, 0.2f)};

            pairsOnlyTestCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(pairsOnlyTestCase));
        }

        {
            TReadDatasetTestCase floatAndCatFeaturesTestCase;
            TSrcData srcData;
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tNum\tfloat0\n"
                "3\tCateg\tGender1\n"
                "4\tNum\tfloat2\n"
                "5\tCateg\tCountry3\n"
                "6\tNum\tfloat4\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n"
                "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n"
                "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n"
                "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n"
                "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n"
                "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tNum\tfloat0\n"
                "3\tCateg\tGender1\n"
                "4\tNum\tfloat2\n"
                "5\tCateg\tCountry3\n"
                "6\tNum\tfloat4\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n"
                "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n"
                "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n"
                "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n"
                "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n"
                "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"
            );
            srcData.DsvFileHasHeader = false;
            srcData.GroupWeightsFileData = AsStringBuf(
                "query0\t1.0\n"
                "query1\t0.0\n"
                "Query 2\t0.5"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, true, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tGroupId\n"
                "2\tNum\tfloat0\n"
                "3\tCateg\tGender1\n"
                "4\tNum\tfloat2\n"
                "5\tCateg\tCountry3\n"
                "6\tNum\tfloat4\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n"
                "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n"
                "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n"
                "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n"
                "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n"
                "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tTimestamp"
            );
            srcData.DatasetFileData = AsStringBuf(
                "Target\tTimestamp\tFeat0\tFeat1\n"
                "0\t10\t0.1\t0.2\n"
                "1\t10\t0.97\t0.82\n"
                "0\t20\t0.13\t0.22\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tTimestamp"
            );
            srcData.DatasetFileData = AsStringBuf(
                "Target\tTimestamp\tFeat0\tFeat1\n"
                "0\t20\t0.1\t0.2\n"
                "1\t10\t0.97\t0.82\n"
                "0\t20\t0.13\t0.22\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tTimestamp"
            );
            srcData.DatasetFileData = AsStringBuf(
                "Target\tTimestamp\tFeat0\tFeat1\n"
                "0\t20\t0.1\t0.2\n"
                "1\t20\t0.97\t0.82\n"
                "0\t20\t0.13\t0.22\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tNum\tfloat0\n"
                "2\tCateg\tGender1\n"
                "3\tNum\tfloat2\n"
                "4\tCateg\tCountry3\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\t0.1\tNan\t0.2\tGermany\n"
                "0.22\t\t\tNA\tRussia\n"
                "0.341\tnan\tMale\t0.22\tN/A\n"
                "None\t0.14\tMale\tNULL\tFinland\n"
                "0.01\tna\tFemale\tNaN\tUSA\n"
                "0.0\t0.66\t#NA\t0.1\tNone\n"
                "N/A\tNone\tFemale\t0.12\tNULL\n"
                "0.11\t-1.#QNAN\tN/a\t1.#IND\t1.#IND\n"
                "-\t#N/A N/A\t#N/A N/A\t-\t-\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tNum\tfloat0\n"
                "2\tCateg\tPlace1\n"
                "3\tNum\tfloat2\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0\t0\t\"Berlin, Germany\"\t0\n"
                "0\t0\t\"\"\"Skolkovo\"\", Russia\"\t0\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tNum\tfloat0\n"
                "2\tCateg\tPlace1\n"
                "3\tNum\tfloat2\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "Target\t\"just \"\"float\"\"\"\t\"Place, Country\"\t\"second\tfloat\"\n"
                "0\t0\t\"Berlin, Germany\"\t0\n"
                "0\t0\t\"\"\"Skolkovo\"\", Russia\"\t0\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tText\ttext0\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tWhat\n"
                "0.22\tnoise\n"
                "0.34\tannoys\n"
                "0.23\tan oyster\n"
                "0.99\t\n"
                "0.01\tmost\n"
                "0.02\t?\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* baselineColumn */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tText\tName\n"
                "2\tCateg\tCountry\n"
                "3\tNum\tAge\n"
                "4\tText\tFavouriteMusic\n"
                "5\tCateg\tGender\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tSpiderman\tUSA\t18\tjazz\tMale\n"
                "0.22\tWonderwoman\tEngland\t20\tsoul\tFemale\n"
                "0.34\tBatman\tUSA\t35\tclassical\tMale\n"
                "0.23\tCow\tRussia\t5\tNaN\t-\n"
                "0.99\tFaramir\tGondor\t500\tfolk\tMale\n"
                "0.01\tPotter\t,.?!#$\t5\tblues\tMale\n"
                "0.02\tCollins\tEngland\t50\t-\tMale\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* baselineColumn */ Nothing(), &featureId);
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

    Y_UNIT_TEST(ReadDatasetWithExternalFeatureNames) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase externalFeatureNamesTestCase;
            TSrcData srcData;
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "1\tText\n"
                "2\tCateg\n"
                "3\tNum\n"
                "4\tText\n"
                "5\tCateg\n"
            );
            srcData.DatasetFileData = AsStringBuf(
                "0.12\tSpiderman\tUSA\t18\tjazz\tMale\n"
                "0.22\tWonderwoman\tEngland\t20\tsoul\tFemale\n"
                "0.34\tBatman\tUSA\t35\tclassical\tMale\n"
                "0.23\tCow\tRussia\t5\tNaN\t-\n"
                "0.99\tFaramir\tGondor\t500\tfolk\tMale\n"
                "0.01\tPotter\t,.?!#$\t5\tblues\tMale\n"
                "0.02\tCollins\tEngland\t50\t-\tMale\n"
            );
            srcData.FeatureNamesFileData = AsStringBuf(
                "0\tName\n"
                "1\tCountry\n"
                "2\tAge\n"
                "3\tFavouriteMusic\n"
                "4\tGender\n"
            );
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* baselineColumn */ Nothing(), &featureId);
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
            srcData.CdFileData = AsStringBuf("0\tTarget\n"
                "1\tText\tName\n"
                "2\tCateg\tCountry\n"
                "3\tNum\tAge\n"
                "4\tText\tFavouriteMusic\n"
                "5\tCateg\tGender\n");
            srcData.DatasetFileData = AsStringBuf("0.12\tSpiderman\tUSA\t18\tjazz\tMale\n"
                "0.22\tWonderwoman\tEngland\t20\tsoul\tFemale\n"
                "0.34\tBatman\tUSA\t35\tclassical\tMale\n"
                "0.23\tCow\tRussia\t5\tNaN\t-\n"
                "0.99\tFaramir\tGondor\t500\tfolk\tMale\n"
                "0.01\tPotter\t,.?!#$\t5\tblues\tMale\n"
                "0.02\tCollins\tEngland\t50\t-\tMale\n");
            srcData.FeatureNamesFileData = AsStringBuf("0\tName\n"
                "1\tCountry\n"
                "2\tAge\n"
                "3\tFavouriteMusic\n"
                "4\tGender\n");
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

            expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, /* baselineColumn */ Nothing(), &featureId);
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
}
