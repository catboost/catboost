
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/loader.h>

#include <catboost/libs/data/ut/lib/for_data_provider.h>
#include <catboost/libs/data/ut/lib/for_loader.h>

#include <catboost/libs/column_description/cd_parser.h>

#include <util/system/compiler.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


template <class TConsumer>
inline void ReadAndProceedPoolInBlocks(
    const TPathWithScheme& dsvFilePath,
    const TDsvFormatOptions& dsvFormatOptions,
    const TPathWithScheme& cdFilePath,
    ui32 blockSize,
    TConsumer&& poolConsumer,
    NPar::ILocalExecutor* localExecutor) {

    const auto loadSubset = TDatasetSubset::MakeColumns();
    auto datasetLoader = GetProcessor<IDatasetLoader>(
        dsvFilePath, // for choosing processor

        // processor args
        TDatasetLoaderPullArgs {
            dsvFilePath,

            TDatasetLoaderCommonArgs {
                /*PairsFilePath*/TPathWithScheme(),
                /*GraphFilePath*/TPathWithScheme(),
                /*GroupWeightsFilePath=*/TPathWithScheme(),
                /*BaselineFilePath=*/TPathWithScheme(),
                /*TimestampsFilePath*/TPathWithScheme(),
                /*FeatureNamesPath*/TPathWithScheme(),
                /*PoolMetaInfoPath*/TPathWithScheme(),
                /* ClassLabels */{},
                dsvFormatOptions,
                MakeCdProviderFromFile(cdFilePath),
                /*ignoredFeatures*/ {},
                EObjectsOrder::Undefined,
                blockSize,
                loadSubset,
                /*LoadColumnsAsString*/ false,
                /*LoadSampleIds*/ false,
                /*ForceUnitPairWeights*/ false,
                localExecutor
            }
        }
    );

    THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
        datasetLoader->GetVisitorType(),
        TDataProviderBuilderOptions{},
        loadSubset,
        localExecutor
    );
    CB_ENSURE_INTERNAL(
        dataProviderBuilder,
        "Failed to create data provider builder for visitor of type " << datasetLoader->GetVisitorType()
    );

    NCB::IRawObjectsOrderDatasetLoader* rawObjectsOrderDatasetLoader
        = dynamic_cast<NCB::IRawObjectsOrderDatasetLoader*>(datasetLoader.Get());

    UNIT_ASSERT(rawObjectsOrderDatasetLoader);

    // process in blocks
    NCB::IRawObjectsOrderDataVisitor* visitor = dynamic_cast<NCB::IRawObjectsOrderDataVisitor*>(
        dataProviderBuilder.Get()
    );
    UNIT_ASSERT(visitor);

    while (rawObjectsOrderDatasetLoader->DoBlock(visitor)) {
        auto result = dataProviderBuilder->GetResult();
        if (result) {
            poolConsumer(std::move(result));
        }
    }
    auto lastResult = dataProviderBuilder->GetLastResult();
    if (lastResult) {
        poolConsumer(std::move(lastResult));
    }
}


Y_UNIT_TEST_SUITE(ProcessDataBlocksFromDsv) {
    struct TTestCase {
        TSrcData SrcData;
        TVector<TExpectedRawData> ExpectedData;
        ui32 BlockSize;
    };


    void Test(const TTestCase& testCase) {
        TReadDatasetMainParams readDatasetMainParams;

        // TODO(akhropov): temporarily use THolder until TTempFile move semantic are fixed
        TVector<THolder<TTempFile>> srcDataFiles;

        SaveSrcData(testCase.SrcData, &readDatasetMainParams, &srcDataFiles);

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(3);

        ui32 currentPart = 0;

        ReadAndProceedPoolInBlocks(
            readDatasetMainParams.PoolPath,
            TDsvFormatOptions{testCase.SrcData.DsvFileHasHeader, '\t'},
            readDatasetMainParams.ColumnarPoolFormatParams.CdFilePath,
            testCase.BlockSize,
            [&] (TDataProviderPtr dataProvider) {
                Compare<TRawObjectsDataProvider>(
                    std::move(dataProvider),
                    testCase.ExpectedData[currentPart],
                    true
                );
                ++currentPart;
            },
            &localExecutor
        );
        UNIT_ASSERT_VALUES_EQUAL((size_t)currentPart, testCase.ExpectedData.size());
    }

    static TSrcData GetNonGroupedSrcData() {
        TSrcData srcData;
        srcData.CdFileData = TStringBuf(
            "0\tTarget\n"
            "1\tTimestamp\n"
            "2\tBaseline\n"
            "3\tBaseline\n"
            "4\tWeight\n"
            "5\tNum\tfloat0\n"
            "6\tCateg\tGender1\n"
            "7\tNum\tfloat2\n"
            "8\tCateg\tCountry3\n"
            "9\tNum\tfloat4\n"
        );
        srcData.DatasetFileData = TStringBuf(
            "0.12\t0\t0.0\t0.1\t0.5\t0.1\tMale\t0.2\tGermany\t0.11\n"
            "0.22\t1\t0.12\t0.23\t0.22\t0.97\tFemale\t0.82\tRussia\t0.33\n"
            "0.34\t1\t0.1\t0.11\t0.67\t0.81\tMale\t0.22\tUSA\t0.23\n"
            "0.42\t2\t0.17\t0.29\t0.8\t0.52\tMale\t0.18\tFinland\t0.1\n"

            "0.01\t3\t0.2\t0.12\t1.0\t0.33\tFemale\t0.67\tUSA\t0.17\n"
            "0.0\t4\t0.1\t0.2\t0.89\t0.66\tFemale\t0.1\tUK\t0.31\n"
            "0.11\t7\t0.11\t0.33\t0.8\t0.97\tMale\t0.37\tGreece\t0.82\n"
            "0.21\t10\t0.81\t0.31\t0.72\t0.81\tMale\t0.19\tItaly\t0.44\n"

            "0.92\t20\t0.32\t0.13\t0.66\t0.22\tFemale\t0.28\tGermany\t0.1\n"
            "0.04\t30\t0.01\t0.19\t0.51\t0.0\tFemale\t0.77\tUK\t0.62\n"
        );
        srcData.DsvFileHasHeader = false;
        srcData.ObjectsOrder = EObjectsOrder::Undefined;

        return srcData;
    }

    static TExpectedRawData GetNonGroupedCommonExpectedData() {
        TExpectedRawData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Label, ""},
            {EColumn::Timestamp, ""},
            {EColumn::Baseline, ""},
            {EColumn::Baseline, ""},
            {EColumn::Weight, ""},
            {EColumn::Num, "float0"},
            {EColumn::Categ, "Gender1"},
            {EColumn::Num, "float2"},
            {EColumn::Categ, "Country3"},
            {EColumn::Num, "float4"},
        };

        TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3", "float4"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);

        return expectedData;
    }

    static TSrcData GetGroupedSrcData() {
        TSrcData srcData;
        srcData.CdFileData = TStringBuf(
            "0\tTarget\n"
            "1\tTimestamp\n"
            "2\tGroupId\n"
            "3\tSubgroupId\n"
            "4\tGroupWeight\n"
            "5\tBaseline\n"
            "6\tBaseline\n"
            "7\tWeight\n"
            "8\tNum\tfloat0\n"
            "9\tCateg\tGender1\n"
            "10\tNum\tfloat2\n"
            "11\tCateg\tCountry3\n"
            "12\tNum\tfloat4\n"
        );
        srcData.DatasetFileData = TStringBuf(
            "0.12\t0\tgroup0\tsubgroup0\t1.0\t0.0\t0.1\t0.5\t0.1\tMale\t0.2\tGermany\t0.11\n"
            "0.22\t0\tgroup0\tsubgroup1\t1.0\t0.12\t0.23\t0.22\t0.97\tFemale\t0.82\tRussia\t0.33\n"
            "0.34\t0\tgroup0\tsubgroup0\t1.0\t0.1\t0.11\t0.67\t0.81\tMale\t0.22\tUSA\t0.23\n"

            "0.42\t10\tgroup1\tsubgroup2\t0.0\t0.17\t0.29\t0.8\t0.52\tMale\t0.18\tFinland\t0.1\n"
            "0.01\t10\tgroup1\tsubgroup3\t0.0\t0.2\t0.12\t1.0\t0.33\tFemale\t0.67\tUSA\t0.17\n"
            "0.0\t10\tgroup1\tsubgroup0\t0.0\t0.1\t0.2\t0.89\t0.66\tFemale\t0.1\tUK\t0.31\n"

            "0.11\t30\tgroup2\tsubgroup10\t0.2\t0.11\t0.33\t0.8\t0.97\tMale\t0.37\tGreece\t0.82\n"
            "0.21\t30\tgroup2\tsubgroup4\t0.2\t0.81\t0.31\t0.72\t0.81\tMale\t0.19\tItaly\t0.44\n"
            "0.92\t20\tgroup3\tsubgroup5\t0.3\t0.32\t0.13\t0.66\t0.22\tFemale\t0.28\tGermany\t0.1\n"

            "0.04\t10\tgroup4\tsubgroup7\t0.5\t0.01\t0.19\t0.51\t0.0\tFemale\t0.77\tUK\t0.62\n"
        );
        srcData.DsvFileHasHeader = false;
        srcData.ObjectsOrder = EObjectsOrder::Undefined;

        return srcData;
    }

    static TExpectedRawData GetGroupedCommonExpectedData() {
        TExpectedRawData expectedData;

        TDataColumnsMetaInfo dataColumnsMetaInfo;
        dataColumnsMetaInfo.Columns = {
            {EColumn::Label, ""},
            {EColumn::Timestamp, ""},
            {EColumn::GroupId, ""},
            {EColumn::SubgroupId, ""},
            {EColumn::GroupWeight, ""},
            {EColumn::Baseline, ""},
            {EColumn::Baseline, ""},
            {EColumn::Weight, ""},
            {EColumn::Num, "float0"},
            {EColumn::Categ, "Gender1"},
            {EColumn::Num, "float2"},
            {EColumn::Categ, "Country3"},
            {EColumn::Num, "float4"},
        };

        TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3", "float4"};

        expectedData.MetaInfo = TDataMetaInfo(std::move(dataColumnsMetaInfo), ERawTargetType::String, false, false, false, false, false, false, /* additionalBaselineCount */ Nothing(), &featureId);

        return expectedData;
    }

    Y_UNIT_TEST(TestNonGroupedBigBlock) {
        TTestCase testCase;

        testCase.SrcData = GetNonGroupedSrcData();

        TExpectedRawData expectedData = GetNonGroupedCommonExpectedData();

        expectedData.Objects.Order = EObjectsOrder::Ordered;

        expectedData.Objects.Timestamp = TVector<ui64>{0, 1, 1, 2, 3, 4, 7, 10, 20, 30};
        expectedData.Objects.FloatFeatures = {
            TVector<float>{0.1f, 0.97f, 0.81f, 0.52f, 0.33f, 0.66f, 0.97f, 0.81f, 0.22f, 0.0f},
            TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f, 0.37f, 0.19f, 0.28f, 0.77f},
            TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f, 0.82f, 0.44f, 0.1f, 0.62f}
        };

        expectedData.Objects.CatFeatures = {
            TVector<TStringBuf>{
                "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Male", "Female", "Female"
            },
            TVector<TStringBuf>{
                "Germany", "Russia", "USA", "Finland", "USA", "UK", "Greece", "Italy", "Germany", "UK"
            }
        };

        expectedData.ObjectsGrouping = TObjectsGrouping(10);

        expectedData.Target.TargetType = ERawTargetType::String;

        TVector<TVector<TString>> rawTarget{
            {"0.12", "0.22", "0.34", "0.42", "0.01", "0.0", "0.11", "0.21", "0.92", "0.04"}
        };
        expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
        expectedData.Target.Baseline = {
            {0.0f, 0.12f, 0.1f, 0.17f, 0.2f, 0.1f, 0.11f, 0.81f, 0.32f, 0.01f},
            {0.1f, 0.23f, 0.11f, 0.29f, 0.12f, 0.2f, 0.33f, 0.31f, 0.13f, 0.19f}
        };
        expectedData.Target.Weights = TWeights<float>(
            {0.5f, 0.22f, 0.67f, 0.8f, 1.0f, 0.89f, 0.8f, 0.72f, 0.66f, 0.51f}
        );
        expectedData.Target.GroupWeights = TWeights<float>(10);

        testCase.ExpectedData = {expectedData};
        testCase.BlockSize = 100;

        Test(testCase);
    }

    Y_UNIT_TEST(TestNonGroupedMultiBlock) {
        TTestCase testCase;

        testCase.SrcData = GetNonGroupedSrcData();

        {
            TExpectedRawData expectedData = GetNonGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.Timestamp = TVector<ui64>{0, 1, 1, 2};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.81f, 0.52f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Male", "Male"},
                TVector<TStringBuf>{"Germany", "Russia", "USA", "Finland"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(4);

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.42"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.0f, 0.12f, 0.1f, 0.17f}, {0.1f, 0.23f, 0.11f, 0.29f}};
            expectedData.Target.Weights = TWeights<float>({0.5f, 0.22f, 0.67f, 0.8f});
            expectedData.Target.GroupWeights = TWeights<float>(4);

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetNonGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.Timestamp = TVector<ui64>{3, 4, 7, 10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.33f, 0.66f, 0.97f, 0.81f},
                TVector<float>{0.67f, 0.1f, 0.37f, 0.19f},
                TVector<float>{0.17f, 0.31f, 0.82f, 0.44f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Female", "Female", "Male", "Male"},
                TVector<TStringBuf>{"USA", "UK", "Greece", "Italy"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(4);

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.01", "0.0", "0.11", "0.21"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.2f, 0.1f, 0.11f, 0.81f}, {0.12f, 0.2f, 0.33f, 0.31f}};
            expectedData.Target.Weights = TWeights<float>({1.0f, 0.89f, 0.8f, 0.72f});
            expectedData.Target.GroupWeights = TWeights<float>(4);

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetNonGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.Timestamp = TVector<ui64>{20, 30};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.22f, 0.0f},
                TVector<float>{0.28f, 0.77f},
                TVector<float>{0.1f, 0.62f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Female", "Female"},
                TVector<TStringBuf>{"Germany", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(2);

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.92", "0.04"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.32f, 0.01f}, {0.13f, 0.19f}};
            expectedData.Target.Weights = TWeights<float>({0.66f, 0.51f});
            expectedData.Target.GroupWeights = TWeights<float>(2);

            testCase.ExpectedData.push_back(std::move(expectedData));
        }

        testCase.BlockSize = 4;

        Test(testCase);
    }

    Y_UNIT_TEST(TestGroupedBigBlock) {
        TTestCase testCase;

        testCase.SrcData = GetGroupedSrcData();

        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "group0",
                "group0",
                "group0",
                "group1",
                "group1",
                "group1",
                "group2",
                "group2",
                "group3"
            };
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
                "subgroup0",
                "subgroup1",
                "subgroup0",
                "subgroup2",
                "subgroup3",
                "subgroup0",
                "subgroup10",
                "subgroup4",
                "subgroup5"
            };
            expectedData.Objects.Timestamp = TVector<ui64>{0, 0, 0, 10, 10, 10, 30, 30, 20};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.81f, 0.52f, 0.33f, 0.66f, 0.97f, 0.81f, 0.22f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f, 0.37f, 0.19f, 0.28f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f, 0.82f, 0.44f, 0.1f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{
                    "Male", "Female", "Male", "Male", "Female", "Female", "Male", "Male", "Female"
                },
                TVector<TStringBuf>{
                    "Germany", "Russia", "USA", "Finland", "USA", "UK", "Greece", "Italy", "Germany"
                }
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 3}, {3, 6}, {6, 8}, {8, 9}}
            );

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{
                "0.12", "0.22", "0.34", "0.42", "0.01", "0.0", "0.11", "0.21", "0.92"
            }};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {
                {0.0f, 0.12f, 0.1f, 0.17f, 0.2f, 0.1f, 0.11f, 0.81f, 0.32f},
                {0.1f, 0.23f, 0.11f, 0.29f, 0.12f, 0.2f, 0.33f, 0.31f, 0.13f}
            };
            expectedData.Target.Weights = TWeights<float>(
                {0.5f, 0.22f, 0.67f, 0.8f, 1.0f, 0.89f, 0.8f, 0.72f, 0.66f}
            );
            expectedData.Target.GroupWeights = TWeights<float>(
                {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.3f}
            );

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group4"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup7"};
            expectedData.Objects.Timestamp = TVector<ui64>{10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.0f},
                TVector<float>{0.77f},
                TVector<float>{0.62f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Female"},
                TVector<TStringBuf>{"UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 1}}
            );

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.04"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.01f}, {0.19f}};
            expectedData.Target.Weights = TWeights<float>(TVector<float>{0.51f});
            expectedData.Target.GroupWeights = TWeights<float>(TVector<float>{0.5f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }

        testCase.BlockSize = 100;

        Test(testCase);
    }

    Y_UNIT_TEST(TestGroupedMultiBlock) {
        TTestCase testCase;

        testCase.SrcData = GetGroupedSrcData();

        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group0", "group0", "group0"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup0", "subgroup1", "subgroup0"};
            expectedData.Objects.Timestamp = TVector<ui64>{0, 0, 0};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.81f},
                TVector<float>{0.2f, 0.82f, 0.22f},
                TVector<float>{0.11f, 0.33f, 0.23f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Male"},
                TVector<TStringBuf>{"Germany", "Russia", "USA"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.0f, 0.12f, 0.1f}, {0.1f, 0.23f, 0.11f}};
            expectedData.Target.Weights = TWeights<float>({0.5f, 0.22f, 0.67f});
            expectedData.Target.GroupWeights = TWeights<float>({1.0f, 1.0f, 1.0f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group1", "group1", "group1"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup2", "subgroup3", "subgroup0"};
            expectedData.Objects.Timestamp = TVector<ui64>{10, 10, 10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.52f, 0.33f, 0.66f},
                TVector<float>{0.18f, 0.67f, 0.1f},
                TVector<float>{0.1f, 0.17f, 0.31f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Female"},
                TVector<TStringBuf>{"Finland", "USA", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.42", "0.01", "0.0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.17f, 0.2f, 0.1f}, {0.29f, 0.12f, 0.2f}};
            expectedData.Target.Weights = TWeights<float>({0.8f, 1.0f, 0.89f});
            expectedData.Target.GroupWeights = TWeights<float>({0.0f, 0.0f, 0.0f}, "GroupWeights", true);

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group2", "group2", "group3"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
                "subgroup10",
                "subgroup4",
                "subgroup5"
            };
            expectedData.Objects.Timestamp = TVector<ui64>{30, 30, 20};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.97f, 0.81f, 0.22f},
                TVector<float>{0.37f, 0.19f, 0.28f},
                TVector<float>{0.82f, 0.44f, 0.1f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Male", "Female"},
                TVector<TStringBuf>{"Greece", "Italy", "Germany"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.11", "0.21", "0.92"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.11f, 0.81f, 0.32f}, {0.33f, 0.31f, 0.13f}};
            expectedData.Target.Weights = TWeights<float>({0.8f, 0.72f, 0.66f});
            expectedData.Target.GroupWeights = TWeights<float>({0.2f, 0.2f, 0.3f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group4"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup7"};
            expectedData.Objects.Timestamp = TVector<ui64>{10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.0f},
                TVector<float>{0.77f},
                TVector<float>{0.62f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Female"},
                TVector<TStringBuf>{"UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 1}}
            );

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.04"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.01f}, {0.19f}};
            expectedData.Target.Weights = TWeights<float>(TVector<float>{0.51f});
            expectedData.Target.GroupWeights = TWeights<float>(TVector<float>{0.5f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }

        testCase.BlockSize = 4;

        Test(testCase);
    }

    Y_UNIT_TEST(TestGroupedBlockSmallerThanGroupSize) {
        TTestCase testCase;

        testCase.SrcData = GetGroupedSrcData();

        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group0", "group0", "group0"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup0", "subgroup1", "subgroup0"};
            expectedData.Objects.Timestamp = TVector<ui64>{0, 0, 0};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.81f},
                TVector<float>{0.2f, 0.82f, 0.22f},
                TVector<float>{0.11f, 0.33f, 0.23f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Male"},
                TVector<TStringBuf>{"Germany", "Russia", "USA"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.0f, 0.12f, 0.1f}, {0.1f, 0.23f, 0.11f}};
            expectedData.Target.Weights = TWeights<float>({0.5f, 0.22f, 0.67f});
            expectedData.Target.GroupWeights = TWeights<float>({1.0f, 1.0f, 1.0f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group1", "group1", "group1"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup2", "subgroup3", "subgroup0"};
            expectedData.Objects.Timestamp = TVector<ui64>{10, 10, 10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.52f, 0.33f, 0.66f},
                TVector<float>{0.18f, 0.67f, 0.1f},
                TVector<float>{0.1f, 0.17f, 0.31f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Female"},
                TVector<TStringBuf>{"Finland", "USA", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.42", "0.01", "0.0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.17f, 0.2f, 0.1f}, {0.29f, 0.12f, 0.2f}};
            expectedData.Target.Weights = TWeights<float>({0.8f, 1.0f, 0.89f});
            expectedData.Target.GroupWeights = TWeights<float>({0.0f, 0.0f, 0.0f}, "GroupWeights", true);

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group2", "group2", "group3"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
                "subgroup10",
                "subgroup4",
                "subgroup5"
            };
            expectedData.Objects.Timestamp = TVector<ui64>{30, 30, 20};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.97f, 0.81f, 0.22f},
                TVector<float>{0.37f, 0.19f, 0.28f},
                TVector<float>{0.82f, 0.44f, 0.1f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Male", "Female"},
                TVector<TStringBuf>{"Greece", "Italy", "Germany"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.11", "0.21", "0.92"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.11f, 0.81f, 0.32f}, {0.33f, 0.31f, 0.13f}};
            expectedData.Target.Weights = TWeights<float>({0.8f, 0.72f, 0.66f});
            expectedData.Target.GroupWeights = TWeights<float>({0.2f, 0.2f, 0.3f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group4"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup7"};
            expectedData.Objects.Timestamp = TVector<ui64>{10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.0f},
                TVector<float>{0.77f},
                TVector<float>{0.62f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Female"},
                TVector<TStringBuf>{"UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 1}}
            );

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.04"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.01f}, {0.19f}};
            expectedData.Target.Weights = TWeights<float>(TVector<float>{0.51f});
            expectedData.Target.GroupWeights = TWeights<float>(TVector<float>{0.5f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }

        testCase.BlockSize = 2;

        Test(testCase);
    }

    Y_UNIT_TEST(TestGroupedGroupOverSeveralBlocks) {
        TTestCase testCase;

        TSrcData srcData;
        testCase.SrcData.CdFileData = TStringBuf(
            "0\tTarget\n"
            "1\tTimestamp\n"
            "2\tGroupId\n"
            "3\tSubgroupId\n"
            "4\tGroupWeight\n"
            "5\tBaseline\n"
            "6\tBaseline\n"
            "7\tWeight\n"
            "8\tNum\tfloat0\n"
            "9\tCateg\tGender1\n"
            "10\tNum\tfloat2\n"
            "11\tCateg\tCountry3\n"
            "12\tNum\tfloat4\n"
        );
        testCase.SrcData.DatasetFileData = TStringBuf(
            "0.12\t0\tgroup0\tsubgroup0\t1.0\t0.0\t0.1\t0.5\t0.1\tMale\t0.2\tGermany\t0.11\n"
            "0.22\t0\tgroup0\tsubgroup1\t1.0\t0.12\t0.23\t0.22\t0.97\tFemale\t0.82\tRussia\t0.33\n"

            "0.34\t0\tgroup0\tsubgroup1\t1.0\t0.1\t0.11\t0.67\t0.81\tMale\t0.22\tUSA\t0.23\n"
            "0.42\t10\tgroup0\tsubgroup1\t1.0\t0.17\t0.29\t0.8\t0.52\tMale\t0.18\tFinland\t0.1\n"

            "0.01\t10\tgroup0\tsubgroup2\t1.0\t0.2\t0.12\t1.0\t0.33\tFemale\t0.67\tUSA\t0.17\n"
            "0.0\t10\tgroup0\tsubgroup2\t1.0\t0.1\t0.2\t0.89\t0.66\tFemale\t0.1\tUK\t0.31\n"

            "0.11\t30\tgroup0\tsubgroup10\t1.0\t0.11\t0.33\t0.8\t0.97\tMale\t0.37\tGreece\t0.82\n"
            "0.21\t30\tgroup1\tsubgroup4\t0.2\t0.81\t0.31\t0.72\t0.81\tMale\t0.19\tItaly\t0.44\n"

            "0.92\t20\tgroup1\tsubgroup5\t0.2\t0.32\t0.13\t0.66\t0.22\tFemale\t0.28\tGermany\t0.1\n"
            "0.04\t10\tgroup1\tsubgroup7\t0.2\t0.01\t0.19\t0.51\t0.0\tFemale\t0.77\tUK\t0.62\n"
        );
        testCase.SrcData.DsvFileHasHeader = false;
        testCase.SrcData.ObjectsOrder = EObjectsOrder::Undefined;


        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Ordered;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "group0",
                "group0",
                "group0",
                "group0",
                "group0",
                "group0",
                "group0"
            };
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{
                "subgroup0",
                "subgroup1",
                "subgroup1",
                "subgroup1",
                "subgroup2",
                "subgroup2",
                "subgroup10"
            };
            expectedData.Objects.Timestamp = TVector<ui64>{0, 0, 0, 10, 10, 10, 30};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.1f, 0.97f, 0.81f, 0.52f, 0.33f, 0.66f, 0.97f},
                TVector<float>{0.2f, 0.82f, 0.22f, 0.18f, 0.67f, 0.1f, 0.37f},
                TVector<float>{0.11f, 0.33f, 0.23f, 0.1f, 0.17f, 0.31f, 0.82f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Male", "Male", "Female", "Female", "Male"},
                TVector<TStringBuf>{"Germany", "Russia", "USA", "Finland", "USA", "UK", "Greece"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 7}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.12", "0.22", "0.34", "0.42", "0.01", "0.0", "0.11"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {
                {0.0f, 0.12f, 0.1f, 0.17f, 0.2f, 0.1f, 0.11f},
                {0.1f, 0.23f, 0.11f, 0.29f, 0.12f, 0.2f, 0.33f}
            };
            expectedData.Target.Weights = TWeights<float>({0.5f, 0.22f, 0.67f, 0.8f, 1.0f, 0.89f, 0.8f});
            expectedData.Target.GroupWeights = TWeights<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }
        {
            TExpectedRawData expectedData = GetGroupedCommonExpectedData();

            expectedData.Objects.Order = EObjectsOrder::Undefined;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{"group1", "group1", "group1"};
            expectedData.Objects.SubgroupIds = TVector<TStringBuf>{"subgroup4", "subgroup5", "subgroup7"};
            expectedData.Objects.Timestamp = TVector<ui64>{30, 20, 10};
            expectedData.Objects.FloatFeatures = {
                TVector<float>{0.81f, 0.22f, 0.0f},
                TVector<float>{0.19f, 0.28f, 0.77f},
                TVector<float>{0.44f, 0.1f, 0.62f}
            };
            expectedData.Objects.CatFeatures = {
                TVector<TStringBuf>{"Male", "Female", "Female"},
                TVector<TStringBuf>{"Italy", "Germany", "UK"}
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(TVector<TGroupBounds>{{0, 3}});

            expectedData.Target.TargetType = ERawTargetType::String;

            TVector<TVector<TString>> rawTarget{{"0.21", "0.92", "0.04"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Baseline = {{0.81f, 0.32f, 0.01f}, {0.31f, 0.13f, 0.19f}};
            expectedData.Target.Weights = TWeights<float>({0.72f, 0.66f, 0.51f});
            expectedData.Target.GroupWeights = TWeights<float>({0.2f, 0.2f, 0.2f});

            testCase.ExpectedData.push_back(std::move(expectedData));
        }

        testCase.BlockSize = 2;

        Test(testCase);
    }
}
