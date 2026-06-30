
#include <catboost/libs/data/data_provider.h>

#include <catboost/libs/data/ut/lib/for_objects.h>
#include <catboost/libs/data/ut/lib/for_target.h>

#include <library/cpp/binsaver/util_stream_io.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


template <class TTObjectsDataProvider>
static void CreateQuantizedObjectsDataProviderTestData(
    bool hasPairs,
    TDataMetaInfo* metaInfo,
    TObjectsGroupingPtr* objectsGrouping,
    TIntrusivePtr<TTObjectsDataProvider>* objectsData
) {
    TDataColumnsMetaInfo dataColumnsMetaInfo;
    dataColumnsMetaInfo.Columns = {
        {EColumn::Label, ""},
        {EColumn::GroupId, ""},
        {EColumn::SubgroupId, ""},
        {EColumn::Timestamp, ""},
        {EColumn::Num, "float0"},
        {EColumn::Categ, "Gender1"},
        {EColumn::Num, "float2"},
        {EColumn::Categ, "Country3"},
        {EColumn::Num, "float4"},
    };

    TVector<TString> featureId = {"float0", "Gender1", "float2", "Country3", "float4"};

    *metaInfo = TDataMetaInfo(
        std::move(dataColumnsMetaInfo),
        ERawTargetType::String,
        /*hasAdditionalGroupWeight*/ false,
        /*hasTimestamps*/ false,
        hasPairs,
        /*hasGraph*/ false,
        /*forceUnitAutoPairWeights*/ false,
        /*loadSampleIds*/ false,
        /*additionalBaselineCount*/ Nothing(),
        &featureId
    );

    *objectsGrouping = MakeIntrusive<TObjectsGrouping>(
        TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
    );

    TCommonObjectsData commonObjectsData;
    commonObjectsData.FeaturesLayout = metaInfo->FeaturesLayout;
    commonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
        TIndexedSubset<ui32>{0, 4, 3, 1, 7, 8}
    );

    commonObjectsData.Order = EObjectsOrder::RandomShuffled;
    commonObjectsData.GroupIds.GetMaybeNumData() = TVector<TGroupId>{
        CalcGroupIdFor("query0"),
        CalcGroupIdFor("query0"),
        CalcGroupIdFor("query1"),
        CalcGroupIdFor("Query 2"),
        CalcGroupIdFor("Query 2"),
        CalcGroupIdFor("Query 2")
    };

    commonObjectsData.SubgroupIds.GetMaybeNumData() = TVector<TSubgroupId>{0, 12, 18, 21, 0, 2};
    commonObjectsData.Timestamp = TVector<ui64>{10, 20, 10, 30, 50, 70};


    TQuantizedObjectsData quantizedObjectsData;

    TVector<TVector<ui8>> quantizedFloatFeatures = {
        TVector<ui8>{1, 1, 0, 0, 3, 2, 4, 4, 2, 0, 3, 4, 1},
        TVector<ui8>{0, 0, 2, 3, 1, 1, 4, 4, 2, 2, 0, 3, 1},
        TVector<ui8>{5, 1, 2, 0, 4, 3, 0, 5, 2, 1, 3, 3, 4}
    };

    TVector<TVector<ui32>> quantizedCatFeatures = {
        TVector<ui32>{0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 3, 1},
        TVector<ui32>{0, 1, 2, 3, 4, 5, 2, 0, 2, 3, 6, 7, 8}
    };

    InitQuantizedFeatures(
        quantizedFloatFeatures,
        commonObjectsData.SubsetIndexing.Get(),
        {0, 2, 4},
        &quantizedObjectsData.FloatFeatures
    );

    InitQuantizedFeatures(
        quantizedCatFeatures,
        commonObjectsData.SubsetIndexing.Get(),
        {1, 3},
        &quantizedObjectsData.CatFeatures
    );


    NCatboostOptions::TBinarizationOptions binarizationOptions(
        EBorderSelectionType::GreedyLogSum,
        4,
        ENanMode::Min
    );

    quantizedObjectsData.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
        *metaInfo->FeaturesLayout,
        TConstArrayRef<ui32>(),
        binarizationOptions
    );

    TVector<TVector<float>> borders = {
        {0.1149999946f, 0.4449999928f, 0.7849999666f, 1.049999952f},
        {0.1299999952f, 0.1949999928f, 0.875f, 1.565000057f},
        {
            std::numeric_limits<float>::lowest(),
            0.0549999997f,
            0.1550000012f,
            0.3199999928f,
            0.6349999905f
        }
    };
    TVector<ENanMode> nanModes = {ENanMode::Forbidden, ENanMode::Forbidden, ENanMode::Min};

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
        },
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
        }
    };

    for (auto i : xrange(3)) {
        auto floatFeatureIdx = TFloatFeatureIdx(i);
        quantizedObjectsData.QuantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(borders[i]));
        quantizedObjectsData.QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanModes[i]);
    }

    for (auto i : xrange(2)) {
        auto catFeatureIdx = TCatFeatureIdx(i);
        quantizedObjectsData.QuantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(
            catFeatureIdx,
            std::move(expectedPerfectHash[i])
        );
    }

    quantizedObjectsData.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
        *metaInfo->FeaturesLayout,
        TVector<TExclusiveFeaturesBundle>()
    );
    quantizedObjectsData.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
        *metaInfo->FeaturesLayout,
        *quantizedObjectsData.QuantizedFeaturesInfo,
        quantizedObjectsData.ExclusiveFeatureBundlesData,
        true
    );
    quantizedObjectsData.FeaturesGroupsData = TFeatureGroupsData(
        *metaInfo->FeaturesLayout,
        TVector<TFeaturesGroup>()
    );

    *objectsData = MakeIntrusive<TQuantizedObjectsDataProvider>(
        *objectsGrouping,
        std::move(commonObjectsData),
        std::move(quantizedObjectsData),
        true,
        Nothing()
    );
}


static TRawTargetData CreateRawTargetData() {
    TRawTargetData targetData;
    targetData.TargetType = ERawTargetType::String;
    TVector<TVector<TString>> rawTarget{{"0.2", "0.5", "1.0", "0.0", "0.8", "0.3"}};
    targetData.Target.assign(rawTarget.begin(), rawTarget.end());
    targetData.Baseline = TVector<TVector<float>>{
        TVector<float>{0.12f, 0.0f, 0.11f, 0.31f, 0.2f, 0.9f}
    };
    targetData.Weights = TWeights<float>{TVector<float>{1.0f, 0.0f, 0.2f, 0.12f, 0.45f, 0.89f}};
    targetData.GroupWeights = TWeights<float>(6);
    targetData.Pairs = TFlatPairsInfo{TPair{0, 1, 0.1f}, TPair{3, 4, 1.0f}, TPair{3, 5, 2.0f}};

    return targetData;
}


static TRawTargetData CreateRawMultiTargetData() {
    TRawTargetData targetData;
    targetData.TargetType = ERawTargetType::String;
    TVector<TVector<TString>> rawTarget{{"0.2", "0.5", "1.0", "0.0", "0.8", "0.3"}, {"-0.2", "-0.5", "-1.0", "-0.0", "-0.8", "-0.3"}};
    targetData.Target.assign(rawTarget.begin(), rawTarget.end());
    targetData.Baseline = TVector<TVector<float>>{
        TVector<float>{0.12f, 0.0f, 0.11f, 0.31f, 0.2f, 0.9f}
    };
    targetData.Weights = TWeights<float>{TVector<float>{1.0f, 0.0f, 0.2f, 0.12f, 0.45f, 0.89f}};
    targetData.GroupWeights = TWeights<float>(6);
    targetData.Pairs = TFlatPairsInfo{TPair{0, 1, 0.1f}, TPair{3, 4, 1.0f}, TPair{3, 5, 2.0f}};

    return targetData;
}


Y_UNIT_TEST_SUITE(TDataProviderTemplate) {
    template <class TTObjectsDataProvider, class TTargetFactory>
    void TestEqual(TTargetFactory targetFactory) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(3);

        THolder<TDataProviderTemplate<TTObjectsDataProvider>> dataProviders[2];
        for (auto i : xrange(2)) {
            TDataMetaInfo metaInfo;
            TObjectsGroupingPtr objectsGrouping;
            TIntrusivePtr<TTObjectsDataProvider> objectsData;

            CreateQuantizedObjectsDataProviderTestData(true, &metaInfo, &objectsGrouping, &objectsData);

            TRawTargetData rawTargetData = targetFactory();

            dataProviders[i] = MakeHolder<TDataProviderTemplate<TTObjectsDataProvider>>(
                std::move(metaInfo),
                std::move(objectsData),
                objectsGrouping,
                TRawTargetDataProvider(objectsGrouping, std::move(rawTargetData), false, false, &localExecutor)
            );
        }

        UNIT_ASSERT_EQUAL(*dataProviders[0], *dataProviders[1]);

        dataProviders[1]->RawTargetData.SetWeights(TVector<float>{1.0f, 0.0f, 2.0f, 1.1f, 0.67f, 0.0f});

        UNIT_ASSERT_UNEQUAL(*dataProviders[0], *dataProviders[1]);
    }

    Y_UNIT_TEST(Equal) {
        TestEqual<TQuantizedObjectsDataProvider>(CreateRawTargetData);
        TestEqual<TQuantizedObjectsDataProvider>(CreateRawMultiTargetData);
    }
}



Y_UNIT_TEST_SUITE(TProcessedDataProviderTemplate) {

    TVector<TSharedVector<float>> MakeTarget(const TVector<TVector<float>>& target) {
        auto processedTarget = TVector<TSharedVector<float>>();
        processedTarget.reserve(target.size());
        for (const auto& subTarget : target) {
            processedTarget.emplace_back(MakeAtomicShared<TVector<float>>(subTarget));
        }
        return processedTarget;
    }

    template <class TTObjectsDataProvider>
    void Compare(
        const TProcessedDataProviderTemplate<TTObjectsDataProvider>& lhs,
        const TProcessedDataProviderTemplate<TTObjectsDataProvider>& rhs
    ) {
        UNIT_ASSERT_EQUAL(lhs.MetaInfo, rhs.MetaInfo);
        UNIT_ASSERT_EQUAL(*lhs.ObjectsGrouping, *rhs.ObjectsGrouping);
        NCB::NDataNewUT::Compare(*lhs.ObjectsData, *rhs.ObjectsData);
        UNIT_ASSERT_EQUAL(*lhs.TargetData, *rhs.TargetData);
    }

    template <class TTObjectsDataProvider>
    void TestSerializationCase(
        TProcessedDataProviderTemplate<TTObjectsDataProvider>& trainingDataProvider
    ) {
        TBuffer buffer;

        {
            TBufferOutput out(buffer);
            SerializeToArcadiaStream(out, trainingDataProvider);
        }

        TProcessedDataProviderTemplate<TTObjectsDataProvider> trainingDataProvider2;

        {
            TBufferInput in(buffer);
            SerializeFromStream(in, trainingDataProvider2);
        }

        Compare(trainingDataProvider, trainingDataProvider2);
    }


    template <class TTObjectsDataProvider>
    void TestSerialization() {
        TProcessedDataProviderTemplate<TTObjectsDataProvider> trainingDataProvider;

        CreateQuantizedObjectsDataProviderTestData(
            false,
            &trainingDataProvider.MetaInfo,
            &trainingDataProvider.ObjectsGrouping,
            &trainingDataProvider.ObjectsData
        );

        TSharedWeights<float> weights = Share(
            TWeights<float>({1.0f, 2.0f, 1.0f, 1.2f, 2.1f, 0.0f})
        );

        auto targets = MakeTarget({{0.0f, 0.2f, 0.1f, 0.3f, 0.12f, 0.0f}});

        // 3 classes
        auto multiClassTargets = MakeTarget({{1.f, 0.0f, 1.0f, 2.0f, 2.0f, 0.0f}});

        TVector<TSharedVector<float>> baselines = {
            ShareVector<float>({0.1f, 0.2f, 0.4f, 1.0f, 2.1f, 3.3f}),
            ShareVector<float>({0.0f, 0.11f, 0.04f, 0.12f, 0.6f, 0.82f}),
            ShareVector<float>({0.22f, 0.71f, 0.0f, 0.05f, 0.0f, 0.0f}),
        };


        TVector<TQueryInfo> groupInfo(3);

        groupInfo[0].Begin = 0;
        groupInfo[0].End = 2;
        groupInfo[0].Weight = 2.0f;
        groupInfo[0].SubgroupId = {0, 12};

        groupInfo[1].Begin = 2;
        groupInfo[1].End = 3;
        groupInfo[1].Weight = 1.0f;
        groupInfo[1].SubgroupId = {18};

        groupInfo[2].Begin = 3;
        groupInfo[2].End = 6;
        groupInfo[2].Weight = 3.0f;
        groupInfo[2].SubgroupId = {21, 0, 2};
        groupInfo[2].Competitors = {
            { TCompetitor(2, 1.0f) },
            { TCompetitor(0, 1.0f), TCompetitor(2, 3.0f) },
            {}
        };

        TSharedVector<TQueryInfo> sharedGroupInfo = ShareVector(std::move(groupInfo));


        // BinClass
        {
            TProcessedTargetData processedTargetData;
            processedTargetData.TargetsClassCount.emplace("", 2);
            processedTargetData.Targets.emplace("", targets);
            processedTargetData.Weights.emplace("", weights);
            processedTargetData.Baselines.emplace("", TVector<TSharedVector<float>>(1, baselines[0]));

            trainingDataProvider.TargetData = MakeIntrusive<TTargetDataProvider>(
                trainingDataProvider.ObjectsGrouping,
                std::move(processedTargetData)
            );

            TestSerializationCase(trainingDataProvider);
        }

        // MultiClass
        {
            TProcessedTargetData processedTargetData;
            processedTargetData.TargetsClassCount.emplace("", 3);
            processedTargetData.Targets.emplace("", targets);
            processedTargetData.Weights.emplace("", weights);
            processedTargetData.Baselines.emplace("", baselines);

            trainingDataProvider.TargetData = MakeIntrusive<TTargetDataProvider>(
                trainingDataProvider.ObjectsGrouping,
                std::move(processedTargetData)
            );

            TestSerializationCase(trainingDataProvider);
        }

        // GroupwiseRanking
        {
            TProcessedTargetData processedTargetData;
            processedTargetData.Targets.emplace("", targets);
            processedTargetData.Weights.emplace("", weights);
            processedTargetData.Baselines.emplace("", TVector<TSharedVector<float>>(1, baselines[1]));
            processedTargetData.GroupInfos.emplace("", sharedGroupInfo);

            trainingDataProvider.TargetData = MakeIntrusive<TTargetDataProvider>(
                trainingDataProvider.ObjectsGrouping,
                std::move(processedTargetData)
            );

            TestSerializationCase(trainingDataProvider);
        }

        // GroupPairwiseRanking
        {
            TProcessedTargetData processedTargetData;
            processedTargetData.Baselines.emplace("", TVector<TSharedVector<float>>(1, baselines[0]));
            processedTargetData.GroupInfos.emplace("", sharedGroupInfo);

            trainingDataProvider.TargetData = MakeIntrusive<TTargetDataProvider>(
                trainingDataProvider.ObjectsGrouping,
                std::move(processedTargetData)
            );

            TestSerializationCase(trainingDataProvider);
        }
    }

    template <class TTObjectsDataProvider>
    void TestMultiTargetSerialization() {
        TProcessedDataProviderTemplate<TTObjectsDataProvider> trainingDataProvider;

        CreateQuantizedObjectsDataProviderTestData(
            false,
            &trainingDataProvider.MetaInfo,
            &trainingDataProvider.ObjectsGrouping,
            &trainingDataProvider.ObjectsData
        );

        TSharedWeights<float> weights = Share(
            TWeights<float>({1.0f, 2.0f, 1.0f, 1.2f, 2.1f, 0.0f})
        );

        auto targets = MakeTarget({{0.0f, 0.2f, 0.1f, 0.3f, 0.12f, 0.0f}, {-0.0f, -0.2f, -0.1f, -0.3f, -0.12f, -0.0f}});

        TVector<TSharedVector<float>> baselines = {
            ShareVector<float>({0.1f, 0.2f, 0.4f, 1.0f, 2.1f, 3.3f}),
            ShareVector<float>({0.0f, 0.11f, 0.04f, 0.12f, 0.6f, 0.82f}),
            ShareVector<float>({0.22f, 0.71f, 0.0f, 0.05f, 0.0f, 0.0f}),
        };

        {
            TProcessedTargetData processedTargetData;
            processedTargetData.Targets.emplace("", targets);
            processedTargetData.Weights.emplace("", weights);
            processedTargetData.Baselines.emplace("", TVector<TSharedVector<float>>(1, baselines[0]));

            trainingDataProvider.TargetData = MakeIntrusive<TTargetDataProvider>(
                trainingDataProvider.ObjectsGrouping,
                std::move(processedTargetData)
            );

            TestSerializationCase(trainingDataProvider);
        }
    }

    Y_UNIT_TEST(Serialization) {
        TestSerialization<TQuantizedObjectsDataProvider>();
    }

    Y_UNIT_TEST(MultiTargetSerialization) {
        TestMultiTargetSerialization<TQuantizedObjectsDataProvider>();
    }
}
