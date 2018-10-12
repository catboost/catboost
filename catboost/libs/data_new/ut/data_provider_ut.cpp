
#include <catboost/libs/data_new/data_provider.h>

#include <catboost/libs/data_new/ut/lib/for_objects.h>
#include <catboost/libs/data_new/ut/lib/for_target.h>

#include <library/binsaver/util_stream_io.h>

#include <library/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(TTrainingDataProviderTemplate) {

    template <class TTObjectsDataProvider>
    void Compare(
        const TTrainingDataProviderTemplate<TTObjectsDataProvider>& lhs,
        const TTrainingDataProviderTemplate<TTObjectsDataProvider>& rhs
    ) {
        UNIT_ASSERT_EQUAL(lhs.MetaInfo, rhs.MetaInfo);
        UNIT_ASSERT_EQUAL(*lhs.ObjectsGrouping, *rhs.ObjectsGrouping);
        NCB::NDataNewUT::Compare(*lhs.ObjectsData, *rhs.ObjectsData);
        CompareTargetDataProviders(lhs.TargetData, rhs.TargetData);
    }

    template <class TTObjectsDataProvider>
    void TestSerialization() {
        TTrainingDataProviderTemplate<TTObjectsDataProvider> trainingDataProvider;

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

        trainingDataProvider.MetaInfo = TDataMetaInfo(
            std::move(dataColumnsMetaInfo),
            false,
            false,
            &featureId
        );

        trainingDataProvider.ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(
            TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 6}}
        );

        TCommonObjectsData commonObjectsData;
        commonObjectsData.FeaturesLayout = trainingDataProvider.MetaInfo.FeaturesLayout;
        commonObjectsData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1, 7, 8}
        );

        commonObjectsData.Order = EObjectsOrder::RandomShuffled;
        commonObjectsData.GroupIds = TVector<TGroupId>{
            CalcGroupIdFor("query0"),
            CalcGroupIdFor("query0"),
            CalcGroupIdFor("query1"),
            CalcGroupIdFor("Query 2"),
            CalcGroupIdFor("Query 2"),
            CalcGroupIdFor("Query 2")
        };

        commonObjectsData.SubgroupIds = TVector<TSubgroupId>{0, 12, 18, 21, 0, 2};
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
            trainingDataProvider.MetaInfo.FeaturesLayout,
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

        TVector<TMap<ui32, ui32>> expectedPerfectHash = {
            {{12, 0}, {25, 1}, {10, 2}, {8, 3}, {165, 4}, {1, 5}, {0, 6}, {112, 7}, {23, 8}},
            {{256, 0}, {45, 1}, {9, 2}, {110, 3}, {50, 4}, {10, 5}, {257, 6}, {90, 7}, {0, 8}}
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

        trainingDataProvider.ObjectsData = MakeIntrusive<TTObjectsDataProvider>(
            trainingDataProvider.ObjectsGrouping,
            std::move(commonObjectsData),
            std::move(quantizedObjectsData),
            true,
            Nothing()
        );

        TSharedWeights<float> weights = Share(
            TWeights<float>({1.0f, 2.0f, 1.0f, 1.2f, 2.1f, 0.0f})
        );

        TSharedVector<float> targets = ShareVector<float>({0.0f, 0.2f, 0.1f, 0.3f, 0.12f, 0.0f});
        TVector<TSharedVector<float>> baselines = {
            ShareVector<float>({0.1f, 0.2f, 0.4f, 1.0f, 2.1f, 3.3f}),
            ShareVector<float>({0.0f, 0.11f, 0.04f, 0.12f, 0.6f, 0.82f})
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


        trainingDataProvider.TargetData.emplace(
            TTargetDataSpecification(ETargetType::BinClass),
            MakeIntrusive<TBinClassTarget>(
                "",
                trainingDataProvider.ObjectsGrouping,
                targets,
                weights,
                baselines[0]
            )
        );

        trainingDataProvider.TargetData.emplace(
            TTargetDataSpecification(ETargetType::MultiClass),
            MakeIntrusive<TMultiClassTarget>(
                "",
                trainingDataProvider.ObjectsGrouping,
                targets,
                weights,
                TVector<TSharedVector<float>>(baselines)
            )
        );
        trainingDataProvider.TargetData.emplace(
            TTargetDataSpecification(ETargetType::Regression),
            MakeIntrusive<TRegressionTarget>(
                "",
                trainingDataProvider.ObjectsGrouping,
                targets,
                weights,
                baselines[0]
            )
        );

        trainingDataProvider.TargetData.emplace(
            TTargetDataSpecification(ETargetType::GroupwiseRanking),
            MakeIntrusive<TGroupwiseRankingTarget>(
                "",
                trainingDataProvider.ObjectsGrouping,
                targets,
                weights,
                baselines[1], // take 2nd baseline just for diversity with BinClassTarget
                sharedGroupInfo
            )
        );

        trainingDataProvider.TargetData.emplace(
            TTargetDataSpecification(ETargetType::GroupPairwiseRanking),
            MakeIntrusive<TGroupPairwiseRankingTarget>(
                "",
                trainingDataProvider.ObjectsGrouping,
                baselines[0],
                sharedGroupInfo
            )
        );


        TBuffer buffer;

        {
            TBufferOutput out(buffer);
            SerializeToStream(out, trainingDataProvider);
        }

        TTrainingDataProviderTemplate<TTObjectsDataProvider> trainingDataProvider2;

        {
            TBufferInput in(buffer);
            SerializeFromStream(in, trainingDataProvider2);
        }

        Compare(trainingDataProvider, trainingDataProvider2);
    }

    Y_UNIT_TEST(Serialization) {
        TestSerialization<TQuantizedObjectsDataProvider>();
        TestSerialization<TQuantizedForCPUObjectsDataProvider>();
    }
}
