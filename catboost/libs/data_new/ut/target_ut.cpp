#include "util.h"

#include <catboost/libs/data_new/target.h>

#include <catboost/libs/data_new/ut/lib/for_target.h>

#include <catboost/libs/data_new/util.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/hash.h>
#include <util/generic/xrange.h>

#include <utility>

#include <library/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(TRawTargetData) {
    // rawTargetData passed by value intentionally to be moved into created provider
    TRawTargetDataProvider CreateProviderSimple(
        ui32 objectCount,
        TRawTargetData rawTargetData
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        return TRawTargetDataProvider(
            MakeIntrusive<TObjectsGrouping>(objectCount),
            std::move(rawTargetData),
            false,
            &localExecutor
        );
    }

    TRawTargetDataProvider CreateProviderSimple(
        const TVector<TGroupBounds>& groupBounds,
        TRawTargetData rawTargetData
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        return TRawTargetDataProvider(
            MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>(groupBounds)),
            std::move(rawTargetData),
            false,
            &localExecutor
        );
    }


    Y_UNIT_TEST(Empty) {
        TRawTargetData rawTargetData;
        CreateProviderSimple(0, rawTargetData);
    }

    Y_UNIT_TEST(Target) {
        {
            TVector<TVector<TString>> goodTargets;
            goodTargets.push_back({});
            goodTargets.push_back({"0", "1", "0", "1"});
            goodTargets.push_back({"0.0", "1.0", "3.8"});
            goodTargets.push_back({"male", "male", "male", "female"});

            for (const auto& target : goodTargets) {
                TRawTargetData rawTargetData;
                rawTargetData.Target = target;
                rawTargetData.SetTrivialWeights(target.size());

                auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                UNIT_ASSERT(Equal(*rawTargetDataProvider.GetTarget(), target));
            }
        }

        {
            TVector<TVector<TString>> badTargets;
            badTargets.push_back({"0", "", "0", "1"});

            for (const auto& target : badTargets) {
                TRawTargetData rawTargetData;
                rawTargetData.Target = target;
                rawTargetData.SetTrivialWeights(target.size());

                UNIT_ASSERT_EXCEPTION(
                    CreateProviderSimple(target.size(), rawTargetData),
                    TCatBoostException
                );
            }
        }
    }

    Y_UNIT_TEST(Baseline) {
        TVector<TString> target = {"0", "1", "0", "1"};

        {
            TVector<TVector<TVector<float>>> goodBaselines;
            goodBaselines.push_back({}); // Baseline field will not be set
            goodBaselines.push_back({
                {0.0f, 0.1f, 0.3f, 0.2f}
            });
            goodBaselines.push_back({
                {0.0f, 0.1f, 0.3f, 0.2f},
                {1.0f, 2.1f, 1.3f, 2.2f}
            });

            for (const auto& baseline : goodBaselines) {
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    if (!baseline.empty()) {
                        rawTargetData.Baseline = baseline;
                    }
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                    auto baseline2 = rawTargetDataProvider.GetBaseline();
                    if (baseline.empty()) {
                        UNIT_ASSERT(!baseline2);
                    } else {
                        UNIT_ASSERT(baseline2);
                        UNIT_ASSERT_EQUAL(baseline.size(), baseline2->size());

                        for (auto i : xrange(baseline.size())) {
                            UNIT_ASSERT(Equal((*baseline2)[i], baseline[i]));
                        }
                    }
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                    TVector<TConstArrayRef<float>> baselineView(baseline.size());
                    for (auto i : xrange(baseline.size())) {
                        baselineView[i] = baseline[i];
                    }

                    rawTargetDataProvider.SetBaseline(baselineView);
                    auto baseline2 = rawTargetDataProvider.GetBaseline();

                    if (baseline.empty()) {
                        UNIT_ASSERT(!baseline2);
                    } else {
                        UNIT_ASSERT(baseline2);
                        UNIT_ASSERT(Equal(*baseline2, baselineView));
                    }
                }
            }
        }

        {
            TVector<TVector<TVector<float>>> badBaselines;
            badBaselines.push_back({{}});
            badBaselines.push_back({
                {0.0f, 0.1f, 0.3f}
            });
            badBaselines.push_back({
                {0.0f, 0.1f, 0.3f},
                {1.0f, 2.1f, 1.3f, 2.2f}
            });

            for (const auto& baseline : badBaselines) {
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.Baseline = baseline;
                    rawTargetData.SetTrivialWeights(target.size());

                    UNIT_ASSERT_EXCEPTION(
                        CreateProviderSimple(target.size(), rawTargetData),
                        TCatBoostException
                    );
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                    TVector<TConstArrayRef<float>> baselineView(baseline.size());
                    for (auto i : xrange(baseline.size())) {
                        baselineView[i] = baseline[i];
                    }

                    UNIT_ASSERT_EXCEPTION(
                        rawTargetDataProvider.SetBaseline(baselineView),
                        TCatBoostException
                    );
                }
            }
        }
    }

    Y_UNIT_TEST(Weight) {
        TVector<TString> target = {"0", "1", "2", "1", "2"};

        // trivial
        {
            TRawTargetData rawTargetData;
            rawTargetData.Target = target;
            rawTargetData.SetTrivialWeights(target.size());

            auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

            UNIT_ASSERT(rawTargetDataProvider.GetWeights().IsTrivial());
        }

        // good
        {
            TVector<float> weight = {1.0, 0.0, 2.0, 3.0, 1.0};
            {
                TRawTargetData rawTargetData;
                rawTargetData.Target = target;
                rawTargetData.Weights = TWeights<float>(TVector<float>(weight));
                rawTargetData.GroupWeights = TWeights<float>(target.size());

                auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                UNIT_ASSERT(!rawTargetDataProvider.GetWeights().IsTrivial());
                UNIT_ASSERT(Equal(rawTargetDataProvider.GetWeights().GetNonTrivialData(), weight));
            }

            // check Set
            {
                TRawTargetData rawTargetData;
                rawTargetData.Target = target;
                rawTargetData.SetTrivialWeights(target.size());

                auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                rawTargetDataProvider.SetWeights(weight);

                UNIT_ASSERT(!rawTargetDataProvider.GetWeights().IsTrivial());
                UNIT_ASSERT(Equal(rawTargetDataProvider.GetWeights().GetNonTrivialData(), weight));
            }
        }

        // bad
        {
            TVector<TVector<float>> badWeights;
            badWeights.push_back( {} ); // wrong size
            badWeights.push_back( {1.0, 0.0, 2.0} ); // wrong size
            badWeights.push_back( {1.0, -2.0, 3.0, 4.0, -1.0} ); // negative weights

            for (const auto& weight : badWeights) {
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.GroupWeights = TWeights<float>(target.size());

                    if (weight.size() != target.size()) {
                        rawTargetData.Weights = TWeights<float>(TVector<float>(weight));

                        UNIT_ASSERT_EXCEPTION(
                            CreateProviderSimple(target.size(), rawTargetData),
                            TCatBoostException
                        );
                    } else {
                        UNIT_ASSERT_EXCEPTION(
                            [&]() {
                                rawTargetData.Weights = TWeights<float>(TVector<float>(weight));
                            } (),
                            TCatBoostException
                        );
                    }
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                    UNIT_ASSERT_EXCEPTION(rawTargetDataProvider.SetWeights(weight), TCatBoostException);
                }
            }
        }
    }

    Y_UNIT_TEST(GroupWeight) {
        TVector<TString> target = {"0", "1", "2", "1", "2"};

        {
            TVector<TVector<float>> goodGroupWeights;
            goodGroupWeights.push_back( {1.0f, 1.0f, 2.0f, 2.0f, 0.0f} );

            TVector<TGroupBounds> groupBounds = {{0, 2}, {2, 4}, {4, 5}};

            for (const auto& groupWeights : goodGroupWeights) {
                if (!groupWeights.empty()) {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.Weights = TWeights<float>(target.size());
                    rawTargetData.GroupWeights = TWeights<float>(TVector<float>(groupWeights));

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    UNIT_ASSERT(!rawTargetDataProvider.GetGroupWeights().IsTrivial());
                    UNIT_ASSERT(
                        Equal(rawTargetDataProvider.GetGroupWeights().GetNonTrivialData(), groupWeights)
                    );
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    rawTargetDataProvider.SetGroupWeights(groupWeights);

                    UNIT_ASSERT_VALUES_EQUAL(
                        groupWeights.empty(),
                        rawTargetDataProvider.GetGroupWeights().IsTrivial()
                    );
                    if (!groupWeights.empty()) {
                        UNIT_ASSERT(
                            Equal(rawTargetDataProvider.GetGroupWeights().GetNonTrivialData(), groupWeights)
                        );
                    }
                }
            }
        }

        {
            TVector<TVector<float>> badGroupWeights;
            badGroupWeights.push_back( {1.0f, 1.0f, 3.0f, 3.0f, -1.0f} ); // negative weights
            badGroupWeights.push_back( {1.0f, 1.0f, 2.0f} ); // wrong size
            badGroupWeights.push_back( {} ); // wrong size
            badGroupWeights.push_back( {1.0, 1.0, 3.0, 2.0, 1.0} ); // different in one group

            TVector<TGroupBounds> groupBounds = {{0, 2}, {2, 4}, {4, 5}};

            for (auto testSetIdx : xrange(badGroupWeights.size())) {
                const auto& groupWeights = badGroupWeights[testSetIdx];
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.Weights = TWeights<float>(target.size());

                    if (testSetIdx != 0) {
                        rawTargetData.GroupWeights = TWeights<float>(TVector<float>(groupWeights));

                        UNIT_ASSERT_EXCEPTION(
                            CreateProviderSimple(groupBounds, rawTargetData),
                            TCatBoostException
                        );
                    } else {
                        UNIT_ASSERT_EXCEPTION(
                            [&]() {
                                rawTargetData.GroupWeights = TWeights<float>(TVector<float>(groupWeights));
                            } (),
                            TCatBoostException
                        );
                    }

                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target;
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    UNIT_ASSERT_EXCEPTION(
                        rawTargetDataProvider.SetGroupWeights(groupWeights),
                        TCatBoostException
                    );
                }
            }
        }
    }

    Y_UNIT_TEST(Pairs) {
        // use only for Set testing, we need non-empty pairs or target for TRawTargetDataProvider creation
        TVector<TString> target = {"0", "1", "2", "1", "2"};

        TVector<TGroupBounds> groupBounds = {{0, 2}, {2, 4}, {4, 5}};

        {
            TVector<TVector<TPair>> goodPairs;
            goodPairs.push_back( {TPair(0, 1, 0.0f), TPair(2, 3, 1.0f)} );

            for (const auto& pairs : goodPairs) {
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.SetTrivialWeights(5);
                    rawTargetData.Pairs = pairs;

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    UNIT_ASSERT(EqualAsMultiSets(rawTargetDataProvider.GetPairs(), pairs));
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target; // for TRawTargetDataProvider creation
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    rawTargetDataProvider.SetPairs(pairs);

                    UNIT_ASSERT(EqualAsMultiSets(rawTargetDataProvider.GetPairs(), pairs));
                }
            }
        }

        {
            TVector<TVector<TPair>> badPairs;
            badPairs.push_back( {TPair(6, 0, 0.0f), TPair(1, 2, 1.0f)} ); // bad indices
            badPairs.push_back( {TPair(0, 1, 0.0f), TPair(1, 12, 1.0f), TPair(2, 12, 1.0f)} ); // bad indices
            badPairs.push_back( {TPair(0, 2, 0.0f), TPair(1, 3, 1.0f)} ); // not in one group
            badPairs.push_back( {TPair(0, 1, -2.2f)} ); // bad weight
            badPairs.push_back( {TPair(1, 1, 2.2f)} ); // winnerId = loserId

            for (const auto& pairs : badPairs) {
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.SetTrivialWeights(5);
                    rawTargetData.Pairs = pairs;

                    UNIT_ASSERT_EXCEPTION(
                        CreateProviderSimple(groupBounds, rawTargetData),
                        TCatBoostException
                    );
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.Target = target; // for TRawTargetDataProvider creation
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    UNIT_ASSERT_EXCEPTION(rawTargetDataProvider.SetPairs(pairs), TCatBoostException);
                }
            }
        }
    }

    Y_UNIT_TEST(GetSubset) {
        TVector<TRawTargetData> rawTargetDataVector;

        {
            TRawTargetData rawTargetData;
            rawTargetData.Target = {"0", "1", "1", "0", "1", "0"};
            rawTargetData.SetTrivialWeights(6);
            rawTargetData.Baseline = {
                {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f},
                {1.0f, 2.1f, 1.3f, 2.2f, 3.3f, 4.7f}
            };

            rawTargetDataVector.push_back(rawTargetData);
        }

        {
            TRawTargetData rawTargetData;
            rawTargetData.Target = {"0.0", "1.0", "1.0", "0.0", "1.0", "0.0", "1.0f", "0.5", "0.8"};
            rawTargetData.Baseline = {{0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f}};
            rawTargetData.Weights = TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.8f, 0.9f, 0.1f});
            rawTargetData.GroupWeights = TWeights<float>(
                {1.0f, 3.0f, 2.0f, 2.1f, 2.1f, 2.1f, 0.0f, 1.1f, 1.1f}
            );
            rawTargetData.Pairs = {TPair(7, 8, 0.0f), TPair(3, 5, 1.0f), TPair(3, 4, 2.0f)};

            rawTargetDataVector.push_back(rawTargetData);
        }

        TVector<TObjectsGroupingPtr> targetDataGroupingVector;
        targetDataGroupingVector.push_back(MakeIntrusive<TObjectsGrouping>(ui32(6)));
        targetDataGroupingVector.push_back(
            MakeIntrusive<TObjectsGrouping>(
                TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 3}, {3, 6}, {6, 7}, {7, 9}}
            )
        );


        TVector<TArraySubsetIndexing<ui32>> subsetVector;
        TVector<EObjectsOrder> subsetOrdersVector;
        subsetVector.emplace_back(TFullSubset<ui32>(6));
        subsetOrdersVector.emplace_back(EObjectsOrder::Ordered);
        subsetVector.emplace_back(TIndexedSubset<ui32>{2, 3});
        subsetOrdersVector.emplace_back(EObjectsOrder::Undefined);

        using TExpectedMapIndex = std::pair<size_t, size_t>;

        // (rawTargetDataVector idx, subsetVector idx) -> expectedResult
        THashMap<TExpectedMapIndex, std::pair<TRawTargetData, TObjectsGroupingPtr>> expectedResults;

        expectedResults[TExpectedMapIndex(0, 0)] = std::make_pair(
            rawTargetDataVector[0],
            targetDataGroupingVector[0]
        );
        expectedResults[TExpectedMapIndex(1, 0)] = std::make_pair(
            rawTargetDataVector[1],
            targetDataGroupingVector[1]
        );

        {
            TRawTargetData rawTargetData;
            rawTargetData.Target = {"1", "0"};
            rawTargetData.SetTrivialWeights(2);
            rawTargetData.Baseline = {
                {0.3f, 0.2f},
                {1.3f, 2.2f}
            };

            expectedResults[TExpectedMapIndex(0, 1)] = std::make_pair(
                rawTargetData,
                MakeIntrusive<TObjectsGrouping>(ui32(2))
            );
        }

        {
            TRawTargetData rawTargetData;
            rawTargetData.Target = {"1.0", "0.0", "1.0", "0.0"};
            rawTargetData.Baseline = {{0.3f, 0.2f, 0.35f, 0.8f}};
            rawTargetData.Weights = TWeights<float>({2.0f, 3.0f, 0.0f, 1.0f});
            rawTargetData.GroupWeights = TWeights<float>({2.0f, 2.1f, 2.1f, 2.1f});
            rawTargetData.Pairs = {TPair(1, 3, 1.0f), TPair(1, 2, 2.0f)};

            expectedResults[TExpectedMapIndex(1, 1)] = std::make_pair(
                rawTargetData,
                MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 1}, {1, 4}})
            );
        }

        for (auto rawTargetDataIdx : xrange(rawTargetDataVector.size())) {
            for (auto subsetIdx : xrange(subsetVector.size())) {
                // copy to move to TRawTargetDataProvider ctor
                TRawTargetData rawTargetData = rawTargetDataVector[rawTargetDataIdx];

                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(2);

                TRawTargetDataProvider rawTargetDataProvider(
                    targetDataGroupingVector[rawTargetDataIdx],
                    std::move(rawTargetData),
                    false,
                    &localExecutor
                );

                TObjectsGroupingSubset objectsGroupingSubset = GetSubset(
                    rawTargetDataProvider.GetObjectsGrouping(),
                    TArraySubsetIndexing<ui32>(subsetVector[subsetIdx]),
                    subsetOrdersVector[subsetIdx]
                );

                TRawTargetDataProvider subsetDataProvider =
                    rawTargetDataProvider.GetSubset(objectsGroupingSubset, &localExecutor);

                auto expectedSubsetData = expectedResults[TExpectedMapIndex(rawTargetDataIdx, subsetIdx)];

                TRawTargetData expectedSubsetRawTargetData = expectedSubsetData.first;
                TObjectsGroupingPtr expectedSubsetGrouping = expectedSubsetData.second;

                TRawTargetDataProvider expectedSubsetDataProvider(
                    expectedSubsetGrouping,
                    std::move(expectedSubsetRawTargetData),
                    false,
                    &localExecutor
                );

#define COMPARE_DATA_PROVIDER_FIELD(FIELD) \
                UNIT_ASSERT_EQUAL( \
                    subsetDataProvider.Get##FIELD(), \
                    expectedSubsetDataProvider.Get##FIELD() \
                );

                COMPARE_DATA_PROVIDER_FIELD(Target);
                COMPARE_DATA_PROVIDER_FIELD(Baseline);
                COMPARE_DATA_PROVIDER_FIELD(Weights);
                COMPARE_DATA_PROVIDER_FIELD(GroupWeights);

#undef COMPARE_DATA_PROVIDER_FIELD

                UNIT_ASSERT(
                    EqualAsMultiSets(subsetDataProvider.GetPairs(), expectedSubsetDataProvider.GetPairs())
                );

                UNIT_ASSERT_EQUAL(
                    *subsetDataProvider.GetObjectsGrouping(),
                    *expectedSubsetDataProvider.GetObjectsGrouping()
                );
            }
        }
    }
}



Y_UNIT_TEST_SUITE(TTargetDataProvider) {

    // subsets are fixed: first is always FullSubset, second is always TIndexedSubset<ui32>{2, 3}
    // TComparisonFunc must accept two TTargetDataProviders and check their equality
    void TestGetSubsets(
        const TVector<TTargetDataProviders>& targetsVector,
        const TVector<TTargetDataProviders>& expectedSecondSubsets,

        // nondefault values used for checking TObjectsGroupingSubset with permutations inside group
        TMaybe<TObjectsGroupingSubset> secondObjectsGroupingSubset = Nothing()
    ) {
        TVector<TArraySubsetIndexing<ui32>> subsetVector;
        TVector<EObjectsOrder> subsetOrdersVector;
        subsetVector.emplace_back(TFullSubset<ui32>(6));
        subsetOrdersVector.emplace_back(EObjectsOrder::Ordered);
        subsetVector.emplace_back(TIndexedSubset<ui32>{2, 3});
        subsetOrdersVector.emplace_back(EObjectsOrder::Undefined);

        using TExpectedMapIndex = std::pair<size_t, size_t>;

        // (targetVector idx, subsetVector idx) -> expectedResult
        THashMap<TExpectedMapIndex, TTargetDataProviders> expectedResults;

        for (auto targetVectorIdx : xrange(targetsVector.size())) {
            expectedResults[TExpectedMapIndex(targetVectorIdx, 0)] = targetsVector[targetVectorIdx];
            expectedResults[TExpectedMapIndex(targetVectorIdx, 1)] = expectedSecondSubsets[targetVectorIdx];
        }

        for (auto targetVectorIdx : xrange(targetsVector.size())) {
            for (auto subsetIdx : xrange(subsetVector.size())) {
                TObjectsGroupingSubset objectsGroupingSubset =
                    ((subsetIdx == 1) && secondObjectsGroupingSubset) ?
                    std::move(*secondObjectsGroupingSubset)
                    : GetSubset(
                        /* get object grouping from the first element of targetsVector[targetVectorIdx],
                            they all should be equal in all vector elements
                        */
                        targetsVector[targetVectorIdx].begin()->second->GetObjectsGrouping(),
                        TArraySubsetIndexing<ui32>(subsetVector[subsetIdx]),
                        subsetOrdersVector[subsetIdx]
                    );

                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(2);

                TTargetDataProviders subsetTargets = GetSubsets(
                    targetsVector[targetVectorIdx],
                    objectsGroupingSubset,
                    &localExecutor
                );

                TTargetDataProviders expectedSubsetTargets =
                    expectedResults[TExpectedMapIndex(targetVectorIdx, subsetIdx)];

                CompareTargetDataProviders(subsetTargets, expectedSubsetTargets);
            }
        }
    }

    template <class TTarget>
    void TestGetSubset(
        const TVector<TTarget>& targetVector,
        const TVector<TTarget>& expectedSecondSubsets,

        // nondefault values used for checking TObjectsGroupingSubset with permutations inside group
        TMaybe<TObjectsGroupingSubset> secondObjectsGroupingSubset = Nothing()
    ) {
        // converts to TestGetSubsets format
        TVector<TTargetDataProviders> targetVector2;
        TVector<TTargetDataProviders> expectedSecondSubsets2;
        for (auto i : xrange(targetVector.size())) {
            targetVector2.emplace_back(
                TTargetDataProviders{
                    {targetVector[i].GetSpecification(), MakeIntrusive<TTarget>(targetVector[i])}
                }
            );
            expectedSecondSubsets2.emplace_back(
                TTargetDataProviders{
                    {
                        expectedSecondSubsets[i].GetSpecification(),
                        MakeIntrusive<TTarget>(expectedSecondSubsets[i])
                    }
                }
            );
        }

        TestGetSubsets(targetVector2, expectedSecondSubsets2, std::move(secondObjectsGroupingSubset));
    }


    Y_UNIT_TEST(TBinClassTarget_GetSubset) {
        TVector<TBinClassTarget> targetVector;
        TVector<TBinClassTarget> expectedSecondSubsets;

        targetVector.push_back(
            TBinClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(6)),
                /*baseline*/ nullptr
            )
        );
        expectedSecondSubsets.push_back(
            TBinClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(2)),
                /*baseline*/ nullptr
            )
        );

        targetVector.push_back(
            TBinClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f})),
                /*baseline*/ ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f})
            )
        );
        expectedSecondSubsets.push_back(
            TBinClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>( {1.0f, 0.0f} ),
                /*weights*/ Share( TWeights<float>({2.0f, 3.0f}) ),
                /*baseline*/ ShareVector<float>( {0.3f, 0.2f} )
            )
        );

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TMultiClassTarget_GetSubset) {
        TVector<TMultiClassTarget> targetVector;
        TVector<TMultiClassTarget> expectedSecondSubsets;

        targetVector.push_back(
            TMultiClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*classCount*/ui32(2),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(6)),
                /*baseline*/ {},
                /*isForGpu*/ false
            )
        );
        expectedSecondSubsets.push_back(
            TMultiClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*classCount*/ui32(2),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(2)),
                /*baseline*/ {},
                /*isForGpu*/ false
            )
        );

        targetVector.push_back(
            TMultiClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*classCount*/ ui32(2),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f})),
                /*baseline*/ {
                    ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f}),
                    ShareVector<float>({1.0f, 2.1f, 1.3f, 2.2f, 3.3f, 4.7f})
                },
                /*isForGpu*/ false
            )
        );
        expectedSecondSubsets.push_back(
            TMultiClassTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*classCount*/ ui32(2),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({2.0f, 3.0f})),
                /*baseline*/ {ShareVector<float>({0.3f, 0.2f}), ShareVector<float>({1.3f, 2.2f})},
                /*isForGpu*/ false
            )
        );

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TRegressionTarget_GetSubset) {
        TVector<TRegressionTarget> targetVector;
        TVector<TRegressionTarget> expectedSecondSubsets;

        targetVector.push_back(
            TRegressionTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(6)),
                /*baseline*/ nullptr
            )
        );
        expectedSecondSubsets.push_back(
            TRegressionTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(2)),
                /*baseline*/ nullptr
            )
        );

        targetVector.push_back(
            TRegressionTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f})),
                /*baseline*/ ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f})
            )
        );
        expectedSecondSubsets.push_back(
            TRegressionTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({2.0f, 3.0f})),
                /*baseline*/ ShareVector<float>({0.3f, 0.2f})
            )
        );

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TGroupwiseRankingTarget_GetSubset) {
        TVector<TGroupwiseRankingTarget> targetVector;
        TVector<TGroupwiseRankingTarget> expectedSecondSubsets;

        targetVector.push_back(
            TGroupwiseRankingTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(
                    TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 3}, {3, 6}, {6, 7}, {7, 9}}
                ),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.8f}),
                /*weight*/ Share(TWeights<float>(9)),
                /*baseline*/ nullptr,
                /*groupInfo*/ ShareVector<TQueryInfo>(
                    {
                        TQueryInfo(0, 1),
                        TQueryInfo(1, 2),
                        TQueryInfo(2, 3),
                        TQueryInfo(3, 6),
                        TQueryInfo(6, 7),
                        TQueryInfo(7, 9)
                    }
                )
            )
        );
        expectedSecondSubsets.push_back(
            TGroupwiseRankingTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 1}, {1, 4}}),
                /*target*/ ShareVector<float>({1.0f, 0.0f, 1.0f, 0.0f}),
                /*weight*/ Share(TWeights<float>(4)),
                /*baseline*/ nullptr,
                /*groupInfo*/ ShareVector<TQueryInfo>({TQueryInfo(0, 1), TQueryInfo(1, 4)})
            )
        );

        {
            TVector<TQueryInfo> groupInfo(6);

            groupInfo[0].Begin = 0;
            groupInfo[0].End = 1;
            groupInfo[0].Weight = 2.0f;
            groupInfo[0].SubgroupId = {0};

            groupInfo[1].Begin = 1;
            groupInfo[1].End = 3;
            groupInfo[1].Weight = 1.0f;
            groupInfo[1].SubgroupId = {1, 1};

            groupInfo[2].Begin = 3;
            groupInfo[2].End = 5;
            groupInfo[2].Weight = 3.0f;
            groupInfo[2].SubgroupId = {3, 4};

            groupInfo[3].Begin = 5;
            groupInfo[3].End = 6;
            groupInfo[3].Weight = 1.0f;
            groupInfo[3].SubgroupId = {7};

            groupInfo[4].Begin = 6;
            groupInfo[4].End = 8;
            groupInfo[4].Weight = 4.0f;
            groupInfo[4].SubgroupId = {8, 9};

            groupInfo[5].Begin = 8;
            groupInfo[5].End = 9;
            groupInfo[5].Weight = 0.0f;
            groupInfo[5].SubgroupId = {10};


            targetVector.push_back(
                TGroupwiseRankingTarget(
                    "",
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 3}, {3, 5}, {5, 6}, {6, 8}, {8, 9}}
                    ),
                    /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.8f}),
                    /*weight*/ Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.8f, 0.9f, 0.1f})),
                    /*baseline*/ ShareVector<float>(
                        {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f}
                    ),
                    ShareVector<TQueryInfo>(std::move(groupInfo))
                )
            );
        }
        {
            TVector<TQueryInfo> groupInfo(2);

            groupInfo[0].Begin = 0;
            groupInfo[0].End = 2;
            groupInfo[0].Weight = 3.0f;
            groupInfo[0].SubgroupId = {3, 4};

            groupInfo[1].Begin = 2;
            groupInfo[1].End = 3;
            groupInfo[1].Weight = 1.0f;
            groupInfo[1].SubgroupId = {7};

            expectedSecondSubsets.push_back(
                TGroupwiseRankingTarget(
                    "",
                    MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 2}, {2, 3}}),
                    /*target*/ ShareVector<float>({0.0f, 1.0f, 0.0f}),
                    /*weight*/ Share(TWeights<float>({3.0f, 0.0f, 1.0f})),
                    /*baseline*/ ShareVector<float>({0.2f, 0.35f, 0.8f}),
                    ShareVector<TQueryInfo>(std::move(groupInfo))
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TGroupPairwiseRankingTarget_GetSubset) {
        TVector<TGroupPairwiseRankingTarget> targetVector;
        TVector<TGroupPairwiseRankingTarget> expectedSecondSubsets;

        {
            TVector<TQueryInfo> groupInfo(6);

            groupInfo[0].Begin = 0;
            groupInfo[0].End = 1;
            groupInfo[0].Weight = 2.0f;
            groupInfo[0].SubgroupId = {0};

            groupInfo[1].Begin = 1;
            groupInfo[1].End = 2;
            groupInfo[1].Weight = 1.0f;
            groupInfo[1].SubgroupId = {1};

            groupInfo[2].Begin = 2;
            groupInfo[2].End = 6;
            groupInfo[2].Weight = 3.0f;
            groupInfo[2].SubgroupId = {2, 2, 3, 3};
            groupInfo[2].Competitors = {
                { TCompetitor(1, 1.0f) },
                { TCompetitor(2, 1.0f), TCompetitor(3, 3.0f) },
                { TCompetitor(1, 2.0f) },
                {}
            };

            groupInfo[3].Begin = 6;
            groupInfo[3].End = 7;
            groupInfo[3].Weight = 2.0f;
            groupInfo[3].SubgroupId = {0};

            groupInfo[4].Begin = 7;
            groupInfo[4].End = 9;
            groupInfo[4].Weight = 2.2f;
            groupInfo[4].SubgroupId = {0, 1};
            groupInfo[4].Competitors = {
                { TCompetitor(1, 1.0f) },
                {}
            };

            groupInfo[5].Begin = 9;
            groupInfo[5].End = 11;
            groupInfo[5].Weight = 1.3f;
            groupInfo[5].SubgroupId = {0, 0};


            targetVector.push_back(
                TGroupPairwiseRankingTarget(
                    "",
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 6}, {6, 7}, {7, 9}, {9, 11}}
                    ),
                    /*baseline*/ ShareVector<float>(
                        {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f, 0.0f, 1.0f}
                    ),
                    ShareVector<TQueryInfo>(std::move(groupInfo))
                )
            );
        }

        {
            TVector<TQueryInfo> groupInfo(2);

            groupInfo[0].Begin = 0;
            groupInfo[0].End = 4;
            groupInfo[0].Weight = 3.0f;
            groupInfo[0].SubgroupId = {2, 2, 3, 3};
            groupInfo[0].Competitors = {
                { TCompetitor(1, 1.0f) },
                { TCompetitor(2, 1.0f), TCompetitor(3, 3.0f) },
                { TCompetitor(1, 2.0f) },
                {}
            };

            groupInfo[1].Begin = 4;
            groupInfo[1].End = 5;
            groupInfo[1].Weight = 2.0f;
            groupInfo[1].SubgroupId = {0};

            expectedSecondSubsets.push_back(
                TGroupPairwiseRankingTarget(
                    "",
                    MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 4}, {4, 5}}),
                    /*baseline*/ ShareVector<float>({0.3f, 0.2f, 0.35f, 0.8f, 0.12f}),
                    ShareVector<TQueryInfo>(std::move(groupInfo))
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TGroupPairwiseRankingTarget_GetSubsetWithShuffle) {
        TVector<TGroupPairwiseRankingTarget> targetVector;
        TVector<TGroupPairwiseRankingTarget> expectedSecondSubsets;

        {
            TVector<TQueryInfo> groupInfo(6);

            groupInfo[0].Begin = 0;
            groupInfo[0].End = 1;
            groupInfo[0].Weight = 2.0f;
            groupInfo[0].SubgroupId = {0};

            groupInfo[1].Begin = 1;
            groupInfo[1].End = 2;
            groupInfo[1].Weight = 1.0f;
            groupInfo[1].SubgroupId = {1};

            groupInfo[2].Begin = 2;
            groupInfo[2].End = 6;
            groupInfo[2].Weight = 3.0f;
            groupInfo[2].SubgroupId = {2, 2, 3, 3};
            groupInfo[2].Competitors = {
                { TCompetitor(1, 1.0f) },
                { TCompetitor(2, 1.0f), TCompetitor(3, 3.0f) },
                { TCompetitor(1, 2.0f) },
                {}
            };

            groupInfo[3].Begin = 6;
            groupInfo[3].End = 7;
            groupInfo[3].Weight = 2.0f;
            groupInfo[3].SubgroupId = {0};

            groupInfo[4].Begin = 7;
            groupInfo[4].End = 9;
            groupInfo[4].Weight = 2.2f;
            groupInfo[4].SubgroupId = {0, 1};
            groupInfo[4].Competitors = {
                { TCompetitor(1, 1.0f) },
                {}
            };

            groupInfo[5].Begin = 9;
            groupInfo[5].End = 11;
            groupInfo[5].Weight = 1.3f;
            groupInfo[5].SubgroupId = {0, 0};


            targetVector.push_back(
                TGroupPairwiseRankingTarget(
                    "",
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 6}, {6, 7}, {7, 9}, {9, 11}}
                    ),
                    /*baseline*/ ShareVector<float>(
                        {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f, 0.0f, 1.0f}
                    ),
                    ShareVector<TQueryInfo>(std::move(groupInfo))
                )
            );
        }

        {
            TVector<TQueryInfo> groupInfo(6);

            groupInfo[0].Begin = 0;
            groupInfo[0].End = 4;
            groupInfo[0].Weight = 3.0f;
            groupInfo[0].SubgroupId = {3, 2, 2, 3};
            groupInfo[0].Competitors = {
                {},
                { TCompetitor(2, 1.0f) },
                { TCompetitor(3, 1.0f), TCompetitor(0, 3.0f) },
                { TCompetitor(2, 2.0f) }
            };

            groupInfo[1].Begin = 4;
            groupInfo[1].End = 6;
            groupInfo[1].Weight = 2.2f;
            groupInfo[1].SubgroupId = {1, 0};
            groupInfo[1].Competitors = {
                {},
                { TCompetitor(0, 1.0f) }
            };

            groupInfo[2].Begin = 6;
            groupInfo[2].End = 8;
            groupInfo[2].Weight = 1.3f;
            groupInfo[2].SubgroupId = {0, 0};

            groupInfo[3].Begin = 8;
            groupInfo[3].End = 9;
            groupInfo[3].Weight = 2.0f;
            groupInfo[3].SubgroupId = {0};

            groupInfo[4].Begin = 9;
            groupInfo[4].End = 10;
            groupInfo[4].Weight = 2.0f;
            groupInfo[4].SubgroupId = {0};

            groupInfo[5].Begin = 10;
            groupInfo[5].End = 11;
            groupInfo[5].Weight = 1.0f;
            groupInfo[5].SubgroupId = {1};


            expectedSecondSubsets.push_back(
                TGroupPairwiseRankingTarget(
                    "",
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 4}, {4, 6}, {6, 8}, {8, 9}, {9, 10}, {10, 11}}
                    ),
                    /*baseline*/ ShareVector<float>(
                        {0.8f, 0.3f, 0.2f, 0.35f, 0.87f, 0.67f, 0.0f, 1.0f, 0.0f, 0.12f, 0.1f}
                    ),
                    ShareVector<TQueryInfo>(std::move(groupInfo))
                )
            );
        }

        TRestorableFastRng64 rand(0);

        TestGetSubset(
            targetVector,
            expectedSecondSubsets,
            Shuffle(targetVector.back().GetObjectsGrouping(), 1, &rand)
        );
    }

    Y_UNIT_TEST(TSimpleTarget_GetSubset) {
        TVector<TSimpleTarget> targetVector;
        TVector<TSimpleTarget> expectedSecondSubsets;

        targetVector.push_back(
            TSimpleTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f})
            )
        );
        expectedSecondSubsets.push_back(
            TSimpleTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f})
            )
        );

        targetVector.push_back(
            TSimpleTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f})
            )
        );
        expectedSecondSubsets.push_back(
            TSimpleTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f})
            )
        );

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }


    Y_UNIT_TEST(TUserDefinedTarget_GetSubset) {
        TVector<TUserDefinedTarget> targetVector;
        TVector<TUserDefinedTarget> expectedSecondSubsets;

        targetVector.push_back(
            TUserDefinedTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(6))
            )
        );
        expectedSecondSubsets.push_back(
            TUserDefinedTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>(2))
            )
        );

        targetVector.push_back(
            TUserDefinedTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(6)),
                /*target*/ ShareVector<float>({0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f}))
            )
        );
        expectedSecondSubsets.push_back(
            TUserDefinedTarget(
                "",
                MakeIntrusive<TObjectsGrouping>(ui32(2)),
                /*target*/ ShareVector<float>({1.0f, 0.0f}),
                /*weights*/ Share(TWeights<float>({2.0f, 3.0f}))
            )
        );

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }


    TWeights<float> CreateWeights() {
        return TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.98f, 0.11f, 0.43f, 0.24f, 0.2f});
    }

    TVector<float> CreateTarget() {
        return {0.12f, 1.9f, 2.1f, 0.22f, 1.1f, 1.12f, 0.32f, 0.5f, 0.8f, 0.9f, 0.22f};
    }

    TVector<TVector<float>> CreateBaseline() {
        return {
            {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f, 0.33f, 0.92f},
            {1.0f, 1.2f, 2.3f, 0.25f, 0.65f, 0.82f, 0.0f, 0.11f, 0.82f, 0.29f, 0.1f}
        };
    }

    TVector<TQueryInfo> CreateGroupInfo() {
        TVector<TQueryInfo> groupInfo(6);

        groupInfo[0].Begin = 0;
        groupInfo[0].End = 1;
        groupInfo[0].Weight = 2.0f;
        groupInfo[0].SubgroupId = {0};

        groupInfo[1].Begin = 1;
        groupInfo[1].End = 2;
        groupInfo[1].Weight = 1.0f;
        groupInfo[1].SubgroupId = {1};

        groupInfo[2].Begin = 2;
        groupInfo[2].End = 6;
        groupInfo[2].Weight = 3.0f;
        groupInfo[2].SubgroupId = {2, 2, 3, 3};
        groupInfo[2].Competitors = {
            { TCompetitor(1, 1.0f) },
            { TCompetitor(2, 1.0f), TCompetitor(3, 3.0f) },
            { TCompetitor(1, 2.0f) },
            {}
        };

        groupInfo[3].Begin = 6;
        groupInfo[3].End = 7;
        groupInfo[3].Weight = 2.0f;
        groupInfo[3].SubgroupId = {0};

        groupInfo[4].Begin = 7;
        groupInfo[4].End = 9;
        groupInfo[4].Weight = 2.2f;
        groupInfo[4].SubgroupId = {0, 1};
        groupInfo[4].Competitors = {
            { TCompetitor(1, 1.0f) },
            {}
        };

        groupInfo[5].Begin = 9;
        groupInfo[5].End = 11;
        groupInfo[5].Weight = 1.3f;
        groupInfo[5].SubgroupId = {0, 0};

        return groupInfo;
    }


    void CreateTargetDataProviders(
        TTargetDataProviders* targetDataProviders,
        TTargetDataProviders* expectedSubsetTargetDataProviders
    ) {
        TObjectsGroupingPtr objectsGrouping = MakeIntrusive<TObjectsGrouping>(
            TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 6}, {6, 7}, {7, 9}, {9, 11}}
        );
        TObjectsGroupingPtr expectedSubsetObjectsGrouping = MakeIntrusive<TObjectsGrouping>(
            TVector<TGroupBounds>{{0, 4}, {4, 5}}
        );

        TSharedWeights<float> weights = Share(CreateWeights());
        TSharedWeights<float> expectedSubsetWeights = Share(TWeights<float>({2.0f, 3.0f, 0.0f, 1.0f, 0.98f}));

        TSharedVector<float> targets = ShareVector<float>(CreateTarget());
        TSharedVector<float> expectedSubsetTargets = ShareVector<float>({2.1f, 0.22f, 1.1f, 1.12f, 0.32f});

        auto baselinesSrc = CreateBaseline();
        TVector<TSharedVector<float>> baselines = {
            ShareVector(TVector<float>(baselinesSrc[0])),
            ShareVector(TVector<float>(baselinesSrc[1]))
        };

        TVector<TSharedVector<float>> expectedSubsetBaselines = {
            ShareVector<float>({0.3f, 0.2f, 0.35f, 0.8f, 0.12f}),
            ShareVector<float>({2.3f, 0.25f, 0.65f, 0.82f, 0.0f})
        };

        TSharedVector<TQueryInfo> sharedGroupInfo = ShareVector<TQueryInfo>(CreateGroupInfo());


        TVector<TQueryInfo> expectedSubsetGroupInfo(2);

        expectedSubsetGroupInfo[0].Begin = 0;
        expectedSubsetGroupInfo[0].End = 4;
        expectedSubsetGroupInfo[0].Weight = 3.0f;
        expectedSubsetGroupInfo[0].SubgroupId = {2, 2, 3, 3};
        expectedSubsetGroupInfo[0].Competitors = {
            { TCompetitor(1, 1.0f) },
            { TCompetitor(2, 1.0f), TCompetitor(3, 3.0f) },
            { TCompetitor(1, 2.0f) },
            {}
        };

        expectedSubsetGroupInfo[1].Begin = 4;
        expectedSubsetGroupInfo[1].End = 5;
        expectedSubsetGroupInfo[1].Weight = 2.0f;
        expectedSubsetGroupInfo[1].SubgroupId = {0};

        TSharedVector<TQueryInfo> expectedSubsetSharedGroupInfo = ShareVector<TQueryInfo>(
            std::move(expectedSubsetGroupInfo)
        );


        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::BinClass),
            MakeIntrusive<TBinClassTarget>(
                "",
                objectsGrouping,
                targets,
                weights,
                baselines[0]
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::BinClass),
            MakeIntrusive<TBinClassTarget>(
                "",
                expectedSubsetObjectsGrouping,
                expectedSubsetTargets,
                expectedSubsetWeights,
                expectedSubsetBaselines[0]
            )
        );

        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::MultiClass),
            MakeIntrusive<TMultiClassTarget>(
                "",
                objectsGrouping,
                /*classCount*/ ui32(2),
                targets,
                weights,
                TVector<TSharedVector<float>>(baselines),
                /*isForGpu*/ false
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::MultiClass),
            MakeIntrusive<TMultiClassTarget>(
                "",
                expectedSubsetObjectsGrouping,
                /*classCount*/ ui32(2),
                expectedSubsetTargets,
                expectedSubsetWeights,
                TVector<TSharedVector<float>>(expectedSubsetBaselines),
                /*isForGpu*/ false
            )
        );

        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::Regression),
            MakeIntrusive<TRegressionTarget>(
                "",
                objectsGrouping,
                targets,
                weights,
                baselines[0]
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::Regression),
            MakeIntrusive<TRegressionTarget>(
                "",
                expectedSubsetObjectsGrouping,
                expectedSubsetTargets,
                expectedSubsetWeights,
                expectedSubsetBaselines[0]
            )
        );

        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::GroupwiseRanking),
            MakeIntrusive<TGroupwiseRankingTarget>(
                "",
                objectsGrouping,
                targets,
                weights,
                baselines[1], // take 2nd baseline just for diversity with BinClassTarget
                sharedGroupInfo
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::GroupwiseRanking),
            MakeIntrusive<TGroupwiseRankingTarget>(
                "",
                expectedSubsetObjectsGrouping,
                expectedSubsetTargets,
                expectedSubsetWeights,
                expectedSubsetBaselines[1],
                expectedSubsetSharedGroupInfo
            )
        );

        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::GroupPairwiseRanking),
            MakeIntrusive<TGroupPairwiseRankingTarget>(
                "",
                objectsGrouping,
                baselines[0],
                sharedGroupInfo
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::GroupPairwiseRanking),
            MakeIntrusive<TGroupPairwiseRankingTarget>(
                "",
                expectedSubsetObjectsGrouping,
                expectedSubsetBaselines[0],
                expectedSubsetSharedGroupInfo
            )
        );

        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::Simple),
            MakeIntrusive<TSimpleTarget>(
                "",
                objectsGrouping,
                targets
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::Simple),
            MakeIntrusive<TSimpleTarget>(
                "",
                expectedSubsetObjectsGrouping,
                expectedSubsetTargets
            )
        );

        targetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::UserDefined),
            MakeIntrusive<TUserDefinedTarget>(
                "",
                objectsGrouping,
                targets,
                weights
            )
        );
        expectedSubsetTargetDataProviders->emplace(
            TTargetDataSpecification(ETargetType::UserDefined),
            MakeIntrusive<TUserDefinedTarget>(
                "",
                expectedSubsetObjectsGrouping,
                expectedSubsetTargets,
                expectedSubsetWeights
            )
        );
    }


    Y_UNIT_TEST(GetSubsets) {
        TTargetDataProviders targetDataProviders;
        TTargetDataProviders expectedSubsetTargetDataProviders;

        CreateTargetDataProviders(&targetDataProviders, &expectedSubsetTargetDataProviders);

        TestGetSubsets({targetDataProviders}, {expectedSubsetTargetDataProviders});
    }


    void CompareCompatibilityWeights(
        TConstArrayRef<float> oldWeights,
        const TWeights<float>& expectedWeights)
    {
        if (expectedWeights.IsTrivial()) {
            UNIT_ASSERT(oldWeights.empty());
        } else {
            UNIT_ASSERT_EQUAL(oldWeights, expectedWeights.GetNonTrivialData());
        }
    }


    Y_UNIT_TEST(TTargetDataProvidersCompatibilityFunctions) {
        TTargetDataProviders targetDataProviders;
        TTargetDataProviders expectedSubsetTargetDataProviders; // in fact unused here

        CreateTargetDataProviders(&targetDataProviders, &expectedSubsetTargetDataProviders);

        // get subsets that are compatible with compatibility functions

#define COMPARE_COMPATIBILITY_FIELD(targetDataProviders, FIELD) \
            UNIT_ASSERT(Equal(Get##FIELD(targetDataProviders), Create##FIELD()));


        {
            TTargetDataProviders onlyBinClass;
            onlyBinClass.emplace(
                TTargetDataSpecification(ETargetType::BinClass),
                targetDataProviders[TTargetDataSpecification(ETargetType::BinClass)]
            );

            COMPARE_COMPATIBILITY_FIELD(onlyBinClass, Target);
            CompareCompatibilityWeights(GetWeights(onlyBinClass), CreateWeights());
            UNIT_ASSERT_EQUAL(
                GetBaseline(onlyBinClass), TVector<TConstArrayRef<float>>{CreateBaseline()[0]}
            );
            UNIT_ASSERT_EQUAL(GetGroupInfo(onlyBinClass), TConstArrayRef<TQueryInfo>());
        }

        {
            TTargetDataProviders onlyMultiClass;
            onlyMultiClass.emplace(
                TTargetDataSpecification(ETargetType::MultiClass),
                targetDataProviders[TTargetDataSpecification(ETargetType::MultiClass)]
            );

            COMPARE_COMPATIBILITY_FIELD(onlyMultiClass, Target);
            CompareCompatibilityWeights(GetWeights(onlyMultiClass), CreateWeights());

            auto expectedBaseline = CreateBaseline();
            UNIT_ASSERT_EQUAL(
                GetBaseline(onlyMultiClass),
                (TVector<TConstArrayRef<float>>{expectedBaseline[0], expectedBaseline[1]})
            );
            UNIT_ASSERT_EQUAL(GetGroupInfo(onlyMultiClass), TConstArrayRef<TQueryInfo>());
        }

        {
            TTargetDataProviders onlyRegression;
            onlyRegression.emplace(
                TTargetDataSpecification(ETargetType::Regression),
                targetDataProviders[TTargetDataSpecification(ETargetType::Regression)]
            );

            COMPARE_COMPATIBILITY_FIELD(onlyRegression, Target);
            CompareCompatibilityWeights(GetWeights(onlyRegression), CreateWeights());

            UNIT_ASSERT_EQUAL(
                GetBaseline(onlyRegression), TVector<TConstArrayRef<float>>{CreateBaseline()[0]}
            );
            UNIT_ASSERT_EQUAL(GetGroupInfo(onlyRegression), TConstArrayRef<TQueryInfo>());
        }

        {
            TTargetDataProviders onlyGroupwiseRanking;
            onlyGroupwiseRanking.emplace(
                TTargetDataSpecification(ETargetType::GroupwiseRanking),
                targetDataProviders[TTargetDataSpecification(ETargetType::GroupwiseRanking)]
            );

            COMPARE_COMPATIBILITY_FIELD(onlyGroupwiseRanking, Target);
            CompareCompatibilityWeights(GetWeights(onlyGroupwiseRanking), CreateWeights());

            UNIT_ASSERT_EQUAL(
                GetBaseline(onlyGroupwiseRanking), TVector<TConstArrayRef<float>>{CreateBaseline()[1]}
            );
            COMPARE_COMPATIBILITY_FIELD(onlyGroupwiseRanking, GroupInfo);
        }

        {
            TTargetDataProviders onlyGroupPairwiseRanking;
            onlyGroupPairwiseRanking.emplace(
                TTargetDataSpecification(ETargetType::GroupPairwiseRanking),
                targetDataProviders[TTargetDataSpecification(ETargetType::GroupPairwiseRanking)]
            );

            UNIT_ASSERT_EXCEPTION(GetTarget(onlyGroupPairwiseRanking), TCatBoostException);
            UNIT_ASSERT_EQUAL(GetWeights(onlyGroupPairwiseRanking), TConstArrayRef<float>());

            UNIT_ASSERT_EQUAL(
                GetBaseline(onlyGroupPairwiseRanking), TVector<TConstArrayRef<float>>{CreateBaseline()[0]}
            );
            COMPARE_COMPATIBILITY_FIELD(onlyGroupPairwiseRanking, GroupInfo);
        }

        {
            TTargetDataProviders onlySimple;
            onlySimple.emplace(
                TTargetDataSpecification(ETargetType::Simple),
                targetDataProviders[TTargetDataSpecification(ETargetType::Simple)]
            );

            COMPARE_COMPATIBILITY_FIELD(onlySimple, Target);
            UNIT_ASSERT_EQUAL(GetWeights(onlySimple), TConstArrayRef<float>());
            UNIT_ASSERT_EQUAL(GetBaseline(onlySimple), TVector<TConstArrayRef<float>>());
            UNIT_ASSERT_EQUAL(GetGroupInfo(onlySimple), TConstArrayRef<TQueryInfo>());
        }

        {
            TTargetDataProviders onlyUserDefined;
            onlyUserDefined.emplace(
                TTargetDataSpecification(ETargetType::UserDefined),
                targetDataProviders[TTargetDataSpecification(ETargetType::UserDefined)]
            );

            COMPARE_COMPATIBILITY_FIELD(onlyUserDefined, Target);
            CompareCompatibilityWeights(GetWeights(onlyUserDefined),  CreateWeights());
            UNIT_ASSERT_EQUAL(GetBaseline(onlyUserDefined), TVector<TConstArrayRef<float>>());
            UNIT_ASSERT_EQUAL(GetGroupInfo(onlyUserDefined), TConstArrayRef<TQueryInfo>());
        }

        {
            // regression and grouppairwise should work together

            TTargetDataProviders regressionAndGroupPairwise;
            regressionAndGroupPairwise.emplace(
                TTargetDataSpecification(ETargetType::Regression),
                targetDataProviders[TTargetDataSpecification(ETargetType::Regression)]
            );
            regressionAndGroupPairwise.emplace(
                TTargetDataSpecification(ETargetType::GroupPairwiseRanking),
                targetDataProviders[TTargetDataSpecification(ETargetType::GroupPairwiseRanking)]
            );

            COMPARE_COMPATIBILITY_FIELD(regressionAndGroupPairwise, Target);
            CompareCompatibilityWeights(GetWeights(regressionAndGroupPairwise), CreateWeights());

            UNIT_ASSERT_EQUAL(
                GetBaseline(regressionAndGroupPairwise), TVector<TConstArrayRef<float>>{CreateBaseline()[0]}
            );
            COMPARE_COMPATIBILITY_FIELD(regressionAndGroupPairwise, GroupInfo);
        }

        {
            // bad mix of single and multiclass

            TTargetDataProviders badMix;
            badMix.emplace(
                TTargetDataSpecification(ETargetType::BinClass),
                targetDataProviders[TTargetDataSpecification(ETargetType::BinClass)]
            );
            badMix.emplace(
                TTargetDataSpecification(ETargetType::MultiClass),
                targetDataProviders[TTargetDataSpecification(ETargetType::MultiClass)]
            );

            UNIT_ASSERT_EXCEPTION(GetBaseline(badMix), TCatBoostException);
        }


#undef COMPARE_COMPATIBILITY_FIELD

    }
}

