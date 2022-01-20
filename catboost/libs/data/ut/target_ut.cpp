#include "util.h"

#include <catboost/libs/data/pairs.h>
#include <catboost/libs/data/target.h>

#include <catboost/libs/data/ut/lib/for_target.h>

#include <catboost/libs/data/util.h>

#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/helpers/maybe.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/hash.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <utility>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;


Y_UNIT_TEST_SUITE(TRawTargetData) {
    bool Equal(TConstArrayRef<TConstArrayRef<TString>>& matrixA, TConstArrayRef<TConstArrayRef<TString>>& matrixB) {
        return matrixA == matrixB;
    }

    bool Equal(TConstArrayRef<TString>& vectorA, TConstArrayRef<TString>& vectorB) {
        return vectorA == vectorB;
    }

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
            false,
            &localExecutor
        );
    }


    Y_UNIT_TEST(Empty) {
        TRawTargetData rawTargetData;
        CreateProviderSimple(0, rawTargetData);
    }

    void TestStringRawTarget(const TVector<TString>& rawTarget) {
        TRawTargetData rawTargetData;
        rawTargetData.TargetType = ERawTargetType::String;
        rawTargetData.Target = {rawTarget};
        rawTargetData.SetTrivialWeights(rawTarget.size());

        auto rawTargetDataProvider = CreateProviderSimple(rawTarget.size(), rawTargetData);

        const TVector<TString>* rawTargetStringData = std::get_if<TVector<TString>>(
            *rawTargetDataProvider.GetOneDimensionalTarget()
        );
        UNIT_ASSERT(rawTargetStringData);
        UNIT_ASSERT_VALUES_EQUAL(*rawTargetStringData, rawTarget);
    }

    template <class T>
    void TestNumericRawTarget(TVector<T>&& rawTarget, const TVector<float>& expectedTarget) {
        const ui32 objectCount = (ui32)rawTarget.size();
        TRawTargetData rawTargetData;
        rawTargetData.TargetType = ERawTargetType::Float;
        rawTargetData.SetTrivialWeights(objectCount);
        rawTargetData.Target.resize(1);
        rawTargetData.Target[0]
            = (ITypedSequencePtr<float>)MakeIntrusive<TTypeCastArrayHolder<float, T>>(std::move(rawTarget));

        auto rawTargetDataProvider = CreateProviderSimple(objectCount, rawTargetData);

        const ITypedSequencePtr<float>* rawTargetFloatData = std::get_if<ITypedSequencePtr<float>>(
            *rawTargetDataProvider.GetOneDimensionalTarget()
        );
        UNIT_ASSERT(rawTargetFloatData);
        UNIT_ASSERT_VALUES_EQUAL(ToVector(**rawTargetFloatData), expectedTarget);
    }

    Y_UNIT_TEST(Target) {
        {
            TestStringRawTarget({"0", "1", "0", "1"});
            TestStringRawTarget({"0.0", "1.0", "3.8"});
            TestStringRawTarget({"male", "male", "male", "female"});
        }

        {
            TestNumericRawTarget<float>({0.0f, 0.12f, 0.1f, 0.0f}, {0.0f, 0.12f, 0.1f, 0.0f});
            TestNumericRawTarget<double>({0.0, 0.12, 0.1, 0.0}, {0.0f, 0.12f, 0.1f, 0.0f});
            TestNumericRawTarget<int>({0, 1, -1, -3}, {0.0f, 1.0f, -1.f, -3.0f});
            TestNumericRawTarget<ui64>({15, 0, 100, 34567}, {15.0f, 0.0f, 100.0f, 34567.0f});
        }

        {
            TVector<TVector<TString>> badTargets;
            badTargets.push_back({"0", "", "0", "1"});

            for (const auto& target : badTargets) {
                TRawTargetData rawTargetData;
                rawTargetData.TargetType = ERawTargetType::String;
                rawTargetData.Target.resize(1);
                rawTargetData.Target[0] = target;
                rawTargetData.SetTrivialWeights(target.size());

                UNIT_ASSERT_EXCEPTION(
                    CreateProviderSimple(target.size(), rawTargetData),
                    TCatBoostException
                );
            }
        }
    }

    Y_UNIT_TEST(MultiTarget) {
        {
            TVector<TVector<TVector<TString>>> goodMultiTargets;
            goodMultiTargets.push_back({{}});
            goodMultiTargets.push_back({{"0", "1", "0", "1"}, {"1", "0", "1", "0"}});
            goodMultiTargets.push_back({{"0.0", "1.0", "3.8"}, {"2.0", "0.5", "1.3"}});
            goodMultiTargets.push_back({{"male", "male", "male", "female"}, {"male", "female", "female", "male"}});

            for (const auto& target : goodMultiTargets) {
                const auto docCount = target.empty() ? 0 : target[0].size();
                TRawTargetData rawTargetData;
                rawTargetData.TargetType = ERawTargetType::String;
                rawTargetData.Target.assign(target.begin(), target.end());
                rawTargetData.SetTrivialWeights(docCount);

                auto rawTargetDataProvider = CreateProviderSimple(docCount, rawTargetData);
                TMaybeData<TConstArrayRef<TRawTarget>> maybeTargetData = rawTargetDataProvider.GetTarget();
                if (target.empty()) {
                    UNIT_ASSERT(!maybeTargetData.Defined());
                } else {
                    UNIT_ASSERT(maybeTargetData.Defined());
                    TConstArrayRef<TRawTarget> targetData = *maybeTargetData;
                    UNIT_ASSERT_VALUES_EQUAL(targetData.size(), target.size());

                    for (auto i : xrange(target.size())) {
                        UNIT_ASSERT_EQUAL(std::get<TVector<TString>>(targetData[i]), target[i]);
                    }
                }
            }
        }

        {
            TVector<TVector<TVector<TString>>> badMultiTargets;
            badMultiTargets.push_back({{"0", "", "0", "1"}, {"1", "1", "1", "0"}});

            for (const auto& target : badMultiTargets) {
                const auto docCount = target.empty() ? 0 : target[0].size();
                TRawTargetData rawTargetData;
                rawTargetData.TargetType = ERawTargetType::String;
                rawTargetData.Target.assign(target.begin(), target.end());
                rawTargetData.SetTrivialWeights(docCount);

                UNIT_ASSERT_EXCEPTION(
                    CreateProviderSimple(docCount, rawTargetData),
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = {target};
            rawTargetData.SetTrivialWeights(target.size());

            auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

            UNIT_ASSERT(rawTargetDataProvider.GetWeights().IsTrivial());
        }

        // good
        {
            TVector<float> weight = {1.0, 0.0, 2.0, 3.0, 1.0};
            {
                TRawTargetData rawTargetData;
                rawTargetData.TargetType = ERawTargetType::String;
                rawTargetData.Target = {target};
                rawTargetData.Weights = TWeights<float>(TVector<float>(weight));
                rawTargetData.GroupWeights = TWeights<float>(target.size());

                auto rawTargetDataProvider = CreateProviderSimple(target.size(), rawTargetData);

                UNIT_ASSERT(!rawTargetDataProvider.GetWeights().IsTrivial());
                UNIT_ASSERT(Equal(rawTargetDataProvider.GetWeights().GetNonTrivialData(), weight));
            }

            // check Set
            {
                TRawTargetData rawTargetData;
                rawTargetData.TargetType = ERawTargetType::String;
                rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target};
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
                    rawTargetData.Pairs = TRawPairsData(pairs);

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    UNIT_ASSERT(
                        Equal(
                            rawTargetDataProvider.GetPairs(),
                            TMaybeData<TRawPairsData>(pairs),
                            EqualWithoutOrder
                        )
                    );
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target}; // for TRawTargetDataProvider creation
                    rawTargetData.SetTrivialWeights(target.size());

                    auto rawTargetDataProvider = CreateProviderSimple(groupBounds, rawTargetData);

                    rawTargetDataProvider.SetPairs(pairs);

                    UNIT_ASSERT(
                        Equal(
                            rawTargetDataProvider.GetPairs(),
                            TMaybeData<TRawPairsData>(pairs),
                            EqualWithoutOrder
                        )
                    );
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
                    rawTargetData.Pairs = TRawPairsData(pairs);

                    UNIT_ASSERT_EXCEPTION(
                        CreateProviderSimple(groupBounds, rawTargetData),
                        TCatBoostException
                    );
                }

                // check Set
                {
                    TRawTargetData rawTargetData;
                    rawTargetData.TargetType = ERawTargetType::String;
                    rawTargetData.Target = {target}; // for TRawTargetDataProvider creation
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
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = { TVector<TString>{"0", "1", "1", "0", "1", "0"} };
            rawTargetData.SetTrivialWeights(6);
            rawTargetData.Baseline = {
                {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f},
                {1.0f, 2.1f, 1.3f, 2.2f, 3.3f, 4.7f}
            };

            rawTargetDataVector.push_back(rawTargetData);
        }

        {
            TRawTargetData rawTargetData;
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = {
                TVector<TString>{"0.0", "1.0", "1.0", "0.0", "1.0", "0.0", "1.0f", "0.5", "0.8"}
            };
            rawTargetData.Baseline = {{0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f}};
            rawTargetData.Weights = TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.8f, 0.9f, 0.1f});
            rawTargetData.GroupWeights = TWeights<float>(
                {1.0f, 3.0f, 2.0f, 2.1f, 2.1f, 2.1f, 0.0f, 1.1f, 1.1f}
            );
            rawTargetData.Pairs = TFlatPairsInfo{TPair(7, 8, 0.0f), TPair(3, 5, 1.0f), TPair(3, 4, 2.0f)};

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
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = { TVector<TString>{"1", "0"} };
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
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = { TVector<TString>{"1.0", "0.0", "1.0", "0.0"} };
            rawTargetData.Baseline = {{0.3f, 0.2f, 0.35f, 0.8f}};
            rawTargetData.Weights = TWeights<float>({2.0f, 3.0f, 0.0f, 1.0f});
            rawTargetData.GroupWeights = TWeights<float>({2.0f, 2.1f, 2.1f, 2.1f});
            rawTargetData.Pairs = TFlatPairsInfo{TPair(1, 3, 1.0f), TPair(1, 2, 2.0f)};

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
                    Equal(
                        subsetDataProvider.GetPairs(),
                        expectedSubsetDataProvider.GetPairs(),
                        EqualWithoutOrder
                    )
                );

                UNIT_ASSERT_EQUAL(
                    *subsetDataProvider.GetObjectsGrouping(),
                    *expectedSubsetDataProvider.GetObjectsGrouping()
                );
            }
        }
    }

    Y_UNIT_TEST(GetMultiTargetSubset) {
        TVector<TRawTargetData> rawTargetDataVector;

        {
            TRawTargetData rawTargetData;
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = {
                TVector<TString>{"0", "1", "1", "0", "1", "0"},
                TVector<TString>{"1", "0", "0", "1", "0", "1"}
            };
            rawTargetData.SetTrivialWeights(6);
            rawTargetData.Baseline = {
                {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f},
                {1.0f, 2.1f, 1.3f, 2.2f, 3.3f, 4.7f}
            };

            rawTargetDataVector.push_back(rawTargetData);
        }

        {
            TRawTargetData rawTargetData;
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = {
                TVector<TString>{"0.0", "1.0", "1.0", "0.0", "1.0", "0.0", "1.0f", "0.5", "0.8"},
                TVector<TString>{"-0.0", "-1.0", "-1.0", "-0.0", "-1.0", "-0.0", "-1.0f", "-0.5", "-0.8"}
            };
            rawTargetData.Baseline = {{0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f}};
            rawTargetData.Weights = TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.8f, 0.9f, 0.1f});
            rawTargetData.GroupWeights = TWeights<float>(
                {1.0f, 3.0f, 2.0f, 2.1f, 2.1f, 2.1f, 0.0f, 1.1f, 1.1f}
            );
            rawTargetData.Pairs = TFlatPairsInfo{TPair(7, 8, 0.0f), TPair(3, 5, 1.0f), TPair(3, 4, 2.0f)};

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
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = {TVector<TString>{"1", "0"}, TVector<TString>{"0", "1"}};
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
            rawTargetData.TargetType = ERawTargetType::String;
            rawTargetData.Target = {
                TVector<TString>{"1.0", "0.0", "1.0", "0.0"},
                TVector<TString>{"-1.0", "-0.0", "-1.0", "-0.0"}
            };
            rawTargetData.Baseline = {{0.3f, 0.2f, 0.35f, 0.8f}};
            rawTargetData.Weights = TWeights<float>({2.0f, 3.0f, 0.0f, 1.0f});
            rawTargetData.GroupWeights = TWeights<float>({2.0f, 2.1f, 2.1f, 2.1f});
            rawTargetData.Pairs = TFlatPairsInfo{TPair(1, 3, 1.0f), TPair(1, 2, 2.0f)};

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
                    Equal(
                        subsetDataProvider.GetPairs(),
                        expectedSubsetDataProvider.GetPairs(),
                        EqualWithoutOrder
                    )
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

    TVector<TSharedVector<float>> MakeTarget(const TVector<TVector<float>>& target) {
        auto processedTarget = TVector<TSharedVector<float>>();
        processedTarget.reserve(target.size());
        for (const auto& subTarget : target) {
            processedTarget.emplace_back(MakeAtomicShared<TVector<float>>(subTarget));
        }
        return processedTarget;
    }

    // subsets are fixed: first is always FullSubset, second is always TIndexedSubset<ui32>{2, 3}
    // TComparisonFunc must accept two TTargetDataProviders and check their equality
    void TestGetSubset(
        const TVector<TTargetDataProviderPtr>& targetsVector,
        const TVector<TTargetDataProviderPtr>& expectedSecondSubsets,

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
        THashMap<TExpectedMapIndex, TTargetDataProviderPtr> expectedResults;

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
                        targetsVector[targetVectorIdx]->GetObjectsGrouping(),
                        TArraySubsetIndexing<ui32>(subsetVector[subsetIdx]),
                        subsetOrdersVector[subsetIdx]
                    );

                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(2);

                TTargetDataProviderPtr subsetTarget = targetsVector[targetVectorIdx]->GetSubset(
                    objectsGroupingSubset,
                    &localExecutor
                );

                TTargetDataProviderPtr expectedSubsetTarget =
                    expectedResults[TExpectedMapIndex(targetVectorIdx, subsetIdx)];

                UNIT_ASSERT_EQUAL(*subsetTarget, *expectedSubsetTarget);
            }
        }
    }


    Y_UNIT_TEST(MultiTarget_GetSubset) {
        TVector<TTargetDataProviderPtr> targetVector;
        TVector<TTargetDataProviderPtr> expectedSecondSubsets;

        {
            TProcessedTargetData data;
            data.Targets.emplace("", MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f}}));
            data.Weights.emplace("", Share(TWeights<float>(6)));

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(6)),
                    std::move(data)
                )
            );
        }
        {
            TProcessedTargetData data;
            data.Targets.emplace("", MakeTarget({{1.0f, 0.0f}, {0.0f, 1.0f}}));
            data.Weights.emplace("", Share(TWeights<float>(2)));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(2)),
                    std::move(data)
                )
            );
        }

        {
            TProcessedTargetData data;
            data.Targets.emplace("", MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f})));
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(1, ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f}))
            );

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(6)),
                    std::move(data)
                )
            );
        }
        {
            TProcessedTargetData data;
            data.Targets.emplace("", MakeTarget({{1.0f, 0.0f}, {0.0f, 1.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({2.0f, 3.0f})));
            data.Baselines.emplace("", TVector<TSharedVector<float>>(1, ShareVector<float>({0.3f, 0.2f})));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(2)),
                    std::move(data)
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(BinClass_GetSubset) {
        TVector<TTargetDataProviderPtr> targetVector;
        TVector<TTargetDataProviderPtr> expectedSecondSubsets;

        {
            TProcessedTargetData data;
            data.TargetsClassCount.emplace("", 2);
            data.Targets.emplace("", MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>(6)));

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(6)),
                    std::move(data)
                )
            );
        }
        {
            TProcessedTargetData data;
            data.TargetsClassCount.emplace("", 2);
            data.Targets.emplace("", MakeTarget({{1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>(2)));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(2)),
                    std::move(data)
                )
            );
        }

        {
            TProcessedTargetData data;
            data.TargetsClassCount.emplace("", 2);
            data.Targets.emplace("", MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f})));
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(1, ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f}))
            );

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(6)),
                    std::move(data)
                )
            );
        }
        {
            TProcessedTargetData data;
            data.TargetsClassCount.emplace("", 2);
            data.Targets.emplace("", MakeTarget({{1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({2.0f, 3.0f})));
            data.Baselines.emplace("", TVector<TSharedVector<float>>(1, ShareVector<float>({0.3f, 0.2f})));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(2)),
                    std::move(data)
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TMultiClassTarget_GetSubset) {
        TVector<TTargetDataProviderPtr> targetVector;
        TVector<TTargetDataProviderPtr> expectedSecondSubsets;

        {
            TProcessedTargetData data;
            data.TargetsClassCount.emplace("", 2);
            data.Targets.emplace("", MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f})));
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>{
                    ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f}),
                    ShareVector<float>({1.0f, 2.1f, 1.3f, 2.2f, 3.3f, 4.7f})
                }
            );

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(6)),
                    std::move(data)
                )
            );
        }
        {
            TProcessedTargetData data;
            data.TargetsClassCount.emplace("", 2);
            data.Targets.emplace("", MakeTarget({{1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({2.0f, 3.0f})));
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>{
                    ShareVector<float>({0.3f, 0.2f}),
                    ShareVector<float>({1.3f, 2.2f})
                }
            );

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(ui32(2)),
                    std::move(data)
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TGroupwiseRankingTarget_GetSubset) {
        TVector<TTargetDataProviderPtr> targetVector;
        TVector<TTargetDataProviderPtr> expectedSecondSubsets;

        {
            TProcessedTargetData data;

            data.Targets.emplace(
                "",
                MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.8f}})
            );
            data.Weights.emplace("", Share(TWeights<float>(9)));
            data.GroupInfos.emplace(
                "",
                ShareVector<TQueryInfo>(
                    {
                        TQueryInfo(0, 1),
                        TQueryInfo(1, 2),
                        TQueryInfo(2, 3),
                        TQueryInfo(3, 6),
                        TQueryInfo(6, 7),
                        TQueryInfo(7, 9)
                    }
                )
            );

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 3}, {3, 6}, {6, 7}, {7, 9}}
                    ),
                    std::move(data)
                )
            );
        }
        {
            TProcessedTargetData data;

            data.Targets.emplace("", MakeTarget({{1.0f, 0.0f, 1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>(4)));
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>({TQueryInfo(0, 1), TQueryInfo(1, 4)}));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 1}, {1, 4}}),
                    std::move(data)
                )
            );
        }

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

            TProcessedTargetData data;
            data.Targets.emplace(
                "",
                MakeTarget({{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.8f}})
            );
            data.Weights.emplace(
                "",
                Share(TWeights<float>({1.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 0.8f, 0.9f, 0.1f}))
            );
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(
                    1,
                    ShareVector<float>({0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f})
                )
            );
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>(std::move(groupInfo)));

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 3}, {3, 5}, {5, 6}, {6, 8}, {8, 9}}
                    ),
                    std::move(data)
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

            TProcessedTargetData data;
            data.Targets.emplace("", MakeTarget({{0.0f, 1.0f, 0.0f}}));
            data.Weights.emplace("", Share(TWeights<float>({3.0f, 0.0f, 1.0f})));
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(1, ShareVector<float>({0.2f, 0.35f, 0.8f}))
            );
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>(std::move(groupInfo)));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 2}, {2, 3}}),
                    std::move(data)
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TGroupPairwiseRankingTarget_GetSubset) {
        TVector<TTargetDataProviderPtr> targetVector;
        TVector<TTargetDataProviderPtr> expectedSecondSubsets;

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

            TProcessedTargetData data;
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(
                    1,
                    ShareVector<float>(
                        {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f, 0.0f, 1.0f}
                    )
                )
            );
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>(std::move(groupInfo)));

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 6}, {6, 7}, {7, 9}, {9, 11}}
                    ),
                    std::move(data)
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

            TProcessedTargetData data;
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(
                    1,
                    ShareVector<float>({0.3f, 0.2f, 0.35f, 0.8f, 0.12f})
                )
            );
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>(std::move(groupInfo)));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>{{0, 4}, {4, 5}}),
                    std::move(data)
                )
            );
        }

        TestGetSubset(
            targetVector,
            expectedSecondSubsets
        );
    }

    Y_UNIT_TEST(TGroupPairwiseRankingTarget_GetSubsetWithShuffle) {
        TVector<TTargetDataProviderPtr> targetVector;
        TVector<TTargetDataProviderPtr> expectedSecondSubsets;

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


            TProcessedTargetData data;
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(
                    1,
                    ShareVector<float>(
                        {0.0f, 0.1f, 0.3f, 0.2f, 0.35f, 0.8f, 0.12f, 0.67f, 0.87f, 0.0f, 1.0f}
                    )
                )
            );
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>(std::move(groupInfo)));

            targetVector.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 1}, {1, 2}, {2, 6}, {6, 7}, {7, 9}, {9, 11}}
                    ),
                    std::move(data)
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


            TProcessedTargetData data;
            data.Baselines.emplace(
                "",
                TVector<TSharedVector<float>>(
                    1,
                    ShareVector<float>({0.8f, 0.3f, 0.2f, 0.35f, 0.87f, 0.67f, 0.0f, 1.0f, 0.0f, 0.12f, 0.1f})
                )
            );
            data.GroupInfos.emplace("", ShareVector<TQueryInfo>(std::move(groupInfo)));

            expectedSecondSubsets.push_back(
                MakeIntrusive<TTargetDataProvider>(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 4}, {4, 6}, {6, 8}, {8, 9}, {9, 10}, {10, 11}}
                    ),
                    std::move(data)
                )
            );
        }

        TRestorableFastRng64 rand(0);

        TestGetSubset(
            targetVector,
            expectedSecondSubsets,
            Shuffle(targetVector.back()->GetObjectsGrouping(), 1, &rand)
        );
    }
}

