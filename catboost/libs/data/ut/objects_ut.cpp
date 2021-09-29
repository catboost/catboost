#include "util.h"

#include <catboost/libs/data/objects.h>

#include <catboost/libs/data/cat_feature_perfect_hash_helper.h>
#include <catboost/libs/data/ut/lib/for_objects.h>

#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/random/fast.h>
#include <util/random/shuffle.h>
#include <util/system/info.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>

#include <algorithm>


using namespace NCB;
using namespace NCB::NDataNewUT;


template <class TTObjectsDataProvider>
static TTObjectsDataProvider GetMaybeSubsetDataProvider(
    TTObjectsDataProvider&& objectsDataProvider,
    TMaybe<TArraySubsetIndexing<ui32>> subsetForGetSubset,
    TMaybe<EObjectsOrder> objectsOrderForGetSubset,
    NPar::ILocalExecutor* localExecutor
) {
    if (subsetForGetSubset.Defined()) {
        TObjectsGroupingSubset objectsGroupingSubset = GetSubset(
            objectsDataProvider.GetObjectsGrouping(),
            TArraySubsetIndexing<ui32>(*subsetForGetSubset),
            *objectsOrderForGetSubset
        );

        auto subsetDataProvider = objectsDataProvider.GetSubset(
            objectsGroupingSubset,
            NSystemInfo::TotalMemorySize(),
            localExecutor
        );
        objectsDataProvider = std::move(
            dynamic_cast<TTObjectsDataProvider&>(*subsetDataProvider.Get())
        );
    }
    return std::move(objectsDataProvider);
}

template <class T>
bool Equal(TMaybeData<TConstArrayRef<T>> lhs, TMaybeData<TVector<T>> rhs) {
    if (!lhs) {
        return !rhs;
    }
    if (!rhs) {
        return false;
    }
    return Equal(*lhs, *rhs);
}


Y_UNIT_TEST_SUITE(TRawObjectsData) {
    Y_UNIT_TEST(CommonObjectsData) {
        TVector<TObjectsGrouping> srcObjectsGroupings;
        TVector<TCommonObjectsData> srcDatas;

        {
            srcObjectsGroupings.push_back(TObjectsGrouping(ui32(4)));

            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(4)
            );

            srcDatas.push_back(commonData);
        }

        {
            srcObjectsGroupings.push_back(
                TObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 4}, {4, 6}})
            );

            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();

            TVector<TIndexRange<ui32>> indexRanges{{7, 10}, {2, 3}, {4, 6}};
            TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));

            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TRangesSubset<ui32>(savedIndexRanges)
            );
            commonData.GroupIds.GetMaybeNumData() = {0, 0, 2, 3, 4, 4};

            srcDatas.push_back(commonData);
        }

        {
            srcObjectsGroupings.push_back(
                TObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 3}, {3, 4}})
            );

            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TIndexedSubset<ui32>{0, 4, 3, 1}
            );
            commonData.GroupIds.GetMaybeNumData() = {0, 0, 3, 1};
            commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};

            srcDatas.push_back(commonData);
        }

        {
            srcObjectsGroupings.push_back(TObjectsGrouping(ui32(4)));

            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(4)
            );
            commonData.Timestamp = {100, 0, 20, 10};

            srcDatas.push_back(commonData);
        }

        for (auto i : xrange(srcDatas.size())) {
            const auto& srcData = srcDatas[i];

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            TRawObjectsDataProvider rawObjectsDataProvider(
                Nothing(),
                TCommonObjectsData{srcData},
                TRawObjectsData(),
                false,
                &localExecutor
            );

            UNIT_ASSERT_VALUES_EQUAL(
                rawObjectsDataProvider.GetObjectCount(),
                srcData.SubsetIndexing->Size()
            );
            UNIT_ASSERT_EQUAL(
                *rawObjectsDataProvider.GetObjectsGrouping(),
                srcObjectsGroupings[i]
            );

#define COMPARE_DATA_PROVIDER_FIELD(FIELD) \
            UNIT_ASSERT(Equal(rawObjectsDataProvider.Get##FIELD(), srcData.FIELD.GetMaybeNumData()));

            COMPARE_DATA_PROVIDER_FIELD(GroupIds)
            COMPARE_DATA_PROVIDER_FIELD(SubgroupIds)

#undef COMPARE_DATA_PROVIDER_FIELD
            UNIT_ASSERT(Equal(rawObjectsDataProvider.GetTimestamp(), srcData.Timestamp));

            UNIT_ASSERT_EQUAL(*rawObjectsDataProvider.GetFeaturesLayout(), *srcData.FeaturesLayout);
        }
    }

    Y_UNIT_TEST(BadCommonObjectsData) {
        TVector<TCommonObjectsData> srcs;

        {
            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(4)
            );
            commonData.GroupIds.GetMaybeNumData() = TVector<TGroupId>{0};

            srcs.push_back(commonData);
        }

        {
            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();

            TVector<TIndexRange<ui32>> indexRanges{{7, 10}, {2, 3}, {4, 6}};
            TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));

            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TRangesSubset<ui32>(savedIndexRanges)
            );
            commonData.GroupIds.GetMaybeNumData() = {0, 0, 2, 3, 4, 0};

            srcs.push_back(commonData);
        }

        {
            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TIndexedSubset<ui32>{0, 4, 3, 1}
            );
            commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};

            srcs.push_back(commonData);
        }

        {
            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TIndexedSubset<ui32>{0, 4, 3, 1}
            );
            commonData.GroupIds.GetMaybeNumData() = {0, 0, 1, 1};
            commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3};

            srcs.push_back(commonData);
        }

        {
            TCommonObjectsData commonData;
            commonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>();
            commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                TFullSubset<ui32>(4)
            );
            commonData.Timestamp = {0, 10};

            srcs.push_back(commonData);
        }

        for (auto& src : srcs) {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(2);

            UNIT_ASSERT_EXCEPTION(
                TRawObjectsDataProvider(
                    Nothing(),
                    std::move(src),
                    TRawObjectsData(),
                    false,
                    &localExecutor
                ),
                TCatBoostException
            );
        }
    }

    TRawObjectsDataProvider CreateRawObjectsDataProvider(
        const TVector<TVector<float>>& srcFloatFeatures,
        const TVector<TVector<ui32>>& srcCatFeatures,
        const TVector<THashMap<ui32, TString>>& catFeaturesHashToString,
        const TCommonObjectsData& commonData,
        std::pair<bool, bool> useFeatureTypes,
        NPar::ILocalExecutor* localExecutor
    ) {
        TRawObjectsData data;
        ui32 featureId = 0;

        TVector<ui32> catFeatureIndices;

        if (useFeatureTypes.first) {
            InitFeatures(srcFloatFeatures, *commonData.SubsetIndexing, &featureId, &data.FloatFeatures);
        }
        TCommonObjectsData commonDataCopy = commonData;
        if (useFeatureTypes.second) {
            ui32 catFeaturesIndicesStart = featureId;

            commonDataCopy.CatFeaturesHashToString = MakeAtomicShared<TVector<THashMap<ui32, TString>>>(
                catFeaturesHashToString
            );
            InitFeatures(srcCatFeatures, *commonData.SubsetIndexing, &featureId, &data.CatFeatures);

            for (ui32 idx : xrange(catFeaturesIndicesStart, featureId)) {
                catFeatureIndices.push_back(idx);
            }
        }

        TFeaturesLayout featuresLayout(featureId, catFeatureIndices, {}, {}, {});
        commonDataCopy.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(featuresLayout);

        return TRawObjectsDataProvider(
            Nothing(),
            std::move(commonDataCopy),
            std::move(data),
            /*skipCheck*/ false,
            localExecutor
        );
    }

    using TCatHashToString = std::pair<ui32, TString>;

    TVector<TVector<float>> GetFloatFeatures1() {
         return TVector<TVector<float>>{
            {0.f, 1.f, 2.f, 2.3f, 0.82f, 0.67f},
            {0.22f, 0.3f, 0.16f, 0.f, 0.2f, 0.11f},
            {0.31f, 1.0f, 0.23f, 0.89f, 0.0f, 0.9f}
        };
    }

    void GetCatFeaturesWithHash1(
        TVector<TVector<ui32>>* catFeatures,
        TVector<THashMap<ui32, TString>>* catFeaturesHashToString
    ) {
        catFeaturesHashToString->clear();
        {
            THashMap<ui32, TString> hashToString = {
                TCatHashToString(0x0, "0x0"),
                TCatHashToString(0x12, "0x12"),
                TCatHashToString(0x0F, "0x0F"),
                TCatHashToString(0x23, "0x23"),
                TCatHashToString(0x11, "0x11"),
                TCatHashToString(0x03, "0x03")
            };
            catFeaturesHashToString->push_back(hashToString);
        }
        {
            THashMap<ui32, TString> hashToString = {
                TCatHashToString(0xAB, "0xAB"),
                TCatHashToString(0xBF, "0xBF"),
                TCatHashToString(0x04, "0x04"),
                TCatHashToString(0x20, "0x20"),
                TCatHashToString(0x78, "0x78"),
                TCatHashToString(0xFA, "0xFA")
            };
            catFeaturesHashToString->push_back(hashToString);
        }

        *catFeatures = TVector<TVector<ui32>>{
            {0x0, 0x12, 0x0F, 0x23, 0x11, 0x03},
            {0xAB, 0xBF, 0x04, 0x20, 0x78, 0xFA}
        };
    }


    Y_UNIT_TEST(Equal) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1}
        );
        commonData.Order = EObjectsOrder::RandomShuffled;
        commonData.GroupIds.GetMaybeNumData() = {0, 0, 1, 1};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};

        TVector<TVector<float>> srcFloatFeatures1 = GetFloatFeatures1();

        TVector<TVector<ui32>> srcCatFeatures1;
        TVector<THashMap<ui32, TString>> catFeaturesHashToString1;

        GetCatFeaturesWithHash1(&srcCatFeatures1, &catFeaturesHashToString1);


        TVector<std::pair<bool, bool>> useFeatureTypes = {
            std::make_pair(true, false),
            std::make_pair(false, true),
            std::make_pair(true, true)
        };

        TVector<TRawObjectsDataProviderPtr> objectDataProviders;

        for (auto i : xrange(useFeatureTypes.size())) {
            objectDataProviders.push_back(
                MakeIntrusive<TRawObjectsDataProvider>(
                    CreateRawObjectsDataProvider(
                        srcFloatFeatures1,
                        srcCatFeatures1,
                        catFeaturesHashToString1,
                        commonData,
                        useFeatureTypes[i],
                        &localExecutor
                    )
                )
            );

            for (auto j : xrange(i + 1)) {
                UNIT_ASSERT_VALUES_EQUAL((*(objectDataProviders[i]) == (*objectDataProviders[j])), i == j);
            }
        }

        TVector<TVector<float>> srcFloatFeatures1WithNans = {
            {0.f, 1.f, 2.f, 2.3f, std::numeric_limits<float>::quiet_NaN(), 0.67f},
            {0.22f, 0.3f, 0.16f, 0.f, 0.2f, 0.11f},
            {std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.23f, 0.89f, 0.0f, 0.9f}
        };

        TRawObjectsDataProvider objectDataProviderWithNans =
            CreateRawObjectsDataProvider(
                srcFloatFeatures1WithNans,
                /*srcCatFeatures*/ {},
                /*catFeaturesHashToString*/ {},
                commonData,
                std::make_pair(true, false),
                &localExecutor
            );

        UNIT_ASSERT_EQUAL(objectDataProviderWithNans, objectDataProviderWithNans);

        for (const auto& objectDataProviderPtr : objectDataProviders) {
            UNIT_ASSERT_UNEQUAL(*objectDataProviderPtr, objectDataProviderWithNans);
        }
    }


    void TestSubsetFeatures(
        TMaybe<TArraySubsetIndexing<ui32>> subsetForGetSubset,
        TMaybe<EObjectsOrder> objectsOrderForGetSubset,
        const TObjectsGrouping& expectedObjectsGrouping,
        const TCommonObjectsData& commonData,

        // SubsetIndexing is not checked, only TRawObjectsDataProvider's fields
        const TCommonObjectsData& expectedCommonData,
        const TVector<TVector<float>>& srcFloatFeatures,
        const TVector<TVector<float>>& subsetFloatFeatures,
        const TVector<THashMap<ui32, TString>>& catFeaturesHashToString,
        const TVector<TVector<ui32>>& srcCatFeatures,
        const TVector<TVector<ui32>>& subsetCatFeatures
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        // (use float, use cat) pairs
        for (auto useFeatureTypes : {
            std::make_pair(true, false),
            std::make_pair(false, true),
            std::make_pair(true, true)
        }) {
            auto objectsDataProvider = GetMaybeSubsetDataProvider(
                CreateRawObjectsDataProvider(
                    srcFloatFeatures,
                    srcCatFeatures,
                    catFeaturesHashToString,
                    commonData,
                    useFeatureTypes,
                    &localExecutor
                ),
                subsetForGetSubset,
                objectsOrderForGetSubset,
                &localExecutor
            );

            ui32 featureCount = 0;
            TVector<ui32> catFeatureIndices;

            UNIT_ASSERT_EQUAL(expectedObjectsGrouping, *objectsDataProvider.GetObjectsGrouping());

            if (useFeatureTypes.first) {
                featureCount += srcFloatFeatures.size();

                for (auto i : xrange(subsetFloatFeatures.size())) {
                    UNIT_ASSERT(
                        Equal<float>(
                            *(*objectsDataProvider.GetFloatFeature(i))->ExtractValues(&localExecutor),
                            subsetFloatFeatures[i]
                        )
                    );
                }
            }
            if (useFeatureTypes.second) {
                catFeatureIndices.resize(srcCatFeatures.size());
                std::iota(catFeatureIndices.begin(), catFeatureIndices.end(), featureCount);

                featureCount += srcCatFeatures.size();

                for (auto i : xrange(subsetCatFeatures.size())) {
                    UNIT_ASSERT(
                        Equal<ui32>(
                            *(*objectsDataProvider.GetCatFeature(i))->ExtractValues(&localExecutor),
                            subsetCatFeatures[i]
                        )
                    );
                }
            }

            TFeaturesLayout featuresLayout(featureCount, catFeatureIndices, {}, {}, {});

#define COMPARE_DATA_PROVIDER_FIELD(FIELD) \
            UNIT_ASSERT(Equal(objectsDataProvider.Get##FIELD(), expectedCommonData.FIELD.GetMaybeNumData()));

            COMPARE_DATA_PROVIDER_FIELD(GroupIds)
            COMPARE_DATA_PROVIDER_FIELD(SubgroupIds)

#undef COMPARE_DATA_PROVIDER_FIELD
            UNIT_ASSERT(Equal(objectsDataProvider.GetTimestamp(), expectedCommonData.Timestamp));


            UNIT_ASSERT_EQUAL(*objectsDataProvider.GetFeaturesLayout(), featuresLayout);
            UNIT_ASSERT_VALUES_EQUAL(objectsDataProvider.GetOrder(), expectedCommonData.Order);
        }
    }

    Y_UNIT_TEST(FullSubset) {
        TObjectsGrouping expectedObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 4}});

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TFullSubset<ui32>{4}
        );
        commonData.Order = EObjectsOrder::Ordered;
        commonData.GroupIds.GetMaybeNumData() = {0, 0, 1, 1};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};
        commonData.Timestamp = {10, 0, 100, 20};

        TVector<TVector<float>> floatFeatures = {
            {0.f, 1.f, 2.f, 2.3f},
            {0.22f, 0.3f, 0.16f, 0.f}
        };

        TVector<THashMap<ui32, TString>> catFeaturesHashToString;
        {
            THashMap<ui32, TString> hashToString = {
                TCatHashToString(0x0, "0x0"),
                TCatHashToString(0x12, "0x12"),
                TCatHashToString(0x0F, "0x0F"),
                TCatHashToString(0x23, "0x23")
            };
            catFeaturesHashToString.push_back(hashToString);
        }
        {
            THashMap<ui32, TString> hashToString = {
                TCatHashToString(0xAB, "0xAB"),
                TCatHashToString(0xBF, "0xBF"),
                TCatHashToString(0x04, "0x04"),
                TCatHashToString(0x20, "0x20")
            };
            catFeaturesHashToString.push_back(hashToString);
        }

        TVector<TVector<ui32>> catFeatures = {
            {0x0, 0x12, 0x0F, 0x23},
            {0xAB, 0xBF, 0x04, 0x20}
        };

        TestSubsetFeatures(
            Nothing(),
            Nothing(),
            expectedObjectsGrouping,
            commonData,
            commonData,
            floatFeatures,
            floatFeatures,
            catFeaturesHashToString,
            catFeatures,
            catFeatures
        );
    }

    Y_UNIT_TEST(Subset) {
        TObjectsGrouping expectedObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 4}});

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1}
        );
        commonData.Order = EObjectsOrder::RandomShuffled;
        commonData.GroupIds.GetMaybeNumData() = {0, 0, 1, 1};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};

        TVector<TVector<float>> srcFloatFeatures = GetFloatFeatures1();

        TVector<TVector<float>> subsetFloatFeatures = {
            {0.f, 0.82f, 2.3f, 1.f},
            {0.22f, 0.2f, 0.f, 0.3f},
            {0.31f, 0.0f, 0.89f, 1.0f}
        };

        TVector<TVector<ui32>> srcCatFeatures;
        TVector<THashMap<ui32, TString>> catFeaturesHashToString;

        GetCatFeaturesWithHash1(&srcCatFeatures, &catFeaturesHashToString);

        TVector<TVector<ui32>> subsetCatFeatures = {
            {0x0, 0x11, 0x23, 0x12},
            {0xAB, 0x78, 0x20, 0xBF}
        };

        TestSubsetFeatures(
            Nothing(),
            Nothing(),
            expectedObjectsGrouping,
            commonData,
            commonData,
            srcFloatFeatures,
            subsetFloatFeatures,
            catFeaturesHashToString,
            srcCatFeatures,
            subsetCatFeatures
        );
    }

    Y_UNIT_TEST(SubsetCompositionTrivialGrouping) {
        TArraySubsetIndexing<ui32> subsetForGetSubset(TIndexedSubset<ui32>{3,1});
        EObjectsOrder objectsOrderForGetSubset = EObjectsOrder::RandomShuffled;

        TObjectsGrouping expectedSubsetObjectsGrouping(ui32(2));

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1}
        );
        commonData.Order = EObjectsOrder::RandomShuffled;

        TVector<TVector<float>> srcFloatFeatures = GetFloatFeatures1();

        TVector<TVector<float>> subsetFloatFeatures = { {1.f, 0.82f}, {0.3f, 0.2f}, {1.0f, 0.0f} };

        TVector<TVector<ui32>> srcCatFeatures;
        TVector<THashMap<ui32, TString>> catFeaturesHashToString;

        GetCatFeaturesWithHash1(&srcCatFeatures, &catFeaturesHashToString);

        TVector<TVector<ui32>> subsetCatFeatures = {{0x12, 0x11}, {0xBF, 0x78}};

        TCommonObjectsData expectedCommonData;
        expectedCommonData.Order = EObjectsOrder::RandomShuffled;

        TestSubsetFeatures(
            subsetForGetSubset,
            objectsOrderForGetSubset,
            expectedSubsetObjectsGrouping,
            commonData,
            expectedCommonData,
            srcFloatFeatures,
            subsetFloatFeatures,
            catFeaturesHashToString,
            srcCatFeatures,
            subsetCatFeatures
        );
    }

    Y_UNIT_TEST(SubsetCompositionNonTrivialGrouping) {
        TArraySubsetIndexing<ui32> subsetForGetSubset(TIndexedSubset<ui32>{3,1});
        // expected indices of objects in src features arrays are: 6 8 9 4 3

        EObjectsOrder objectsOrderForGetSubset = EObjectsOrder::RandomShuffled;

        TObjectsGrouping expectedSubsetObjectsGrouping(TVector<TGroupBounds>{{0, 3}, {3, 5}});

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1, 2, 6, 8, 9}
        );
        commonData.Order = EObjectsOrder::Undefined;
        commonData.GroupIds.GetMaybeNumData() = {0, 1, 1, 2, 2, 3, 3, 3};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1, 7, 0, 2, 4};
        commonData.Timestamp = {10, 20, 15, 30, 25, 16, 22, 36};

        TCommonObjectsData expectedSubsetCommonData;
        expectedSubsetCommonData.Order = EObjectsOrder::RandomShuffled;
        expectedSubsetCommonData.GroupIds.GetMaybeNumData() = {3, 3, 3, 1, 1};
        expectedSubsetCommonData.SubgroupIds.GetMaybeNumData() = {0, 2, 4, 2, 3};
        expectedSubsetCommonData.Timestamp = {16, 22, 36, 20, 15};

        TVector<TVector<float>> srcFloatFeatures = {
            {0.f, 1.f, 2.f, 2.3f, 0.82f, 0.67f, 0.72f, 0.13f, 0.56f, 0.01f, 0.22f},
            {0.22f, 0.3f, 0.16f, 0.f, 0.2f, 0.11f, 0.98f, 0.22f, 0.73f, 0.01f, 0.64f},
            {0.31f, 1.0f, 0.23f, 0.89f, 0.0f, 0.9f, 0.45f, 0.72f, 0.88f, 0.11f, 0.03f},
        };


        TVector<TVector<float>> subsetFloatFeatures = {
            {0.72f, 0.56f, 0.01f, 0.82f, 2.3f},
            {0.98f, 0.73f, 0.01f, 0.2f, 0.f},
            {0.45f, 0.88f, 0.11f, 0.0f, 0.89f}
        };

        TVector<THashMap<ui32, TString>> catFeaturesHashToString;
        {
            THashMap<ui32, TString> hashToString = {
                TCatHashToString(0x0, "0x0"),
                TCatHashToString(0x12, "0x12"),
                TCatHashToString(0x0F, "0x0F"),
                TCatHashToString(0x23, "0x23"),
                TCatHashToString(0x11, "0x11"),
                TCatHashToString(0x03, "0x03"),
                TCatHashToString(0x18, "0x18"),
                TCatHashToString(0xA3, "0xA3"),
                TCatHashToString(0x0B, "0x0B"),
                TCatHashToString(0x34, "0x34"),
                TCatHashToString(0x71, "0x71"),
            };
            catFeaturesHashToString.push_back(hashToString);
        }
        {
            THashMap<ui32, TString> hashToString = {
                TCatHashToString(0xAB, "0xAB"),
                TCatHashToString(0xBF, "0xBF"),
                TCatHashToString(0x04, "0x04"),
                TCatHashToString(0x20, "0x20"),
                TCatHashToString(0x78, "0x78"),
                TCatHashToString(0xFA, "0xFA"),
                TCatHashToString(0xAC, "0xAC"),
                TCatHashToString(0x91, "0x91"),
                TCatHashToString(0x02, "0x02"),
                TCatHashToString(0x99, "0x99"),
                TCatHashToString(0xAA, "0xAA")
            };
            catFeaturesHashToString.push_back(hashToString);
        }

        TVector<TVector<ui32>> srcCatFeatures = {
            {0x00, 0x12, 0x0F, 0x23, 0x11, 0x03, 0x18, 0xA3, 0x0B, 0x34, 0x71},
            {0xAB, 0xBF, 0x04, 0x20, 0x78, 0xFA, 0xAC, 0x91, 0x02, 0x99, 0xAA}
        };


        TVector<TVector<ui32>> subsetCatFeatures = {
            {0x18, 0x0B, 0x34, 0x11, 0x23},
            {0xAC, 0x02, 0x99, 0x78, 0x20}
        };


        TestSubsetFeatures(
            subsetForGetSubset,
            objectsOrderForGetSubset,
            expectedSubsetObjectsGrouping,
            commonData,
            expectedSubsetCommonData,
            srcFloatFeatures,
            subsetFloatFeatures,
            catFeaturesHashToString,
            srcCatFeatures,
            subsetCatFeatures
        );
    }

}


Y_UNIT_TEST_SUITE(TQuantizedObjectsData) {

    TVector<ui32> GenerateSrcHashedCatData(ui32 uniqValues) {
        TFastRng<ui32> rng(0);

        TVector<ui32> result;

        for (ui32 value : xrange(uniqValues)) {
            // repeat each value from 1 to 10 times, simple rand() is ok for this purpose
            ui32 repetitionCount = rng.Uniform(1, 11) ;//ui32(1 + (std::rand() % 10));
            for (auto i : xrange(repetitionCount)) {
                Y_UNUSED(i);
                result.push_back(value);
            }
        }

        Shuffle(result.begin(), result.end(), rng);

        return result;
    }


    void TestSubsetFeatures(
        TMaybe<TArraySubsetIndexing<ui32>> subsetForGetSubset,
        TMaybe<EObjectsOrder> objectsOrderForGetSubset,
        const TObjectsGrouping& expectedObjectsGrouping,
        const TCommonObjectsData& commonData,

        // SubsetIndexing is not checked, only TRawObjectsDataProvider's fields
        const TCommonObjectsData& expectedCommonData,

        // for getting bitsPerKey for GPU
        const TVector<ui32>& srcFloatFeatureBinCounts,
        const TVector<TVector<ui8>>& srcFloatFeatures,
        const TVector<TVector<ui8>>& subsetFloatFeatures,

        // for initialization of uniq values data in TQuantizedFeaturesInfo
        const TVector<ui32>& srcUniqHashedCatValues,
        const TVector<TVector<ui32>>& srcCatFeatures,
        const TVector<TVector<ui32>>& subsetCatFeatures,

        // (useFloatFeatures, useCatFeatures) -> checkSum
        const THashMap<std::pair<bool, bool>, ui32> expectedUsedFeatureTypesToCheckSum
    ) {
        for (auto taskType : NCB::NDataNewUT::GetTaskTypes()) {
            // (use float, use cat) pairs
            for (auto useFeatureTypes : {
                std::make_pair(true, false),
                std::make_pair(false, true),
                std::make_pair(true, true)
            }) {
                TQuantizedObjectsData data;

                TVector<ui32> catFeatureIndices;
                if (useFeatureTypes.second) {
                    ui32 catFeatureFlatIdx = useFeatureTypes.first ? (ui32)srcFloatFeatures.size() : 0;
                    for (auto i : xrange(srcCatFeatures.size())) {
                        Y_UNUSED(i);
                        catFeatureIndices.push_back(catFeatureFlatIdx++);
                    }
                }

                TFeaturesLayout featuresLayout(
                    ui32(
                        (useFeatureTypes.first ? srcFloatFeatures.size() : 0)
                        + (useFeatureTypes.second ? srcCatFeatures.size() : 0)
                    ),
                    catFeatureIndices,
                    {},
                    {},
                    {}
                );

                auto featuresLayoutPtr = MakeIntrusive<TFeaturesLayout>(featuresLayout);

                TCommonObjectsData commonDataCopy(commonData);
                commonDataCopy.FeaturesLayout = featuresLayoutPtr;

                data.QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                    featuresLayout,
                    TConstArrayRef<ui32>(),
                    NCatboostOptions::TBinarizationOptions()
                );

                data.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                    featuresLayout,
                    TVector<TExclusiveFeaturesBundle>()
                );
                data.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                    featuresLayout,
                    *data.QuantizedFeaturesInfo,
                    data.ExclusiveFeatureBundlesData,
                    true
                );
                data.FeaturesGroupsData = TFeatureGroupsData(
                    featuresLayout,
                    TVector<TFeaturesGroup>()
                );

                ui32 featureId = 0;

                if (useFeatureTypes.first) {
                    for (auto floatFeatureIdx : xrange(srcFloatFeatures.size())) {
                        const auto& floatFeature = srcFloatFeatures[floatFeatureIdx];
                        ui32 bitsPerKey =
                            (taskType == ETaskType::CPU) ?
                            8 :
                            IntLog2(srcFloatFeatureBinCounts[floatFeatureIdx]);

                        auto storage = TMaybeOwningArrayHolder<ui64>::CreateOwning(
                            CompressVector<ui64>(floatFeature.data(), floatFeature.size(), bitsPerKey)
                        );

                        data.FloatFeatures.emplace_back(
                            MakeHolder<TQuantizedFloatValuesHolder>(
                                featureId,
                                TCompressedArray(floatFeature.size(), bitsPerKey, storage),
                                commonData.SubsetIndexing.Get()
                            )
                        );
                        ++featureId;
                    }
                }

                if (useFeatureTypes.second) {
                    TCatFeaturesPerfectHashHelper catFeaturesPerfectHashHelper(
                        data.QuantizedFeaturesInfo
                    );

                    for (auto catFeatureIdx : xrange(srcCatFeatures.size())) {
                        const auto& catFeature = srcCatFeatures[catFeatureIdx];
                        auto hashedCatValues = TMaybeOwningConstArrayHolder<ui32>::CreateOwning(
                            GenerateSrcHashedCatData(srcUniqHashedCatValues[catFeatureIdx])
                        );

                        TArraySubsetIndexing<ui32> fullSubsetForUpdatingPerfectHash(
                            TFullSubset<ui32>((*hashedCatValues).size())
                        );

                        catFeaturesPerfectHashHelper.UpdatePerfectHashAndMaybeQuantize(
                            TCatFeatureIdx(catFeatureIdx),
                            TTypeCastArraySubset<ui32, ui32>(
                                hashedCatValues,
                                &fullSubsetForUpdatingPerfectHash
                            ),
                            /*mapMostFrequentValueTo0*/ false,
                            /*hashedCatDefaultValue*/ Nothing(),
                            /*quantizedDefaultBinFraction*/ Nothing(),
                            /*dstBins*/ Nothing()
                        );

                        ui32 bitsPerKey =
                            (taskType == ETaskType::CPU) ?
                            32 :
                            IntLog2(
                                catFeaturesPerfectHashHelper.GetUniqueValuesCounts(
                                    TCatFeatureIdx(catFeatureIdx)
                                ).OnAll
                            );

                        auto storage = TMaybeOwningArrayHolder<ui64>::CreateOwning(
                            CompressVector<ui64>(catFeature.data(), catFeature.size(), bitsPerKey)
                        );

                        data.CatFeatures.emplace_back(
                            MakeHolder<TQuantizedCatValuesHolder>(
                                featureId,
                                TCompressedArray(catFeature.size(), bitsPerKey, storage),
                                commonData.SubsetIndexing.Get()
                            )
                        );

                        ++featureId;
                    }
                }

                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(2);

                THolder<TQuantizedObjectsDataProvider> objectsDataProvider;

                objectsDataProvider = MakeHolder<TQuantizedObjectsDataProvider>(
                    GetMaybeSubsetDataProvider(
                        TQuantizedObjectsDataProvider(
                            Nothing(),
                            std::move(commonDataCopy),
                            std::move(data),
                            false,
                            &localExecutor
                        ),
                        subsetForGetSubset,
                        objectsOrderForGetSubset,
                        &localExecutor
                    )
                );

                UNIT_ASSERT_EQUAL(
                    *objectsDataProvider->GetObjectsGrouping(),
                    expectedObjectsGrouping
                );

                if (useFeatureTypes.first) {
                    for (auto i : xrange(subsetFloatFeatures.size())) {
                        UNIT_ASSERT_EQUAL(
                            subsetFloatFeatures[i],
                            (*objectsDataProvider->GetFloatFeature(i))->ExtractValues<ui8>(
                                &localExecutor
                            )
                        );
                    }
                }
                if (useFeatureTypes.second) {
                    for (auto i : xrange(subsetCatFeatures.size())) {
                        UNIT_ASSERT_EQUAL(
                            subsetCatFeatures[i],
                            (*objectsDataProvider->GetCatFeature(i))->ExtractValues<ui32>(&localExecutor)
                        );
                    }
                }
                if (taskType == ETaskType::CPU) {
                    auto& quantizedForCPUObjectsDataProvider =
                        dynamic_cast<TQuantizedObjectsDataProvider&>(*objectsDataProvider);

                    if (useFeatureTypes.first) {
                        for (auto i : xrange(subsetFloatFeatures.size())) {
                            UNIT_ASSERT(
                                Equal<ui8>(
                                    (**quantizedForCPUObjectsDataProvider.GetNonPackedFloatFeature(i))
                                        .ExtractValues<ui8>(&localExecutor),
                                    subsetFloatFeatures[i]
                                )
                            );
                        }
                    }

                    if (useFeatureTypes.second) {
                        for (auto i : xrange(subsetCatFeatures.size())) {
                            UNIT_ASSERT(
                                Equal<ui32>(
                                    (**quantizedForCPUObjectsDataProvider.GetNonPackedCatFeature(i))
                                        .ExtractValues<ui32>(&localExecutor),
                                    subsetCatFeatures[i]
                                )
                            );

                            UNIT_ASSERT_VALUES_EQUAL(
                                quantizedForCPUObjectsDataProvider.GetCatFeatureUniqueValuesCounts(i).OnAll,
                                srcUniqHashedCatValues[i]
                            );
                        }
                    }
                }

#define COMPARE_DATA_PROVIDER_FIELD(FIELD) \
            UNIT_ASSERT(Equal(objectsDataProvider->Get##FIELD(), expectedCommonData.FIELD.GetMaybeNumData()));

            COMPARE_DATA_PROVIDER_FIELD(GroupIds)
            COMPARE_DATA_PROVIDER_FIELD(SubgroupIds)

#undef COMPARE_DATA_PROVIDER_FIELD
                UNIT_ASSERT(Equal(objectsDataProvider->GetTimestamp(), expectedCommonData.Timestamp));

                UNIT_ASSERT_EQUAL(*objectsDataProvider->GetFeaturesLayout(), featuresLayout);
                UNIT_ASSERT_VALUES_EQUAL(objectsDataProvider->GetOrder(), expectedCommonData.Order);

                UNIT_ASSERT_VALUES_EQUAL(
                    objectsDataProvider->CalcFeaturesCheckSum(&localExecutor),
                    expectedUsedFeatureTypesToCheckSum.at(useFeatureTypes)
                );
            }
        }
    }


    Y_UNIT_TEST(FullSubset) {
        TObjectsGrouping expectedObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 4}});

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TFullSubset<ui32>{4}
        );
        commonData.Order = EObjectsOrder::Ordered;
        commonData.GroupIds.GetMaybeNumData() = {0, 0, 1, 1};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};
        commonData.Timestamp = {10, 0, 100, 20};

        TVector<ui32> srcFloatFeatureBinCounts = {32, 256};

        TVector<TVector<ui8>> floatFeatures = {{0x01, 0x12, 0x11, 0x03}, {0x22, 0x10, 0x01, 0xAF}};

        TVector<ui32> srcUniqHashedCatValues = {128, 511};

        TVector<TVector<ui32>> catFeatures = {{0x0, 0x02, 0x0F, 0x03}, {0xAB, 0xBF, 0x04, 0x20}};

        THashMap<std::pair<bool, bool>, ui32> expectedUsedFeatureTypesToCheckSum = {
            {{true, false}, 330653220},
            {{false, true}, 1741468018},
            {{true, true}, 3156525411}
        };

        TestSubsetFeatures(
            Nothing(),
            Nothing(),
            expectedObjectsGrouping,
            commonData,
            commonData,
            srcFloatFeatureBinCounts,
            floatFeatures,
            floatFeatures,
            srcUniqHashedCatValues,
            catFeatures,
            catFeatures,
            expectedUsedFeatureTypesToCheckSum
        );
    }

    Y_UNIT_TEST(Subset) {
        TObjectsGrouping expectedObjectsGrouping(TVector<TGroupBounds>{{0, 2}, {2, 4}});

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1}
        );
        commonData.Order = EObjectsOrder::RandomShuffled;
        commonData.GroupIds.GetMaybeNumData() = {0, 0, 1, 1};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1};

        TVector<ui32> srcFloatFeatureBinCounts = {64, 256, 256};

        TVector<TVector<ui8>> srcFloatFeatures = {
            {0x0, 0x12, 0x0F, 0x23, 0x11, 0x01},
            {0xAB, 0xBF, 0x04, 0x20, 0xAA, 0x12},
            {0x10, 0x02, 0x01, 0xFA, 0xFF, 0x11}
        };

        TVector<TVector<ui8>> subsetFloatFeatures = {
            {0x0, 0x11, 0x23, 0x12},
            {0xAB, 0xAA, 0x20, 0xBF},
            {0x10, 0xFF, 0xFA, 0x02}
        };

        TVector<ui32> srcUniqHashedCatValues = {128, 511};

        TVector<TVector<ui32>> srcCatFeatures = {
            {0x0, 0x02, 0x0F, 0x03, 0x01, 0x03},
            {0xAB, 0xBF, 0x04, 0x20, 0x78, 0xFA}
        };

        TVector<TVector<ui32>> subsetCatFeatures = {{0x0, 0x01, 0x03, 0x02}, {0xAB, 0x78, 0x20, 0xBF}};

        THashMap<std::pair<bool, bool>, ui32> expectedUsedFeatureTypesToCheckSum = {
            {{true, false}, 2582600868},
            {{false, true}, 2202839997},
            {{true, true}, 1279137029}
        };


        TestSubsetFeatures(
            Nothing(),
            Nothing(),
            expectedObjectsGrouping,
            commonData,
            commonData,
            srcFloatFeatureBinCounts,
            srcFloatFeatures,
            subsetFloatFeatures,
            srcUniqHashedCatValues,
            srcCatFeatures,
            subsetCatFeatures,
            expectedUsedFeatureTypesToCheckSum
        );
    }

    Y_UNIT_TEST(SubsetCompositionTrivialGrouping) {
        TArraySubsetIndexing<ui32> subsetForGetSubset(TIndexedSubset<ui32>{3,1});
        EObjectsOrder objectsOrderForGetSubset = EObjectsOrder::RandomShuffled;

        TObjectsGrouping expectedSubsetObjectsGrouping(ui32(2));

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1}
        );
        commonData.Order = EObjectsOrder::RandomShuffled;

        TVector<ui32> srcFloatFeatureBinCounts = {64, 256, 256};

        TVector<TVector<ui8>> srcFloatFeatures = {
            {0x0, 0x12, 0x0F, 0x23, 0x11, 0x01},
            {0xAB, 0xBF, 0x04, 0x20, 0xAA, 0x12},
            {0x10, 0x02, 0x01, 0xFA, 0xFF, 0x11}
        };

        TVector<TVector<ui8>> subsetFloatFeatures = {{0x12, 0x11}, {0xBF, 0xAA}, {0x02, 0xFF}};

        TVector<ui32> srcUniqHashedCatValues = {128, 511};

        TVector<TVector<ui32>> srcCatFeatures = {
            {0x00, 0x02, 0x0F, 0x03, 0x01, 0x03},
            {0xAB, 0xBF, 0x04, 0x20, 0x78, 0xFA}
        };

        TVector<TVector<ui32>> subsetCatFeatures = {{0x02, 0x01}, {0xBF, 0x78}};

        TCommonObjectsData expectedCommonData;
        expectedCommonData.Order = EObjectsOrder::RandomShuffled;

        THashMap<std::pair<bool, bool>, ui32> expectedUsedFeatureTypesToCheckSum = {
            {{true, false}, 952444266},
            {{false, true}, 2146138296},
            {{true, true}, 3778734499}
        };

        TestSubsetFeatures(
            subsetForGetSubset,
            objectsOrderForGetSubset,
            expectedSubsetObjectsGrouping,
            commonData,
            expectedCommonData,
            srcFloatFeatureBinCounts,
            srcFloatFeatures,
            subsetFloatFeatures,
            srcUniqHashedCatValues,
            srcCatFeatures,
            subsetCatFeatures,
            expectedUsedFeatureTypesToCheckSum
        );
    }

    Y_UNIT_TEST(SubsetCompositionNonTrivialGrouping) {
        TArraySubsetIndexing<ui32> subsetForGetSubset(TIndexedSubset<ui32>{3,1});
        // expected indices of objects in src features arrays are: 6 8 9 4 3

        EObjectsOrder objectsOrderForGetSubset = EObjectsOrder::RandomShuffled;

        TObjectsGrouping expectedSubsetObjectsGrouping(TVector<TGroupBounds>{{0, 3}, {3, 5}});

        TCommonObjectsData commonData;

        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TIndexedSubset<ui32>{0, 4, 3, 1, 2, 6, 8, 9}
        );
        commonData.Order = EObjectsOrder::Undefined;
        commonData.GroupIds.GetMaybeNumData() = {0, 1, 1, 2, 2, 3, 3, 3};
        commonData.SubgroupIds.GetMaybeNumData() = {0, 2, 3, 1, 7, 0, 2, 4};
        commonData.Timestamp = {10, 20, 15, 30, 25, 16, 22, 36};

        TCommonObjectsData expectedSubsetCommonData;
        expectedSubsetCommonData.Order = EObjectsOrder::RandomShuffled;
        expectedSubsetCommonData.GroupIds.GetMaybeNumData() = {3, 3, 3, 1, 1};
        expectedSubsetCommonData.SubgroupIds.GetMaybeNumData() = {0, 2, 4, 2, 3};
        expectedSubsetCommonData.Timestamp = {16, 22, 36, 20, 15};


        TVector<ui32> srcFloatFeatureBinCounts = {64, 256, 256};

        TVector<TVector<ui8>> srcFloatFeatures = {
            {0x00, 0x12, 0x0F, 0x23, 0x11, 0x01, 0x32, 0x18, 0x22, 0x05, 0x19},
            {0xAB, 0xBF, 0x04, 0x20, 0xAA, 0x12, 0xF2, 0xEE, 0x18, 0x00, 0x90},
            {0x10, 0x02, 0x01, 0xFA, 0xFF, 0x11, 0xFA, 0xFB, 0xAA, 0xAB, 0x00}
        };

        TVector<TVector<ui8>> subsetFloatFeatures = {
            {0x32, 0x22, 0x05, 0x11, 0x23},
            {0xF2, 0x18, 0x00, 0xAA, 0x20},
            {0xFA, 0xAA, 0xAB, 0xFF, 0xFA}
        };

        TVector<ui32> srcUniqHashedCatValues = {128, 511};

        TVector<TVector<ui32>> srcCatFeatures = {
            {0x00, 0x02, 0x0F, 0x03, 0x01, 0x03, 0x72, 0x6B, 0x5A, 0x11, 0x04},
            {0xAB, 0xBF, 0x04, 0x20, 0x78, 0xFA, 0xFF, 0x78, 0x89, 0xFA, 0x3B}
        };

        TVector<TVector<ui32>> subsetCatFeatures = {
            {0x72, 0x5A, 0x11, 0x01, 0x03},
            {0xFF, 0x89, 0xFA, 0x78, 0x20}
        };

        THashMap<std::pair<bool, bool>, ui32> expectedUsedFeatureTypesToCheckSum = {
            {{true, false}, 2838800885},
            {{false, true}, 1744160829},
            {{true, true}, 2395442154}
        };

        TestSubsetFeatures(
            subsetForGetSubset,
            objectsOrderForGetSubset,
            expectedSubsetObjectsGrouping,
            commonData,
            expectedSubsetCommonData,
            srcFloatFeatureBinCounts,
            srcFloatFeatures,
            subsetFloatFeatures,
            srcUniqHashedCatValues,
            srcCatFeatures,
            subsetCatFeatures,
            expectedUsedFeatureTypesToCheckSum
        );

    }
}
