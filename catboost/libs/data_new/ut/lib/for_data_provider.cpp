#include "for_data_provider.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data_new/util.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/array_ref.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>

#include <functional>


namespace NCB {
    namespace NDataNewUT {

    template <class T>
    void Compare(const TMaybeData<TConstArrayRef<T>>& lhs, const TMaybe<TVector<T>>& rhs) {
        if (lhs) {
            UNIT_ASSERT(rhs);
            UNIT_ASSERT(Equal(*lhs, *rhs));
        } else {
            UNIT_ASSERT(!rhs);
        }
    }


    void CompareGroupIds(
        const TMaybeData<TConstArrayRef<TGroupId>>& lhs,
        const TMaybe<TVector<TStringBuf>>& rhs
    ) {
        if (lhs) {
            UNIT_ASSERT(rhs);
            const size_t size = lhs->size();
            UNIT_ASSERT_VALUES_EQUAL(size, rhs->size());
            for (auto i : xrange(size)) {
                UNIT_ASSERT_VALUES_EQUAL((*lhs)[i], CalcGroupIdFor((*rhs)[i]));
            }
        } else {
            UNIT_ASSERT(!rhs);
        }
    }

    void CompareSubgroupIds(
        const TMaybeData<TConstArrayRef<TSubgroupId>>& lhs,
        const TMaybe<TVector<TStringBuf>>& rhs
    ) {
        if (lhs) {
            UNIT_ASSERT(rhs);
            const size_t size = lhs->size();
            UNIT_ASSERT_VALUES_EQUAL(size, rhs->size());
            for (auto i : xrange(size)) {
                UNIT_ASSERT_VALUES_EQUAL((*lhs)[i], CalcSubgroupIdFor((*rhs)[i]));
            }
        } else {
            UNIT_ASSERT(!rhs);
        }
    }

    template <EFeatureType FeatureType, class TValue, class TFeatureData>
    void CompareFeatures(
        const TFeaturesLayout& featuresLayout,

        // getFeatureFunc and getExpectedFeatureFunc accept perTypeFeatureIdx as a param
        std::function<TMaybeData<const TFeatureData*>(ui32)> getFeatureFunc,
        std::function<TMaybe<TVector<TValue>>(ui32)> getExpectedFeatureFunc,
        std::function<bool(const TVector<TValue>&, const TFeatureData&)> areEqualFunc
    ) {
        const ui32 perTypeFeatureCount = featuresLayout.GetFeatureCount(FeatureType);

        for (auto perTypeFeatureIdx : xrange(perTypeFeatureCount)) {
            auto maybeFeatureData = getFeatureFunc(perTypeFeatureIdx);
            auto expectedMaybeFeatureData = getExpectedFeatureFunc(perTypeFeatureIdx);
            bool isAvailable = featuresLayout.GetInternalFeatureMetaInfo(
                perTypeFeatureIdx,
                FeatureType
            ).IsAvailable;

            UNIT_ASSERT(!isAvailable || maybeFeatureData);
            UNIT_ASSERT(!isAvailable || expectedMaybeFeatureData);
            if (isAvailable) {
                UNIT_ASSERT(areEqualFunc(*expectedMaybeFeatureData, **maybeFeatureData));
            }
        }
    }

    void CompareObjectsData(
        const TRawObjectsDataProvider& objectsData,
        const TExpectedRawData& expectedData,
        bool catFeaturesHashCanContainExtraData
    ) {
        UNIT_ASSERT_EQUAL(objectsData.GetObjectCount(), expectedData.ObjectsGrouping.GetObjectCount());
        UNIT_ASSERT_EQUAL(*objectsData.GetObjectsGrouping(), expectedData.ObjectsGrouping);
        UNIT_ASSERT_EQUAL(*objectsData.GetFeaturesLayout(), *expectedData.MetaInfo.FeaturesLayout);
        UNIT_ASSERT_VALUES_EQUAL(objectsData.GetOrder(), expectedData.Objects.Order);

        CompareGroupIds(objectsData.GetGroupIds(), expectedData.Objects.GroupIds);
        CompareSubgroupIds(objectsData.GetSubgroupIds(), expectedData.Objects.SubgroupIds);
        Compare(objectsData.GetTimestamp(), expectedData.Objects.Timestamp);

        CompareFeatures<EFeatureType::Float, float, TFloatValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&] (ui32 floatFeatureIdx) {
                return objectsData.GetFloatFeature(floatFeatureIdx);
            },
            /*getExpectedFeatureFunc*/ [&] (ui32 floatFeatureIdx) {
                UNIT_ASSERT(floatFeatureIdx < expectedData.Objects.FloatFeatures.size());
                return expectedData.Objects.FloatFeatures[floatFeatureIdx];
            },
            /*areEqualFunc*/ [&](const TVector<float>& lhs, const TFloatValuesHolder& rhs) {
                return EqualWithNans<float>(lhs, rhs.GetArrayData());
            }
        );

        const ui32 catFeatureCount = objectsData.GetFeaturesLayout()->GetCatFeatureCount();
        TVector<THashMap<ui32, TString>> expectedCatFeaturesHashToString(catFeatureCount);

        CompareFeatures<EFeatureType::Categorical, ui32, THashedCatValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&] (ui32 catFeatureIdx) {return objectsData.GetCatFeature(catFeatureIdx);},
            /*getExpectedFeatureFunc*/ [&] (ui32 catFeatureIdx) -> TMaybe<TVector<ui32>> {
                if (expectedData.Objects.CatFeatures[catFeatureIdx]) {
                    TVector<ui32> hashedCategoricalValues;
                    for (const auto& stringValue : *expectedData.Objects.CatFeatures[catFeatureIdx]) {
                        ui32 hashValue = (ui32)CalcCatFeatureHash(stringValue);
                        expectedCatFeaturesHashToString[catFeatureIdx][hashValue] = TString(stringValue);
                        hashedCategoricalValues.push_back(hashValue);
                    }
                    return hashedCategoricalValues;
                } else {
                    return Nothing();
                }
            },
            /*areEqualFunc*/ [&](const TVector<ui32>& lhs, const THashedCatValuesHolder& rhs) {
                return Equal<ui32>(lhs, rhs.GetArrayData());
            }
        );

        for (auto catFeatureIdx : xrange(catFeatureCount)) {
            if (catFeaturesHashCanContainExtraData) {
                // check that hashes for expected data are present in objectsData.GetCatFeaturesHashToString
                const auto& catFeaturesHashToString = objectsData.GetCatFeaturesHashToString(catFeatureIdx);
                for (const auto& [key, value] : expectedCatFeaturesHashToString[catFeatureIdx]) {
                    auto it = catFeaturesHashToString.find(key);
                    UNIT_ASSERT(it != catFeaturesHashToString.end());
                    UNIT_ASSERT_VALUES_EQUAL(value, it->second);
                }
            } else {
                UNIT_ASSERT_VALUES_EQUAL(
                    objectsData.GetCatFeaturesHashToString(catFeatureIdx),
                    expectedCatFeaturesHashToString[catFeatureIdx]
                );
            }
        }
    }

    void CompareObjectsData(
        const TQuantizedObjectsDataProvider& objectsData,
        const TExpectedQuantizedData& expectedData,
        bool /*catFeaturesHashCanContainExtraData*/
    ) {
        UNIT_ASSERT_EQUAL(objectsData.GetObjectCount(), expectedData.ObjectsGrouping.GetObjectCount());
        UNIT_ASSERT_EQUAL(*objectsData.GetObjectsGrouping(), expectedData.ObjectsGrouping);
        UNIT_ASSERT_EQUAL(*objectsData.GetFeaturesLayout(), *expectedData.MetaInfo.FeaturesLayout);

        Compare(objectsData.GetGroupIds(), expectedData.Objects.GroupIds);
        Compare(objectsData.GetSubgroupIds(), expectedData.Objects.SubgroupIds);
        Compare(objectsData.GetTimestamp(), expectedData.Objects.Timestamp);

        NPar::TLocalExecutor localExecutor;

        CompareFeatures<EFeatureType::Float, ui8, IQuantizedFloatValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&] (ui32 floatFeatureIdx) {
                return objectsData.GetFloatFeature(floatFeatureIdx);
            },
            /*getExpectedFeatureFunc*/ [&] (ui32 floatFeatureIdx) {
                UNIT_ASSERT(floatFeatureIdx < expectedData.Objects.FloatFeatures.size());
                return expectedData.Objects.FloatFeatures[floatFeatureIdx];
            },
            /*areEqualFunc*/ [&](const TVector<ui8>& lhs, const IQuantizedFloatValuesHolder& rhs) {
                return Equal<ui8>(*rhs.ExtractValues(&localExecutor), lhs);
            }
        );

        CompareFeatures<EFeatureType::Categorical, ui32, IQuantizedCatValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&] (ui32 catFeatureIdx) { return objectsData.GetCatFeature(catFeatureIdx); },
            /*getExpectedFeatureFunc*/ [&] (ui32 catFeatureIdx) {
                UNIT_ASSERT(catFeatureIdx < expectedData.Objects.CatFeatures.size());
                return expectedData.Objects.CatFeatures[catFeatureIdx];
            },
            /*areEqualFunc*/ [&](const TVector<ui32>& lhs, const IQuantizedCatValuesHolder& rhs) {
                return Equal<ui32>(*rhs.ExtractValues(&localExecutor), lhs);
            }
        );
        UNIT_ASSERT_EQUAL(
            *objectsData.GetQuantizedFeaturesInfo(),
            *expectedData.Objects.QuantizedFeaturesInfo
        );

        UNIT_ASSERT_VALUES_EQUAL(
            objectsData.GetQuantizedFeaturesInfo()->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(),
            expectedData.Objects.MaxCategoricalFeaturesUniqValuesOnLearn
        );
    }

    void CompareObjectsData(
        const TQuantizedForCPUObjectsDataProvider& objectsData,
        const TExpectedQuantizedData& expectedData,
        bool /*catFeaturesHashCanContainExtraData*/
    ) {
        CompareObjectsData((const TQuantizedObjectsDataProvider&)objectsData, expectedData);

        const auto& featuresLayout = *objectsData.GetFeaturesLayout();

        for (auto floatFeatureIdx : xrange(featuresLayout.GetFloatFeatureCount())) {
            auto expectedMaybeBinaryIndex
                = expectedData.Objects.PackedBinaryFeaturesData
                    .FloatFeatureToPackedBinaryIndex[floatFeatureIdx];

            UNIT_ASSERT_EQUAL(
                objectsData.GetFloatFeatureToPackedBinaryIndex(TFloatFeatureIdx(floatFeatureIdx)),
                expectedMaybeBinaryIndex
            );
            UNIT_ASSERT_VALUES_EQUAL(
                objectsData.IsFeaturePackedBinary(TFloatFeatureIdx(floatFeatureIdx)),
                expectedMaybeBinaryIndex.Defined()
            );
        }

        const ui32 catFeatureCount = featuresLayout.GetFeatureCount(EFeatureType::Categorical);

        UNIT_ASSERT(!catFeatureCount || expectedData.Objects.CatFeatureUniqueValuesCounts);

        for (auto catFeatureIdx : xrange(catFeatureCount)) {
            auto expectedMaybeBinaryIndex
                = expectedData.Objects.PackedBinaryFeaturesData
                    .CatFeatureToPackedBinaryIndex[catFeatureIdx];

            UNIT_ASSERT_EQUAL(
                objectsData.GetCatFeatureToPackedBinaryIndex(TCatFeatureIdx(catFeatureIdx)),
                expectedMaybeBinaryIndex
            );
            UNIT_ASSERT_VALUES_EQUAL(
                objectsData.IsFeaturePackedBinary(TCatFeatureIdx(catFeatureIdx)),
                expectedMaybeBinaryIndex.Defined()
            );

            if (!featuresLayout.GetInternalFeatureMetaInfo(
                    catFeatureIdx,
                    EFeatureType::Categorical
                ).IsAvailable)
            {
                continue;
            }

            UNIT_ASSERT_EQUAL(
                objectsData.GetCatFeatureUniqueValuesCounts(catFeatureIdx),
                (*expectedData.Objects.CatFeatureUniqueValuesCounts)[catFeatureIdx]
            );
        }

        UNIT_ASSERT_EQUAL(
            objectsData.GetPackedBinaryFeaturesSize(),
            expectedData.Objects.PackedBinaryFeaturesData.PackedBinaryToSrcIndex.size()
        );

        for (auto packedBinaryFeatureLinearIdx : xrange(objectsData.GetPackedBinaryFeaturesSize())) {
            UNIT_ASSERT_EQUAL(
                objectsData.GetPackedBinaryFeatureSrcIndex(
                    TPackedBinaryIndex::FromLinearIdx(packedBinaryFeatureLinearIdx)),
                expectedData.Objects.PackedBinaryFeaturesData
                    .PackedBinaryToSrcIndex[packedBinaryFeatureLinearIdx]
            );
        }

        UNIT_ASSERT_EQUAL(
            objectsData.GetBinaryFeaturesPacksSize(),
            expectedData.Objects.PackedBinaryFeaturesData.SrcData.size()
        );

        for (auto packIdx : xrange(objectsData.GetBinaryFeaturesPacksSize())) {
            UNIT_ASSERT(
                Equal(
                    *expectedData.Objects.PackedBinaryFeaturesData.SrcData[packIdx],
                    objectsData.GetBinaryFeaturesPack(packIdx)
                )
            );
        }

    }

    void CompareTargetData(
        const TRawTargetDataProvider& targetData,
        const TObjectsGrouping& expectedObjectsGrouping,
        const TRawTargetData& expectedData
    ) {
        TRawTargetDataProvider expectedTargetData(
            MakeIntrusive<TObjectsGrouping>(expectedObjectsGrouping),
            TRawTargetData(expectedData),
            true,
            Nothing()
        );

        UNIT_ASSERT_EQUAL(targetData, expectedTargetData);
    }

    }
}
