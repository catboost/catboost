#include "for_data_provider.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/sparse_columns.h>
#include <catboost/libs/data/util.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/array_ref.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>

#include <algorithm>
#include <functional>


namespace NCB {
    namespace NDataNewUT {

    template <class T1, class T2>
    void Compare(const TMaybeData<TConstArrayRef<T1>>& lhs, const TMaybe<TVector<T2>>& rhs) {
        if (lhs) {
            UNIT_ASSERT(rhs);
            UNIT_ASSERT(std::equal(lhs->begin(), lhs->end(), rhs->begin(), rhs->end()));
        } else {
            UNIT_ASSERT(!rhs);
        }
    }


    void CompareGroupIds(
        const TMaybeData<TConstArrayRef<TGroupId>>& lhs,
        const TMaybe<TVector<TStringBuf>>& rhs,
        bool treatExpectedDataAsIntegers = false
    ) {
        if (lhs) {
            UNIT_ASSERT(rhs);
            const size_t size = lhs->size();
            UNIT_ASSERT_VALUES_EQUAL(size, rhs->size());
            for (auto i : xrange(size)) {
                if (treatExpectedDataAsIntegers) {
                    UNIT_ASSERT_VALUES_EQUAL((*lhs)[i], FromString<TGroupId>((*rhs)[i]));
                } else {
                    UNIT_ASSERT_VALUES_EQUAL((*lhs)[i], CalcGroupIdFor((*rhs)[i]));
                }
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
        std::function<TMaybe<TExpectedFeatureColumn<TValue>>(ui32)> getExpectedFeatureFunc,
        std::function<bool(const TExpectedFeatureColumn<TValue>&, const TFeatureData&)> areEqualFunc
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

    template <class T, EFeatureValuesType ValuesType>
    bool SimpleEqual(
        const TExpectedFeatureColumn<T>& lhs,
        const ITypedFeatureValuesHolder<T, ValuesType>& rhs
    ) {
        if (const auto* lhsDenseData = std::get_if<TVector<T>>(&lhs)) {
            return Equal<T>(*rhs.ExtractValues(&NPar::LocalExecutor()), *lhsDenseData);
        } else {
            const auto& lhsSparseArray = std::get<TConstPolymorphicValuesSparseArray<T, ui32>>(lhs);
            using TColumn = ITypedFeatureValuesHolder<T, ValuesType>;
            if (const auto* rhsSparseArrayHolder
                    = dynamic_cast<const TSparsePolymorphicArrayValuesHolder<TColumn>*>(&rhs))
            {
                const auto& rhsSparseArray = rhsSparseArrayHolder->GetData();
                // compare field-by-field because lhsSparseArray and rhsSparseArray have different types (const and non-const)
                return (*lhsSparseArray.GetIndexing() == *rhsSparseArray.GetIndexing()) &&
                    AreBlockedSequencesEqual<T, T>(
                        lhsSparseArray.GetNonDefaultValues().GetImpl().GetBlockIterator(),
                        rhsSparseArray.GetNonDefaultValues().GetImpl().GetBlockIterator()
                    ) &&
                    (lhsSparseArray.GetDefaultValue() == rhsSparseArray.GetDefaultValue());
            } else {
                UNIT_FAIL("bad column type for sparse data");
            }
            Y_UNREACHABLE();
        }
    }

    template <class T, EFeatureValuesType ValuesType>
    bool SimpleEqual(
        const TExpectedFeatureColumn<T>& lhs,
        const IQuantizedFeatureValuesHolder<T, ValuesType>& rhs
    ) {
        if (const auto* lhsDenseData = std::get_if<TVector<T>>(&lhs)) {
            return Equal<T>(rhs.template ExtractValues<T>(&NPar::LocalExecutor()), *lhsDenseData);
        } else {
            using TColumn = IQuantizedFeatureValuesHolder<T, ValuesType>;
            const auto& lhsSparseArray = std::get<TConstPolymorphicValuesSparseArray<T, ui32>>(lhs);
            if (const auto* rhsSparseArrayHolder
                           = dynamic_cast<const TSparseCompressedValuesHolderImpl<TColumn>*>(&rhs))
            {
                const auto& rhsSparseArray = rhsSparseArrayHolder->GetData();
                TVector<T> lhsValues = lhsSparseArray.ExtractValues();
                return (*lhsSparseArray.GetIndexing() == *rhsSparseArray.GetIndexing()) &&
                    // switch comparison sides because TCompressionArray has operator==(TConstArrayRef<T>)
                    (rhsSparseArray.GetNonDefaultValues() == TConstArrayRef<T>(lhsValues)) &&
                    (lhsSparseArray.GetDefaultValue() == rhsSparseArray.GetDefaultValue());
            } else {
                UNIT_FAIL("bad column type for sparse data");
            }
            Y_UNREACHABLE();
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

        if (expectedData.MetaInfo.StoreStringColumns) {
            Compare(objectsData.GetStringGroupIds(), expectedData.Objects.GroupIds);
            Compare(objectsData.GetStringSubgroupIds(), expectedData.Objects.SubgroupIds);
        } else {
            CompareGroupIds(
                objectsData.GetGroupIds(),
                expectedData.Objects.GroupIds,
                expectedData.Objects.TreatGroupIdsAsIntegers
            );
            CompareSubgroupIds(objectsData.GetSubgroupIds(), expectedData.Objects.SubgroupIds);
        }

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
            /*areEqualFunc*/ [&](const TExpectedFeatureColumn<float>& lhs, const TFloatValuesHolder& rhs) {
                if (const auto* lhsDenseData = std::get_if<TVector<float>>(&lhs)) {
                    auto rhsValues = rhs.ExtractValues(&NPar::LocalExecutor());
                    return std::equal(
                        lhsDenseData->begin(),
                        lhsDenseData->end(),
                        rhsValues.begin(),
                        rhsValues.end(),
                        EqualWithNans<float>
                    );
                } else {
                    const auto& lhsSparseArray = std::get<TConstPolymorphicValuesSparseArray<float, ui32>>(lhs);

                    const auto* rhsSparseArrayHolder = dynamic_cast<const TFloatSparseValuesHolder*>(&rhs);
                    UNIT_ASSERT(rhsSparseArrayHolder);
                    const auto& rhsSparseArray = rhsSparseArrayHolder->GetData();
                    if (!(*lhsSparseArray.GetIndexing() == *rhsSparseArray.GetIndexing())) {
                        return false;
                    }
                    const auto& lhsNonDefaultValues = lhsSparseArray.GetNonDefaultValues();
                    const auto& rhsNonDefaultValues = rhsSparseArray.GetNonDefaultValues();
                    return AreBlockedSequencesEqual<float, float>(
                        lhsNonDefaultValues.GetImpl().GetBlockIterator(),
                        rhsNonDefaultValues.GetImpl().GetBlockIterator(),
                        EqualWithNans<float>
                    );
                }
            }
        );

        const ui32 catFeatureCount = objectsData.GetFeaturesLayout()->GetCatFeatureCount();
        TVector<THashMap<ui32, TString>> expectedCatFeaturesHashToString(catFeatureCount);

        CompareFeatures<EFeatureType::Categorical, ui32, THashedCatValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&] (ui32 catFeatureIdx) {return objectsData.GetCatFeature(catFeatureIdx);},
            /*getExpectedFeatureFunc*/ [&] (ui32 catFeatureIdx) -> TMaybe<TExpectedFeatureColumn<ui32>> {
                if (expectedData.Objects.CatFeatures[catFeatureIdx]) {
                    if (const auto* denseData = std::get_if<TVector<TStringBuf>>(expectedData.Objects.CatFeatures[catFeatureIdx].Get())) {
                        TVector<ui32> hashedCategoricalValues;
                        for (const auto& stringValue : *denseData) {
                            ui32 hashValue = (ui32)CalcCatFeatureHash(stringValue);
                            expectedCatFeaturesHashToString[catFeatureIdx][hashValue] = TString(stringValue);
                            hashedCategoricalValues.push_back(hashValue);
                        }
                        return hashedCategoricalValues;
                    } else {
                        const auto& sparseData = std::get<TConstPolymorphicValuesSparseArray<TStringBuf, ui32>>(
                            *expectedData.Objects.CatFeatures[catFeatureIdx]
                        );

                        TVector<ui32> hashedNonDefaultCategoricalValues;

                        sparseData.GetNonDefaultValues().GetImpl().ForEach(
                            [&] (TStringBuf stringValue) {
                                ui32 hashValue = (ui32)CalcCatFeatureHash(stringValue);
                                expectedCatFeaturesHashToString[catFeatureIdx][hashValue] = TString(stringValue);
                                hashedNonDefaultCategoricalValues.push_back(hashValue);
                            }
                        );

                        const TStringBuf stringDefaultValue = sparseData.GetDefaultValue();
                        const ui32 hashedDefaultValue = (ui32)CalcCatFeatureHash(stringDefaultValue);
                        expectedCatFeaturesHashToString[catFeatureIdx][hashedDefaultValue]
                            = TString(stringDefaultValue);

                        return MakeConstPolymorphicValuesSparseArray<ui32>(
                            sparseData.GetIndexing(),
                            TMaybeOwningConstArrayHolder<ui32>::CreateOwning(
                                std::move(hashedNonDefaultCategoricalValues)
                            ),
                            ui32(hashedDefaultValue)
                        );
                    }
                } else {
                    return Nothing();
                }
            },
            /*areEqualFunc*/ [&](const TExpectedFeatureColumn<ui32>& lhs, const THashedCatValuesHolder& rhs) {
                return SimpleEqual(lhs, rhs);
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

        CompareFeatures<EFeatureType::Text, TStringBuf, TStringTextValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&](ui32 textFeatureIdx) {return objectsData.GetTextFeature(textFeatureIdx);},
            /*getExpectedFeatureFunc*/ [&](ui32 textFeatureIdx)
                {return *expectedData.Objects.TextFeatures[textFeatureIdx];},
            /*areEqualFunc*/ [&](const TExpectedFeatureColumn<TStringBuf>& lhs, const TStringTextValuesHolder& rhs) {
                if (const auto* lhsDenseData = std::get_if<TVector<TStringBuf>>(&lhs)) {
                    TMaybeOwningArrayHolder<TString> rhsValues = rhs.ExtractValues(&NPar::LocalExecutor());
                    return std::equal(
                        lhsDenseData->begin(),
                        lhsDenseData->end(),
                        rhsValues.begin(),
                        rhsValues.end()
                    );
                } else {
                    const auto& lhsSparseArray = std::get<TConstPolymorphicValuesSparseArray<TStringBuf, ui32>>(lhs);
                    const auto* rhsSparseArrayHolder = dynamic_cast<const TStringTextSparseValuesHolder*>(&rhs);
                    UNIT_ASSERT(rhsSparseArrayHolder);
                    const auto& rhsSparseArray = rhsSparseArrayHolder->GetData();
                    if (!(*lhsSparseArray.GetIndexing() == *rhsSparseArray.GetIndexing()) ||
                        (lhsSparseArray.GetDefaultValue() != rhsSparseArray.GetDefaultValue()))
                    {
                        return false;
                    }
                    const auto& lhsNonDefaultValues = lhsSparseArray.GetNonDefaultValues();
                    const auto& rhsNonDefaultValues = rhsSparseArray.GetNonDefaultValues();
                    return AreBlockedSequencesEqual<TStringBuf, TString>(
                        lhsNonDefaultValues.GetImpl().GetBlockIterator(),
                        rhsNonDefaultValues.GetImpl().GetBlockIterator()
                    );
                }
            }
        );

        CompareFeatures<EFeatureType::Embedding, TVector<float>, TEmbeddingValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&](ui32 embeddingFeatureIdx) {
                return objectsData.GetEmbeddingFeature(embeddingFeatureIdx);
            },
            /*getExpectedFeatureFunc*/ [&](ui32 embeddingFeatureIdx)
                {return *expectedData.Objects.EmbeddingFeatures[embeddingFeatureIdx];},
            /*areEqualFunc*/ [&](
                const TExpectedFeatureColumn<TVector<float>>& lhs,
                const TEmbeddingValuesHolder& rhs
            ) {
                if (const auto* lhsDenseData = std::get_if<TVector<TVector<float>>>(&lhs)) {
                    TMaybeOwningArrayHolder<TMaybeOwningConstArrayHolder<float>> rhsValues
                        = rhs.ExtractValues(&NPar::LocalExecutor());
                    return std::equal(
                        lhsDenseData->begin(),
                        lhsDenseData->end(),
                        rhsValues.begin(),
                        rhsValues.end(),
                        [&] (const TVector<float>& lhs1, const TMaybeOwningConstArrayHolder<float>& rhs1) {
                            return std::equal(
                                lhs1.begin(),
                                lhs1.end(),
                                rhs1.begin(),
                                rhs1.end(),
                                EqualWithNans<float>
                            );
                        }
                    );
                } else {
                    return false;
                }
            }
        );
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
            /*areEqualFunc*/ [&](const TExpectedFeatureColumn<ui8>& lhs, const IQuantizedFloatValuesHolder& rhs) {
                return SimpleEqual(lhs, rhs);
            }
        );

        CompareFeatures<EFeatureType::Categorical, ui32, IQuantizedCatValuesHolder>(
            *objectsData.GetFeaturesLayout(),
            /*getFeatureFunc*/ [&] (ui32 catFeatureIdx) { return objectsData.GetCatFeature(catFeatureIdx); },
            /*getExpectedFeatureFunc*/ [&] (ui32 catFeatureIdx) {
                UNIT_ASSERT(catFeatureIdx < expectedData.Objects.CatFeatures.size());
                return expectedData.Objects.CatFeatures[catFeatureIdx];
            },
            /*areEqualFunc*/ [&](const TExpectedFeatureColumn<ui32>& lhs, const IQuantizedCatValuesHolder& rhs) {
                return SimpleEqual(lhs, rhs);
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

        const auto& featuresLayout = *objectsData.GetFeaturesLayout();

        for (auto floatFeatureIdx : xrange(featuresLayout.GetFloatFeatureCount())) {
            auto flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(floatFeatureIdx, EFeatureType::Float);
            auto expectedMaybeBinaryIndex
                = expectedData.Objects.PackedBinaryFeaturesData
                    .FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx];

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
            auto flatFeatureIdx
                = featuresLayout.GetExternalFeatureIdx(catFeatureIdx, EFeatureType::Categorical);
            auto expectedMaybeBinaryIndex
                = expectedData.Objects.PackedBinaryFeaturesData
                    .FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx];

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
            UNIT_ASSERT_EQUAL(
                expectedData.Objects.PackedBinaryFeaturesData.SrcData[packIdx]->ExtractValues<ui8>(&localExecutor),
                objectsData.GetBinaryFeaturesPack(packIdx).ExtractValues<ui8>(&localExecutor)
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
            /*forceUnitAutoPairWeights*/ true,
            Nothing()
        );

        UNIT_ASSERT_EQUAL(targetData, expectedTargetData);
    }

    }
}
