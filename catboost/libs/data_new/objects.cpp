#include "objects.h"
#include "util.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/compare.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/stream/format.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <numeric>
#include <type_traits>


using namespace NCB;


static TMaybe<TVector<ui32>> GetSrcArrayPermutation(
    const TArraySubsetIndexing<ui32>& subsetIndexing,
    NPar::TLocalExecutor* localExecutor
) {
    switch (subsetIndexing.index()) {
        case TVariantIndexV<TFullSubset<ui32>, TArraySubsetIndexing<ui32>::TBase>:
            {
                TVector<ui32> srcArrayPermutation;
                srcArrayPermutation.yresize(subsetIndexing.Size());
                std::iota(srcArrayPermutation.begin(), srcArrayPermutation.end(), 0);
                return MakeMaybe(std::move(srcArrayPermutation));
            }
        case TVariantIndexV<TRangesSubset<ui32>, TArraySubsetIndexing<ui32>::TBase>:
        case TVariantIndexV<TIndexedSubset<ui32>, TArraySubsetIndexing<ui32>::TBase>:
            {
                TVector<ui32> subsetIndices;
                subsetIndices.yresize(subsetIndexing.Size());
                ui32 minIdx = Max<ui32>();
                subsetIndexing.ForEach( // not ParallelForEach to avoid minIdx synchronization issues
                    [&] (ui32 idx, ui32 srcIdx) {
                        subsetIndices[idx] = srcIdx;
                        if (srcIdx < minIdx) {
                            minIdx = srcIdx;
                        }
                    }
                );
                NPar::ParallelFor(
                    *localExecutor,
                    0,
                    SafeIntegerCast<int>(subsetIndices.size()),
                    [minIdx, &subsetIndices] (int i) {
                        subsetIndices[i] -= minIdx;
                    }
                );

                if (!IsPermutation(TVector<ui32>(subsetIndices))) {
                    return Nothing();
                }

                return MakeMaybe(InvertPermutation(subsetIndices));
            }
    }
    Y_UNREACHABLE();
    return Nothing(); // just to silence compiler warnings
}


void NCB::CheckGroupIds(
    ui32 objectCount,
    TMaybeData<TConstArrayRef<TGroupId>> groupIds,
    TMaybe<TObjectsGroupingPtr> objectsGrouping
) {
    if (!groupIds) {
        return;
    }
    auto groupIdsData = *groupIds;

    CheckDataSize(groupIdsData.size(), (size_t)objectCount, "group Ids", false);


    TVector<TGroupId> groupGroupIds;
    TGroupBounds currentGroupBounds(0); // used only if objectsGrouping is defined

    if (objectsGrouping.Defined()) {
        CheckDataSize(
            groupIdsData.size(),
            (size_t)(*objectsGrouping)->GetObjectCount(),
            "group Ids",
            false,
            "objectGrouping's object count",
            true
        );

        groupGroupIds.reserve((*objectsGrouping)->GetGroupCount());
        currentGroupBounds = (*objectsGrouping)->GetGroup(0);
    }

    TGroupId lastGroupId = groupIdsData[0];
    groupGroupIds.emplace_back(lastGroupId);

    // using ui32 for counters/indices here is safe because groupIdsData' size was checked above
    for (auto objectIdx : xrange(ui32(1), ui32(groupIdsData.size()))) {
        if (groupIdsData[objectIdx] != lastGroupId) {
            if (objectsGrouping.Defined()) {
                CB_ENSURE_INTERNAL(
                    objectIdx == currentGroupBounds.End,
                    "objectsGrouping and grouping by groupId have different ends for group #"
                    << (groupGroupIds.size() - 1)
                );
                currentGroupBounds = (*objectsGrouping)->GetGroup((ui32)groupGroupIds.size());
            }

            lastGroupId = groupIdsData[objectIdx];
            groupGroupIds.emplace_back(lastGroupId);
        }
    }

    Sort(groupGroupIds);
    auto it = std::adjacent_find(groupGroupIds.begin(), groupGroupIds.end());
    CB_ENSURE(it == groupGroupIds.end(), "group Ids are not consecutive");
}


TObjectsGrouping NCB::CreateObjectsGroupingFromGroupIds(
    ui32 objectCount,
    TMaybeData<TConstArrayRef<TGroupId>> groupIds
) {
    if (!groupIds) {
        return TObjectsGrouping(objectCount);
    }
    auto groupIdsData = *groupIds;

    CheckDataSize(groupIdsData.size(), (size_t)objectCount, "group Ids", false);

    TVector<TGroupBounds> groupBounds;
    {
        TVector<TGroupId> groupGroupIds;

        ui32 lastGroupBegin = 0;
        TGroupId lastGroupId = groupIdsData[0];
        groupGroupIds.emplace_back(lastGroupId);

        // using ui32 for counters/indices here is safe because groupIdsData' size was checked above
        for (auto objectIdx : xrange(ui32(1), ui32(groupIdsData.size()))) {
            if (groupIdsData[objectIdx] != lastGroupId) {
                lastGroupId = groupIdsData[objectIdx];
                groupGroupIds.emplace_back(lastGroupId);
                groupBounds.emplace_back(lastGroupBegin, objectIdx);
                lastGroupBegin = objectIdx;
            }
        }
        groupBounds.emplace_back(lastGroupBegin, ui32(groupIdsData.size()));

        // check that there're no groupId duplicates
        Sort(groupGroupIds);
        auto it = std::adjacent_find(groupGroupIds.begin(), groupGroupIds.end());
        CB_ENSURE(it == groupGroupIds.end(), "group Ids are not consecutive");
    }

    return TObjectsGrouping(std::move(groupBounds), true);
}


bool NCB::TCommonObjectsData::operator==(const NCB::TCommonObjectsData& rhs) const {
    return (*FeaturesLayout == *rhs.FeaturesLayout) && (Order == rhs.Order) && (GroupIds == rhs.GroupIds) &&
        (SubgroupIds == rhs.SubgroupIds) && (Timestamp == rhs.Timestamp) &&
        ArePointeesEqual(CatFeaturesHashToString, rhs.CatFeaturesHashToString);
}


void NCB::TCommonObjectsData::PrepareForInitialization(
    const TDataMetaInfo& metaInfo,
    ui32 objectCount,
    ui32 prevTailCount
) {
    FeaturesLayout = metaInfo.FeaturesLayout;

    NCB::PrepareForInitialization(metaInfo.HasGroupId, objectCount, prevTailCount, &GroupIds);
    NCB::PrepareForInitialization(metaInfo.HasSubgroupIds, objectCount, prevTailCount, &SubgroupIds);
    NCB::PrepareForInitialization(metaInfo.HasTimestamp, objectCount, prevTailCount, &Timestamp);

    const size_t catFeatureCount = (size_t)metaInfo.FeaturesLayout->GetCatFeatureCount();
    if (catFeatureCount) {
        if (!CatFeaturesHashToString) {
            CatFeaturesHashToString = MakeAtomicShared<TVector<THashMap<ui32, TString>>>();
        }
        CatFeaturesHashToString->resize(catFeatureCount);
    }
}


void NCB::TCommonObjectsData::CheckAllExceptGroupIds() const {
    if (SubgroupIds) {
        CB_ENSURE(
            GroupIds,
            "non-empty SubgroupIds when GroupIds is not defined"
        );
        CheckDataSize(SubgroupIds->size(), GroupIds->size(), "Subgroup Ids", false, "Group Ids size");
    }
    if (Timestamp) {
        CheckDataSize(Timestamp->size(), (size_t)SubsetIndexing->Size(), "Timestamp");
    }
}

void NCB::TCommonObjectsData::Check(TMaybe<TObjectsGroupingPtr> objectsGrouping) const {
    CB_ENSURE_INTERNAL(FeaturesLayout, "FeaturesLayout is undefined");
    if (objectsGrouping.Defined()) {
        CheckDataSize(
            (*objectsGrouping)->GetObjectCount(),
            SubsetIndexing->Size(),
            "objectsGrouping's object count",
            false,
            "SubsetIndexing's Size"
        );
    }
    CheckGroupIds(SubsetIndexing->Size(), GroupIds, objectsGrouping);
    CheckAllExceptGroupIds();
}

NCB::TCommonObjectsData NCB::TCommonObjectsData::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) const {
    TCommonObjectsData result;
    result.ResourceHolders = ResourceHolders;
    result.FeaturesLayout = FeaturesLayout;
    result.Order = Combine(Order, objectsGroupingSubset.GetObjectSubsetOrder());

    result.CatFeaturesHashToString = CatFeaturesHashToString;

    TVector<std::function<void()>> tasks;

    tasks.emplace_back(
        [&, this]() {
            result.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                Compose(*SubsetIndexing, objectsGroupingSubset.GetObjectsIndexing())
            );
        }
    );

    tasks.emplace_back(
        [&, this]() {
            result.GroupIds = GetSubsetOfMaybeEmpty<TGroupId>(
                (TMaybeData<TConstArrayRef<TGroupId>>)GroupIds,
                objectsGroupingSubset.GetObjectsIndexing(),
                localExecutor
            );
        }
    );
    tasks.emplace_back(
        [&, this]() {
            result.SubgroupIds = GetSubsetOfMaybeEmpty<TSubgroupId>(
                (TMaybeData<TConstArrayRef<TSubgroupId>>)SubgroupIds,
                objectsGroupingSubset.GetObjectsIndexing(),
                localExecutor
            );
        }
    );
    tasks.emplace_back(
        [&, this]() {
            result.Timestamp = GetSubsetOfMaybeEmpty<ui64>(
                (TMaybeData<TConstArrayRef<ui64>>)Timestamp,
                objectsGroupingSubset.GetObjectsIndexing(),
                localExecutor
            );
        }
    );

    ExecuteTasksInParallel(&tasks, localExecutor);

    return result;
}

void NCB::TCommonObjectsData::Load(TFeaturesLayoutPtr featuresLayout, ui32 objectCount, IBinSaver* binSaver) {
    FeaturesLayout = featuresLayout;
    SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(TFullSubset<ui32>(objectCount));
    LoadMulti(binSaver, &Order, &GroupIds, &SubgroupIds, &Timestamp);
    AddWithShared(binSaver, &CatFeaturesHashToString);
}

void NCB::TCommonObjectsData::SaveNonSharedPart(IBinSaver* binSaver) const {
    SaveMulti(binSaver, Order, GroupIds, SubgroupIds, Timestamp);
    AddWithShared(
        binSaver,
        const_cast<TAtomicSharedPtr<TVector<THashMap<ui32, TString>>>*>(&CatFeaturesHashToString)
    );
}


NCB::TObjectsDataProvider::TObjectsDataProvider(
    // if not defined - call CreateObjectsGroupingFromGroupIds
    TMaybe<TObjectsGroupingPtr> objectsGrouping,
    TCommonObjectsData&& commonData,
    bool skipCheck
) {
    if (objectsGrouping.Defined()) {
        if (!skipCheck) {
            commonData.Check(objectsGrouping);
        }
        ObjectsGrouping = std::move(*objectsGrouping);
    } else {
        if (!skipCheck) {
            commonData.CheckAllExceptGroupIds();
        }
        ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(
            CreateObjectsGroupingFromGroupIds(
                commonData.SubsetIndexing->Size(),
                commonData.GroupIds
            )
        );
    }
    CommonData = std::move(commonData);

    if ((CommonData.Order == EObjectsOrder::Undefined) && CommonData.Timestamp) {
        const auto& timestamps = *CommonData.Timestamp;
        if ((ObjectsGrouping->GetObjectCount() > 1) &&
            std::is_sorted(timestamps.begin(), timestamps.end()) &&
            (timestamps.front() != timestamps.back()))
        {
            CommonData.Order = EObjectsOrder::Ordered;
        }
    }
}

void NCB::TObjectsDataProvider::SetGroupIds(TConstArrayRef<TGroupId> groupIds) {
    ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(
        CreateObjectsGroupingFromGroupIds(GetObjectCount(), groupIds) // groupIds data size is checked inside
    );
    if (!CommonData.GroupIds) {
        CommonData.GroupIds.ConstructInPlace(groupIds.begin(), groupIds.end());
    } else {
        CommonData.GroupIds->assign(groupIds.begin(), groupIds.end());
    }
}

void NCB::TObjectsDataProvider::SetSubgroupIds(TConstArrayRef<TSubgroupId> subgroupIds) {
    CheckDataSize(subgroupIds.size(), (size_t)GetObjectCount(), "subgroupIds");
    if (!CommonData.SubgroupIds) {
        CommonData.SubgroupIds.ConstructInPlace(subgroupIds.begin(), subgroupIds.end());
    } else {
        CommonData.SubgroupIds->assign(subgroupIds.begin(), subgroupIds.end());
    }
}

template <class T, EFeatureValuesType TType>
static bool AreFeaturesValuesEqual(
    const TTypedFeatureValuesHolder<T, TType>& lhs,
    const TTypedFeatureValuesHolder<T, TType>& rhs
) {
    auto lhsValues = lhs.ExtractValues(&NPar::LocalExecutor());
    auto rhsValues = rhs.ExtractValues(&NPar::LocalExecutor());

    if constexpr (std::is_floating_point<T>::value) {
        return std::equal(
            lhsValues.begin(),
            lhsValues.end(),
            rhsValues.begin(),
            rhsValues.end(),
            EqualWithNans<T>
        );
    } else {
        return *lhsValues == *rhsValues;
    }
}


template <class TFeaturesValuesHolder>
static bool AreFeaturesValuesEqual(
    const THolder<TFeaturesValuesHolder>& lhs,
    const THolder<TFeaturesValuesHolder>& rhs
) {
    if (!lhs) {
        return !rhs;
    }
    if (lhs->GetSize() != rhs->GetSize()) {
        return false;
    }
    return AreFeaturesValuesEqual(*lhs, *rhs);

}


template <class TFeaturesValuesHolder>
static bool AreFeaturesValuesEqual(
    const TVector<THolder<TFeaturesValuesHolder>>& lhs,
    const TVector<THolder<TFeaturesValuesHolder>>& rhs
) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (auto featureIdx : xrange(lhs.size())) {
        if (!AreFeaturesValuesEqual(
                lhs[featureIdx],
                rhs[featureIdx]
            ))
        {
            return false;
        }
    }
    return true;
}


bool NCB::TRawObjectsData::operator==(const NCB::TRawObjectsData& rhs) const {
    return AreFeaturesValuesEqual(FloatFeatures, rhs.FloatFeatures) &&
        AreFeaturesValuesEqual(CatFeatures, rhs.CatFeatures) &&
        AreFeaturesValuesEqual(TextFeatures, rhs.TextFeatures);
}

void NCB::TRawObjectsData::PrepareForInitialization(const TDataMetaInfo& metaInfo) {
    // FloatFeatures, CatFeatures and TextFeatures members are initialized at the end of building
    FloatFeatures.clear();
    FloatFeatures.resize((size_t)metaInfo.FeaturesLayout->GetFloatFeatureCount());

    CatFeatures.clear();
    const size_t catFeatureCount = (size_t)metaInfo.FeaturesLayout->GetCatFeatureCount();
    CatFeatures.resize(catFeatureCount);

    TextFeatures.clear();
    TextFeatures.resize((size_t)metaInfo.FeaturesLayout->GetTextFeatureCount());
}


template <class TFeaturesColumn>
static void CheckDataSizes(
    ui32 objectCount,
    const TFeaturesLayout& featuresLayout,
    const EFeatureType featureType,
    const TVector<THolder<TFeaturesColumn>>& featuresData
) {
    CheckDataSize(
        featuresData.size(),
        (size_t)featuresLayout.GetFeatureCount(featureType),
        TStringBuilder() << "ObjectDataProvider's " << featureType << " features",
        false,
        TStringBuilder() << "featureLayout's " << featureType << " features size",
        true
    );

    for (auto featureIdx : xrange(featuresData.size())) {
        TString dataDescription =
            TStringBuilder() << "ObjectDataProvider's " << featureType << " feature #" << featureIdx;

        auto dataPtr = featuresData[featureIdx].Get();
        bool isAvailable = featuresLayout.GetInternalFeatureMetaInfo(featureIdx, featureType).IsAvailable;
        if (isAvailable) {
            CB_ENSURE_INTERNAL(
                dataPtr,
                dataDescription << " is null despite being available in featuresLayout"
            );
            CheckDataSize(
                dataPtr->GetSize(),
                objectCount,
                dataDescription,
                /*dataCanBeEmpty*/ false,
                "object count",
                /*internalCheck*/ true
            );
        }
    }
}


void NCB::TRawObjectsData::Check(
    ui32 objectCount,
    const TFeaturesLayout& featuresLayout,
    const TVector<THashMap<ui32, TString>>* catFeaturesHashToString,
    NPar::TLocalExecutor* localExecutor
) const {
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Float, FloatFeatures);

    if (CatFeatures.size()) {
        CheckDataSize(
            catFeaturesHashToString ? catFeaturesHashToString->size() : 0,
            CatFeatures.size(),
            "CatFeaturesHashToString",
            /*dataCanBeEmpty*/ false,
            "CatFeatures size",
            /*internalCheck*/ true
        );
    }
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Categorical, CatFeatures);
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Text, TextFeatures);

    localExecutor->ExecRangeWithThrow(
        [&] (int catFeatureIdx) {
            auto* catFeaturePtr = CatFeatures[catFeatureIdx].Get();
            if (catFeaturePtr) {
                const auto& hashToStringMap = (*catFeaturesHashToString)[catFeatureIdx];
                if (hashToStringMap.empty()) {
                    return;
                }

                auto checkValue = [&] (ui32 objectIdx, ui32 hashValue) {
                    CB_ENSURE_INTERNAL(
                        hashToStringMap.contains(hashValue),
                        "catFeature #" << catFeatureIdx << ", object #" << objectIdx << ": value "
                        << Hex(hashValue) << " is missing from CatFeaturesHashToString"
                    );
                };

                if (const auto* catArrayData
                        = dynamic_cast<const THashedCatArrayValuesHolder*>(catFeaturePtr))
                {
                    catArrayData->GetArrayData().ParallelForEach(checkValue, localExecutor);
                } else {
                    CB_ENSURE_INTERNAL(false, "unknown THashedCatValuesHolder subtype");
                }
            }
        },
        0,
        SafeIntegerCast<int>(CatFeatures.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}


template <class T, EFeatureValuesType TType, class TGetAggregatedColumn>
static void CreateSubsetFeatures(
    TConstArrayRef<THolder<TTypedFeatureValuesHolder<T, TType>>> src,
    const TFeaturesArraySubsetIndexing* subsetIndexing,

    /* flatFeatureIdx -> THolder<TTypedFeatureValuesHolder<T, TType>>
     * returns packed or bundled or grouped column data, returns nullptr if not packed or bundled or grouped
     */
    TGetAggregatedColumn&& getAggregatedData,
    //std::function<THolder<TTypedFeatureValuesHolder<T, TType>>(ui32)> getAggregatedData,
    TVector<THolder<TTypedFeatureValuesHolder<T, TType>>>* dst
) {
    dst->clear();
    dst->reserve(src.size());
    for (const auto& feature : src) {
        auto* srcDataPtr = feature.Get();
        if (srcDataPtr) {
            THolder<TTypedFeatureValuesHolder<T, TType>> aggregatedColumn
                = getAggregatedData(srcDataPtr->GetId());
            if (aggregatedColumn) {
                dst->push_back(std::move(aggregatedColumn));
                continue;
            }

            if (const auto* cloneableSrcDataPtr
                = dynamic_cast<const TCloneableWithSubsetIndexingValuesHolder<T, TType>*>(srcDataPtr))
            {
                dst->push_back( cloneableSrcDataPtr->CloneWithNewSubsetIndexing(subsetIndexing) );
            } else {
                CB_ENSURE_INTERNAL(false, "CreateSubsetFeatures: unsupported src column type");
            }
        } else {
            dst->push_back(nullptr);
        }
    }
}

template <class T, EFeatureValuesType TType>
static void CreateSubsetFeatures(
    TConstArrayRef<THolder<TTypedFeatureValuesHolder<T, TType>>> src,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    TVector<THolder<TTypedFeatureValuesHolder<T, TType>>>* dst
) {
    ::CreateSubsetFeatures(
        src,
        subsetIndexing,
        /*getPackedOrBundledData*/ [] (ui32) { return nullptr; },
        dst
    );
}


TObjectsDataProviderPtr NCB::TRawObjectsDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) const {
    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );

    TRawObjectsData subsetData;

    auto createSubsetFeatures = [&] (const auto& srcFeatures, auto* dstFeatures) {
        CreateSubsetFeatures(
            MakeConstArrayRef(srcFeatures),
            subsetCommonData.SubsetIndexing.Get(),
            dstFeatures
        );
    };

    createSubsetFeatures(Data.FloatFeatures, &subsetData.FloatFeatures);
    createSubsetFeatures(Data.CatFeatures, &subsetData.CatFeatures);
    createSubsetFeatures(Data.TextFeatures, &subsetData.TextFeatures);

    return MakeIntrusive<TRawObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}

TObjectsDataProviderPtr NCB::TRawObjectsDataProvider::GetFeaturesSubset(
    const TVector<ui32>& /*ignoredFeatures*/,
    NPar::TLocalExecutor* /*localExecutor*/
) const {
    CB_ENSURE(false, "Not implemented");
}

template <class T, EFeatureValuesType TType>
static void CreateConsecutiveFeaturesData(
    const TVector<THolder<TTypedFeatureValuesHolder<T, TType>>>& srcFeatures,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<TTypedFeatureValuesHolder<T, TType>>>* dstFeatures
) {
    dstFeatures->resize(srcFeatures.size());
    localExecutor->ExecRangeWithThrow(
        [&] (int featureIdx) {
            auto* srcDataPtr = srcFeatures[featureIdx].Get();
            if (!srcDataPtr) {
                return;
            }

            auto dstStorage = srcDataPtr->ExtractValues(localExecutor);

            if constexpr ((TType == EFeatureValuesType::Float) ||
                (TType == EFeatureValuesType::HashedCategorical) ||
                (TType == EFeatureValuesType::StringText) ||
                (TType == EFeatureValuesType::TokenizedText))
            {
                (*dstFeatures)[featureIdx] = MakeHolder<TArrayValuesHolder<T, TType>>(
                    srcDataPtr->GetId(),
                    TMaybeOwningArrayHolder<const T>::CreateOwningReinterpretCast(dstStorage),
                    subsetIndexing
                );
            } else if constexpr ((TType == EFeatureValuesType::QuantizedFloat) ||
                (TType == EFeatureValuesType::PerfectHashedCategorical))
            {
                (*dstFeatures)[featureIdx] = MakeHolder<TCompressedValuesHolderImpl<T, TType>>(
                    srcDataPtr->GetId(),
                    TCompressedArray(
                        srcDataPtr->GetSize(),
                        CHAR_BIT * sizeof(T),
                        TMaybeOwningArrayHolder<ui64>::CreateOwningReinterpretCast(dstStorage)
                    ),
                    subsetIndexing
                );
            } else {
                CB_ENSURE_INTERNAL(false, "Unsupported FeatureValuesType=" << TType);
            }
        },
        0,
        SafeIntegerCast<int>(srcFeatures.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

TIntrusiveConstPtr<TRawObjectsDataProvider>
    NCB::TRawObjectsDataProvider::GetWithPermutedConsecutiveArrayFeaturesData(
        NPar::TLocalExecutor* localExecutor,
        TMaybe<TVector<ui32>>* srcArrayPermutation
    ) const {
        if (CommonData.SubsetIndexing->IsConsecutive()) {
            *srcArrayPermutation = Nothing();
            // TODO(akhropov): proper IntrusivePtr interface to avoid const_cast
            return TIntrusiveConstPtr<TRawObjectsDataProvider>(const_cast<TRawObjectsDataProvider*>(this));
        }

        *srcArrayPermutation = GetSrcArrayPermutation(*CommonData.SubsetIndexing, localExecutor);
        if (*srcArrayPermutation) {
            return TIntrusiveConstPtr<TRawObjectsDataProvider>(
                dynamic_cast<TRawObjectsDataProvider*>(
                    this->GetSubset(
                        GetGroupingSubsetFromObjectsSubset(
                            ObjectsGrouping,
                            TArraySubsetIndexing<ui32>(TVector<ui32>(**srcArrayPermutation)),
                            CommonData.Order
                        ),
                        localExecutor
                    ).Get()
                )
            );
        }

        TCommonObjectsData dstCommonData = CommonData;
        dstCommonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TFullSubset<ui32>(ObjectsGrouping->GetObjectCount())
        );

        TRawObjectsData dstData;

        TVector<std::function<void()>> tasks;

        tasks.push_back(
            [&] () {
                CreateConsecutiveFeaturesData(
                    Data.FloatFeatures,
                    dstCommonData.SubsetIndexing.Get(),
                    localExecutor,
                    &dstData.FloatFeatures
                );
            }
        );

        tasks.push_back(
            [&] () {
                CreateConsecutiveFeaturesData(
                    Data.CatFeatures,
                    dstCommonData.SubsetIndexing.Get(),
                    localExecutor,
                    &dstData.CatFeatures
                );
            }
        );

        tasks.push_back(
            [&] () {
                CreateConsecutiveFeaturesData(
                    Data.TextFeatures,
                    dstCommonData.SubsetIndexing.Get(),
                    localExecutor,
                    &dstData.TextFeatures
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);

        *srcArrayPermutation = Nothing();

        return MakeIntrusiveConst<TRawObjectsDataProvider>(
            ObjectsGrouping,
            std::move(dstCommonData),
            std::move(dstData),
            /*skipCheck*/ true,
            Nothing()
        );
    }


void NCB::TRawObjectsDataProvider::SetGroupIds(TConstArrayRef<TStringBuf> groupStringIds) {
    CheckDataSize(groupStringIds.size(), (size_t)GetObjectCount(), "group Ids");

    TVector<TGroupId> newGroupIds;
    newGroupIds.yresize(groupStringIds.size());
    for (auto i : xrange(groupStringIds.size())) {
        newGroupIds[i] = CalcGroupIdFor(groupStringIds[i]);
    }

    ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(
        CreateObjectsGroupingFromGroupIds(GetObjectCount(), (TConstArrayRef<TGroupId>)newGroupIds)
    );
    CommonData.GroupIds = std::move(newGroupIds);
}

void NCB::TRawObjectsDataProvider::SetSubgroupIds(TConstArrayRef<TStringBuf> subgroupStringIds) {
    CheckDataSize(subgroupStringIds.size(), (size_t)GetObjectCount(), "subgroup Ids");
    CB_ENSURE(
        CommonData.GroupIds,
        "non-empty subgroupStringIds when GroupIds is not defined"
    );

    TVector<TSubgroupId> newSubgroupIds;
    newSubgroupIds.yresize(subgroupStringIds.size());
    for (auto i : xrange(subgroupStringIds.size())) {
        newSubgroupIds[i] = CalcSubgroupIdFor(subgroupStringIds[i]);
    }
    CommonData.SubgroupIds = std::move(newSubgroupIds);
}


bool NCB::TQuantizedObjectsData::operator==(const NCB::TQuantizedObjectsData& rhs) const {
    return AreFeaturesValuesEqual(FloatFeatures, rhs.FloatFeatures) &&
        AreFeaturesValuesEqual(CatFeatures, rhs.CatFeatures) &&
        AreFeaturesValuesEqual(TextFeatures, rhs.TextFeatures);
}


void NCB::TQuantizedObjectsData::PrepareForInitialization(
    const TDataMetaInfo& metaInfo,
    const NCatboostOptions::TBinarizationOptions& binarizationOptions,
    const TMap<ui32, NCatboostOptions::TBinarizationOptions>& perFloatFeatureQuantization
) {
    // FloatFeatures and CatFeatures members are initialized at the end of building
    FloatFeatures.clear();
    FloatFeatures.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());

    CatFeatures.clear();
    const ui32 catFeatureCount = metaInfo.FeaturesLayout->GetCatFeatureCount();
    CatFeatures.resize(catFeatureCount);

    TextFeatures.clear();
    const ui32 textFeatureCount = metaInfo.FeaturesLayout->GetTextFeatureCount();
    TextFeatures.resize(textFeatureCount);

    if (!QuantizedFeaturesInfo) {
        QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *metaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            binarizationOptions,
            perFloatFeatureQuantization,
            /*floatFeaturesAllowNansInTestOnly*/true,

            // be conservative here, it will be reset using SetAllowWriteFiles later if needed
            /*allowWriteFiles*/false
        );
    }
}


void NCB::TQuantizedObjectsData::Check(
    ui32 objectCount,
    const TFeaturesLayout& featuresLayout,
    NPar::TLocalExecutor* localExecutor
) const {
    /* localExecutor is a parameter here to make
     * TQuantizedObjectsData::Check and TQuantizedObjectsData::Check have the same interface
     */
    Y_UNUSED(localExecutor);

    CB_ENSURE(QuantizedFeaturesInfo.Get(), "NCB::TQuantizedObjectsData::QuantizedFeaturesInfo is not initialized");

    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Float, FloatFeatures);
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Categorical, CatFeatures);
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Text, TextFeatures);
}


NCB::TObjectsDataProviderPtr NCB::TQuantizedObjectsDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) const {
    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );

    TQuantizedObjectsData subsetData;

    auto createSubsetFeatures = [&] (const auto& srcFeatures, auto* dstFeatures) {
        CreateSubsetFeatures(
            MakeConstArrayRef(srcFeatures),
            subsetCommonData.SubsetIndexing.Get(),
            dstFeatures
        );
    };

    createSubsetFeatures(Data.FloatFeatures, &subsetData.FloatFeatures);
    createSubsetFeatures(Data.CatFeatures, &subsetData.CatFeatures);
    createSubsetFeatures(Data.TextFeatures, &subsetData.TextFeatures);

    subsetData.QuantizedFeaturesInfo = Data.QuantizedFeaturesInfo;

    return MakeIntrusive<TQuantizedObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


NCB::TObjectsDataProviderPtr NCB::TQuantizedObjectsDataProvider::GetFeaturesSubset(
    const TVector<ui32>& ignoredFeatures,
    NPar::TLocalExecutor* localExecutor
) const {
    const auto& objectsGroupingSubset = ::GetGroupingSubsetFromObjectsSubset(
        ObjectsGrouping,
        TArraySubsetIndexing(TFullSubset<ui32>(GetObjectCount())),
        EObjectsOrder::Ordered);

    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );
    subsetCommonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*subsetCommonData.FeaturesLayout);
    subsetCommonData.FeaturesLayout->IgnoreExternalFeatures(ignoredFeatures);

    TQuantizedObjectsData subsetData;

    auto createSubsetFeatures = [&] (const auto& srcFeatures, auto* dstFeatures) {
        CreateSubsetFeatures(
            MakeConstArrayRef(srcFeatures),
            subsetCommonData.SubsetIndexing.Get(),
            dstFeatures
        );
    };

    createSubsetFeatures(Data.FloatFeatures, &subsetData.FloatFeatures);
    createSubsetFeatures(Data.CatFeatures, &subsetData.CatFeatures);
    createSubsetFeatures(Data.TextFeatures, &subsetData.TextFeatures);

    subsetData.QuantizedFeaturesInfo = Data.QuantizedFeaturesInfo;

    return MakeIntrusive<TQuantizedObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


TIntrusiveConstPtr<TQuantizedObjectsDataProvider>
    NCB::TQuantizedObjectsDataProvider::GetWithPermutedConsecutiveArrayFeaturesData(
        NPar::TLocalExecutor* localExecutor,
        TMaybe<TVector<ui32>>* srcArrayPermutation
    ) const {
        if (CommonData.SubsetIndexing->IsConsecutive()) {
            *srcArrayPermutation = Nothing();
            // TODO(akhropov): proper IntrusivePtr interface to avoid const_cast
            return TIntrusiveConstPtr<TQuantizedObjectsDataProvider>(
                const_cast<TQuantizedObjectsDataProvider*>(this)
            );
        }

        if (AllFeaturesDataIsCompressedArrays()) {
            *srcArrayPermutation = GetSrcArrayPermutation(*CommonData.SubsetIndexing, localExecutor);
            if (*srcArrayPermutation) {
                return TIntrusiveConstPtr<TQuantizedObjectsDataProvider>(
                    dynamic_cast<TQuantizedObjectsDataProvider*>(
                        this->GetSubset(
                            GetGroupingSubsetFromObjectsSubset(
                                ObjectsGrouping,
                                TArraySubsetIndexing<ui32>(TVector<ui32>(**srcArrayPermutation)),
                                CommonData.Order
                            ),
                            localExecutor
                        ).Get()
                    )
                );
            }
        }

        TCommonObjectsData dstCommonData = CommonData;
        dstCommonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
            TFullSubset<ui32>(ObjectsGrouping->GetObjectCount())
        );

        TQuantizedObjectsData dstData;

        TVector<std::function<void()>> tasks;

        tasks.push_back(
            [&] () {
                CreateConsecutiveFeaturesData(
                    Data.FloatFeatures,
                    dstCommonData.SubsetIndexing.Get(),
                    localExecutor,
                    &dstData.FloatFeatures
                );
            }
        );

        tasks.push_back(
            [&] () {
                CreateConsecutiveFeaturesData(
                    Data.CatFeatures,
                    dstCommonData.SubsetIndexing.Get(),
                    localExecutor,
                    &dstData.CatFeatures
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);

        dstData.QuantizedFeaturesInfo = Data.QuantizedFeaturesInfo;
        dstData.CachedFeaturesCheckSum = Data.CachedFeaturesCheckSum;

        *srcArrayPermutation = Nothing();

        return MakeIntrusiveConst<TQuantizedObjectsDataProvider>(
            ObjectsGrouping,
            std::move(dstCommonData),
            std::move(dstData),
            /*skipCheck*/ true,
            Nothing()
        );
    }


template <class T, EFeatureValuesType FeatureValuesType>
static ui32 CalcCompressedFeatureChecksum(
    ui32 checkSum,
    const TCompressedValuesHolderImpl<T, FeatureValuesType>& columnData
) {
    TConstCompressedArraySubset compressedDataSubset = columnData.GetCompressedData();

    auto consecutiveSubsetBegin = compressedDataSubset.GetSubsetIndexing()->GetConsecutiveSubsetBegin();
    const ui32 columnValuesBitWidth = columnData.GetBitsPerKey();
    if (consecutiveSubsetBegin.Defined()) {
        ui8 byteSize = columnValuesBitWidth / 8;
        return UpdateCheckSum(
            checkSum,
            MakeArrayRef(
                compressedDataSubset.GetSrc()->GetRawPtr() + *consecutiveSubsetBegin * byteSize,
                compressedDataSubset.Size())
        );
    }

    if (columnValuesBitWidth == 8) {
        columnData.ForEach([&](ui32 /*idx*/, ui8 element) {
            checkSum = UpdateCheckSum(checkSum, element);
        });
    } else if (columnValuesBitWidth == 16) {
        columnData.ForEach([&](ui32 /*idx*/, ui16 element) {
            checkSum = UpdateCheckSum(checkSum, element);
        });
    } else {
        Y_ASSERT(columnValuesBitWidth == 32);
        columnData.ForEach([&](ui32 /*idx*/, ui32 element) {
            checkSum = UpdateCheckSum(checkSum, element);
        });
    }
    return checkSum;
}

template <EFeatureType FeatureType, class T, EFeatureValuesType FeatureValuesType>
static ui32 CalcFeatureValuesCheckSum(
    ui32 init,
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>& featuresData,
    NPar::TLocalExecutor* localExecutor)
{
    const ui32 emptyColumnDataForCrc = 0;
    TVector<ui32> checkSums(featuresLayout.GetFeatureCount(FeatureType), 0);
    ParallelFor(
        *localExecutor,
        0,
        featuresLayout.GetFeatureCount(FeatureType),
        [&] (ui32 perTypeFeatureIdx) {
            if (featuresLayout.GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType).IsAvailable) {
                // TODO(espetrov,akhropov): remove workaround below MLTOOLS-3604
                if (featuresData[perTypeFeatureIdx].Get() == nullptr) {
                    return;
                }
                auto compressedValuesFeatureData = dynamic_cast<const TCompressedValuesHolderImpl<T, FeatureValuesType>*>(
                    featuresData[perTypeFeatureIdx].Get()
                );
                if (compressedValuesFeatureData) {
                    checkSums[perTypeFeatureIdx] = CalcCompressedFeatureChecksum(0, *compressedValuesFeatureData);
                } else {
                    const auto repackedHolder = featuresData[perTypeFeatureIdx]->ExtractValues(localExecutor);
                    checkSums[perTypeFeatureIdx] = UpdateCheckSum(0, *repackedHolder);
                }
            } else {
                checkSums[perTypeFeatureIdx] = UpdateCheckSum(0, emptyColumnDataForCrc);
            }
        }
    );
    ui32 checkSum = init;
    for (ui32 featureCheckSum : checkSums) {
        checkSum = UpdateCheckSum(checkSum, featureCheckSum);
    }
    return checkSum;
}

ui32 NCB::TQuantizedObjectsDataProvider::CalcFeaturesCheckSum(NPar::TLocalExecutor* localExecutor) const {
    if (!Data.CachedFeaturesCheckSum) {
        ui32 checkSum = 0;

        checkSum = Data.QuantizedFeaturesInfo->CalcCheckSum();
        checkSum = CalcFeatureValuesCheckSum<EFeatureType::Float>(
            checkSum,
            *CommonData.FeaturesLayout,
            Data.FloatFeatures,
            localExecutor
        );
        checkSum = CalcFeatureValuesCheckSum<EFeatureType::Categorical>(
            checkSum,
            *CommonData.FeaturesLayout,
            Data.CatFeatures,
            localExecutor
        );

        Data.CachedFeaturesCheckSum = checkSum;
    }
    return *Data.CachedFeaturesCheckSum;
}

bool NCB::TQuantizedObjectsDataProvider::AllFeaturesDataIsCompressedArrays() const {
    for (const auto& floatFeatureHolder : Data.FloatFeatures) {
        if (floatFeatureHolder && !dynamic_cast<const TQuantizedFloatValuesHolder*>(floatFeatureHolder.Get())) {
            return false;
        }
    }
    for (const auto& catFeatureHolder : Data.CatFeatures) {
        if (catFeatureHolder && !dynamic_cast<const TQuantizedCatValuesHolder*>(catFeatureHolder.Get())) {
            return false;
        }
    }

    return true;
}


static TCompressedArray LoadAsCompressedArray(IBinSaver* binSaver) {
    ui32 objectCount;
    ui32 bitsPerKey;
    binSaver->AddMulti(objectCount, bitsPerKey);

    TVector<ui64> storage;
    IBinSaver::TStoredSize compressedStorageVectorSize;
    LoadMulti(binSaver, &compressedStorageVectorSize);
    storage.yresize(compressedStorageVectorSize);
    LoadArrayData<ui64>(storage, binSaver);

    return TCompressedArray(
        objectCount,
        bitsPerKey,
        TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(storage))
    );
}

enum class ESavedColumnType : ui8 {
    PackedBinary = 0,
    BundlePart = 1,
    Dense = 3
};


template <class T, EFeatureValuesType FeatureValuesType>
static void LoadNonBundledColumnData(
    ui32 flatFeatureIdx,
    const TFeaturesArraySubsetIndexing* newSubsetIndexing,
    IBinSaver* binSaver,
    THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>* column
) {
    *column = MakeHolder<TCompressedValuesHolderImpl<T, FeatureValuesType>>(
        flatFeatureIdx,
        LoadAsCompressedArray(binSaver),
        newSubsetIndexing
    );
}


template <EFeatureType FeatureType, class T, EFeatureValuesType FeatureValuesType>
static void LoadFeatures(
    const TFeaturesLayout& featuresLayout,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    const TMaybe<TPackedBinaryFeaturesData*> packedBinaryFeaturesData,
    const TMaybe<TExclusiveFeatureBundlesData*> exclusiveFeatureBundlesData,
    const TMaybe<TFeatureGroupsData*> featureGroupsData,
    IBinSaver* binSaver,
    TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>* dst
) {
    TVector<TMaybe<TPackedBinaryIndex>>* featureToPackedBinaryIndex;
    if (packedBinaryFeaturesData) {
        featureToPackedBinaryIndex = &(**packedBinaryFeaturesData).FlatFeatureIndexToPackedBinaryIndex;
    } else {
        featureToPackedBinaryIndex = nullptr;
    }

    TVector<TMaybe<TExclusiveBundleIndex>>* featureToBundlePart;
    if (exclusiveFeatureBundlesData) {
        featureToBundlePart = &(**exclusiveFeatureBundlesData).FlatFeatureIndexToBundlePart;
    } else {
        featureToBundlePart = nullptr;
    }

    TVector<TMaybe<TFeaturesGroupIndex>>* featureToGroupPart;
    if (featureGroupsData) {
        featureToGroupPart = &(**featureGroupsData).FlatFeatureIndexToGroupPart;
    } else {
        featureToGroupPart = nullptr;
    }

    dst->clear();
    dst->resize(featuresLayout.GetFeatureCount(FeatureType));

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            ui32 flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(*featureIdx, FeatureType);

            ui32 id;
            ESavedColumnType savedColumnType;
            LoadMulti(binSaver, &id, (ui8*)&savedColumnType);

            CB_ENSURE_INTERNAL(
                flatFeatureIdx == id,
                "deserialized feature id (" << id << ") is not equal to expected flatFeatureIdx ("
                << flatFeatureIdx << ")"
            );


            if (featureToPackedBinaryIndex && (*featureToPackedBinaryIndex)[flatFeatureIdx]) {
                TPackedBinaryIndex packedBinaryIndex = *((*featureToPackedBinaryIndex)[flatFeatureIdx]);

                ui8 bitIdx = 0;
                binSaver->Add(0, &bitIdx);

                CB_ENSURE_INTERNAL(
                    packedBinaryIndex.BitIdx == bitIdx,
                    "deserialized bitIdx (" << bitIdx << ") is not equal to expected packedBinaryIndex.BitIdx ("
                    << packedBinaryIndex.BitIdx << ")"
                );

                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<T, FeatureValuesType>>(
                    flatFeatureIdx,
                    (**packedBinaryFeaturesData).SrcData[packedBinaryIndex.PackIdx].Get(),
                    packedBinaryIndex.BitIdx
                );
            } else if (featureToBundlePart && (*featureToBundlePart)[flatFeatureIdx]) {
                TExclusiveBundleIndex exclusiveBundleIndex = *((*featureToBundlePart)[flatFeatureIdx]);

                const auto& metaData =
                    (**exclusiveFeatureBundlesData).MetaData[exclusiveBundleIndex.BundleIdx];

                TBoundsInBundle boundsInBundle;
                binSaver->Add(0, &boundsInBundle);

                CB_ENSURE_INTERNAL(
                    metaData.Parts[exclusiveBundleIndex.InBundleIdx].Bounds == boundsInBundle,
                    "deserialized " << LabeledOutput(boundsInBundle) << " are not equal to expected "
                    LabeledOutput(metaData.Parts[exclusiveBundleIndex.InBundleIdx].Bounds)
                );

                (*dst)[*featureIdx] = MakeHolder<TBundlePartValuesHolderImpl<T, FeatureValuesType>>(
                    flatFeatureIdx,
                    (**exclusiveFeatureBundlesData).SrcData[exclusiveBundleIndex.BundleIdx].Get(),
                    metaData.Parts[exclusiveBundleIndex.InBundleIdx].Bounds
                );
            } else if (featureToGroupPart && (*featureToGroupPart)[flatFeatureIdx]) {
                const TFeaturesGroupIndex featuresGroupIndex = *((*featureToGroupPart)[flatFeatureIdx]);

                (*dst)[*featureIdx] = MakeHolder<TFeaturesGroupPartValuesHolderImpl<T, FeatureValuesType>>(
                    flatFeatureIdx,
                    (**featureGroupsData).SrcData[featuresGroupIndex.GroupIdx].Get(),
                    featuresGroupIndex.InGroupIdx
                );
            } else {
                LoadNonBundledColumnData(
                    flatFeatureIdx,
                    subsetIndexing,
                    binSaver,
                    &((*dst)[*featureIdx])
                );
            }
        }
    );
}

void NCB::TQuantizedObjectsData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    const NCB::TFeaturesLayout& featuresLayout,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    IBinSaver* binSaver
) {
    QuantizedFeaturesInfo = quantizedFeaturesInfo;
    LoadFeatures<EFeatureType::Float>(
        featuresLayout,
        subsetIndexing,
        /*packedBinaryFeaturesData*/ Nothing(),
        /*exclusiveFeatureBundlesData*/ Nothing(),
        /*featureGroupsData*/ Nothing(),
        binSaver,
        &FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        featuresLayout,
        subsetIndexing,
        /*packedBinaryFeaturesData*/ Nothing(),
        /*exclusiveFeatureBundlesData*/ Nothing(),
        /*featureGroupsData*/ Nothing(),
        binSaver,
        &CatFeatures
    );
    LoadMulti(binSaver, &CachedFeaturesCheckSum);
}


template <class T>
static void SaveAsCompressedArray(TConstArrayRef<T> values, IBinSaver* binSaver) {
    constexpr ui8 paddingBuffer[sizeof(ui64)-1] = {0};

    const ui32 objectCount = values.size();
    const ui32 bytesPerKey = sizeof(T);
    const ui32 bitsPerKey = bytesPerKey * CHAR_BIT;

    TIndexHelper<ui64> indexHelper(bitsPerKey);

    // save values to be deserialiable as a TVector<ui64>

    const IBinSaver::TStoredSize compressedStorageVectorSize = indexHelper.CompressedSize(objectCount);
    SaveMulti(binSaver, objectCount, bitsPerKey, compressedStorageVectorSize);

    // pad to ui64-alignment to make it deserializable as CompressedArray storage
    const size_t paddingSize =
        size_t(compressedStorageVectorSize)*sizeof(ui64) - size_t(bytesPerKey)*objectCount;

    SaveArrayData(values, binSaver);
    if (paddingSize) {
        SaveArrayData(TConstArrayRef<ui8>(paddingBuffer, paddingSize), binSaver);
    }
}


template <class T, EFeatureValuesType FeatureValuesType>
static void SaveColumnData(
    const TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    NPar::TLocalExecutor* localExecutor,
    IBinSaver* binSaver
) {
    if (auto* packedBinaryValues
            = dynamic_cast<const TPackedBinaryValuesHolderImpl<T, FeatureValuesType>*>(&column))
    {
        SaveMulti(binSaver, ESavedColumnType::PackedBinary, packedBinaryValues->GetBitIdx());
    } else if (auto* bundlePartValues
                   = dynamic_cast<const TBundlePartValuesHolderImpl<T, FeatureValuesType>*>(&column))
    {
        SaveMulti(binSaver, ESavedColumnType::BundlePart, bundlePartValues->GetBoundsInBundle());
    } else {
        /* TODO(akhropov): specialize for TCompressedValuesHolderImpl
         * useful if in fact bitsPerKey < sizeof(T) * CHAR_BIT
         */
        SaveMulti(binSaver, ESavedColumnType::Dense);
        SaveAsCompressedArray<T>(*(column.ExtractValues(localExecutor)), binSaver);
    }
}


template <EFeatureType FeatureType, class T, EFeatureValuesType FeatureValuesType>
static void SaveFeatures(
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>& src,
    NPar::TLocalExecutor* localExecutor,
    IBinSaver* binSaver
) {
    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const TTypedFeatureValuesHolder<T, FeatureValuesType>* column = src[*featureIdx].Get();

            SaveMulti(binSaver, column->GetId());
            SaveColumnData(*column, localExecutor, binSaver);
        }
    );
}

void NCB::TQuantizedObjectsData::SaveNonSharedPart(
    const TFeaturesLayout& featuresLayout,
    IBinSaver* binSaver
) const {
    NPar::TLocalExecutor localExecutor;

    SaveFeatures<EFeatureType::Float>(
        featuresLayout,
        FloatFeatures,
        &localExecutor,
        binSaver
    );
    SaveFeatures<EFeatureType::Categorical>(
        featuresLayout,
        CatFeatures,
        &localExecutor,
        binSaver
    );
    SaveMulti(binSaver, CachedFeaturesCheckSum);
}


void NCB::DbgDumpQuantizedFeatures(
    const NCB::TQuantizedObjectsDataProvider& quantizedObjectsDataProvider,
    IOutputStream* out
) {
    const auto& featuresLayout = *quantizedObjectsDataProvider.GetFeaturesLayout();

    NPar::TLocalExecutor localExecutor;

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx floatFeatureIdx) {
            const auto values = (*quantizedObjectsDataProvider.GetFloatFeature(*floatFeatureIdx))
                ->ExtractValues(&localExecutor);

            for (auto objectIdx : xrange((*values).size())) {
                (*out) << "(floatFeature=" << *floatFeatureIdx << ',' << LabeledOutput(objectIdx)
                    << ").bin=" << ui32(values[objectIdx]) << Endl;
            }
        }
    );

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            const auto values = (*quantizedObjectsDataProvider.GetCatFeature(*catFeatureIdx))
                ->ExtractValues(&localExecutor);

            for (auto objectIdx : xrange((*values).size())) {
                (*out) << "(catFeature=" << *catFeatureIdx << ',' << LabeledOutput(objectIdx)
                    << ").bin=" << ui32(values[objectIdx]) << Endl;
            }
        }
    );
}


NCB::TExclusiveFeatureBundlesData::TExclusiveFeatureBundlesData(
    const NCB::TFeaturesLayout& featuresLayout,
    TVector<NCB::TExclusiveFeaturesBundle>&& metaData
)
    : MetaData(std::move(metaData))
{
    FlatFeatureIndexToBundlePart.resize(featuresLayout.GetExternalFeatureCount());

    for (ui32 bundleIdx : xrange(SafeIntegerCast<ui32>(MetaData.size()))) {
        const auto& bundle = MetaData[bundleIdx];
        for (ui32 inBundleIdx : xrange(SafeIntegerCast<ui32>(bundle.Parts.size()))) {
            TExclusiveBundleIndex exclusiveBundleIndex(bundleIdx, inBundleIdx);
            const auto& bundlePart = bundle.Parts[inBundleIdx];
            const ui32 flatFeatureIdx
                = featuresLayout.GetExternalFeatureIdx(bundlePart.FeatureIdx, bundlePart.FeatureType);
            FlatFeatureIndexToBundlePart[flatFeatureIdx] = exclusiveBundleIndex;
        }
    }
}


void NCB::TExclusiveFeatureBundlesData::GetSubset(
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    TExclusiveFeatureBundlesData* subsetData
) const {
    subsetData->FlatFeatureIndexToBundlePart = FlatFeatureIndexToBundlePart;
    subsetData->MetaData = MetaData;

    CreateSubsetFeatures(MakeConstArrayRef(SrcData), subsetIndexing, &(subsetData->SrcData));
}


void NCB::TExclusiveFeatureBundlesData::Save(
    NPar::TLocalExecutor* localExecutor,
    IBinSaver* binSaver
) const {
    Y_ASSERT(!binSaver->IsReading());
    Y_ASSERT(MetaData.size() == SrcData.size());

    SaveMulti(
        binSaver,
        FlatFeatureIndexToBundlePart,
        MetaData
    );

    for (const auto& srcDataElement : SrcData) {
        SaveColumnData(*srcDataElement, localExecutor, binSaver);
    }
}


void NCB::TExclusiveFeatureBundlesData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    IBinSaver* binSaver
) {
    Y_ASSERT(binSaver->IsReading());

    LoadMulti(
        binSaver,
        &FlatFeatureIndexToBundlePart,
        &MetaData
    );

    SrcData.resize(MetaData.size());
    for (auto& srcDataElement : SrcData) {
        ESavedColumnType savedColumnType;
        LoadMulti(binSaver, (ui8*)&savedColumnType);
        LoadNonBundledColumnData(
            /*flatFeatureIdx*/ 0, // not actually used later
            subsetIndexing,
            binSaver,
            &srcDataElement
        );
    }
}


NCB::TFeatureGroupsData::TFeatureGroupsData(
    const NCB::TFeaturesLayout& featuresLayout,
    TVector<NCB::TFeaturesGroup>&& metaData
)
    : MetaData(std::move(metaData))
{
    FlatFeatureIndexToGroupPart.resize(featuresLayout.GetExternalFeatureCount());

    for (ui32 groupIdx : xrange(SafeIntegerCast<ui32>(MetaData.size()))) {
        const auto& group = MetaData[groupIdx];
        for (ui32 inGroupIdx : xrange(SafeIntegerCast<ui32>(group.Parts.size()))) {
            TFeaturesGroupIndex featuresGroupIndex{groupIdx, inGroupIdx};
            const auto& groupPart = group.Parts[inGroupIdx];
            const ui32 flatFeatureIdx
                = featuresLayout.GetExternalFeatureIdx(groupPart.FeatureIdx, groupPart.FeatureType);
            FlatFeatureIndexToGroupPart[flatFeatureIdx] = featuresGroupIndex;
        }
    }
}


void NCB::TFeatureGroupsData::GetSubset(
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    TFeatureGroupsData* subsetData
) const {
    subsetData->FlatFeatureIndexToGroupPart = FlatFeatureIndexToGroupPart;
    subsetData->MetaData = MetaData;

    CreateSubsetFeatures(MakeConstArrayRef(SrcData), subsetIndexing, &(subsetData->SrcData));
}


void NCB::TFeatureGroupsData::Save(
    NPar::TLocalExecutor* localExecutor,
    IBinSaver* binSaver
) const {
    Y_ASSERT(!binSaver->IsReading());
    Y_ASSERT(MetaData.size() == SrcData.size());

    SaveMulti(
        binSaver,
        FlatFeatureIndexToGroupPart,
        MetaData
    );

    for (const auto& srcDataElement : SrcData) {
        SaveColumnData(*srcDataElement, localExecutor, binSaver);
    }
}


void NCB::TFeatureGroupsData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    IBinSaver* binSaver
) {
    Y_ASSERT(binSaver->IsReading());

    LoadMulti(
        binSaver,
        &FlatFeatureIndexToGroupPart,
        &MetaData
    );

    SrcData.resize(MetaData.size());
    for (auto& srcDataElement : SrcData) {
        ESavedColumnType savedColumnType;
        LoadMulti(binSaver, (ui8*)&savedColumnType);
        LoadNonBundledColumnData(
            /*flatFeatureIdx*/ 0, // not actually used later
            subsetIndexing,
            binSaver,
            &srcDataElement
        );
    }
}


NCB::TPackedBinaryFeaturesData::TPackedBinaryFeaturesData(
    const TFeaturesLayout& featuresLayout,
    const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    const TExclusiveFeatureBundlesData& exclusiveFeatureBundlesData,
    bool dontPack
) {
    FlatFeatureIndexToPackedBinaryIndex.resize(featuresLayout.GetExternalFeatureCount());

    if (dontPack) {
        return;
    }

    auto addIfNotBundled = [&] (EFeatureType featureType, ui32 perTypeFeatureIdx) {
        const ui32 flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, featureType);
        if (!exclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx]) {
            FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx]
                = TPackedBinaryIndex::FromLinearIdx(SafeIntegerCast<ui32>(PackedBinaryToSrcIndex.size()));
            PackedBinaryToSrcIndex.emplace_back(featureType, perTypeFeatureIdx);
        }
    };

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx floatFeatureIdx) {
            if (quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size() == 1) {
                addIfNotBundled(EFeatureType::Float, *floatFeatureIdx);
            }
        }
    );
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            if (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnAll == 2) {
                addIfNotBundled(EFeatureType::Categorical, *catFeatureIdx);
            }
        }
    );
    SrcData.resize(CeilDiv(PackedBinaryToSrcIndex.size(), sizeof(TBinaryFeaturesPack) * CHAR_BIT));
}

void NCB::TPackedBinaryFeaturesData::GetSubset(
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    TPackedBinaryFeaturesData* subsetData
) const {
    subsetData->FlatFeatureIndexToPackedBinaryIndex = FlatFeatureIndexToPackedBinaryIndex;
    subsetData->PackedBinaryToSrcIndex = PackedBinaryToSrcIndex;

    CreateSubsetFeatures(MakeConstArrayRef(SrcData), subsetIndexing, &(subsetData->SrcData));
}

void NCB::TPackedBinaryFeaturesData::Save(NPar::TLocalExecutor* localExecutor, IBinSaver* binSaver) const {
    Y_ASSERT(!binSaver->IsReading());

    SaveMulti(
        binSaver,
        FlatFeatureIndexToPackedBinaryIndex,
        PackedBinaryToSrcIndex
    );

    auto srcDataSize = SafeIntegerCast<IBinSaver::TStoredSize>(SrcData.size());
    binSaver->Add(0, &srcDataSize);

    for (const auto& srcDataElement : SrcData) {
        SaveColumnData(*srcDataElement, localExecutor, binSaver);
    }
}

void NCB::TPackedBinaryFeaturesData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    IBinSaver* binSaver
) {
    Y_ASSERT(binSaver->IsReading());

    LoadMulti(
        binSaver,
        &FlatFeatureIndexToPackedBinaryIndex,
        &PackedBinaryToSrcIndex
    );

    IBinSaver::TStoredSize srcDataSize = 0;
    binSaver->Add(0, &srcDataSize);
    SrcData.resize(srcDataSize);
    for (auto& srcDataElement : SrcData) {
        ESavedColumnType savedColumnType;
        LoadMulti(binSaver, (ui8*)&savedColumnType);
        LoadNonBundledColumnData(
            /*flatFeatureIdx*/ 0, // not actually used later
            subsetIndexing,
            binSaver,
            &srcDataElement
        );
    }
}


TString NCB::DbgDumpMetaData(const NCB::TPackedBinaryFeaturesData& packedBinaryFeaturesData) {
    TStringBuilder sb;
    sb << "FlatFeatureIndexToPackedBinaryIndex="
       << NCB::DbgDumpWithIndices(packedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex, true)
       << "PackedBinaryToSrcIndex=[";

    const auto& packedBinaryToSrcIndex = packedBinaryFeaturesData.PackedBinaryToSrcIndex;
    if (!packedBinaryToSrcIndex.empty()) {
        sb << Endl;
        for (auto linearIdx : xrange(packedBinaryToSrcIndex.size())) {
            auto packedBinaryIndex = NCB::TPackedBinaryIndex::FromLinearIdx(linearIdx);
            const auto& srcIndex = packedBinaryToSrcIndex[linearIdx];
            sb << "LinearIdx=" << linearIdx << "," << DbgDump(packedBinaryIndex) << " : FeatureType="
               << srcIndex.first << ",FeatureIdx=" << srcIndex.second << Endl;
        }
        sb << Endl;
    }
    sb << "]\n";

    return sb;
}


void NCB::TQuantizedForCPUObjectsData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    const TFeaturesLayout& featuresLayout,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    IBinSaver* binSaver
) {
    PackedBinaryFeaturesData.Load(subsetIndexing, binSaver);
    ExclusiveFeatureBundlesData.Load(subsetIndexing, binSaver);
    FeaturesGroupsData.Load(subsetIndexing, binSaver);
    Data.QuantizedFeaturesInfo = quantizedFeaturesInfo;
    LoadFeatures<EFeatureType::Float>(
        featuresLayout,
        subsetIndexing,
        &PackedBinaryFeaturesData,
        &ExclusiveFeatureBundlesData,
        &FeaturesGroupsData,
        binSaver,
        &Data.FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        featuresLayout,
        subsetIndexing,
        &PackedBinaryFeaturesData,
        &ExclusiveFeatureBundlesData,
        &FeaturesGroupsData,
        binSaver,
        &Data.CatFeatures
    );
    LoadMulti(binSaver, &Data.CachedFeaturesCheckSum);
}


NCB::TQuantizedForCPUObjectsDataProvider::TQuantizedForCPUObjectsDataProvider(
    TMaybe<TObjectsGroupingPtr> objectsGrouping,
    TCommonObjectsData&& commonData,
    TQuantizedForCPUObjectsData&& data,
    bool skipCheck,
    TMaybe<NPar::TLocalExecutor*> localExecutor
)
    : TQuantizedObjectsDataProvider(
        std::move(objectsGrouping),
        std::move(commonData),
        std::move(data.Data),
        skipCheck,
        localExecutor
      )
{
    if (!skipCheck) {
        Check(data.PackedBinaryFeaturesData, data.ExclusiveFeatureBundlesData, data.FeaturesGroupsData);
    }
    PackedBinaryFeaturesData = std::move(data.PackedBinaryFeaturesData);
    ExclusiveFeatureBundlesData = std::move(data.ExclusiveFeatureBundlesData);
    FeaturesGroupsData = std::move(data.FeaturesGroupsData);

    CatFeatureUniqueValuesCounts.yresize(Data.CatFeatures.size());
    for (auto catFeatureIdx : xrange(Data.CatFeatures.size())) {
        CatFeatureUniqueValuesCounts[catFeatureIdx] =
            Data.QuantizedFeaturesInfo->GetUniqueValuesCounts(TCatFeatureIdx(catFeatureIdx));
    }
}


// if data is not packed or bundled or grouped - return empty holder
template <class T, EFeatureValuesType FeatureValuesType>
static THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>> GetAggregatedColumn(
    const TQuantizedForCPUObjectsData& data,
    ui32 flatFeatureIdx
) {
    const auto& bundlesData = data.ExclusiveFeatureBundlesData;
    if (auto bundleIndex = bundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx]) {
        const auto& bundleMetaData = bundlesData.MetaData[bundleIndex->BundleIdx];

        return MakeHolder<TBundlePartValuesHolderImpl<T, FeatureValuesType>>(
            flatFeatureIdx,
            bundlesData.SrcData[bundleIndex->BundleIdx].Get(),
            bundleMetaData.Parts[bundleIndex->InBundleIdx].Bounds
        );
    }

    const auto& packedBinaryData = data.PackedBinaryFeaturesData;
    if (auto packedBinaryIndex = packedBinaryData.FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx]) {
        return MakeHolder<TPackedBinaryValuesHolderImpl<T, FeatureValuesType>>(
            flatFeatureIdx,
            packedBinaryData.SrcData[packedBinaryIndex->PackIdx].Get(),
            packedBinaryIndex->BitIdx
        );
    }

    const auto& groupsData = data.FeaturesGroupsData;
    if (auto groupIndex = groupsData.FlatFeatureIndexToGroupPart[flatFeatureIdx]) {
        return MakeHolder<TFeaturesGroupPartValuesHolderImpl<T, FeatureValuesType>>(
            flatFeatureIdx,
            groupsData.SrcData[groupIndex->GroupIdx].Get(),
            groupIndex->InGroupIdx
        );
    }

    return nullptr;
}


NCB::TObjectsDataProviderPtr NCB::TQuantizedForCPUObjectsDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) const {
    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );
    TQuantizedForCPUObjectsData subsetData;

    auto getSubsetForDataPart = [&] (const auto& srcData, auto* dstSubsetData) {
        srcData.GetSubset(
            subsetCommonData.SubsetIndexing.Get(),
            dstSubsetData
        );
    };

    getSubsetForDataPart(PackedBinaryFeaturesData, &subsetData.PackedBinaryFeaturesData);
    getSubsetForDataPart(ExclusiveFeatureBundlesData, &subsetData.ExclusiveFeatureBundlesData);
    getSubsetForDataPart(FeaturesGroupsData, &subsetData.FeaturesGroupsData);

    CreateSubsetFeatures(
        MakeConstArrayRef(Data.FloatFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        [&] (ui32 flatFeatureIdx) {
            return GetAggregatedColumn<ui8, EFeatureValuesType::QuantizedFloat>(
                subsetData,
                flatFeatureIdx
            );
        },
        &subsetData.Data.FloatFeatures
    );

    CreateSubsetFeatures(
        MakeConstArrayRef(Data.CatFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        [&] (ui32 flatFeatureIdx) {
            return GetAggregatedColumn<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                subsetData,
                flatFeatureIdx
            );
        },
        &subsetData.Data.CatFeatures
    );

    CreateSubsetFeatures(
        MakeConstArrayRef(Data.TextFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.Data.TextFeatures
    );

    subsetData.Data.QuantizedFeaturesInfo = Data.QuantizedFeaturesInfo;

    return MakeIntrusive<TQuantizedForCPUObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


NCB::TObjectsDataProviderPtr NCB::TQuantizedForCPUObjectsDataProvider::GetFeaturesSubset(
    const TVector<ui32>& ignoredFeatures,
    NPar::TLocalExecutor* localExecutor
) const {
    const auto& objectsGroupingSubset = ::GetGroupingSubsetFromObjectsSubset(
        ObjectsGrouping,
        TArraySubsetIndexing(TFullSubset<ui32>(GetObjectCount())),
        EObjectsOrder::Ordered);

    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );
    subsetCommonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*subsetCommonData.FeaturesLayout);
    subsetCommonData.FeaturesLayout->IgnoreExternalFeatures(ignoredFeatures);

    TQuantizedForCPUObjectsData subsetData;

    auto getSubsetForDataPart = [&] (const auto& srcData, auto* dstSubsetData) {
        srcData.GetSubset(
            subsetCommonData.SubsetIndexing.Get(),
            dstSubsetData
        );
    };

    getSubsetForDataPart(PackedBinaryFeaturesData, &subsetData.PackedBinaryFeaturesData);
    getSubsetForDataPart(ExclusiveFeatureBundlesData, &subsetData.ExclusiveFeatureBundlesData);
    getSubsetForDataPart(FeaturesGroupsData, &subsetData.FeaturesGroupsData);

    CreateSubsetFeatures(
        MakeConstArrayRef(Data.FloatFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        [&] (ui32 flatFeatureIdx) {
            return GetAggregatedColumn<ui8, EFeatureValuesType::QuantizedFloat>(
                subsetData,
                flatFeatureIdx
            );
        },
        &subsetData.Data.FloatFeatures
    );

    CreateSubsetFeatures(
        MakeConstArrayRef(Data.CatFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        [&] (ui32 flatFeatureIdx) {
            return GetAggregatedColumn<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                subsetData,
                flatFeatureIdx
            );
        },
        &subsetData.Data.CatFeatures
    );

    CreateSubsetFeatures(
        MakeConstArrayRef(Data.TextFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.Data.TextFeatures
    );

    subsetData.Data.QuantizedFeaturesInfo = Data.QuantizedFeaturesInfo;

    return MakeIntrusive<TQuantizedForCPUObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


template <class T, EFeatureValuesType FeatureValuesType>
static void MakeConsecutiveColumnData(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    const TTypedFeatureValuesHolder<T, FeatureValuesType>& src,
    NPar::TLocalExecutor* localExecutor,
    THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>* dst
) {
    using TCompressedValuesHolder = TCompressedValuesHolderImpl<T, FeatureValuesType>;

    if (const auto* srcCompressedValuesHolder = dynamic_cast<const TCompressedValuesHolder*>(&src)) {
        const ui32 objectCount = srcCompressedValuesHolder->GetSize();
        const ui32 bitsPerKey = srcCompressedValuesHolder->GetBitsPerKey();
        TIndexHelper<ui64> indexHelper(bitsPerKey);
        const ui32 dstStorageSize = indexHelper.CompressedSize(objectCount);

        TVector<ui64> storage;
        storage.yresize(dstStorageSize);

        if (bitsPerKey == 8) {
            auto dstBuffer = (ui8*)(storage.data());

            srcCompressedValuesHolder->template GetArrayData<ui8>().ParallelForEach(
                [dstBuffer](ui32 idx, ui8 value) {
                    dstBuffer[idx] = value;
                },
                localExecutor
            );
        } else if (bitsPerKey == 16) {
            auto dstBuffer = (ui16*)(storage.data());

            srcCompressedValuesHolder->template GetArrayData<ui16>().ParallelForEach(
                [dstBuffer](ui32 idx, ui16 value) {
                    dstBuffer[idx] = value;
                },
                localExecutor
            );
        } else {
            auto dstBuffer = (ui32*)(storage.data());

            srcCompressedValuesHolder->template GetArrayData<ui32>().ParallelForEach(
                [dstBuffer](ui32 idx, ui32 value) {
                    dstBuffer[idx] = value;
                },
                localExecutor
            );
        }

        *dst = MakeHolder<TCompressedValuesHolder>(
            srcCompressedValuesHolder->GetId(),
            TCompressedArray(
                objectCount,
                bitsPerKey,
                TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(storage))
            ),
            newSubsetIndexing
        );
    } else {
        CB_ENSURE_INTERNAL(false, "MakeConsecutiveColumnData: Unsupported column type");
    }
}


template <EFeatureType FeatureType, class T, EFeatureValuesType FeatureValuesType>
static void MakeConsecutiveArrayFeatures(
    const TFeaturesLayout& featuresLayout,
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    const TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>& src,
    const TExclusiveFeatureBundlesData& newExclusiveFeatureBundlesData,
    const TPackedBinaryFeaturesData& newPackedBinaryFeaturesData,
    const TFeatureGroupsData& newFeatureGroupsData,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>* dst
) {
    if (&src != dst) {
        dst->clear();
        dst->resize(featuresLayout.GetFeatureCount(FeatureType));
    }

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const auto* srcColumn = src[*featureIdx].Get();

            if (auto maybeExclusiveFeaturesBundleIndex
                    = newExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[srcColumn->GetId()])
            {
                const auto& bundleMetaData
                    = newExclusiveFeatureBundlesData.MetaData[maybeExclusiveFeaturesBundleIndex->BundleIdx];

                (*dst)[*featureIdx] = MakeHolder<TBundlePartValuesHolderImpl<T, FeatureValuesType>>(
                    srcColumn->GetId(),
                    newExclusiveFeatureBundlesData.SrcData[maybeExclusiveFeaturesBundleIndex->BundleIdx].Get(),
                    bundleMetaData.Parts[maybeExclusiveFeaturesBundleIndex->InBundleIdx].Bounds
                );
            } else if (auto maybePackedBinaryIndex
                           = newPackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[srcColumn->GetId()])
            {
                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<T, FeatureValuesType>>(
                    srcColumn->GetId(),
                    newPackedBinaryFeaturesData.SrcData[maybePackedBinaryIndex->PackIdx].Get(),
                    maybePackedBinaryIndex->BitIdx
                );
            } else if (auto maybeFeaturesGroupIndex
                                = newFeatureGroupsData.FlatFeatureIndexToGroupPart[srcColumn->GetId()])
            {
                (*dst)[*featureIdx] = MakeHolder<TFeaturesGroupPartValuesHolderImpl<T, FeatureValuesType>>(
                    srcColumn->GetId(),
                    newFeatureGroupsData.SrcData[maybeFeaturesGroupIndex->GroupIdx].Get(),
                    maybeFeaturesGroupIndex->InGroupIdx
                );

            } else {
                MakeConsecutiveColumnData(
                    newSubsetIndexing,
                    *srcColumn,
                    localExecutor,
                    &((*dst)[*featureIdx])
                );
            }
        }
    );
}



static void EnsureConsecutiveExclusiveFeatureBundles(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    NCB::TExclusiveFeatureBundlesData* exclusiveFeatureBundlesData
) {
    for (auto& srcDataElement : exclusiveFeatureBundlesData->SrcData) {
        MakeConsecutiveColumnData(
            newSubsetIndexing,
            *srcDataElement,
            localExecutor,
            &srcDataElement
        );
    }
}


static void EnsureConsecutivePackedBinaryFeatures(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<TBinaryPacksHolder>>* packedBinaryFeatures
) {
    for (auto& packedBinaryFeaturesPart : *packedBinaryFeatures) {
        MakeConsecutiveColumnData(
            newSubsetIndexing,
            *packedBinaryFeaturesPart,
            localExecutor,
            &packedBinaryFeaturesPart
        );
    }
}


static void EnsureConsecutiveFeatureGroups(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    NCB::TFeatureGroupsData* featureGroupsData
) {
    for (auto& srcDataElement : featureGroupsData->SrcData) {
        MakeConsecutiveColumnData(
            newSubsetIndexing,
            *srcDataElement,
            localExecutor,
            &srcDataElement
        );
    }
}


void NCB::TQuantizedForCPUObjectsDataProvider::EnsureConsecutiveFeaturesData(
    NPar::TLocalExecutor* localExecutor
) {
    if (GetFeaturesArraySubsetIndexing().IsConsecutive()) {
        return;
    }

    auto newSubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
        TFullSubset<ui32>(GetObjectCount())
    );

    {
        TVector<std::function<void()>> tasks;

        tasks.emplace_back(
            [&] () {
                EnsureConsecutiveExclusiveFeatureBundles(
                    newSubsetIndexing.Get(),
                    localExecutor,
                    &ExclusiveFeatureBundlesData
                );
            }
        );

        tasks.emplace_back(
            [&] () {
                EnsureConsecutivePackedBinaryFeatures(
                    newSubsetIndexing.Get(),
                    localExecutor,
                    &PackedBinaryFeaturesData.SrcData
                );
            }
        );

        tasks.emplace_back(
            [&] () {
                EnsureConsecutiveFeatureGroups(
                    newSubsetIndexing.Get(),
                    localExecutor,
                    &FeaturesGroupsData
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);
    }

    {
        TVector<std::function<void()>> tasks;

        tasks.emplace_back(
            [&] () {
                MakeConsecutiveArrayFeatures<EFeatureType::Float>(
                    *GetFeaturesLayout(),
                    newSubsetIndexing.Get(),
                    Data.FloatFeatures,
                    ExclusiveFeatureBundlesData,
                    PackedBinaryFeaturesData,
                    FeaturesGroupsData,
                    localExecutor,
                    &Data.FloatFeatures
                );
            }
        );
        tasks.emplace_back(
            [&] () {
                MakeConsecutiveArrayFeatures<EFeatureType::Categorical>(
                    *GetFeaturesLayout(),
                    newSubsetIndexing.Get(),
                    Data.CatFeatures,
                    ExclusiveFeatureBundlesData,
                    PackedBinaryFeaturesData,
                    FeaturesGroupsData,
                    localExecutor,
                    &Data.CatFeatures
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);
    }

    CommonData.SubsetIndexing = std::move(newSubsetIndexing);
}


template <class T, EFeatureValuesType FeatureValuesType>
static void CheckFeaturesByType(
    EFeatureType featureType,
    // not TConstArrayRef to allow template parameter deduction
    const TVector<THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>>& data,
    const TExclusiveFeatureBundlesData& exclusiveFeatureBundlesData,
    const TPackedBinaryFeaturesData& packedBinaryFeaturesData,
    const TFeatureGroupsData& featuresGroupsData,
    const TStringBuf featureTypeName
) {
    const auto& packedBinaryToSrcIndex = packedBinaryFeaturesData.PackedBinaryToSrcIndex;
    const auto& bundlesMetaData = exclusiveFeatureBundlesData.MetaData;
    const auto& featuresGroupsMetaData = featuresGroupsData.MetaData;

    for (auto featureIdx : xrange(data.size())) {
        auto* dataPtr = data[featureIdx].Get();
        if (!dataPtr) {
            continue;
        }

        auto maybePackedBinaryIndex
            = packedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[dataPtr->GetId()];
        auto maybeBundlePart
            = exclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[dataPtr->GetId()];
        auto maybeGroupPart
            = featuresGroupsData.FlatFeatureIndexToGroupPart[dataPtr->GetId()];

        CB_ENSURE_INTERNAL(
            maybePackedBinaryIndex.Defined() + maybeBundlePart.Defined() + maybeGroupPart.Defined() <= 1,
            "Data." << featureType << "Features[" << featureIdx
            << "] was mis-included into more than one aggregated column"
        );

        if (maybePackedBinaryIndex) {
            auto requiredTypePtr
                = dynamic_cast<TPackedBinaryValuesHolderImpl<T, FeatureValuesType>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type "
                "TPackedBinaryValuesHolderImpl"
            );

            auto linearPackedBinaryFeatureIdx = maybePackedBinaryIndex->GetLinearIdx();
            CB_ENSURE_INTERNAL(
                linearPackedBinaryFeatureIdx < packedBinaryToSrcIndex.size(),
                "linearPackedBinaryFeatureIdx (" << linearPackedBinaryFeatureIdx << ") is greater than "
                "packedBinaryToSrcIndex.size (" << packedBinaryToSrcIndex.size() << ')'
            );
            auto srcFeature = packedBinaryToSrcIndex[linearPackedBinaryFeatureIdx];
            CB_ENSURE_INTERNAL(
                srcFeature.first == featureType,
                "packedBinaryToSrcIndex[" << linearPackedBinaryFeatureIdx << "] type is not "
                << featureType
            );
            CB_ENSURE_INTERNAL(
                srcFeature.second == featureIdx,
                "packedBinaryToSrcIndex[" << linearPackedBinaryFeatureIdx << "] feature index is not "
                << featureIdx
            );
        } else if (maybeBundlePart) {
            auto requiredTypePtr = dynamic_cast<TBundlePartValuesHolderImpl<T, FeatureValuesType>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "BundlePartValuesHolder"
            );
            CB_ENSURE_INTERNAL(
                (size_t)maybeBundlePart->BundleIdx < bundlesMetaData.size(),
                "featureType=" << featureType << ", featureIdx=" << featureIdx << ": bundleIdx ("
                << maybeBundlePart->BundleIdx << ") is greater than bundles size ("
                << bundlesMetaData.size() << ')'
            );
            const auto& bundleMetaData = bundlesMetaData[maybeBundlePart->BundleIdx];
            CB_ENSURE_INTERNAL(
                bundleMetaData.SizeInBytes == requiredTypePtr->GetBundleSizeInBytes(),
                "Bundled feature: SizeInBytes mismatch between metadata and column data"
            );
            const auto& bundlePart = bundleMetaData.Parts[maybeBundlePart->InBundleIdx];
            CB_ENSURE_INTERNAL(
                (bundlePart.FeatureType == featureType) && (bundlePart.FeatureIdx == featureIdx),
                "Bundled feature: Feature type,index mismatch between metadata and column data"
            );
            CB_ENSURE_INTERNAL(
                bundlePart.Bounds == requiredTypePtr->GetBoundsInBundle(),
                "Bundled feature: Bounds mismatch between metadata and column data"
            );
        } else if (maybeGroupPart) {
            const auto requiredTypePtr = dynamic_cast<TFeaturesGroupPartValuesHolderImpl<T, FeatureValuesType>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                        << featureTypeName << "FeaturesGroupPartValuesHolder"
            );
            CB_ENSURE_INTERNAL(
                (size_t)maybeGroupPart->GroupIdx < featuresGroupsMetaData.size(),
                "featureType=" << featureType << ", featureIdx=" << featureIdx << ": groupIdx ("
                               << maybeGroupPart->GroupIdx << ") is greater than groups size ("
                               << featuresGroupsMetaData.size() << ')'
            );
            const auto& groupMetaData = featuresGroupsMetaData[maybeGroupPart->GroupIdx];
            const auto& groupPart = groupMetaData.Parts[maybeGroupPart->InGroupIdx];
            CB_ENSURE_INTERNAL(
                (groupPart.FeatureType == featureType) && (groupPart.FeatureIdx == featureIdx),
                "Grouped feature: Feature type,index mismatch between metadata and column data"
            );
        } else {
            auto requiredTypePtr = dynamic_cast<TCompressedValuesHolderImpl<T, FeatureValuesType>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "ValuesHolder"
            );
        }
    }
}

bool NCB::TQuantizedForCPUObjectsDataProvider::IsPackingCompatibleWith(
    const NCB::TQuantizedForCPUObjectsDataProvider& rhs
) const {
    return GetQuantizedFeaturesInfo()->IsSupersetOf(*rhs.GetQuantizedFeaturesInfo()) &&
        (PackedBinaryFeaturesData.PackedBinaryToSrcIndex
         == rhs.PackedBinaryFeaturesData.PackedBinaryToSrcIndex) &&
        (ExclusiveFeatureBundlesData.MetaData == rhs.ExclusiveFeatureBundlesData.MetaData) &&
        (FeaturesGroupsData.MetaData == rhs.FeaturesGroupsData.MetaData);
}


void NCB::TQuantizedForCPUObjectsDataProvider::Check(
    const TPackedBinaryFeaturesData& packedBinaryData,
    const TExclusiveFeatureBundlesData& exclusiveFeatureBundlesData,
    const TFeatureGroupsData& featuresGroupsData
) const {
    CheckFeaturesByType(
        EFeatureType::Float,
        Data.FloatFeatures,
        exclusiveFeatureBundlesData,
        packedBinaryData,
        featuresGroupsData,
        "Float"
    );
    CheckFeaturesByType(
        EFeatureType::Categorical,
        Data.CatFeatures,
        exclusiveFeatureBundlesData,
        packedBinaryData,
        featuresGroupsData,
        "Cat"
    );
}


void NCB::TQuantizedForCPUObjectsDataProvider::CheckFeatureIsNotInAggregated(
    EFeatureType featureType,
    const TStringBuf featureTypeName,
    ui32 perTypeFeatureIdx
) const {
    const ui32 flatFeatureIdx = GetFeaturesLayout()->GetExternalFeatureIdx(perTypeFeatureIdx, featureType);
    CB_ENSURE_INTERNAL(
        !PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx],
        "Called TQuantizedForCPUObjectsDataProvider::GetNonPacked" << featureTypeName
        << "Feature for binary packed feature #" << flatFeatureIdx
    );
    CB_ENSURE_INTERNAL(
        !ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx],
        "Called TQuantizedForCPUObjectsDataProvider::GetNonPacked" << featureTypeName
        << "Feature for bundled feature #" << flatFeatureIdx
    );
    CB_ENSURE_INTERNAL(
        !FeaturesGroupsData.FlatFeatureIndexToGroupPart[flatFeatureIdx],
        "Called TQuantizedForCPUObjectsDataProvider::GetNonPacked" << featureTypeName
        << "Feature for grouped feature #" << flatFeatureIdx
    );
}


THashMap<ui32, TString> NCB::MergeCatFeaturesHashToString(const NCB::TObjectsDataProvider& objectsData) {
    THashMap<ui32, TString> result;

    for (auto catFeatureIdx : xrange(objectsData.GetFeaturesLayout()->GetCatFeatureCount())) {
        const auto& perFeatureCatFeaturesHashToString
            = objectsData.GetCatFeaturesHashToString(catFeatureIdx);
        for (const auto& [hashedCatValue, catValueString] : perFeatureCatFeaturesHashToString) {
            // TODO(kirillovs): remove this cast, needed only for MSVC 14.12 compiler bug
            result[(ui32)hashedCatValue] = catValueString;
        }
    }

    return result;
}
