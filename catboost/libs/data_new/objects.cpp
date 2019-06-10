#include "objects.h"
#include "util.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/compare.h>
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
    const TArrayValuesHolder<T, TType>& lhs,
    const TArrayValuesHolder<T, TType>& rhs
) {
    auto lhsArrayData = lhs.GetArrayData();
    auto lhsData = GetSubset<T>(*lhsArrayData.GetSrc(), *lhsArrayData.GetSubsetIndexing());
    return Equal<T>(lhsData, rhs.GetArrayData());
}

template <class IQuantizedValuesHolder>
static bool AreFeaturesValuesEqual(
    const IQuantizedValuesHolder& lhs,
    const IQuantizedValuesHolder& rhs
) {
    return *(lhs.ExtractValues(&NPar::LocalExecutor())) == *(rhs.ExtractValues(&NPar::LocalExecutor()));
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
                catFeaturePtr->GetArrayData().ParallelForEach(
                    [&] (ui32 objectIdx, ui32 hashValue) {
                        CB_ENSURE_INTERNAL(
                            hashToStringMap.contains(hashValue),
                            "catFeature #" << catFeatureIdx << ", object #" << objectIdx << ": value "
                            << Hex(hashValue) << " is missing from CatFeaturesHashToString"
                        );
                    },
                    localExecutor
                );
            }
        },
        0,
        SafeIntegerCast<int>(CatFeatures.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}


template <class T, EFeatureValuesType TType>
static void CreateSubsetFeatures(
    TConstArrayRef<THolder<TArrayValuesHolder<T, TType>>> src,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    TVector<THolder<TArrayValuesHolder<T, TType>>>* dst
) {
    dst->clear();
    dst->reserve(src.size());
    for (const auto& feature : src) {
        auto* srcDataPtr = feature.Get();
        if (srcDataPtr) {
            dst->emplace_back(
                MakeHolder<TArrayValuesHolder<T, TType>>(
                    srcDataPtr->GetId(),
                    *(srcDataPtr->GetArrayData().GetSrc()),
                    subsetIndexing
                )
            );
        } else {
            dst->push_back(nullptr);
        }
    }
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
    CreateSubsetFeatures(
        MakeConstArrayRef(Data.FloatFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.FloatFeatures
    );
    CreateSubsetFeatures(
        MakeConstArrayRef(Data.CatFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.CatFeatures
    );
    CreateSubsetFeatures(
        MakeConstArrayRef(Data.TextFeatures),
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.TextFeatures
    );

    return MakeIntrusive<TRawObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


template <class T, EFeatureValuesType TType>
static void CreateConsecutiveFeaturesData(
    const TVector<THolder<TArrayValuesHolder<T, TType>>>& srcFeatures,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<TArrayValuesHolder<T, TType>>>* dstFeatures
) {
    dstFeatures->resize(srcFeatures.size());
    localExecutor->ExecRangeWithThrow(
        [&] (int featureIdx) {
            if (srcFeatures[featureIdx]) {
                const auto& srcFeatureHolder = *srcFeatures[featureIdx];

                const auto arrayData = srcFeatureHolder.GetArrayData();

                TVector<T> dstStorage = GetSubset<T>(
                    *arrayData.GetSrc(),
                    *arrayData.GetSubsetIndexing(),
                    localExecutor
                );

                (*dstFeatures)[featureIdx] = MakeHolder<TArrayValuesHolder<T, TType>>(
                    srcFeatureHolder.GetId(),
                    TMaybeOwningArrayHolder<const T>::CreateOwning(std::move(dstStorage)),
                    subsetIndexing
                );
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

TVector<float> NCB::TRawObjectsDataProvider::GetFeatureDataOldFormat(ui32 flatFeatureIdx) const {
    TVector<float> result;

    const auto& featuresLayout = *GetFeaturesLayout();
    const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();
    CB_ENSURE(
        flatFeatureIdx < featuresMetaInfo.size(),
        "feature #" << flatFeatureIdx << " is not present in pool"
    );
    const auto& featureMetaInfo = featuresMetaInfo[flatFeatureIdx];
    CB_ENSURE(
        featureMetaInfo.Type != EFeatureType::Text,
        "feature #" << flatFeatureIdx << " has type Text and cannot be converted to float format"
    );
    if (!featureMetaInfo.IsAvailable) {
        return result;
    }

    result.yresize(GetObjectCount());

    if (featureMetaInfo.Type == EFeatureType::Float) {
        const auto& feature = **GetFloatFeature(featuresLayout.GetInternalFeatureIdx(flatFeatureIdx));
        feature.GetArrayData().ForEach([&result](ui32 idx, float value) { result[idx] = value; });
    } else {
        const auto& feature = **GetCatFeature(featuresLayout.GetInternalFeatureIdx(flatFeatureIdx));
        feature.GetArrayData().ForEach(
            [&result](ui32 idx, ui32 value) {
                result[idx] = ConvertCatFeatureHashToFloat(value);
            }
        );
    }

    return result;
}



bool NCB::TQuantizedObjectsData::operator==(const NCB::TQuantizedObjectsData& rhs) const {
    return AreFeaturesValuesEqual(FloatFeatures, rhs.FloatFeatures) &&
        AreFeaturesValuesEqual(CatFeatures, rhs.CatFeatures) &&
        AreFeaturesValuesEqual(TextFeatures, rhs.TextFeatures);
}


void NCB::TQuantizedObjectsData::PrepareForInitialization(
    const TDataMetaInfo& metaInfo,
    const NCatboostOptions::TBinarizationOptions& binarizationOptions,
    const TMap<ui32, NCatboostOptions::TBinarizationOptions>& perFloatFeatureBinarization
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
            perFloatFeatureBinarization,
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


template <class T>
static void CreateSubsetFeatures(
    const TVector<THolder<T>>& src, // not TConstArrayRef to allow template parameter deduction
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    TVector<THolder<T>>* dst
) {
    dst->clear();
    dst->reserve(src.size());
    for (const auto& feature : src) {
        auto* srcDataPtr = feature.Get();
        if (srcDataPtr) {
            dst->emplace_back(srcDataPtr->CloneWithNewSubsetIndexing(subsetIndexing));
        } else {
            dst->push_back(nullptr);
        }
    }
}

TQuantizedObjectsData NCB::TQuantizedObjectsData::GetSubset(
    const TArraySubsetIndexing<ui32>* subsetComposition
) const {
    TQuantizedObjectsData subsetData;
    CreateSubsetFeatures(
        FloatFeatures,
        subsetComposition,
        &subsetData.FloatFeatures
    );
    CreateSubsetFeatures(
        CatFeatures,
        subsetComposition,
        &subsetData.CatFeatures
    );
    CreateSubsetFeatures(
        TextFeatures,
        subsetComposition,
        &subsetData.TextFeatures
    );
    subsetData.QuantizedFeaturesInfo = QuantizedFeaturesInfo;

    return subsetData;
}

NCB::TObjectsDataProviderPtr NCB::TQuantizedObjectsDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) const {
    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );
    TQuantizedObjectsData subsetData = Data.GetSubset(subsetCommonData.SubsetIndexing.Get());

    return MakeIntrusive<TQuantizedObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


template <class IFeatureValuesHolder>
static void CreateConsecutiveFeaturesData(
    const TVector<THolder<IFeatureValuesHolder>>& srcFeatures,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<IFeatureValuesHolder>>* dstFeatures
) {
    using TValueType = typename IFeatureValuesHolder::TValueType;

    dstFeatures->resize(srcFeatures.size());
    localExecutor->ExecRangeWithThrow(
        [&] (int featureIdx) {
            if (srcFeatures[featureIdx]) {
                const auto& srcFeatureHolder = *srcFeatures[featureIdx];

                auto dstStorage = srcFeatureHolder.ExtractValues(localExecutor);

                (*dstFeatures)[featureIdx] = MakeHolder<TCompressedValuesHolderImpl<IFeatureValuesHolder>>(
                    srcFeatureHolder.GetId(),
                    TCompressedArray(
                        srcFeatureHolder.GetSize(),
                        CHAR_BIT * sizeof(TValueType),
                        TMaybeOwningArrayHolder<ui64>::CreateOwningReinterpretCast(dstStorage)
                    ),
                    subsetIndexing
                );
            }
        },
        0,
        SafeIntegerCast<int>(srcFeatures.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
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

template<class TCompressedColumnData>
static ui32 CalcCompressedFeatureChecksum(ui32 checkSum, TCompressedColumnData& columnData) {
    TConstCompressedArraySubset compressedDataSubset = columnData->GetCompressedData();

    auto consecutiveSubsetBegin = compressedDataSubset.GetSubsetIndexing()->GetConsecutiveSubsetBegin();
    const ui32 columnValuesBitWidth = columnData->GetBitsPerKey();
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
        columnData->ForEach([&](ui32 /*idx*/, ui8 element) {
            checkSum = UpdateCheckSum(checkSum, element);
        });
    } else if (columnValuesBitWidth == 16) {
        columnData->ForEach([&](ui32 /*idx*/, ui16 element) {
            checkSum = UpdateCheckSum(checkSum, element);
        });
    } else {
        Y_ASSERT(columnValuesBitWidth == 32);
        columnData->ForEach([&](ui32 /*idx*/, ui32 element) {
            checkSum = UpdateCheckSum(checkSum, element);
        });
    }
    return checkSum;
}

template <EFeatureType FeatureType, class IColumn>
static ui32 CalcFeatureValuesCheckSum(
    ui32 init,
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<IColumn>>& featuresData,
    NPar::TLocalExecutor* localExecutor)
{
    const ui32 emptyColumnDataForCrc = 0;
    TVector<ui32> checkSums(featuresLayout.GetFeatureCount(FeatureType), 0);
    ParallelFor(*localExecutor, 0, featuresLayout.GetFeatureCount(FeatureType), [&] (ui32 perTypeFeatureIdx) {
        if (featuresLayout.GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType).IsAvailable) {
            // TODO(espetrov,akhropov): remove workaround below MLTOOLS-3604
            if (featuresData[perTypeFeatureIdx].Get() == nullptr) {
                return;
            }
            auto compressedValuesFeatureData = dynamic_cast<const TCompressedValuesHolderImpl<IColumn>*>(
                featuresData[perTypeFeatureIdx].Get()
            );
            if (compressedValuesFeatureData) {
                checkSums[perTypeFeatureIdx] = CalcCompressedFeatureChecksum(0, compressedValuesFeatureData);
            } else {
                const auto repackedHolder = featuresData[perTypeFeatureIdx]->ExtractValues(localExecutor);
                checkSums[perTypeFeatureIdx] = UpdateCheckSum(0, *repackedHolder);
            }
        } else {
            checkSums[perTypeFeatureIdx] = UpdateCheckSum(0, emptyColumnDataForCrc);
        }
    });
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


template <EFeatureType FeatureType, class IColumnType>
static void LoadFeatures(
    const TFeaturesLayout& featuresLayout,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    const TMaybe<TPackedBinaryFeaturesData*> packedBinaryFeaturesData,
    const TMaybe<TExclusiveFeatureBundlesData*> exclusiveFeatureBundlesData,
    IBinSaver* binSaver,
    TVector<THolder<IColumnType>>* dst
) {
    const ui32 objectCount = subsetIndexing->Size();

    TVector<TMaybe<TPackedBinaryIndex>>* featureToPackedBinaryIndex;
    if (packedBinaryFeaturesData) {
        featureToPackedBinaryIndex =
          ((FeatureType == EFeatureType::Float) ?
              &(**packedBinaryFeaturesData).FloatFeatureToPackedBinaryIndex :
              &(**packedBinaryFeaturesData).CatFeatureToPackedBinaryIndex );
    } else {
        featureToPackedBinaryIndex = nullptr;
    }

    TVector<TMaybe<TExclusiveBundleIndex>>* featureToBundlePart;
    if (exclusiveFeatureBundlesData) {
        featureToBundlePart =
          ((FeatureType == EFeatureType::Float) ?
              &(**exclusiveFeatureBundlesData).FloatFeatureToBundlePart :
              &(**exclusiveFeatureBundlesData).CatFeatureToBundlePart );
    } else {
        featureToBundlePart = nullptr;
    }

    dst->clear();
    dst->resize(featuresLayout.GetFeatureCount(FeatureType));

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            ui32 flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(*featureIdx, FeatureType);

            ui32 id;
            ui32 size;
            LoadMulti(binSaver, &id, &size);

            CB_ENSURE_INTERNAL(
                flatFeatureIdx == id,
                "deserialized feature id (" << id << ") is not equal to expected flatFeatureIdx ("
                << flatFeatureIdx << ")"
            );
            CheckDataSize(size, objectCount, "column data", false, "object count", true);


            if (featureToPackedBinaryIndex && (*featureToPackedBinaryIndex)[*featureIdx]) {
                TPackedBinaryIndex packedBinaryIndex = *((*featureToPackedBinaryIndex)[*featureIdx]);

                ui8 bitIdx = 0;
                binSaver->Add(0, &bitIdx);

                CB_ENSURE_INTERNAL(
                    packedBinaryIndex.BitIdx == bitIdx,
                    "deserialized bitIdx (" << bitIdx << ") is not equal to expected packedBinaryIndex.BitIdx ("
                    << packedBinaryIndex.BitIdx << ")"
                );

                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<IColumnType>>(
                    flatFeatureIdx,
                    (**packedBinaryFeaturesData).SrcData[packedBinaryIndex.PackIdx],
                    packedBinaryIndex.BitIdx,
                    subsetIndexing
                );
            } else if (featureToBundlePart && (*featureToBundlePart)[*featureIdx]) {
                TExclusiveBundleIndex exclusiveBundleIndex = *((*featureToBundlePart)[*featureIdx]);

                const auto& metaData =
                    (**exclusiveFeatureBundlesData).MetaData[exclusiveBundleIndex.BundleIdx];

                (*dst)[*featureIdx] = MakeHolder<TBundlePartValuesHolderImpl<IColumnType>>(
                    flatFeatureIdx,
                    (**exclusiveFeatureBundlesData).SrcData[exclusiveBundleIndex.BundleIdx],
                    metaData.SizeInBytes,
                    metaData.Parts[exclusiveBundleIndex.InBundleIdx].Bounds,
                    subsetIndexing
                );
            } else {
                ui32 bitsPerKey;
                binSaver->Add(0, &bitsPerKey);

                TVector<ui64> storage;
                LoadMulti(binSaver, &storage);

                (*dst)[*featureIdx] = MakeHolder<TCompressedValuesHolderImpl<IColumnType>>(
                    flatFeatureIdx,
                    TCompressedArray(
                        objectCount,
                        bitsPerKey,
                        TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(storage))
                    ),
                    subsetIndexing
                );
            }
        }
    );
}

void NCB::TQuantizedObjectsData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    IBinSaver* binSaver
) {
    QuantizedFeaturesInfo = quantizedFeaturesInfo;
    LoadFeatures<EFeatureType::Float>(
        *QuantizedFeaturesInfo->GetFeaturesLayout(),
        subsetIndexing,
        /*packedBinaryFeaturesData*/ Nothing(),
        /*exclusiveFeatureBundlesData*/ Nothing(),
        binSaver,
        &FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        *QuantizedFeaturesInfo->GetFeaturesLayout(),
        subsetIndexing,
        /*packedBinaryFeaturesData*/ Nothing(),
        /*exclusiveFeatureBundlesData*/ Nothing(),
        binSaver,
        &CatFeatures
    );
    LoadMulti(binSaver, &CachedFeaturesCheckSum);
}


template <EFeatureType FeatureType, class IColumnType>
static void SaveFeatures(
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<IColumnType>>& src,
    NPar::TLocalExecutor* localExecutor,
    IBinSaver* binSaver
) {
    constexpr ui8 paddingBuffer[sizeof(ui64)-1] = {0};

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const IColumnType* column = src[*featureIdx].Get();

            if (auto* packedBinaryValues
                    = dynamic_cast<const TPackedBinaryValuesHolderImpl<IColumnType>*>(column))
            {
                SaveMulti(binSaver, column->GetId(), column->GetSize(), packedBinaryValues->GetBitIdx());
            } else if (auto* bundlePartValues
                           = dynamic_cast<const TBundlePartValuesHolderImpl<IColumnType>*>(column))
            {
                SaveMulti(binSaver, column->GetId(), column->GetSize());
            } else {
                // TODO(akhropov): replace by repacking (possibly in parts) to compressed array in the future
                const auto values = column->ExtractValues(localExecutor);
                const ui32 objectCount = (*values).size();
                const ui32 bytesPerKey = sizeof(*(*values).data());
                const ui32 bitsPerKey = bytesPerKey * CHAR_BIT;
                SaveMulti(binSaver, column->GetId(), objectCount, bitsPerKey);

                TIndexHelper<ui64> indexHelper(bitsPerKey);

                // save values to be deserialiable as a TVector<ui64>

                const IBinSaver::TStoredSize compressedStorageVectorSize = indexHelper.CompressedSize(objectCount);
                SaveMulti(binSaver, compressedStorageVectorSize);

                // pad to ui64-alignment to make it deserializable as CompressedArray storage
                const size_t paddingSize =
                    size_t(compressedStorageVectorSize)*sizeof(ui64) - size_t(bytesPerKey)*objectCount;

                SaveRawData(*values, binSaver);
                if (paddingSize) {
                    SaveRawData(TConstArrayRef<ui8>(paddingBuffer, paddingSize), binSaver);
                }
            }
        }
    );
}

void NCB::TQuantizedObjectsData::SaveNonSharedPart(IBinSaver* binSaver) const {
    NPar::TLocalExecutor localExecutor;

    SaveFeatures<EFeatureType::Float>(
        *QuantizedFeaturesInfo->GetFeaturesLayout(),
        FloatFeatures,
        &localExecutor,
        binSaver
    );
    SaveFeatures<EFeatureType::Categorical>(
        *QuantizedFeaturesInfo->GetFeaturesLayout(),
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
    const auto& featuresLayout = *quantizedObjectsDataProvider.GetQuantizedFeaturesInfo()->GetFeaturesLayout();

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
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    TVector<NCB::TExclusiveFeaturesBundle>&& metaData
)
    : MetaData(std::move(metaData))
{
    const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();
    FloatFeatureToBundlePart.resize(featuresLayout.GetFloatFeatureCount());
    CatFeatureToBundlePart.resize(featuresLayout.GetCatFeatureCount());

    for (ui32 bundleIdx : xrange(SafeIntegerCast<ui32>(MetaData.size()))) {
        const auto& bundle = MetaData[bundleIdx];
        for (ui32 inBundleIdx : xrange(SafeIntegerCast<ui32>(bundle.Parts.size()))) {
            TExclusiveBundleIndex exclusiveBundleIndex(bundleIdx, inBundleIdx);
            const auto& bundlePart = bundle.Parts[inBundleIdx];
            ((bundlePart.FeatureType == EFeatureType::Float) ?
                FloatFeatureToBundlePart :
                CatFeatureToBundlePart)[bundlePart.FeatureIdx] = exclusiveBundleIndex;
        }
    }
}


void NCB::TExclusiveFeatureBundlesData::Save(
    const TArraySubsetIndexing<ui32>& subsetIndexing,
    IBinSaver* binSaver
) const {
    Y_ASSERT(!binSaver->IsReading());
    Y_ASSERT(MetaData.size() == SrcData.size());

    SaveMulti(
        binSaver,
        FloatFeatureToBundlePart,
        CatFeatureToBundlePart,
        MetaData,
        subsetIndexing.Size()
    );

    for (auto bundleIdx : xrange(SrcData.size())) {
        const auto srcDataElementArray = *(SrcData[bundleIdx]);
        switch (MetaData[bundleIdx].SizeInBytes) {
            case 1:
                subsetIndexing.ForEach(
                    [&](ui32 /*idx*/, ui32 srcIdx) {
                        SaveMulti(binSaver, srcDataElementArray[srcIdx]);
                    }
                );
                break;
            case 2:
                subsetIndexing.ForEach(
                    [&](ui32 /*idx*/, ui32 srcIdx) {
                        SaveMulti(
                            binSaver,
                            srcDataElementArray[2 * srcIdx],
                            srcDataElementArray[2 * srcIdx + 1]
                        );
                    }
                );
                break;
            default:
                ythrow TCatBoostException() << "Wrong features bundle size in bytes : "
                    << MetaData[bundleIdx].SizeInBytes;
        }
    }
}

void NCB::TExclusiveFeatureBundlesData::Load(IBinSaver* binSaver) {
    Y_ASSERT(binSaver->IsReading());

    ui32 objectCount = 0;
    LoadMulti(
        binSaver,
        &FloatFeatureToBundlePart,
        &CatFeatureToBundlePart,
        &MetaData,
        &objectCount
    );

    SrcData.resize(MetaData.size());
    for (auto bundleIdx : xrange(MetaData.size())) {
        TVector<ui8> data;
        const ui32 dataSize = objectCount * MetaData[bundleIdx].SizeInBytes;
        data.yresize(dataSize);
        binSaver->AddRawData(0, data.data(), (i64)dataSize);
        SrcData[bundleIdx] = TMaybeOwningArrayHolder<ui8>::CreateOwning(std::move(data));
    }
}


NCB::TPackedBinaryFeaturesData::TPackedBinaryFeaturesData(
    const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    const TExclusiveFeatureBundlesData& exclusiveFeatureBundlesData,
    bool dontPack
) {
    const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();
    FloatFeatureToPackedBinaryIndex.resize(featuresLayout.GetFloatFeatureCount());
    CatFeatureToPackedBinaryIndex.resize(featuresLayout.GetCatFeatureCount());

    if (dontPack) {
        return;
    }

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx floatFeatureIdx) {
            if (!exclusiveFeatureBundlesData.FloatFeatureToBundlePart[*floatFeatureIdx] &&
                (quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size() == 1))
            {
                FloatFeatureToPackedBinaryIndex[*floatFeatureIdx]
                    = TPackedBinaryIndex::FromLinearIdx(SafeIntegerCast<ui32>(PackedBinaryToSrcIndex.size()));
                PackedBinaryToSrcIndex.emplace_back(EFeatureType::Float, *floatFeatureIdx);
            }
        }
    );
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            if (!exclusiveFeatureBundlesData.CatFeatureToBundlePart[*catFeatureIdx] &&
                (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnAll == 2))
            {
                CatFeatureToPackedBinaryIndex[*catFeatureIdx]
                    = TPackedBinaryIndex::FromLinearIdx(SafeIntegerCast<ui32>(PackedBinaryToSrcIndex.size()));
                PackedBinaryToSrcIndex.emplace_back(EFeatureType::Categorical, *catFeatureIdx);
            }
        }
    );
    SrcData.resize(CeilDiv(PackedBinaryToSrcIndex.size(), sizeof(TBinaryFeaturesPack) * CHAR_BIT));
}

void NCB::TPackedBinaryFeaturesData::Save(
    const TArraySubsetIndexing<ui32>& subsetIndexing,
    IBinSaver* binSaver
) const {
    Y_ASSERT(!binSaver->IsReading());

    SaveMulti(
        binSaver,
        FloatFeatureToPackedBinaryIndex,
        CatFeatureToPackedBinaryIndex,
        PackedBinaryToSrcIndex,
        subsetIndexing.Size()
    );

    auto srcDataSize = SafeIntegerCast<IBinSaver::TStoredSize>(SrcData.size());
    binSaver->Add(0, &srcDataSize);

    for (const auto& srcDataElement : SrcData) {
        const auto srcDataElementArray = *srcDataElement;
        subsetIndexing.ForEach(
            [&](ui32 /*idx*/, ui32 srcIdx) {
                SaveMulti(binSaver, srcDataElementArray[srcIdx]);
            }
        );
    }
}

void NCB::TPackedBinaryFeaturesData::Load(IBinSaver* binSaver) {
    Y_ASSERT(binSaver->IsReading());

    ui32 objectCount = 0;
    LoadMulti(
        binSaver,
        &FloatFeatureToPackedBinaryIndex,
        &CatFeatureToPackedBinaryIndex,
        &PackedBinaryToSrcIndex,
        &objectCount
    );

    IBinSaver::TStoredSize srcDataSize = 0;
    binSaver->Add(0, &srcDataSize);
    SrcData.resize(srcDataSize);

    for (auto srcDataIdx : xrange(srcDataSize)) {
        TVector<TBinaryFeaturesPack> packedData;
        packedData.yresize(objectCount);
        binSaver->AddRawData(0, packedData.data(), (i64)objectCount*sizeof(TBinaryFeaturesPack));

        SrcData[srcDataIdx] = TMaybeOwningArrayHolder<TBinaryFeaturesPack>::CreateOwning(
            std::move(packedData)
        );
    }
}

TPackedBinaryIndex NCB::TPackedBinaryFeaturesData::AddFeature(
    EFeatureType featureType,
    ui32 perTypeFeatureIdx
) {
    auto packedBinaryIndex = TPackedBinaryIndex::FromLinearIdx(PackedBinaryToSrcIndex.size());
    PackedBinaryToSrcIndex.emplace_back(featureType, perTypeFeatureIdx);
    if (featureType == EFeatureType::Float) {
        FloatFeatureToPackedBinaryIndex[perTypeFeatureIdx] = packedBinaryIndex;
    } else if (featureType == EFeatureType::Categorical) {
        CatFeatureToPackedBinaryIndex[perTypeFeatureIdx] = packedBinaryIndex;
    } else {
        CB_ENSURE(false, "Feature type " << featureType << " is not supported in PackedBinaryFeatures");
    }

    return packedBinaryIndex;
}

TString NCB::DbgDumpMetaData(const NCB::TPackedBinaryFeaturesData& packedBinaryFeaturesData) {
    TStringBuilder sb;
    sb << "FloatFeatureToPackedBinaryIndex="
       << NCB::DbgDumpWithIndices(packedBinaryFeaturesData.FloatFeatureToPackedBinaryIndex, true)
       << "CatFeatureToPackedBinaryIndex="
       << NCB::DbgDumpWithIndices(packedBinaryFeaturesData.CatFeatureToPackedBinaryIndex, true)
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
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    IBinSaver* binSaver
) {
    PackedBinaryFeaturesData.Load(binSaver);
    ExclusiveFeatureBundlesData.Load(binSaver);
    Data.QuantizedFeaturesInfo = quantizedFeaturesInfo;
    LoadFeatures<EFeatureType::Float>(
        *(quantizedFeaturesInfo->GetFeaturesLayout()),
        subsetIndexing,
        &PackedBinaryFeaturesData,
        &ExclusiveFeatureBundlesData,
        binSaver,
        &Data.FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        *(quantizedFeaturesInfo->GetFeaturesLayout()),
        subsetIndexing,
        &PackedBinaryFeaturesData,
        &ExclusiveFeatureBundlesData,
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
        Check(data.PackedBinaryFeaturesData, data.ExclusiveFeatureBundlesData);
    }
    PackedBinaryFeaturesData = std::move(data.PackedBinaryFeaturesData);
    ExclusiveFeatureBundlesData = std::move(data.ExclusiveFeatureBundlesData);

    CatFeatureUniqueValuesCounts.yresize(Data.CatFeatures.size());
    for (auto catFeatureIdx : xrange(Data.CatFeatures.size())) {
        CatFeatureUniqueValuesCounts[catFeatureIdx] =
            Data.QuantizedFeaturesInfo->GetUniqueValuesCounts(TCatFeatureIdx(catFeatureIdx));
    }
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
    subsetData.Data = Data.GetSubset(subsetCommonData.SubsetIndexing.Get());
    subsetData.PackedBinaryFeaturesData = PackedBinaryFeaturesData;
    subsetData.ExclusiveFeatureBundlesData = ExclusiveFeatureBundlesData;

    return MakeIntrusive<TQuantizedForCPUObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}


template <EFeatureType FeatureType, class IColumnType>
static void MakeConsecutiveArrayFeatures(
    const TFeaturesLayout& featuresLayout,
    ui32 objectCount,
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    const TVector<THolder<IColumnType>>& src,
    const TVector<TMaybe<TExclusiveBundleIndex>>& featureToExclusiveFeaturesBundleIndex,
    const TExclusiveFeatureBundlesData& newExclusiveFeatureBundlesData,
    const TVector<TMaybe<TPackedBinaryIndex>>& featureToPackedBinaryIndex,
    const TVector<TMaybeOwningArrayHolder<TBinaryFeaturesPack>>& newPackedBinaryFeatures,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<IColumnType>>* dst
) {
    if (&src != dst) {
        dst->clear();
        dst->resize(featuresLayout.GetFeatureCount(FeatureType));
    }

    TVector<std::function<void()>> tasks;

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const auto& srcColumn = *(src[*featureIdx]);

            if (auto maybeExclusiveFeaturesBundleIndex = featureToExclusiveFeaturesBundleIndex[*featureIdx]) {
                const auto& bundleMetaData
                    = newExclusiveFeatureBundlesData.MetaData[maybeExclusiveFeaturesBundleIndex->BundleIdx];

                (*dst)[*featureIdx] = MakeHolder<TBundlePartValuesHolderImpl<IColumnType>>(
                    srcColumn.GetId(),
                    newExclusiveFeatureBundlesData.SrcData[maybeExclusiveFeaturesBundleIndex->BundleIdx],
                    bundleMetaData.SizeInBytes,
                    bundleMetaData.Parts[maybeExclusiveFeaturesBundleIndex->InBundleIdx].Bounds,
                    newSubsetIndexing
                );
            } else if (auto maybePackedBinaryIndex = featureToPackedBinaryIndex[*featureIdx]) {
                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<IColumnType>>(
                    srcColumn.GetId(),
                    newPackedBinaryFeatures[maybePackedBinaryIndex->PackIdx],
                    maybePackedBinaryIndex->BitIdx,
                    newSubsetIndexing
                );
            } else {
                tasks.emplace_back(
                    [&, featureIdx, localExecutor]() {
                        const auto& srcCompressedValuesHolder
                            = dynamic_cast<const TCompressedValuesHolderImpl<IColumnType>&>(srcColumn);
                        const ui32 bitsPerKey = srcCompressedValuesHolder.GetBitsPerKey();
                        TIndexHelper<ui64> indexHelper(bitsPerKey);
                        const ui32 dstStorageSize = indexHelper.CompressedSize(objectCount);

                        TVector<ui64> storage;
                        storage.yresize(dstStorageSize);

                        if (bitsPerKey == 8) {
                            auto dstBuffer = (ui8*)(storage.data());

                            srcCompressedValuesHolder.template GetArrayData<ui8>().ParallelForEach(
                                [&](ui32 idx, ui8 value) {
                                    dstBuffer[idx] = value;
                                },
                                localExecutor
                            );
                        } else if (bitsPerKey == 16) {
                            auto dstBuffer = (ui16*)(storage.data());

                            srcCompressedValuesHolder.template GetArrayData<ui16>().ParallelForEach(
                                [&](ui32 idx, ui16 value) {
                                    dstBuffer[idx] = value;
                                },
                                localExecutor
                            );
                        } else {
                            auto dstBuffer = (ui32*)(storage.data());

                            srcCompressedValuesHolder.template GetArrayData<ui32>().ParallelForEach(
                                [&](ui32 idx, ui32 value) {
                                    dstBuffer[idx] = value;
                                },
                                localExecutor
                            );
                        }

                        (*dst)[*featureIdx] = MakeHolder<TCompressedValuesHolderImpl<IColumnType>>(
                            srcColumn.GetId(),
                            TCompressedArray(
                                objectCount,
                                bitsPerKey,
                                TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(storage))
                            ),
                            newSubsetIndexing
                        );
                    }
                );
            }
        }
    );

    ExecuteTasksInParallel(&tasks, localExecutor);
}


template <class TBundle>
static inline TMaybeOwningArrayHolder<ui8> CreateConsecutiveData(
    const TBundle* srcData,
    const NCB::TFeaturesArraySubsetIndexing& subsetIndexing,
    NPar::TLocalExecutor* localExecutor
) {
    auto storageHolder = MakeIntrusive<TVectorHolder<TBundle>>(
        NCB::GetSubset<TBundle>(srcData, subsetIndexing, localExecutor)
    );

    ui8* dataPtrAsUi8 = (ui8*)storageHolder->Data.data();
    return TMaybeOwningArrayHolder<ui8>::CreateOwning(
        TArrayRef<ui8>(dataPtrAsUi8, subsetIndexing.Size() * sizeof(TBundle)),
        std::move(storageHolder)
    );
}


static void MakeConsecutiveExclusiveFeatureBundles(
    const NCB::TFeaturesArraySubsetIndexing& subsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    NCB::TExclusiveFeatureBundlesData* exclusiveFeatureBundlesData
) {
    TVector<std::function<void()>> tasks;

    for (auto i : xrange(exclusiveFeatureBundlesData->MetaData.size())) {
        tasks.emplace_back(
            [&, i] () {
                auto& bundleData = exclusiveFeatureBundlesData->SrcData[i];

                auto sizeInBytes = exclusiveFeatureBundlesData->MetaData[i].SizeInBytes;
                switch (sizeInBytes) {
                    case 1:
                        bundleData = CreateConsecutiveData((*bundleData).data(), subsetIndexing, localExecutor);
                        break;
                    case 2:
                        bundleData = CreateConsecutiveData(
                            (const ui16*)(*bundleData).data(),
                            subsetIndexing,
                            localExecutor
                        );
                        break;
                    default:
                        CB_ENSURE_INTERNAL(false, "unsupported Bundle SizeInBytes = " << sizeInBytes);
                }
            }
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
}



static void MakeConsecutivePackedBinaryFeatures(
    const NCB::TFeaturesArraySubsetIndexing& subsetIndexing,
    NPar::TLocalExecutor* localExecutor,
    TVector<TMaybeOwningArrayHolder<TBinaryFeaturesPack>>* packedBinaryFeatures
) {
    TVector<std::function<void()>> tasks;

    for (auto i : xrange(packedBinaryFeatures->size())) {
        tasks.emplace_back(
            [&, i] () {
                auto& packedBinaryFeaturesPart = (*packedBinaryFeatures)[i];
                TVector<TBinaryFeaturesPack> consecutiveData = NCB::GetSubset<TBinaryFeaturesPack>(
                    *packedBinaryFeaturesPart,
                    subsetIndexing,
                    localExecutor
                );
                packedBinaryFeaturesPart = TMaybeOwningArrayHolder<TBinaryFeaturesPack>::CreateOwning(
                    std::move(consecutiveData)
                );
            }
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
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
                MakeConsecutiveExclusiveFeatureBundles(
                    GetFeaturesArraySubsetIndexing(),
                    localExecutor,
                    &ExclusiveFeatureBundlesData
                );
            }
        );

        tasks.emplace_back(
            [&] () {
                MakeConsecutivePackedBinaryFeatures(
                    GetFeaturesArraySubsetIndexing(),
                    localExecutor,
                    &PackedBinaryFeaturesData.SrcData
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
                    GetObjectCount(),
                    newSubsetIndexing.Get(),
                    Data.FloatFeatures,
                    ExclusiveFeatureBundlesData.FloatFeatureToBundlePart,
                    ExclusiveFeatureBundlesData,
                    PackedBinaryFeaturesData.FloatFeatureToPackedBinaryIndex,
                    PackedBinaryFeaturesData.SrcData,
                    localExecutor,
                    &Data.FloatFeatures
                );
            }
        );
        tasks.emplace_back(
            [&] () {
                MakeConsecutiveArrayFeatures<EFeatureType::Categorical>(
                    *GetFeaturesLayout(),
                    GetObjectCount(),
                    newSubsetIndexing.Get(),
                    Data.CatFeatures,
                    ExclusiveFeatureBundlesData.CatFeatureToBundlePart,
                    ExclusiveFeatureBundlesData,
                    PackedBinaryFeaturesData.CatFeatureToPackedBinaryIndex,
                    PackedBinaryFeaturesData.SrcData,
                    localExecutor,
                    &Data.CatFeatures
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);
    }

    CommonData.SubsetIndexing = std::move(newSubsetIndexing);
}


template <class TBaseFeatureColumn>
static void CheckFeaturesByType(
    EFeatureType featureType,
    // not TConstArrayRef to allow template parameter deduction
    const TVector<THolder<TBaseFeatureColumn>>& data,
    const TVector<TMaybe<TPackedBinaryIndex>>& featureToPackedBinaryIndex,
    const TVector<std::pair<EFeatureType, ui32>>& packedBinaryToSrcIndex,
    const TVector<TMaybe<TExclusiveBundleIndex>>& featureToBundlePart,
    const TVector<TExclusiveFeaturesBundle>& bundlesMetaData,
    const TStringBuf featureTypeName
) {
    CB_ENSURE_INTERNAL(
        data.size() == featureToPackedBinaryIndex.size(),
        "Data." << featureTypeName << "Features.size() is not equal to PackedBinaryFeaturesData."
        << featureTypeName << "FeatureToPackedBinaryIndex.size()"
    );

    for (auto featureIdx : xrange(data.size())) {
        auto* dataPtr = data[featureIdx].Get();
        if (!dataPtr) {
            continue;
        }

        auto maybePackedBinaryIndex = featureToPackedBinaryIndex[featureIdx];
        auto maybeBundlePart = featureToBundlePart[featureIdx];

        CB_ENSURE_INTERNAL(
            !maybePackedBinaryIndex || !maybeBundlePart,
            "Data." << featureType << "Features[" << featureIdx
            << "] is both binary packed and in exclusive bundle"
        );

        if (maybePackedBinaryIndex) {
            auto requiredTypePtr = dynamic_cast<TPackedBinaryValuesHolderImpl<TBaseFeatureColumn>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "PackedBinaryValuesHolder"
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
            auto requiredTypePtr = dynamic_cast<TBundlePartValuesHolderImpl<TBaseFeatureColumn>*>(dataPtr);
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
        } else {
            auto requiredTypePtr = dynamic_cast<TCompressedValuesHolderImpl<TBaseFeatureColumn>*>(dataPtr);
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
        (ExclusiveFeatureBundlesData.MetaData == rhs.ExclusiveFeatureBundlesData.MetaData);
}


void NCB::TQuantizedForCPUObjectsDataProvider::Check(
    const TPackedBinaryFeaturesData& packedBinaryData,
    const TExclusiveFeatureBundlesData& exclusiveFeatureBundlesData
) const {
    CheckFeaturesByType(
        EFeatureType::Float,
        Data.FloatFeatures,
        packedBinaryData.FloatFeatureToPackedBinaryIndex,
        packedBinaryData.PackedBinaryToSrcIndex,
        exclusiveFeatureBundlesData.FloatFeatureToBundlePart,
        exclusiveFeatureBundlesData.MetaData,
        "Float"
    );
    CheckFeaturesByType(
        EFeatureType::Categorical,
        Data.CatFeatures,
        packedBinaryData.CatFeatureToPackedBinaryIndex,
        packedBinaryData.PackedBinaryToSrcIndex,
        exclusiveFeatureBundlesData.CatFeatureToBundlePart,
        exclusiveFeatureBundlesData.MetaData,
        "Cat"
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
