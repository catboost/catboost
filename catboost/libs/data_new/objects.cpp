#include "objects.h"
#include "util.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/compare.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/ymath.h>
#include <util/stream/format.h>
#include <util/system/yassert.h>

#include <algorithm>


using namespace NCB;


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
        AreFeaturesValuesEqual(CatFeatures, rhs.CatFeatures);
}

void NCB::TRawObjectsData::PrepareForInitialization(const TDataMetaInfo& metaInfo) {
    // FloatFeatures and CatFeatures members are initialized at the end of building
    FloatFeatures.clear();
    FloatFeatures.resize((size_t)metaInfo.FeaturesLayout->GetFloatFeatureCount());

    CatFeatures.clear();
    const size_t catFeatureCount = (size_t)metaInfo.FeaturesLayout->GetCatFeatureCount();
    CatFeatures.resize(catFeatureCount);
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
        (TConstArrayRef<THolder<TFloatValuesHolder>>)Data.FloatFeatures,
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.FloatFeatures
    );
    CreateSubsetFeatures(
        (TConstArrayRef<THolder<THashedCatValuesHolder>>)Data.CatFeatures,
        subsetCommonData.SubsetIndexing.Get(),
        &subsetData.CatFeatures
    );

    return MakeIntrusive<TRawObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
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
        AreFeaturesValuesEqual(CatFeatures, rhs.CatFeatures);
}


void NCB::TQuantizedObjectsData::PrepareForInitialization(
    const TDataMetaInfo& metaInfo,
    const NCatboostOptions::TBinarizationOptions& binarizationOptions
) {
    // FloatFeatures and CatFeatures members are initialized at the end of building
    FloatFeatures.clear();
    FloatFeatures.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());

    CatFeatures.clear();
    const ui32 catFeatureCount = metaInfo.FeaturesLayout->GetCatFeatureCount();
    CatFeatures.resize(catFeatureCount);

    if (!QuantizedFeaturesInfo) {
        QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *metaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            binarizationOptions,
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

template <EFeatureType FeatureType, class T, class IColumn>
static ui32 CalcFeatureValuesCheckSum(
    ui32 init,
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<IColumn>>& featuresData,
    NPar::TLocalExecutor* localExecutor)
{
    ui32 checkSum = init;
    const ui32 emptyColumnDataForCrc = 0;
    for (auto perTypeFeatureIdx : xrange(featuresLayout.GetFeatureCount(FeatureType))) {
        if (featuresLayout.GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType).IsAvailable) {
            auto compressedValuesFeatureData = dynamic_cast<const TCompressedValuesHolderImpl<IColumn>*>(
                featuresData[perTypeFeatureIdx].Get()
            );
            if (compressedValuesFeatureData) {
                if (compressedValuesFeatureData->GetBitsPerKey() == CHAR_BIT*sizeof(T)) {
                    compressedValuesFeatureData->GetArrayData().ForEach([&](ui32 /*idx*/, T element) {
                        checkSum = UpdateCheckSum(checkSum, (ui32)element);
                    });
                } else {
                    compressedValuesFeatureData->GetCompressedData().ForEach([&](ui32 /*idx*/, ui32 element) {
                        checkSum = UpdateCheckSum(checkSum, element);
                    });
                }
            } else {
                const auto valuesFeatureData = featuresData[perTypeFeatureIdx]->ExtractValues(localExecutor);
                for (auto element : *valuesFeatureData) {
                    checkSum = UpdateCheckSum(checkSum, element);
                }
            }
        } else {
            checkSum = UpdateCheckSum(checkSum, emptyColumnDataForCrc);
        }
    }
    return checkSum;
}

ui32 NCB::TQuantizedObjectsDataProvider::CalcFeaturesCheckSum(NPar::TLocalExecutor* localExecutor) const {
    ui32 checkSum = 0;

    checkSum = Data.QuantizedFeaturesInfo->CalcCheckSum();
    checkSum = CalcFeatureValuesCheckSum<EFeatureType::Float, ui8>(
        checkSum,
        *CommonData.FeaturesLayout,
        Data.FloatFeatures,
        localExecutor
    );
    checkSum = CalcFeatureValuesCheckSum<EFeatureType::Categorical, ui32>(
        checkSum,
        *CommonData.FeaturesLayout,
        Data.CatFeatures,
        localExecutor
    );

    return checkSum;

}

template <EFeatureType FeatureType, class IColumnType>
static void LoadFeatures(
    const TFeaturesLayout& featuresLayout,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    const TMaybe<TPackedBinaryFeaturesData*> packedBinaryFeaturesData,
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
        binSaver,
        &FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        *QuantizedFeaturesInfo->GetFeaturesLayout(),
        subsetIndexing,
        /*packedBinaryFeaturesData*/ Nothing(),
        binSaver,
        &CatFeatures
    );
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
}


NCB::TPackedBinaryFeaturesData::TPackedBinaryFeaturesData(
    const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
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
            if (quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size() == 1) {
                FloatFeatureToPackedBinaryIndex[*floatFeatureIdx]
                    = TPackedBinaryIndex::FromLinearIdx(SafeIntegerCast<ui32>(PackedBinaryToSrcIndex.size()));
                PackedBinaryToSrcIndex.emplace_back(EFeatureType::Float, *floatFeatureIdx);
            }
        }
    );
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            if (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnAll == 2) {
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
    } else {
        CatFeatureToPackedBinaryIndex[perTypeFeatureIdx] = packedBinaryIndex;
    }

    return packedBinaryIndex;
}


void NCB::TQuantizedForCPUObjectsData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    IBinSaver* binSaver
) {
    PackedBinaryFeaturesData.Load(binSaver);
    Data.QuantizedFeaturesInfo = quantizedFeaturesInfo;
    LoadFeatures<EFeatureType::Float>(
        *(quantizedFeaturesInfo->GetFeaturesLayout()),
        subsetIndexing,
        &PackedBinaryFeaturesData,
        binSaver,
        &Data.FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        *(quantizedFeaturesInfo->GetFeaturesLayout()),
        subsetIndexing,
        &PackedBinaryFeaturesData,
        binSaver,
        &Data.CatFeatures
    );
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
        Check(data.PackedBinaryFeaturesData);
    }
    PackedBinaryFeaturesData = std::move(data.PackedBinaryFeaturesData);

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
    const TVector<TMaybe<TPackedBinaryIndex>>& featureToPackedBinaryIndex,
    const TVector<TMaybeOwningArrayHolder<TBinaryFeaturesPack>>& newPackedBinaryFeatures,
    NPar::TLocalExecutor* localExecutor,
    TVector<THolder<IColumnType>>* dst
) {
    constexpr ui32 bytesPerKey = sizeof(typename IColumnType::TValueType);
    constexpr ui32 bitsPerKey = bytesPerKey * CHAR_BIT;

    TIndexHelper<ui64> indexHelper(bitsPerKey);
    const ui32 dstStorageSize = indexHelper.CompressedSize(objectCount);

    if (&src != dst) {
        dst->clear();
        dst->resize(featuresLayout.GetFeatureCount(FeatureType));
    }

    TVector<std::function<void()>> tasks;

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const auto& srcColumn = *(src[*featureIdx]);

            if (auto maybePackedBinaryIndex = featureToPackedBinaryIndex[*featureIdx]) {
                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<IColumnType>>(
                    srcColumn.GetId(),
                    newPackedBinaryFeatures[maybePackedBinaryIndex->PackIdx],
                    maybePackedBinaryIndex->BitIdx,
                    newSubsetIndexing
                );
            } else {
                tasks.emplace_back(
                    [&, featureIdx]() {
                        const auto& srcCompressedValuesHolder
                            = dynamic_cast<const TCompressedValuesHolderImpl<IColumnType>&>(srcColumn);

                        TVector<ui64> storage;
                        storage.yresize(dstStorageSize);
                        auto dstBuffer = (typename IColumnType::TValueType*)(storage.data());

                        srcCompressedValuesHolder.GetArrayData().ParallelForEach(
                            [&] (ui32 idx, typename IColumnType::TValueType value) {
                                dstBuffer[idx] = value;
                            },
                            localExecutor
                        );

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

    MakeConsecutivePackedBinaryFeatures(
        GetFeaturesArraySubsetIndexing(),
        localExecutor,
        &PackedBinaryFeaturesData.SrcData
    );

    TVector<std::function<void()>> tasks;

    tasks.emplace_back(
        [&] () {
            MakeConsecutiveArrayFeatures<EFeatureType::Float>(
                *GetFeaturesLayout(),
                GetObjectCount(),
                newSubsetIndexing.Get(),
                Data.FloatFeatures,
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
                PackedBinaryFeaturesData.CatFeatureToPackedBinaryIndex,
                PackedBinaryFeaturesData.SrcData,
                localExecutor,
                &Data.CatFeatures
            );
        }
    );

    ExecuteTasksInParallel(&tasks, localExecutor);

    CommonData.SubsetIndexing = std::move(newSubsetIndexing);
}


template <class TBaseFeatureColumn>
static void CheckFeaturesByType(
    EFeatureType featureType,
    // not TConstArrayRef to allow template parameter deduction
    const TVector<THolder<TBaseFeatureColumn>>& data,
    const TVector<TMaybe<TPackedBinaryIndex>>& featureToPackedBinaryIndex,
    const TVector<std::pair<EFeatureType, ui32>>& packedBinaryToSrcIndex,
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

        if (featureToPackedBinaryIndex[featureIdx]) {
            auto requiredTypePtr = dynamic_cast<TPackedBinaryValuesHolderImpl<TBaseFeatureColumn>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "PackedBinaryValuesHolder"
            );

            auto linearPackedBinaryFeatureIdx = featureToPackedBinaryIndex[featureIdx]->GetLinearIdx();
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
        } else {
            auto requiredTypePtr = dynamic_cast<TCompressedValuesHolderImpl<TBaseFeatureColumn>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "ValuesHolder"
            );
            requiredTypePtr->GetCompressedData().GetSrc()
                ->template CheckIfCanBeInterpretedAsRawArray<typename TBaseFeatureColumn::TValueType>();
        }
    }
}


void NCB::TQuantizedForCPUObjectsDataProvider::Check(const TPackedBinaryFeaturesData& packedBinaryData) const {
    CheckFeaturesByType(
        EFeatureType::Float,
        Data.FloatFeatures,
        packedBinaryData.FloatFeatureToPackedBinaryIndex,
        packedBinaryData.PackedBinaryToSrcIndex,
        "Float"
    );
    CheckFeaturesByType(
        EFeatureType::Categorical,
        Data.CatFeatures,
        packedBinaryData.CatFeatureToPackedBinaryIndex,
        packedBinaryData.PackedBinaryToSrcIndex,
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
