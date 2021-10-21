#include "objects.h"

#include "sparse_columns.h"
#include "util.h"

#include <catboost/libs/cat_feature/cat_feature.h>

#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>
#include <util/stream/format.h>
#include <util/stream/output.h>
#include <util/system/mem_info.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <type_traits>


using namespace NCB;

template <class TGroupIdClass>
static void CheckGroupIds(
    ui32 objectCount,
    TMaybeData<TConstArrayRef<TGroupIdClass>> groupIds,
    TMaybe<TObjectsGroupingPtr> objectsGrouping
) {
    if (!groupIds) {
        return;
    }
    auto groupIdsData = *groupIds;

    CheckDataSize(groupIdsData.size(), (size_t)objectCount, "group Ids", false);


    TVector<TGroupIdClass> groupGroupIds;
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

    TGroupIdClass lastGroupId = groupIdsData[0];
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

static bool HaveMoreThanOneKeyOrAnyValueMismatch(
    const THashMap<ui32, TString>& lhs,
    const THashMap<ui32, TString>& rhs,
    bool* haveKeyMismatch // updated
) {
    for (const auto& lhsElement : lhs) {
        const auto rhsIterator = rhs.find(lhsElement.first);
        if (rhsIterator == rhs.end()) {
            if (*haveKeyMismatch) {
                return true;
            }
            *haveKeyMismatch = true;
        } else if (rhsIterator->second != lhsElement.second) {
            return true;
        }
    }
    return false;
}


static bool AreCatFeaturesHashToStringEqual(
    TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> lhsCatFeaturesHashToString,
    TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> rhsCatFeaturesHashToString,
    bool ignoreSparsity
) {
    if (!lhsCatFeaturesHashToString) {
        return !rhsCatFeaturesHashToString;
    }
    if (!rhsCatFeaturesHashToString) {
        return false;
    }

    if (!ignoreSparsity) {
        return *lhsCatFeaturesHashToString == *rhsCatFeaturesHashToString;
    }

    // compare with sparsity
    if (lhsCatFeaturesHashToString->size() != rhsCatFeaturesHashToString->size()) {
        return false;
    }

    // allow up to one mismatch, can't check default values precisely because we don't know their values here
    for (auto catFeatureIdx : xrange(lhsCatFeaturesHashToString->size())) {
        const auto& lhsCatFeatureHashToString = (*lhsCatFeaturesHashToString)[catFeatureIdx];
        const auto& rhsCatFeatureHashToString = (*rhsCatFeaturesHashToString)[catFeatureIdx];

        bool haveKeyMismatch = false;
        if (HaveMoreThanOneKeyOrAnyValueMismatch(
                lhsCatFeatureHashToString,
                rhsCatFeatureHashToString,
                &haveKeyMismatch))
        {
            return false;
        }
        if (HaveMoreThanOneKeyOrAnyValueMismatch(
                rhsCatFeatureHashToString,
                lhsCatFeatureHashToString,
                &haveKeyMismatch))
        {
            return false;
        }
    }

    return true;
}


bool NCB::TCommonObjectsData::EqualTo(const NCB::TCommonObjectsData& rhs, bool ignoreSparsity) const {
    if (!AreCatFeaturesHashToStringEqual(
        CatFeaturesHashToString,
        rhs.CatFeaturesHashToString,
        ignoreSparsity))
    {
        return false;
    }

    return FeaturesLayout->EqualTo(*rhs.FeaturesLayout, ignoreSparsity) && (Order == rhs.Order) && (StoreStringColumns == rhs.StoreStringColumns) &&
        (SampleId == rhs.SampleId) && (GroupIds == rhs.GroupIds) && (SubgroupIds == rhs.SubgroupIds) && (Timestamp == rhs.Timestamp);
}

void NCB::TCommonObjectsData::SetStoreStringColumns(bool storeStringColumns) {
    SubgroupIds.SetStoreStringColumns(storeStringColumns);
    GroupIds.SetStoreStringColumns(storeStringColumns);
    StoreStringColumns = storeStringColumns;
}

void NCB::TCommonObjectsData::PrepareForInitialization(
    const TDataMetaInfo& metaInfo,
    ui32 objectCount,
    ui32 prevTailCount
) {
    FeaturesLayout = metaInfo.FeaturesLayout;
    if (prevTailCount == 0) {
        SetStoreStringColumns(metaInfo.StoreStringColumns);
    } else {
        SubgroupIds.SetStoreStringColumnsVal(metaInfo.StoreStringColumns);
        GroupIds.SetStoreStringColumnsVal(metaInfo.StoreStringColumns);
        StoreStringColumns = metaInfo.StoreStringColumns;
    }

    if (StoreStringColumns) {
        NCB::PrepareForInitialization(metaInfo.HasGroupId, objectCount, prevTailCount, &GroupIds.GetMaybeStringData());
        NCB::PrepareForInitialization(metaInfo.HasSubgroupIds, objectCount, prevTailCount, &SubgroupIds.GetMaybeStringData());
        NCB::PrepareForInitialization(metaInfo.HasSampleId, objectCount, prevTailCount, &SampleId);
    } else {
        NCB::PrepareForInitialization(metaInfo.HasGroupId, objectCount, prevTailCount, &GroupIds.GetMaybeNumData());
        NCB::PrepareForInitialization(metaInfo.HasSubgroupIds, objectCount, prevTailCount, &SubgroupIds.GetMaybeNumData());
    }
    NCB::PrepareForInitialization(metaInfo.HasTimestamp, objectCount, prevTailCount, &Timestamp);

    const size_t catFeatureCount = (size_t)metaInfo.FeaturesLayout->GetCatFeatureCount();
    if (catFeatureCount) {
        if (!CatFeaturesHashToString) {
            CatFeaturesHashToString = MakeAtomicShared<TVector<THashMap<ui32, TString>>>();
        }
        CatFeaturesHashToString->resize(catFeatureCount);
    }
}

void NCB::TCommonObjectsData::SetBuildersArrayRef(
    const TDataMetaInfo& metaInfo,
    TArrayRef<TGroupId>* numGroupIdsRefPtr,
    TArrayRef<TString>* stringGroupIdsRefPtr,
    TArrayRef<TSubgroupId>* numSubgroupIdsRefPtr,
    TArrayRef<TString>* stringSubgroupIdsRefPtr
) {
    if (StoreStringColumns) {
        if (metaInfo.HasGroupId) {
            *stringGroupIdsRefPtr = GroupIds.GetMaybeStringData().GetRef();
        }
        if (metaInfo.HasSubgroupIds) {
            *stringSubgroupIdsRefPtr = SubgroupIds.GetMaybeStringData().GetRef();
        }
    } else {
        if (metaInfo.HasGroupId) {
            *numGroupIdsRefPtr = GroupIds.GetMaybeNumData().GetRef();
        }
        if (metaInfo.HasSubgroupIds) {
            *numSubgroupIdsRefPtr = SubgroupIds.GetMaybeNumData().GetRef();
        }
    }
}


void NCB::TCommonObjectsData::CheckAllExceptGroupIds() const {
    if (SubgroupIds.IsDefined()) {
        CB_ENSURE(
            GroupIds.IsDefined(),
            "non-empty SubgroupIds when GroupIds is not defined"
        );
        CheckDataSize(SubgroupIds.GetSize(), GroupIds.GetSize(), "Subgroup Ids", false, "Group Ids size");
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
    if (StoreStringColumns) {
        CheckGroupIds<TString>(SubsetIndexing->Size(), GroupIds.GetMaybeStringData(), objectsGrouping);
    } else {
        CheckGroupIds<TGroupId>(SubsetIndexing->Size(), GroupIds.GetMaybeNumData(), objectsGrouping);
    }
    CheckAllExceptGroupIds();
}

NCB::TCommonObjectsData NCB::TCommonObjectsData::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor
) const {
    TCommonObjectsData result;
    result.ResourceHolders = ResourceHolders;
    result.FeaturesLayout = FeaturesLayout;
    result.Order = Combine(Order, objectsGroupingSubset.GetObjectSubsetOrder());
    result.SetStoreStringColumns(StoreStringColumns);

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
            result.GroupIds = GetSubsetFromMaybeStringOrNumIdColumn(
                GroupIds,
                objectsGroupingSubset.GetObjectsIndexing(),
                localExecutor
            );
        }
    );
    tasks.emplace_back(
        [&, this]() {
            result.SampleId = GetSubsetOfMaybeEmpty<TString>(
                (TMaybeData<TConstArrayRef<TString>>) SampleId,
                objectsGroupingSubset.GetObjectsIndexing(),
                localExecutor
            );
        }
    );
    tasks.emplace_back(
        [&, this]() {
            result.SubgroupIds = GetSubsetFromMaybeStringOrNumIdColumn(
                SubgroupIds,
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
    LoadMulti(binSaver, &Order, &StoreStringColumns, &SampleId, &GroupIds, &SubgroupIds, &Timestamp);
    AddWithShared(binSaver, &CatFeaturesHashToString);
}

void NCB::TCommonObjectsData::SaveNonSharedPart(IBinSaver* binSaver) const {
    SaveMulti(binSaver, Order, StoreStringColumns, SampleId, GroupIds, SubgroupIds, Timestamp);
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
            commonData.StoreStringColumns ?
                CreateObjectsGroupingFromGroupIds<TString>(
                    commonData.SubsetIndexing->Size(),
                    commonData.GroupIds.GetMaybeStringData() // Nothing(), Turn off group checks for quantization due to time
                ) :
                CreateObjectsGroupingFromGroupIds<TGroupId>(
                    commonData.SubsetIndexing->Size(),
                    commonData.GroupIds.GetMaybeNumData()
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
        CreateObjectsGroupingFromGroupIds<TGroupId>(GetObjectCount(), groupIds) // groupIds data size is checked inside
    );
    auto& groupIds_ = CommonData.GroupIds.GetMaybeNumData();
    if (!groupIds_) {
        groupIds_.ConstructInPlace(groupIds.begin(), groupIds.end());
    } else {
        groupIds_->assign(groupIds.begin(), groupIds.end());
    }
}

void NCB::TObjectsDataProvider::SetSubgroupIds(TConstArrayRef<TSubgroupId> subgroupIds) {
    CheckDataSize(subgroupIds.size(), (size_t)GetObjectCount(), "subgroupIds");
    auto& subgroupIds_ = CommonData.SubgroupIds.GetMaybeNumData();
    if (!subgroupIds_) {
        subgroupIds_.ConstructInPlace(subgroupIds.begin(), subgroupIds.end());
    } else {
        subgroupIds_->assign(subgroupIds.begin(), subgroupIds.end());
    }
}

void NCB::TObjectsDataProvider::SetTimestamps(TConstArrayRef<ui64> timestamps) {
    CheckDataSize(timestamps.size(), (size_t)GetObjectCount(), "timestamps");
    CommonData.Timestamp.ConstructInPlace(TVector<ui64>(timestamps.begin(), timestamps.end()));
}

TIntrusivePtr<TObjectsDataProvider> NCB::TObjectsDataProvider::GetFeaturesSubset(
    const TVector<ui32>& ignoredFeatures,
    NPar::ILocalExecutor* localExecutor
) const {
    return GetSubsetImpl(
        ::GetGroupingSubsetFromObjectsSubset(
            ObjectsGrouping,
            TArraySubsetIndexing(TFullSubset<ui32>(GetObjectCount())),
            EObjectsOrder::Ordered),
        ignoredFeatures,
        GetMonopolisticFreeCpuRam(),
        localExecutor
    );
}

TIntrusivePtr<TObjectsDataProvider> NCB::TObjectsDataProvider::Clone(
    NPar::ILocalExecutor* localExecutor
) const {
    return GetSubsetImpl(
        ::GetGroupingSubsetFromObjectsSubset(
            ObjectsGrouping,
            TArraySubsetIndexing(TFullSubset<ui32>(GetObjectCount())),
            EObjectsOrder::Ordered),
        /*ignoredFeatures*/ Nothing(),
        GetMonopolisticFreeCpuRam(),
        localExecutor
    );
}


template <class TColumn>
static bool AreFeaturesValuesEqual(
    const TColumn& lhs,
    const TColumn& rhs
) {
    auto lhsValues = lhs.ExtractValues(&NPar::LocalExecutor());
    auto rhsValues = rhs.ExtractValues(&NPar::LocalExecutor());

    if constexpr (std::is_floating_point<typename TColumn::TValueType>::value) {
        return std::equal(
            lhsValues.begin(),
            lhsValues.end(),
            rhsValues.begin(),
            rhsValues.end(),
            EqualWithNans<typename TColumn::TValueType>
        );
    } else {
        return *lhsValues == *rhsValues;
    }
}

template <typename T, EFeatureValuesType ValuesType, typename TBaseInterface = IFeatureValuesHolder>
static bool AreFeaturesValuesEqual(
    const IQuantizedFeatureValuesHolder<T, ValuesType, TBaseInterface>& lhs,
    const IQuantizedFeatureValuesHolder<T, ValuesType, TBaseInterface>& rhs
) {
    auto lhsValues = lhs.template ExtractValues<T>(&NPar::LocalExecutor());
    auto rhsValues = rhs.template ExtractValues<T>(&NPar::LocalExecutor());
    return lhsValues == rhsValues;
}

template <class TFeaturesValuesHolder>
static bool AreFeaturesValuesEqual(
    const THolder<TFeaturesValuesHolder>& lhs,
    const THolder<TFeaturesValuesHolder>& rhs
) {
    if (!lhs) {
        return !rhs;
    }
    if (!rhs) {
        return false;
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
        AreFeaturesValuesEqual(TextFeatures, rhs.TextFeatures) &&
        AreFeaturesValuesEqual(EmbeddingFeatures, rhs.EmbeddingFeatures);
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

    EmbeddingFeatures.clear();
    EmbeddingFeatures.resize((size_t)metaInfo.FeaturesLayout->GetEmbeddingFeatureCount());
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
    NPar::ILocalExecutor* localExecutor
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
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Embedding, EmbeddingFeatures);

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

                if (auto* denseCatFeaturePtr
                        = dynamic_cast<const THashedCatArrayValuesHolder*>(catFeaturePtr))
                {
                    denseCatFeaturePtr->GetData()->ParallelForEach(checkValue, localExecutor);
                } else if (auto* sparseCatFeaturePtr
                               = dynamic_cast<const THashedCatSparseValuesHolder*>(catFeaturePtr))
                {
                    const auto& sparseArray = sparseCatFeaturePtr->GetData();
                    CB_ENSURE_INTERNAL(
                        hashToStringMap.contains(sparseArray.GetDefaultValue()),
                        "catFeature #" << catFeatureIdx << ", default value "
                        << Hex(sparseArray.GetDefaultValue()) << " is missing from CatFeaturesHashToString"
                    );
                    sparseArray.ForEachNonDefault(checkValue);
                } else {
                    CB_ENSURE_INTERNAL(false, "unknown THashedCatValuesHolder subtype");
                }
            }
        },
        0,
        SafeIntegerCast<int>(CatFeatures.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );

    if (objectCount) {
        localExecutor->ExecRangeWithThrow(
            [&] (int embeddingFeatureIdx) {
                auto* embeddingFeaturePtr = EmbeddingFeatures[embeddingFeatureIdx].Get();
                if (embeddingFeaturePtr) {
                    size_t expectedDimension = embeddingFeaturePtr->GetBlockIterator()->Next(1)[0].GetSize();

                    if (auto* denseEmbeddingFeaturePtr
                            = dynamic_cast<const TEmbeddingArrayValuesHolder*>(embeddingFeaturePtr))
                    {
                        denseEmbeddingFeaturePtr->GetData()->ParallelForEach(
                            [&] (ui32 objectIdx, const TConstEmbedding& value) {
                                CB_ENSURE_INTERNAL(
                                    value.GetSize() == expectedDimension,
                                    "Inconsistent dimensions for embedding data for objects #0 and #"
                                    << objectIdx
                                );
                            },
                            localExecutor
                        );
                    } else {
                        CB_ENSURE_INTERNAL(false, "unknown TEmbeddingValuesHolder subtype");
                    }
                }
            },
            0,
            SafeIntegerCast<int>(EmbeddingFeatures.size()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

template <class TColumn, class TGetAggregatedColumn>
static void GetSubsetWithScheduling(
    TConstArrayRef<THolder<TColumn>> src,
    const TFeaturesArraySubsetIndexing* subsetIndexing,

    // needed only for sparse features
    const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,

    /* flatFeatureIdx -> THolder<TColumn>
     * returns packed or bundled or grouped column data, returns nullptr if not packed or bundled or grouped
     */
    TGetAggregatedColumn&& getAggregatedData,
    //std::function<THolder<TColumn>(ui32)> getAggregatedData,
    TResourceConstrainedExecutor* resourceConstrainedExecutor,
    TVector<THolder<TColumn>>* dst
) {

    dst->clear(); // cleanup old data
    dst->resize(src.size());

    TCloningParams cloningParams;
    cloningParams.MakeConsecutive = false;
    cloningParams.SubsetIndexing = subsetIndexing;
    if (subsetInvertedIndexing.Defined()) {
        cloningParams.InvertedSubsetIndexing = subsetInvertedIndexing.Get();
    }

    auto localExecutor = resourceConstrainedExecutor->GetExecutorPtr();
    for (auto i : xrange(src.size())) {
        auto* srcDataPtr = src[i].Get();
        if (!srcDataPtr) {
            continue;
        }

        (*dst)[i] = getAggregatedData(srcDataPtr->GetId());
        if ((*dst)[i]) {
            continue;
        }
        auto dstHolderPtr = &(*dst)[i];
        // TODO(kirillovs): add shortcut here for zero-cost cloneable columns
        resourceConstrainedExecutor->Add(
            {
                srcDataPtr->EstimateMemoryForCloning(cloningParams),
                [srcDataPtr, dstHolderPtr, localExecutor, cloningParams] () {
                    auto clonedColumn = srcDataPtr->CloneWithNewSubsetIndexing(
                        cloningParams,
                        localExecutor
                    );
                    *dstHolderPtr = DynamicHolderCast<TColumn>(
                        srcDataPtr->CloneWithNewSubsetIndexing(
                            cloningParams,
                            localExecutor
                        ),
                        "Column type changed after cloning"
                    );
                }
            }
        );
    }
}

template <class T>
static void GetSubsetWithScheduling(
    TConstArrayRef<THolder<T>> src,
    const TFeaturesArraySubsetIndexing* subsetIndexing,

    // needed only for sparse features
    const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
    TResourceConstrainedExecutor* resourceConstrainedExecutor,
    TVector<THolder<T>>* dst
) {
    ::GetSubsetWithScheduling(
        src,
        subsetIndexing,
        subsetInvertedIndexing,
        /*getPackedOrBundledData*/ [] (ui32) { return nullptr; },
        resourceConstrainedExecutor,
        dst
    );
}


static TResourceConstrainedExecutor CreateCpuRamConstrainedExecutor(
    ui64 cpuRamLimit,
    NPar::ILocalExecutor* localExecutor
) {
    const ui64 cpuRamUsage = NMemInfo::GetMemInfo().RSS;
    OutputWarningIfCpuRamUsageOverLimit(cpuRamUsage, cpuRamLimit);

    return TResourceConstrainedExecutor(
        "CPU RAM",
        cpuRamLimit - Min(cpuRamUsage, cpuRamLimit),
        /*lenientMode*/ true,
        localExecutor
    );
}


TObjectsDataProviderPtr NCB::TRawObjectsDataProvider::GetSubsetImpl(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
    ui64 cpuRamLimit,
    NPar::ILocalExecutor* localExecutor
) const {
    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );

    // needed only for sparse features
    TMaybe<TFeaturesArraySubsetInvertedIndexing> subsetInvertedIndexing;

    if (CommonData.FeaturesLayout->HasSparseFeatures()) {
        subsetInvertedIndexing.ConstructInPlace(
            GetInvertedIndexing(objectsGroupingSubset.GetObjectsIndexing(), GetObjectCount(), localExecutor)
        );
    }

    if (ignoredFeatures.Defined()) {
        subsetCommonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*subsetCommonData.FeaturesLayout);
        subsetCommonData.FeaturesLayout->IgnoreExternalFeatures(*ignoredFeatures);
    }

    auto resourceConstrainedExecutor = CreateCpuRamConstrainedExecutor(cpuRamLimit, localExecutor);

    TRawObjectsData subsetData;

    auto getSubsetWithScheduling = [&] (const auto& srcFeatures, auto* dstFeatures) {
        GetSubsetWithScheduling(
            MakeConstArrayRef(srcFeatures),
            subsetCommonData.SubsetIndexing.Get(),
            subsetInvertedIndexing,
            &resourceConstrainedExecutor,
            dstFeatures
        );
    };

    getSubsetWithScheduling(Data.FloatFeatures, &subsetData.FloatFeatures);
    getSubsetWithScheduling(Data.CatFeatures, &subsetData.CatFeatures);
    getSubsetWithScheduling(Data.TextFeatures, &subsetData.TextFeatures);
    getSubsetWithScheduling(Data.EmbeddingFeatures, &subsetData.EmbeddingFeatures);

    resourceConstrainedExecutor.ExecTasks();

    return MakeIntrusive<TRawObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}

template <class T>
static bool HasDenseData(const TVector<THolder<T>>& columns) {
    for (const auto& columnPtr : columns) {
        if (columnPtr && !columnPtr->IsSparse()) {
            return true;
        }
    }
    return false;
}

template <class T>
static bool HasSparseData(const TVector<THolder<T>>& columns) {
    for (const auto& columnPtr : columns) {
        if (columnPtr && columnPtr->IsSparse()) {
            return true;
        }
    }
    return false;
}

bool NCB::TRawObjectsDataProvider::HasDenseData() const {
    return ::HasDenseData(Data.FloatFeatures) ||
        ::HasDenseData(Data.CatFeatures) ||
        ::HasDenseData(Data.TextFeatures) ||
        ::HasDenseData(Data.EmbeddingFeatures);
}


bool NCB::TRawObjectsDataProvider::HasSparseData() const {
    return ::HasSparseData(Data.FloatFeatures) ||
        ::HasSparseData(Data.CatFeatures) ||
        ::HasSparseData(Data.TextFeatures) ||
        ::HasDenseData(Data.EmbeddingFeatures);
}

void NCB::TRawObjectsDataProvider::SetGroupIds(TConstArrayRef<TStringBuf> groupStringIds) {
    CB_ENSURE_INTERNAL(!CommonData.StoreStringColumns, "Set TGroupIds with StoreStringColumns option");
    CheckDataSize(groupStringIds.size(), (size_t)GetObjectCount(), "group Ids");

    TVector<TGroupId> newGroupIds;
    newGroupIds.yresize(groupStringIds.size());
    for (auto i : xrange(groupStringIds.size())) {
        newGroupIds[i] = CalcGroupIdFor(groupStringIds[i]);
    }

    ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(
        CreateObjectsGroupingFromGroupIds<TGroupId>(GetObjectCount(), (TConstArrayRef<TGroupId>)newGroupIds)
    );
    CommonData.GroupIds.GetMaybeNumData() = std::move(newGroupIds);
}

void NCB::TRawObjectsDataProvider::SetSubgroupIds(TConstArrayRef<TStringBuf> subgroupStringIds) {
    CB_ENSURE_INTERNAL(!CommonData.StoreStringColumns, "Set TSubroupIds with StoreStringColumns option");
    CheckDataSize(subgroupStringIds.size(), (size_t)GetObjectCount(), "subgroup Ids");
    CB_ENSURE(
        CommonData.GroupIds.IsDefined(),
        "non-empty subgroupStringIds when GroupIds is not defined"
    );

    TVector<TSubgroupId> newSubgroupIds;
    newSubgroupIds.yresize(subgroupStringIds.size());
    for (auto i : xrange(subgroupStringIds.size())) {
        newSubgroupIds[i] = CalcSubgroupIdFor(subgroupStringIds[i]);
    }
    CommonData.SubgroupIds.GetMaybeNumData() = std::move(newSubgroupIds);
}


bool NCB::TQuantizedObjectsData::operator==(const NCB::TQuantizedObjectsData& rhs) const {
    return AreFeaturesValuesEqual(FloatFeatures, rhs.FloatFeatures) &&
        AreFeaturesValuesEqual(CatFeatures, rhs.CatFeatures) &&
        AreFeaturesValuesEqual(TextFeatures, rhs.TextFeatures) &&
        AreFeaturesValuesEqual(EmbeddingFeatures, rhs.EmbeddingFeatures);
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

    EmbeddingFeatures.clear();
    const ui32 embeddingFeatureCount = metaInfo.FeaturesLayout->GetEmbeddingFeatureCount();
    EmbeddingFeatures.resize(embeddingFeatureCount);

    if (!QuantizedFeaturesInfo) {
        QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *metaInfo.FeaturesLayout,
            TConstArrayRef<ui32>(),
            binarizationOptions,
            perFloatFeatureQuantization,
            /*floatFeaturesAllowNansInTestOnly*/true
        );
    }
}


void NCB::TQuantizedObjectsData::Check(
    ui32 objectCount,
    const TFeaturesLayout& featuresLayout,
    NPar::ILocalExecutor* localExecutor
) const {
    /* localExecutor is a parameter here to make
     * TQuantizedObjectsData::Check and TQuantizedObjectsData::Check have the same interface
     */
    Y_UNUSED(localExecutor);

    CB_ENSURE(QuantizedFeaturesInfo.Get(), "NCB::TQuantizedObjectsData::QuantizedFeaturesInfo is not initialized");

    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Float, FloatFeatures);
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Categorical, CatFeatures);
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Text, TextFeatures);
    CheckDataSizes(objectCount, featuresLayout, EFeatureType::Embedding, EmbeddingFeatures);
}


bool NCB::TQuantizedObjectsDataProvider::HasDenseData() const {
    return ::HasDenseData(Data.FloatFeatures) ||
        ::HasDenseData(Data.CatFeatures) ||
        ::HasDenseData(Data.TextFeatures) ||
        ::HasDenseData(Data.EmbeddingFeatures);
}

bool NCB::TQuantizedObjectsDataProvider::HasSparseData() const {
    return ::HasSparseData(Data.FloatFeatures) ||
        ::HasSparseData(Data.CatFeatures) ||
        ::HasSparseData(Data.TextFeatures) ||
        ::HasDenseData(Data.EmbeddingFeatures);
}

template <EFeatureType FeatureType, class T>
static ui32 CalcFeatureValuesCheckSum(
    ui32 init,
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<T>>& featuresData,
    NPar::ILocalExecutor* localExecutor)
{
    const ui32 emptyColumnDataForCrc = 0;
    TVector<ui32> checkSums(featuresLayout.GetFeatureCount(FeatureType), 0);
    localExecutor->ExecRangeWithThrow(
        [&] (ui32 perTypeFeatureIdx) {
            if (featuresLayout.GetInternalFeatureMetaInfo(perTypeFeatureIdx, FeatureType).IsAvailable) {
                // TODO(espetrov,akhropov): remove workaround below MLTOOLS-3604
                if (featuresData[perTypeFeatureIdx].Get() == nullptr) {
                    return;
                }
                checkSums[perTypeFeatureIdx] = featuresData[perTypeFeatureIdx]->CalcChecksum(localExecutor);
            } else {
                checkSums[perTypeFeatureIdx] = UpdateCheckSum(0, emptyColumnDataForCrc);
            }
        },
        0, featuresLayout.GetFeatureCount(FeatureType), NPar::TLocalExecutor::WAIT_COMPLETE
    );
    ui32 checkSum = init;
    for (ui32 featureCheckSum : checkSums) {
        checkSum = UpdateCheckSum(checkSum, featureCheckSum);
    }
    return checkSum;
}

ui32 NCB::TQuantizedObjectsDataProvider::CalcFeaturesCheckSum(NPar::ILocalExecutor* localExecutor) const {
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
    Sparse = 2,
    Dense = 3
};


template <class TColumn>
static void LoadNonBundledColumnData(
    ui32 flatFeatureIdx,
    bool isSparse,
    const TFeaturesArraySubsetIndexing* newSubsetIndexing,
    IBinSaver* binSaver,
    THolder<TColumn>* column
) {
    if (isSparse) {
        TSparseCompressedArray<typename TColumn::TValueType, ui32> data;
        LoadMulti(binSaver, &data);
        *column = MakeHolder<TSparseCompressedValuesHolderImpl<TColumn>>(
            flatFeatureIdx,
            std::move(data)
        );
    } else {
        *column = MakeHolder<TCompressedValuesHolderImpl<TColumn>>(
            flatFeatureIdx,
            LoadAsCompressedArray(binSaver),
            newSubsetIndexing
        );
    }
}

template <EFeatureType FeatureType, class TColumn>
static void LoadFeatures(
    const TFeaturesLayout& featuresLayout,
    const TFeaturesArraySubsetIndexing* subsetIndexing,
    const TMaybe<TPackedBinaryFeaturesData*> packedBinaryFeaturesData,
    const TMaybe<TExclusiveFeatureBundlesData*> exclusiveFeatureBundlesData,
    const TMaybe<TFeatureGroupsData*> featureGroupsData,
    IBinSaver* binSaver,
    TVector<THolder<TColumn>>* dst
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

                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<TColumn>>(
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

                (*dst)[*featureIdx] = MakeHolder<TBundlePartValuesHolderImpl<TColumn>>(
                    flatFeatureIdx,
                    (**exclusiveFeatureBundlesData).SrcData[exclusiveBundleIndex.BundleIdx].Get(),
                    metaData.Parts[exclusiveBundleIndex.InBundleIdx].Bounds
                );
            } else if (featureToGroupPart && (*featureToGroupPart)[flatFeatureIdx]) {
                const TFeaturesGroupIndex featuresGroupIndex = *((*featureToGroupPart)[flatFeatureIdx]);

                (*dst)[*featureIdx] = MakeHolder<TFeaturesGroupPartValuesHolderImpl<TColumn>>(
                    flatFeatureIdx,
                    (**featureGroupsData).SrcData[featuresGroupIndex.GroupIdx].Get(),
                    featuresGroupIndex.InGroupIdx
                );
            } else {
                LoadNonBundledColumnData(
                    flatFeatureIdx,
                    savedColumnType == ESavedColumnType::Sparse,
                    subsetIndexing,
                    binSaver,
                    &((*dst)[*featureIdx])
                );
            }
        }
    );
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

template <class TColumn>
static void SaveColumnData(
    const TColumn& column,
    NPar::ILocalExecutor* localExecutor,
    IBinSaver* binSaver
) {
    if (auto* packedBinaryValues
            = dynamic_cast<const TPackedBinaryValuesHolderImpl<TColumn>*>(&column))
    {
        SaveMulti(binSaver, ESavedColumnType::PackedBinary, packedBinaryValues->GetBitIdx());
    } else if (auto* bundlePartValues
                   = dynamic_cast<const TBundlePartValuesHolderImpl<TColumn>*>(&column))
    {
        SaveMulti(binSaver, ESavedColumnType::BundlePart, bundlePartValues->GetBoundsInBundle());
    } else if (
        auto* sparseValues
            = dynamic_cast<const TSparseCompressedValuesHolderImpl<TColumn>*>(&column))
    {
        SaveMulti(binSaver, ESavedColumnType::Sparse, sparseValues->GetData());
    } else {
        /* TODO(akhropov): specialize for TCompressedValuesHolderImpl
         * useful if in fact bitsPerKey < sizeof(T) * CHAR_BIT
         */
        SaveMulti(binSaver, ESavedColumnType::Dense);
        SaveAsCompressedArray<typename TColumn::TValueType>(column.template ExtractValues<typename TColumn::TValueType>(localExecutor), binSaver);
    }
}


template <EFeatureType FeatureType, class TColumn>
static void SaveFeatures(
    const TFeaturesLayout& featuresLayout,
    const TVector<THolder<TColumn>>& src,
    NPar::ILocalExecutor* localExecutor,
    IBinSaver* binSaver
) {
    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const TColumn* column = src[*featureIdx].Get();
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
    PackedBinaryFeaturesData.Save(&localExecutor, binSaver);
    ExclusiveFeatureBundlesData.Save(&localExecutor, binSaver);
    FeaturesGroupsData.Save(&localExecutor, binSaver);
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
            auto feature = quantizedObjectsDataProvider.GetFloatFeature(*floatFeatureIdx);
            if (!feature)
                return;

            (*feature)->ForEachBlock(
                [out, floatFeatureIdx] (size_t blockStartOffset, auto block) {
                    for (auto i : xrange(block.size())) {
                        auto objectIdx = i + blockStartOffset;
                        (*out) << "(floatFeature=" << *floatFeatureIdx << ',' << LabeledOutput(objectIdx)
                            << ").bin=" << ui32(block[i]) << Endl;
                    }
                }
            );
        }
    );

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx catFeatureIdx) {
            auto feature = quantizedObjectsDataProvider.GetCatFeature(*catFeatureIdx);
            if (!feature)
                return;

            (*feature)->ForEachBlock(
                [out, catFeatureIdx] (size_t blockStartOffset, auto block) {
                    for (auto i : xrange(block.size())) {
                        auto objectIdx = i + blockStartOffset;
                        (*out) << "(catFeature=" << *catFeatureIdx << ',' << LabeledOutput(objectIdx)
                            << ").bin=" << ui32(block[i]) << Endl;
                    }
                }
            );
        }
    );
}


NCB::TExclusiveFeatureBundlesData::TExclusiveFeatureBundlesData(
    const NCB::TFeaturesLayout& featuresLayout,
    TVector<NCB::TExclusiveFeaturesBundle>&& metaData
)
    : MetaData(std::move(metaData))
    , SrcData(MetaData.size())
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


void NCB::TExclusiveFeatureBundlesData::GetSubsetWithScheduling(
    const TFeaturesArraySubsetIndexing* subsetIndexing,

    // needed only for sparse features
    const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
    TResourceConstrainedExecutor* resourceConstrainedExecutor,
    TExclusiveFeatureBundlesData* subsetData
) const {
    subsetData->FlatFeatureIndexToBundlePart = FlatFeatureIndexToBundlePart;
    subsetData->MetaData = MetaData;

    ::GetSubsetWithScheduling(
        MakeConstArrayRef(SrcData),
        subsetIndexing,
        subsetInvertedIndexing,
        resourceConstrainedExecutor,
        &(subsetData->SrcData)
    );
}


void NCB::TExclusiveFeatureBundlesData::Save(
    NPar::ILocalExecutor* localExecutor,
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
            savedColumnType == ESavedColumnType::Sparse,
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
    , SrcData(MetaData.size())
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


void NCB::TFeatureGroupsData::GetSubsetWithScheduling(
    const TFeaturesArraySubsetIndexing* subsetIndexing,

    // needed only for sparse features
    const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
    TResourceConstrainedExecutor* resourceConstrainedExecutor,
    TFeatureGroupsData* subsetData
) const {
    subsetData->FlatFeatureIndexToGroupPart = FlatFeatureIndexToGroupPart;
    subsetData->MetaData = MetaData;

    ::GetSubsetWithScheduling(
        MakeConstArrayRef(SrcData),
        subsetIndexing,
        subsetInvertedIndexing,
        resourceConstrainedExecutor,
        &(subsetData->SrcData)
    );

}


void NCB::TFeatureGroupsData::Save(
    NPar::ILocalExecutor* localExecutor,
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
            savedColumnType == ESavedColumnType::Sparse,
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

void NCB::TPackedBinaryFeaturesData::GetSubsetWithScheduling(
    const TFeaturesArraySubsetIndexing* subsetIndexing,

    // needed only for sparse features
    const TMaybe<TFeaturesArraySubsetInvertedIndexing>& subsetInvertedIndexing,
    TResourceConstrainedExecutor* resourceConstrainedExecutor,
    TPackedBinaryFeaturesData* subsetData
) const {
    subsetData->FlatFeatureIndexToPackedBinaryIndex = FlatFeatureIndexToPackedBinaryIndex;
    subsetData->PackedBinaryToSrcIndex = PackedBinaryToSrcIndex;

    ::GetSubsetWithScheduling(
        MakeConstArrayRef(SrcData),
        subsetIndexing,
        subsetInvertedIndexing,
        resourceConstrainedExecutor,
        &(subsetData->SrcData)
    );
}

void NCB::TPackedBinaryFeaturesData::Save(NPar::ILocalExecutor* localExecutor, IBinSaver* binSaver) const {
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
            savedColumnType == ESavedColumnType::Sparse,
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
               << srcIndex.FeatureType << ",FeatureIdx=" << srcIndex.FeatureIdx << Endl;
        }
        sb << Endl;
    }
    sb << "]\n";

    return sb;
}


void NCB::TQuantizedObjectsData::Load(
    const TArraySubsetIndexing<ui32>* subsetIndexing,
    const TFeaturesLayout& featuresLayout,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    IBinSaver* binSaver
) {
    PackedBinaryFeaturesData.Load(subsetIndexing, binSaver);
    ExclusiveFeatureBundlesData.Load(subsetIndexing, binSaver);
    FeaturesGroupsData.Load(subsetIndexing, binSaver);
    QuantizedFeaturesInfo = quantizedFeaturesInfo;
    LoadFeatures<EFeatureType::Float>(
        featuresLayout,
        subsetIndexing,
        &PackedBinaryFeaturesData,
        &ExclusiveFeatureBundlesData,
        &FeaturesGroupsData,
        binSaver,
        &FloatFeatures
    );
    LoadFeatures<EFeatureType::Categorical>(
        featuresLayout,
        subsetIndexing,
        &PackedBinaryFeaturesData,
        &ExclusiveFeatureBundlesData,
        &FeaturesGroupsData,
        binSaver,
        &CatFeatures
    );
    LoadMulti(binSaver, &CachedFeaturesCheckSum);
}


// if data is not packed or bundled or grouped - return empty holder
template <class TColumn>
static THolder<TColumn> GetAggregatedColumn(
    const TQuantizedObjectsData& data,
    ui32 flatFeatureIdx
) {
    const auto& bundlesData = data.ExclusiveFeatureBundlesData;
    if (auto bundleIndex = bundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx]) {
        const auto& bundleMetaData = bundlesData.MetaData[bundleIndex->BundleIdx];

        return MakeHolder<TBundlePartValuesHolderImpl<TColumn>>(
            flatFeatureIdx,
            bundlesData.SrcData[bundleIndex->BundleIdx].Get(),
            bundleMetaData.Parts[bundleIndex->InBundleIdx].Bounds
        );
    }

    const auto& packedBinaryData = data.PackedBinaryFeaturesData;
    if (auto packedBinaryIndex = packedBinaryData.FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx]) {
        return MakeHolder<TPackedBinaryValuesHolderImpl<TColumn>>(
            flatFeatureIdx,
            packedBinaryData.SrcData[packedBinaryIndex->PackIdx].Get(),
            packedBinaryIndex->BitIdx
        );
    }

    const auto& groupsData = data.FeaturesGroupsData;
    if (auto groupIndex = groupsData.FlatFeatureIndexToGroupPart[flatFeatureIdx]) {
        return MakeHolder<TFeaturesGroupPartValuesHolderImpl<TColumn>>(
            flatFeatureIdx,
            groupsData.SrcData[groupIndex->GroupIdx].Get(),
            groupIndex->InGroupIdx
        );
    }

    return nullptr;
}


NCB::TObjectsDataProviderPtr NCB::TQuantizedObjectsDataProvider::GetSubsetImpl(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
    ui64 cpuRamLimit,
    NPar::ILocalExecutor* localExecutor
) const {
    TCommonObjectsData subsetCommonData = CommonData.GetSubset(
        objectsGroupingSubset,
        localExecutor
    );

    TMaybe<TFeaturesArraySubsetInvertedIndexing> subsetInvertedIndexing;
    if (subsetCommonData.FeaturesLayout->HasSparseFeatures()) {
        subsetInvertedIndexing.ConstructInPlace(
            GetInvertedIndexing(objectsGroupingSubset.GetObjectsIndexing(), GetObjectCount(), localExecutor)
        );
    }

    if (ignoredFeatures.Defined()) {
        subsetCommonData.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(*subsetCommonData.FeaturesLayout);
        subsetCommonData.FeaturesLayout->IgnoreExternalFeatures(*ignoredFeatures);
    }

    auto resourceConstrainedExecutor = CreateCpuRamConstrainedExecutor(cpuRamLimit, localExecutor);

    TQuantizedObjectsData subsetData;

    auto getSubsetWithSchedulingForDataPart = [&] (const auto& srcData, auto* dstSubsetData) {
        srcData.GetSubsetWithScheduling(
            subsetCommonData.SubsetIndexing.Get(),
            subsetInvertedIndexing,
            &resourceConstrainedExecutor,
            dstSubsetData
        );
    };

    getSubsetWithSchedulingForDataPart(Data.PackedBinaryFeaturesData, &subsetData.PackedBinaryFeaturesData);
    getSubsetWithSchedulingForDataPart(Data.ExclusiveFeatureBundlesData, &subsetData.ExclusiveFeatureBundlesData);
    getSubsetWithSchedulingForDataPart(Data.FeaturesGroupsData, &subsetData.FeaturesGroupsData);

    resourceConstrainedExecutor.ExecTasks();

    auto getSubsetWithSchedulingForFeaturesPart = [&] (
        const auto& srcData,
        auto&& getPackedOrBundledData,
        auto* dstSubsetData) {

            ::GetSubsetWithScheduling(
                MakeConstArrayRef(srcData),
                subsetCommonData.SubsetIndexing.Get(),
                subsetInvertedIndexing,
                std::move(getPackedOrBundledData),
                &resourceConstrainedExecutor,
                dstSubsetData
            );
        };

    getSubsetWithSchedulingForFeaturesPart(
        Data.FloatFeatures,
        [&] (ui32 flatFeatureIdx) {
            return GetAggregatedColumn<IQuantizedFloatValuesHolder>(
                subsetData,
                flatFeatureIdx
            );
        },
        &subsetData.FloatFeatures
    );

    getSubsetWithSchedulingForFeaturesPart(
        Data.CatFeatures,
        [&] (ui32 flatFeatureIdx) {
            return GetAggregatedColumn<IQuantizedCatValuesHolder>(
                subsetData,
                flatFeatureIdx
            );
        },
        &subsetData.CatFeatures
    );

    getSubsetWithSchedulingForFeaturesPart(
        Data.TextFeatures,
        [] (ui32) { return nullptr; },
        &subsetData.TextFeatures
    );

    getSubsetWithSchedulingForFeaturesPart(
        Data.EmbeddingFeatures,
        [] (ui32) { return nullptr; },
        &subsetData.EmbeddingFeatures
    );

    resourceConstrainedExecutor.ExecTasks();

    subsetData.QuantizedFeaturesInfo = Data.QuantizedFeaturesInfo;

    return MakeIntrusive<TQuantizedObjectsDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetCommonData),
        std::move(subsetData),
        true,
        Nothing()
    );
}

template <class TColumn>
static void MakeConsecutiveIfDenseColumnDataWithScheduling(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    const TColumn& src,
    NPar::ILocalExecutor* localExecutor,
    TVector<std::function<void()>>* tasks,
    THolder<TColumn>* dst
) {
    if (!src.IsSparse()) {
        tasks->emplace_back(
            [&src, newSubsetIndexing, localExecutor, dst]() {
                TCloningParams cloningParams;
                cloningParams.MakeConsecutive = true;
                cloningParams.SubsetIndexing = newSubsetIndexing;
                *dst = DynamicHolderCast<TColumn>(
                    src.CloneWithNewSubsetIndexing(
                        cloningParams,
                        localExecutor
                    ),
                    "Column type changed after cloning"
                );
            }
        );
    } else {
        if (&src != dst->Get()) {
            using TSparseHolder = TSparseCompressedValuesHolderImpl<TColumn>;
            auto srcSparseCompressedValuesHolder = dynamic_cast<const TSparseHolder*>(&src);
            CB_ENSURE_INTERNAL(
                srcSparseCompressedValuesHolder != nullptr,
                "We expect TSparseCompressedValuesHolderImpl, got different type"
            );
            *dst = MakeHolder<TSparseHolder>(
                srcSparseCompressedValuesHolder->GetId(),
                TSparseCompressedArray<typename TColumn::TValueType, ui32>(
                    srcSparseCompressedValuesHolder->GetData()
                )
            );
        }
    }
}


template <EFeatureType FeatureType, class TColumn>
static void MakeConsecutiveIfDenseArrayFeatures(
    const TFeaturesLayout& featuresLayout,
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    const TVector<THolder<TColumn>>& src,
    const TExclusiveFeatureBundlesData& newExclusiveFeatureBundlesData,
    const TPackedBinaryFeaturesData& newPackedBinaryFeaturesData,
    const TFeatureGroupsData& newFeatureGroupsData,
    NPar::ILocalExecutor* localExecutor,
    TVector<THolder<TColumn>>* dst
) {
    if (&src != dst) {
        dst->clear();
        dst->resize(featuresLayout.GetFeatureCount(FeatureType));
    }

    TVector<std::function<void()>> tasks;

    featuresLayout.IterateOverAvailableFeatures<FeatureType>(
        [&] (TFeatureIdx<FeatureType> featureIdx) {
            const auto* srcColumn = src[*featureIdx].Get();

            if (auto maybeExclusiveFeaturesBundleIndex
                    = newExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[srcColumn->GetId()])
            {
                const auto& bundleMetaData
                    = newExclusiveFeatureBundlesData.MetaData[maybeExclusiveFeaturesBundleIndex->BundleIdx];

                (*dst)[*featureIdx] = MakeHolder<TBundlePartValuesHolderImpl<TColumn>>(
                    srcColumn->GetId(),
                    newExclusiveFeatureBundlesData.SrcData[maybeExclusiveFeaturesBundleIndex->BundleIdx].Get(),
                    bundleMetaData.Parts[maybeExclusiveFeaturesBundleIndex->InBundleIdx].Bounds
                );
            } else if (auto maybePackedBinaryIndex
                           = newPackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[srcColumn->GetId()])
            {
                (*dst)[*featureIdx] = MakeHolder<TPackedBinaryValuesHolderImpl<TColumn>>(
                    srcColumn->GetId(),
                    newPackedBinaryFeaturesData.SrcData[maybePackedBinaryIndex->PackIdx].Get(),
                    maybePackedBinaryIndex->BitIdx
                );
            } else if (auto maybeFeaturesGroupIndex
                                = newFeatureGroupsData.FlatFeatureIndexToGroupPart[srcColumn->GetId()])
            {
                (*dst)[*featureIdx] = MakeHolder<TFeaturesGroupPartValuesHolderImpl<TColumn>>(
                    srcColumn->GetId(),
                    newFeatureGroupsData.SrcData[maybeFeaturesGroupIndex->GroupIdx].Get(),
                    maybeFeaturesGroupIndex->InGroupIdx
                );

            } else {
                MakeConsecutiveIfDenseColumnDataWithScheduling(
                    newSubsetIndexing,
                    *srcColumn,
                    localExecutor,
                    &tasks,
                    &((*dst)[*featureIdx])
                );
            }
        }
    );

    ExecuteTasksInParallel(&tasks, localExecutor);
}



static void EnsureConsecutiveIfDenseExclusiveFeatureBundles(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    NPar::ILocalExecutor* localExecutor,
    NCB::TExclusiveFeatureBundlesData* exclusiveFeatureBundlesData
) {
    TVector<std::function<void()>> tasks;

    for (auto& srcDataElement : exclusiveFeatureBundlesData->SrcData) {
        MakeConsecutiveIfDenseColumnDataWithScheduling(
            newSubsetIndexing,
            *srcDataElement,
            localExecutor,
            &tasks,
            &srcDataElement
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
}


static void EnsureConsecutiveIfDensePackedBinaryFeatures(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    NPar::ILocalExecutor* localExecutor,
    TVector<THolder<IBinaryPacksArray>>* packedBinaryFeatures
) {
    TVector<std::function<void()>> tasks;

    for (auto& packedBinaryFeaturesPart : *packedBinaryFeatures) {
        MakeConsecutiveIfDenseColumnDataWithScheduling(
            newSubsetIndexing,
            *packedBinaryFeaturesPart,
            localExecutor,
            &tasks,
            &packedBinaryFeaturesPart
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
}


static void EnsureConsecutiveIfDenseFeatureGroups(
    const NCB::TFeaturesArraySubsetIndexing* newSubsetIndexing,
    NPar::ILocalExecutor* localExecutor,
    NCB::TFeatureGroupsData* featureGroupsData
) {
    TVector<std::function<void()>> tasks;

    for (auto& srcDataElement : featureGroupsData->SrcData) {
        MakeConsecutiveIfDenseColumnDataWithScheduling(
            newSubsetIndexing,
            *srcDataElement,
            localExecutor,
            &tasks,
            &srcDataElement
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
}


void NCB::TQuantizedObjectsDataProvider::EnsureConsecutiveIfDenseFeaturesData(
    NPar::ILocalExecutor* localExecutor
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
                EnsureConsecutiveIfDenseExclusiveFeatureBundles(
                    newSubsetIndexing.Get(),
                    localExecutor,
                    &Data.ExclusiveFeatureBundlesData
                );
            }
        );

        tasks.emplace_back(
            [&] () {
                EnsureConsecutiveIfDensePackedBinaryFeatures(
                    newSubsetIndexing.Get(),
                    localExecutor,
                    &Data.PackedBinaryFeaturesData.SrcData
                );
            }
        );

        tasks.emplace_back(
            [&] () {
                EnsureConsecutiveIfDenseFeatureGroups(
                    newSubsetIndexing.Get(),
                    localExecutor,
                    &Data.FeaturesGroupsData
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);
    }

    {
        TVector<std::function<void()>> tasks;

        tasks.emplace_back(
            [&] () {
                MakeConsecutiveIfDenseArrayFeatures<EFeatureType::Float>(
                    *GetFeaturesLayout(),
                    newSubsetIndexing.Get(),
                    Data.FloatFeatures,
                    Data.ExclusiveFeatureBundlesData,
                    Data.PackedBinaryFeaturesData,
                    Data.FeaturesGroupsData,
                    localExecutor,
                    &Data.FloatFeatures
                );
            }
        );
        tasks.emplace_back(
            [&] () {
                MakeConsecutiveIfDenseArrayFeatures<EFeatureType::Categorical>(
                    *GetFeaturesLayout(),
                    newSubsetIndexing.Get(),
                    Data.CatFeatures,
                    Data.ExclusiveFeatureBundlesData,
                    Data.PackedBinaryFeaturesData,
                    Data.FeaturesGroupsData,
                    localExecutor,
                    &Data.CatFeatures
                );
            }
        );

        ExecuteTasksInParallel(&tasks, localExecutor);
    }

    CommonData.SubsetIndexing = std::move(newSubsetIndexing);
}


template <class TColumn>
static void CheckFeaturesByType(
    EFeatureType featureType,
    // not TConstArrayRef to allow template parameter deduction
    const TVector<THolder<TColumn>>& data,
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
                = dynamic_cast<TPackedBinaryValuesHolderImpl<TColumn>*>(dataPtr);
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
                srcFeature.FeatureType == featureType,
                "packedBinaryToSrcIndex[" << linearPackedBinaryFeatureIdx << "] type is not "
                << featureType
            );
            CB_ENSURE_INTERNAL(
                srcFeature.FeatureIdx == featureIdx,
                "packedBinaryToSrcIndex[" << linearPackedBinaryFeatureIdx << "] feature index is not "
                << featureIdx
            );
        } else if (maybeBundlePart) {
            auto requiredTypePtr = dynamic_cast<TBundlePartValuesHolderImpl<TColumn>*>(dataPtr);
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
            const auto requiredTypePtr = dynamic_cast<TFeaturesGroupPartValuesHolderImpl<TColumn>*>(dataPtr);
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
        } else if (dataPtr->IsSparse()) {
            auto requiredTypePtr
                = dynamic_cast<TSparseCompressedValuesHolderImpl<TColumn>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "SparseValuesHolder"
            );
        } else {
            auto requiredTypePtr = dynamic_cast<TCompressedValuesHolderImpl<TColumn>*>(dataPtr);
            CB_ENSURE_INTERNAL(
                requiredTypePtr,
                "Data." << featureType << "Features[" << featureIdx << "] is not of type TQuantized"
                << featureTypeName << "ValuesHolder"
            );
        }
    }
}

void NCB::TQuantizedObjectsDataProvider::CheckCPUTrainCompatibility() const {
    CheckFeaturesByType(
        EFeatureType::Float,
        Data.FloatFeatures,
        Data.ExclusiveFeatureBundlesData,
        Data.PackedBinaryFeaturesData,
        Data.FeaturesGroupsData,
        "Float"
    );
    CheckFeaturesByType(
        EFeatureType::Categorical,
        Data.CatFeatures,
        Data.ExclusiveFeatureBundlesData,
        Data.PackedBinaryFeaturesData,
        Data.FeaturesGroupsData,
        "Cat"
    );
}


void NCB::TQuantizedObjectsDataProvider::CheckFeatureIsNotInAggregated(
    EFeatureType featureType,
    const TStringBuf featureTypeName,
    ui32 perTypeFeatureIdx
) const {
    const ui32 flatFeatureIdx = GetFeaturesLayout()->GetExternalFeatureIdx(perTypeFeatureIdx, featureType);
    CB_ENSURE_INTERNAL(
        !Data.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx],
        "Called TQuantizedObjectsDataProvider::GetNonPacked" << featureTypeName
        << "Feature for binary packed feature #" << flatFeatureIdx
    );
    CB_ENSURE_INTERNAL(
        !Data.ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx],
        "Called TQuantizedObjectsDataProvider::GetNonPacked" << featureTypeName
        << "Feature for bundled feature #" << flatFeatureIdx
    );
    CB_ENSURE_INTERNAL(
        !Data.FeaturesGroupsData.FlatFeatureIndexToGroupPart[flatFeatureIdx],
        "Called TQuantizedObjectsDataProvider::GetNonPacked" << featureTypeName
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
