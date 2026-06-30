#include "target.h"

#include "util.h"

#include <catboost/libs/helpers/parallel_tasks.h>

#include <library/cpp/binsaver/util_stream_io.h>

#include <util/generic/buffer.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/stream/buffer.h>
#include <util/stream/output.h>
#include <util/string/builder.h>
#include <util/system/compiler.h>

#include <cmath>
#include <functional>
#include <utility>


using namespace NCB;


static void CheckTarget(const TVector<TSharedVector<float>>& target, ui32 objectCount) {
    for (auto i : xrange(target.size())) {
        CheckDataSize(target[i]->size(), (size_t)objectCount, "Target[" + ToString(i) + "]", false);
    }
}

static void CheckContainsOnly01s(const ITypedSequence<float>& typedSequence) {
    typedSequence.ForEach(
        [] (float value) {
            CB_ENSURE_INTERNAL(
                (value == 0.0f) || (value == 1.0f),
                "targetType is Boolean but target values contain non-{0,1} data"
            );
        }
    );
}

static void CheckContainsOnlyIntegers(const ITypedSequence<float>& typedSequence) {
    typedSequence.ForEach(
        [] (float value) {
            float integerPart;
            CB_ENSURE_INTERNAL(
                std::modf(value, &integerPart) == 0.0f,
                "targetType is Integer but target values contain non-integer data"
            );
        }
    );
}


static void CheckRawTarget(ERawTargetType targetType, const TVector<TRawTarget>& target, ui32 objectCount) {
    CB_ENSURE_INTERNAL(
        !target.empty() || (targetType == ERawTargetType::None),
        "Target data is specified but targetType is None"
    );

    for (auto i : xrange(target.size())) {
        if (const ITypedSequencePtr<float>* typedSequence = std::get_if<ITypedSequencePtr<float>>(&target[i])) {
            CB_ENSURE_INTERNAL(
                (targetType == ERawTargetType::Boolean) ||
                (targetType == ERawTargetType::Float) ||
                (targetType == ERawTargetType::Integer),
                "target data contains float values but targetType is " << targetType
            );
            CheckDataSize(
                (*typedSequence)->GetSize(),
                objectCount,
                "Target[" + ToString(i) + "]",
                false
            );
            if (targetType == ERawTargetType::Boolean) {
                CheckContainsOnly01s(**typedSequence);
            } else if (targetType == ERawTargetType::Integer) {
                CheckContainsOnlyIntegers(**typedSequence);
            }
        } else {
            CB_ENSURE_INTERNAL(
                targetType == ERawTargetType::String,
                "target data contains float values but targetType is " << targetType
            );
            const TVector<TString>& stringVector = std::get<TVector<TString>>(target[i]);
            CheckDataSize(stringVector.size(), (size_t)objectCount, "Target[" + ToString(i) + "]", false);
            for (auto j : xrange(stringVector.size())) {
                CB_ENSURE(!stringVector[j].empty(), "Target[" << i << ", " << j << "] is empty");
            }
        }
    }
}


static void CheckOneBaseline(TConstArrayRef<float> baseline, size_t idx, ui32 objectCount) {
    CheckDataSize(
        baseline.size(),
        (size_t)objectCount,
        TStringBuilder() << "Baseline[" << idx << ']'
    );
}


static void CheckBaseline(
    const TVector<TSharedVector<float>>& baseline,
    ui32 objectCount,
    TMaybe<ui32> classCount
) {
    if (baseline.size() == 1) {
        CB_ENSURE_INTERNAL(
            !classCount || (*classCount == 2),
            "One-dimensional baseline with multiple classes"
        );
    } else {
        CB_ENSURE_INTERNAL(classCount, "Multidimensional baseline for non-multiclassification");
        CheckDataSize(baseline.size(), (size_t)*classCount, "Baseline", true, "class count");
    }

    for (auto i : xrange(baseline.size())) {
        CheckOneBaseline(*(baseline[i]), i, objectCount);
    }
}

// groupWeights is indexed by objectIdx
void NCB::CheckGroupWeights(
    TConstArrayRef<float> groupWeights,
    const TObjectsGrouping& objectsGrouping
) {
    CheckDataSize(groupWeights.size(), (size_t)objectsGrouping.GetObjectCount(), "GroupWeights");
    if (!objectsGrouping.IsTrivial()) {
        TConstArrayRef<TGroupBounds> groupsBounds = objectsGrouping.GetNonTrivialGroups();
        for (auto groupBounds : groupsBounds) {
            if (groupBounds.GetSize()) {
                float groupWeight = groupWeights[groupBounds.Begin];
                CB_ENSURE(groupWeight >= 0.f, "groupWeight[" << groupBounds.Begin << "] is negative");
                for (auto objectIdx : xrange(groupBounds.Begin + 1, groupBounds.End)) {
                    CB_ENSURE(
                        FuzzyEquals(groupWeight, groupWeights[objectIdx]),
                        "groupWeight[" << objectIdx << "] = " << groupWeights[objectIdx]
                        << " is not equal to the weight of group's first element "
                        << " (groupWeight[" << groupBounds.Begin << "] = " << groupWeight << ')'
                    );
                }
            }
        }
    }
}

void NCB::CheckPairs(TConstArrayRef<TPair> pairs, const TObjectsGrouping& objectsGrouping) {
    size_t objectCount = objectsGrouping.GetObjectCount();
    for (auto pairIdx : xrange(pairs.size())) {
        const auto& pair = pairs[pairIdx];
        try {
            CB_ENSURE(pair.WinnerId != pair.LoserId, "WinnerId is equal to LoserId");
            CB_ENSURE(pair.WinnerId < objectCount, "WinnerId is not less than object count");
            CB_ENSURE(pair.LoserId < objectCount, "LoserId is not less than object count");
            CB_ENSURE(pair.Weight >= 0.0f, "Weight is negative");

            if (!objectsGrouping.IsTrivial()) {
                ui32 winnerIdGroupIdx = objectsGrouping.GetGroupIdxForObject(pair.WinnerId);
                ui32 loserIdGroupIdx = objectsGrouping.GetGroupIdxForObject(pair.LoserId);

                CB_ENSURE(
                    winnerIdGroupIdx == loserIdGroupIdx,
                    "winner id group #" << winnerIdGroupIdx << " is not equal to loser id group #"
                    << loserIdGroupIdx << " (group ids are numbered from 0 to group count - 1 according to their appearance in dataset)"
                );
            }
        } catch (const TCatBoostException& e) {
            // throw, not ythrow to avoid duplication of line info
            throw TCatBoostException() << "Pair #" << pairIdx << ' ' << HumanReadableDescription(pair) << ": "
                << e.what();
        }
    }
}

static void CheckPairs(TConstArrayRef<TPairInGroup> pairs, const TObjectsGrouping& objectsGrouping) {
    CB_ENSURE(
        !objectsGrouping.IsTrivial(),
        "Pairs in groups are specified, but there's no group info in dataset"
    );

    for (auto pairIdx : xrange(pairs.size())) {
        const auto& pairInGroup = pairs[pairIdx];
        try {
            CB_ENSURE(
                pairInGroup.GroupIdx < objectsGrouping.GetGroupCount(),
                "GroupIdx is not less than total number of groups (" << objectsGrouping.GetGroupCount() << ')'
            );

            CB_ENSURE(
                pairInGroup.WinnerIdxInGroup != pairInGroup.LoserIdxInGroup,
                "WinnerIdxInGroup is equal to LoserIdxInGroup"
            );
            ui32 groupSize = objectsGrouping.GetGroup(pairInGroup.GroupIdx).GetSize();
            auto checkIdx = [&] (auto idx, TStringBuf fieldName) {
                CB_ENSURE(
                    idx < groupSize,
                    fieldName << " (" << idx << ") > group size (" << groupSize << ')'
                );
            };
            checkIdx(pairInGroup.WinnerIdxInGroup, TStringBuf("WinnerIdxInGroup"));
            checkIdx(pairInGroup.LoserIdxInGroup, TStringBuf("LoserIdxInGroup"));

            CB_ENSURE(pairInGroup.Weight >= 0.0f, "Weight is negative");
        } catch (const TCatBoostException& e) {
            // throw, not ythrow to avoid duplication of line info
            throw TCatBoostException() << "Pair #" << pairIdx << ' ' << pairInGroup << ": "
                << e.what();
        }
    }
}


void NCB::CheckOneGroupInfo(const TQueryInfo& groupInfo) {
    CB_ENSURE_INTERNAL(groupInfo.Begin <= groupInfo.End, "CheckOneGroupInfo: Begin > End");
    CB_ENSURE_INTERNAL(groupInfo.Weight >= 0.0f, "CheckOneGroupInfo: Weight is negative");
    if (!groupInfo.SubgroupId.empty()) {
        CB_ENSURE_INTERNAL(
            groupInfo.SubgroupId.size() == (size_t)groupInfo.GetSize(),
            "CheckOneGroupInfo: SubgroupId.size() is not equal to group size"
        );
    }
    if (!groupInfo.Competitors.empty()) {
        CB_ENSURE_INTERNAL(
            groupInfo.Competitors.size() == (size_t)groupInfo.GetSize(),
            "CheckOneGroupInfo: Competitors.size() is not equal to group size"
        );
        for (auto competitorIdx1 : xrange(groupInfo.Competitors.size())) {
            for (auto competitorIdx2 : xrange(groupInfo.Competitors[competitorIdx1].size())) {
                const auto& competitor = groupInfo.Competitors[competitorIdx1][competitorIdx2];
                CB_ENSURE_INTERNAL(
                   competitor.Id < groupInfo.GetSize(),
                    "CheckOneGroupInfo: competitor[" << competitorIdx1 << "][" << competitorIdx2 << "]"
                    << ".Id (" << competitor.Id << ") is not less than group size ("
                    << groupInfo.GetSize() << ')'
                );
                CB_ENSURE_INTERNAL(
                    (size_t)competitor.Id != competitorIdx1,
                    "CheckOneGroupInfo: competitor[" << competitorIdx1 << "][" << competitorIdx2 << "]"
                    << ".Id is equal to its first index"
                );
                CB_ENSURE_INTERNAL(
                    competitor.Weight >= 0.0f,
                    "CheckOneGroupInfo: competitor[" << competitorIdx1 << "][" << competitorIdx2 << "]"
                    ".Weight is negative"
                );
                CB_ENSURE_INTERNAL(
                    competitor.SampleWeight >= 0.0f,
                    "CheckOneGroupInfo: competitor[" << competitorIdx1 << "][" << competitorIdx2 << "]"
                    ".SampleWeight is negative"
                );
            }
        }
    }
}

void NCB::CheckGroupInfo(
    TConstArrayRef<TQueryInfo> groupInfoVector,
    const TObjectsGrouping& objectsGrouping,
    bool mustContainPairData
) {
    CheckDataSize(
        groupInfoVector.size(),
        (size_t)objectsGrouping.GetGroupCount(),
        "groupInfo",
        false,
        "group count",
        true
    );

    bool hasPairData = false;

    for (auto i : xrange(groupInfoVector.size())) {
        const auto& groupInfo = groupInfoVector[i];
        try {
            CB_ENSURE_INTERNAL(
                (TGroupBounds)groupInfo == objectsGrouping.GetGroup(i),
                "bounds " << ((const TGroupBounds&) groupInfo)
                << " are not equal to grouping's corresponding group bounds: "
                << objectsGrouping.GetGroup(i)
            );
            CheckOneGroupInfo(groupInfo);
            if (!groupInfo.Competitors.empty()) {
                hasPairData = true;
            }
        } catch (const TCatBoostException& e) {
            // throw, not ythrow to avoid duplication of line info
            throw TCatBoostException() << "groupInfo[" << i << "]: " << e.what();
        }
    }
    CB_ENSURE_INTERNAL(!mustContainPairData || hasPairData, "groups do not contain pair data");
}

static void CheckGroupWeights(const TWeights<float>& groupWeights, const TObjectsGrouping& objectsGrouping) {
    if (groupWeights.IsTrivial()) {
        CheckDataSize(groupWeights.GetSize(), objectsGrouping.GetObjectCount(), "GroupWeights");
    } else {
        CheckGroupWeights(groupWeights.GetNonTrivialData(), objectsGrouping);
    }
}


bool EqualAsFloatTarget(const ITypedSequencePtr<float>& lhs, const TVector<TString>& rhs) {
    bool haveUnequalElements = false;
    size_t i = 0;
    lhs->ForEach(
        [&] (float lhsElement) {
            if (!FuzzyEquals(lhsElement, FromString<float>(rhs[i++]))) {
                haveUnequalElements = true;
            }
        }
    );
    return !haveUnequalElements;
}

bool Equal(const TRawTarget& lhs, const TRawTarget& rhs) {
    if (const ITypedSequencePtr<float>* lhsTypedSequence = std::get_if<ITypedSequencePtr<float>>(&lhs)) {
        if (const ITypedSequencePtr<float>* rhsTypedSequence = std::get_if<ITypedSequencePtr<float>>(&rhs)) {
            return (*lhsTypedSequence)->EqualTo(**rhsTypedSequence, /*strict*/ false);
        } else {
            return EqualAsFloatTarget(*lhsTypedSequence, std::get<TVector<TString>>(rhs));
        }
    } else {
        const TVector<TString>& lhsStringVector = std::get<TVector<TString>>(lhs);
        if (const TVector<TString>* rhsStringVector = std::get_if<TVector<TString>>(&rhs)) {
            return lhsStringVector == *rhsStringVector;
        } else {
            return EqualAsFloatTarget(std::get<ITypedSequencePtr<float>>(rhs), lhsStringVector);
        }
    }
}

bool Equal(const TVector<TRawTarget>& lhs, const TVector<TRawTarget>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (auto i : xrange(lhs.size())) {
        if (!Equal(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}


bool TRawTargetData::operator==(const TRawTargetData& rhs) const {
    return (TargetType == rhs.TargetType) && Equal(Target, rhs.Target) && (Baseline == rhs.Baseline) &&
        (Weights == rhs.Weights) && (GroupWeights == rhs.GroupWeights) &&
        Equal(Pairs, rhs.Pairs, EqualWithoutOrder);
}


void TRawTargetData::Check(
    const TObjectsGrouping& objectsGrouping,
    NPar::ILocalExecutor* localExecutor
) const {
    ui32 objectCount = objectsGrouping.GetObjectCount();

    // Weights's values have been already checked in it's constructor
    CheckDataSize(Weights.GetSize(), objectCount, "Weights");


    TVector<std::function<void()>> tasks;

    tasks.emplace_back(
        [&, this]() {
            CheckRawTarget(TargetType, Target, objectCount);
        }
    );

    for (auto i : xrange(Baseline.size())) {
        tasks.emplace_back(
            [&, i, this]() {
                CheckOneBaseline(Baseline[i], i, objectCount);
            }
        );
    }

    tasks.emplace_back(
        [&, this]() {
            // GroupWeights's values have been already checked for non-negativeness in it's constructor
            CheckGroupWeights(GroupWeights, objectsGrouping);
        }
    );

    if (Pairs) {
        tasks.emplace_back(
            [&, this]() {
            std::visit(
                    [&] (const auto& pairsData) {
                        CheckPairs(pairsData, objectsGrouping);
                    },
                    *Pairs
                );
            }
        );
    }
    if (Graph) {
        tasks.emplace_back(
            [&, this]() {
                std::visit(
                    [&] (const auto& pairsData) {
                        CheckPairs(pairsData, objectsGrouping);
                    },
                    *Graph
                );
            }
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);
}


void TRawTargetData::PrepareForInitialization(
    const TDataMetaInfo& metaInfo,
    ui32 objectCount,
    ui32 prevTailSize
) {
    TargetType = metaInfo.TargetType;

    // Target is properly initialized at the end of building
    Target.resize(metaInfo.TargetCount);

    Baseline.resize(metaInfo.BaselineCount);
    for (auto& dim : Baseline) {
        NCB::PrepareForInitialization(objectCount, prevTailSize, &dim);
    }

    // if weights are not trivial reset at the end of building
    // here they are set to trivial to clear previous buffers
    SetTrivialWeights(objectCount);

    Pairs.Clear();
}


ERawTargetType TRawTargetDataProvider::GetTargetType() const noexcept {
    return Data.TargetType;
}

void TRawTargetDataProvider::GetNumericTarget(TArrayRef<TArrayRef<float>> dst) const {
    CB_ENSURE(dst.size() == Data.Target.size());
    for (auto targetIdx : xrange(Data.Target.size())) {
        ToArray(*std::get<ITypedSequencePtr<float>>(Data.Target[targetIdx]), dst[targetIdx]);
    }
}

void TRawTargetDataProvider::GetStringTargetRef(TVector<TConstArrayRef<TString>>* dst) const {
    dst->resize(Data.Target.size());
    for (auto targetIdx : xrange(Data.Target.size())) {
        (*dst)[targetIdx] = std::get<TVector<TString>>(Data.Target[targetIdx]);
    }
}


void TRawTargetDataProvider::SetObjectsGrouping(TObjectsGroupingPtr objectsGrouping) {
    CheckDataSize(objectsGrouping->GetObjectCount(), GetObjectCount(), "new objects grouping objects'");
    CB_ENSURE(
        Data.GroupWeights.IsTrivial(),
        "Cannot update objects grouping if target data already has non-trivial group weights"
    );
    CB_ENSURE(
        !Data.Pairs,
        "Cannot update objects grouping if target data already has pairs"
    );
    ObjectsGrouping = objectsGrouping;
}



void TRawTargetDataProvider::SetBaseline(TConstArrayRef<TConstArrayRef<float>> baseline) {
    ui32 objectCount = GetObjectCount();
    TVector<TVector<float>> newBaselineStorage(baseline.size());
    for (auto i : xrange(baseline.size())) {
        CheckOneBaseline(baseline[i], i, objectCount);
        Assign(baseline[i], &newBaselineStorage[i]);
    }
    Data.Baseline = std::move(newBaselineStorage);

    SetBaselineViewFromBaseline();
}

static void GetRawTargetSubset(
    const TRawTarget& src,
    const TArraySubsetIndexing<ui32>& subset,
    NPar::ILocalExecutor* localExecutor,
    TRawTarget* dst
) {
    if (const ITypedSequencePtr<float>* srcTypedSequence = std::get_if<ITypedSequencePtr<float>>(&src)) {
        ITypedArraySubsetPtr<float> typedArraySubset = (*srcTypedSequence)->GetSubset(&subset);
        TVector<float> dstData;
        dstData.yresize(subset.Size());
        TArrayRef<float> dstDataRef = dstData;
        typedArraySubset->ParallelForEach(
            [dstDataRef] (ui32 idx, float value) { dstDataRef[idx] = value; },
            localExecutor
        );
        (*dst) = (ITypedSequencePtr<float>)MakeIntrusive<TTypeCastArrayHolder<float, float>>(
            std::move(dstData)
        );
    } else {
        (*dst) = GetSubset<TString>(std::get<TVector<TString>>(src), subset, localExecutor);
    }
}


static void GetMultidimBaselineSubset(
    const TVector<TVector<float>>& src,
    const TArraySubsetIndexing<ui32>& subset,
    NPar::ILocalExecutor* localExecutor,
    TVector<TVector<float>>* dst
) {
    if (src.empty()) {
        dst->clear();
    } else {
        dst->resize(src.size());

        localExecutor->ExecRangeWithThrow(
            [&] (int baselineIdx) {
                (*dst)[baselineIdx] = NCB::GetSubset<float>(src[baselineIdx], subset, localExecutor);
            },
            0,
            SafeIntegerCast<int>(src.size()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

static TFlatPairsInfo GetPairsSubset(
    TConstArrayRef<TPair> pairs,
    const TObjectsGrouping& objectsGrouping,
    const TObjectsGroupingSubset& objectsGroupingSubset
) {
    TVector<TMaybe<ui32>> srcToDstObjectIndices(objectsGrouping.GetObjectCount());
    objectsGroupingSubset.GetObjectsIndexing().ForEach(
        [&srcToDstObjectIndices] (ui32 idx, ui32 srcIdx) { srcToDstObjectIndices[srcIdx] = idx; }
    );

    TFlatPairsInfo result;
    for (const auto& pair : pairs) {
        const auto& maybeDstWinnerId = srcToDstObjectIndices[pair.WinnerId];
        if (!maybeDstWinnerId) {
            continue;
        }
        const auto& maybeDstLoserId = srcToDstObjectIndices[pair.LoserId];
        if (!maybeDstLoserId) {
            continue;
        }
        result.emplace_back(*maybeDstWinnerId, *maybeDstLoserId, pair.Weight);
    }
    return result;
}


struct TSrcToDstGroupMap {
    ui32 DstGroupIdx;
    TVector<TMaybe<ui32>> InGroupIndicesMap; // srcInGroupIdx -> dstInGroupIdx
};

static TGroupedPairsInfo GetPairsSubset(
    TConstArrayRef<TPairInGroup> pairs,
    const TObjectsGrouping& objectsGrouping,
    const TObjectsGroupingSubset& objectsGroupingSubset
) {
    TVector<TMaybe<ui32>> srcToDstObjectIndices(objectsGrouping.GetObjectCount());
    objectsGroupingSubset.GetObjectsIndexing().ForEach(
        [&srcToDstObjectIndices] (ui32 idx, ui32 srcIdx) { srcToDstObjectIndices[srcIdx] = idx; }
    );

    const TObjectsGrouping& subsetGrouping = *objectsGroupingSubset.GetSubsetGrouping();
    TVector<TMaybe<TSrcToDstGroupMap>> srcToDstGroupMaps(objectsGrouping.GetObjectCount()); // [groupIdx]
    objectsGroupingSubset.GetGroupsIndexing().ForEach(
        [&] (ui32 groupIdx, ui32 srcGroupIdx) {
            TSrcToDstGroupMap srcToDstGroupMap;
            srcToDstGroupMap.DstGroupIdx = groupIdx;

            TGroupBounds srcGroupBounds = objectsGrouping.GetGroup(srcGroupIdx);
            ui32 dstGroupStartIdx = subsetGrouping.GetGroup(groupIdx).Begin;

            srcToDstGroupMap.InGroupIndicesMap.resize(srcGroupBounds.GetSize());
            for (ui32 srcInGroupIdx : xrange(srcGroupBounds.GetSize())) {
                ui32 srcIdx = srcGroupBounds.Begin + srcInGroupIdx;
                TMaybe<ui32> maybeDstObjectIdx = srcToDstObjectIndices[srcIdx];
                if (maybeDstObjectIdx) {
                    srcToDstGroupMap.InGroupIndicesMap[srcInGroupIdx] = *maybeDstObjectIdx - dstGroupStartIdx;
                }
            }
            srcToDstGroupMaps[srcGroupIdx] = std::move(srcToDstGroupMap);
        }
    );

    TGroupedPairsInfo result;
    for (const auto& pair : pairs) {
        const TMaybe<TSrcToDstGroupMap>& srcToDstGroupMap = srcToDstGroupMaps[pair.GroupIdx];
        if (!srcToDstGroupMap) {
            continue;
        }
        const auto& maybeDstWinnerIdxInGroup = srcToDstGroupMap->InGroupIndicesMap[pair.WinnerIdxInGroup];
        if (!maybeDstWinnerIdxInGroup) {
            continue;
        }
        const auto& maybeDstLoserIdxInGroup = srcToDstGroupMap->InGroupIndicesMap[pair.LoserIdxInGroup];
        if (!maybeDstLoserIdxInGroup) {
            continue;
        }
        result.push_back(
            TPairInGroup{
                srcToDstGroupMap->DstGroupIdx,
                *maybeDstWinnerIdxInGroup,
                *maybeDstLoserIdxInGroup,
                pair.Weight
            }
        );
    }
    return result;
}


static void GetPairsSubset(
    // assumes pairs and objectsGrouping consistency has already been checked
    const TRawPairsData& pairs,
    const TObjectsGrouping& objectsGrouping,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    TRawPairsData* result
) {
    if (std::holds_alternative<TFullSubset<ui32>>(objectsGroupingSubset.GetObjectsIndexing())) {
        *result = pairs;
        return;
    }
    std::visit(
        [&] (const auto& pairs) {
            *result = GetPairsSubset(pairs, objectsGrouping, objectsGroupingSubset);
        },
        pairs
    );
}


TRawTargetDataProvider TRawTargetDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor
) const {
    const TArraySubsetIndexing<ui32>& objectsSubsetIndexing = objectsGroupingSubset.GetObjectsIndexing();

    TRawTargetData subsetData;
    subsetData.TargetType = Data.TargetType;

    TVector<std::function<void()>> tasks;

    if (TMaybeData<TConstArrayRef<TRawTarget>> maybeTarget = GetTarget()) {
        TConstArrayRef<TRawTarget> target = *maybeTarget;
        subsetData.Target.resize(target.size());
        for (auto targetIdx : xrange(target.size())) {
            tasks.emplace_back(
                [&, target, targetIdx]() {
                    GetRawTargetSubset(
                        target[targetIdx],
                        objectsSubsetIndexing,
                        localExecutor,
                        &subsetData.Target[targetIdx]
                    );
                }
            );
        }
    }

    tasks.emplace_back(
        [&, this]() {
            GetMultidimBaselineSubset(
                Data.Baseline,
                objectsSubsetIndexing,
                localExecutor,
                &subsetData.Baseline
            );
        }
    );

    tasks.emplace_back(
        [&, this]() {
            subsetData.Weights = Data.Weights.GetSubset(objectsSubsetIndexing, localExecutor);
        }
    );

    tasks.emplace_back(
        [&, this]() {
            subsetData.GroupWeights = Data.GroupWeights.GetSubset(objectsSubsetIndexing, localExecutor);
        }
    );

    if (Data.Pairs) {
        tasks.emplace_back(
            [&, this]() {
                TRawPairsData subsetPairs;
                GetPairsSubset(*Data.Pairs, *ObjectsGrouping, objectsGroupingSubset, &subsetPairs);
                subsetData.Pairs = std::move(subsetPairs);
            }
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);

    return TRawTargetDataProvider(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetData),
        true,
        ForceUnitAutoPairWeights,
        nullptr
    );
}

void TRawTargetDataProvider::AssignWeights(TConstArrayRef<float> src, TWeights<float>* dst) {
    TVector<float> storage;
    Assign(src, &storage);
    *dst = TWeights<float>(
        GetObjectCount(),
        TMaybeOwningArrayHolder<float>::CreateOwning(std::move(storage)),
        true
    );
}


template <class TKey, class TSharedDataPtr>
using TTargetSingleTypeDataCache = THashMap<TKey, TSharedDataPtr>;

/*
 * Serialization using IBinSaver works as following (somewhat similar to subset creation):
 *
 * Save:
 *  1) Collect mapping of unique data ids to data itself in TSerializationTargetDataCache.
 *     Serialize TTargetDataProvider with these data ids instead of actual data
 *     to a temporary binSaver stream.
 *  2) Save data in TSerializationTargetDataCache. Save binSaver stream with data provider
 *     description (created at stage 1) with ids after it.
 *
 *  Load:
 *  1) Load data in TSerializationTargetDataCache.
 *  2) Read data with data provider
 *     description with ids, create actual TTargetDataProvider, initializing with actual shared data loaded
 *     from cache at stage 1.
 *
 */

// key value 0 is special - means this field is optional while serializing
template <class TSharedDataPtr>
using TSerializationTargetSingleTypeDataCache =
    TTargetSingleTypeDataCache<ui64, TSharedDataPtr>;

struct TSerializationTargetDataCache {
    TSerializationTargetSingleTypeDataCache<TSharedVector<float>> Targets;
    TSerializationTargetSingleTypeDataCache<TSharedWeights<float>> Weights;

    // multidim baselines are stored as separate pointers for simplicity
    TSerializationTargetSingleTypeDataCache<TSharedVector<float>> Baselines;
    TSerializationTargetSingleTypeDataCache<TSharedVector<TQueryInfo>> GroupInfos;

public:
    SAVELOAD_WITH_SHARED(Targets, Weights, Baselines, GroupInfos)
};


bool TProcessedTargetData::operator==(const TProcessedTargetData& rhs) const {
    auto compareHashMaps = [&] (const auto& lhs, const auto& rhs, const auto& dataComparator) {
        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (const auto& [name, lhsData] : lhs) {
            auto rhsDataPtr = MapFindPtr(rhs, name);
            if (!rhsDataPtr) {
                return false;
            }
            if (!dataComparator(lhsData, *rhsDataPtr)) {
                return false;
            }
        }
        return true;
    };

    if (!compareHashMaps(
            TargetsClassCount,
            rhs.TargetsClassCount,
            [](const auto& lhs, const auto& rhs) { return lhs == rhs; }))
    {
        return false;
    }

    if (!compareHashMaps(
            Targets,
            rhs.Targets,
            [](const auto& lhs, const auto& rhs) {
                if (lhs.size() != rhs.size()) {
                    return false;
                }
                for (auto i : xrange(lhs.size())) {
                    if (*(lhs[i]) != *(rhs[i])) {
                        return false;
                    }
                }
                return true;
            }
        ))
    {
        return false;
    }

    if (!compareHashMaps(
            Weights,
            rhs.Weights,
            [](const auto& lhs, const auto& rhs) { return *lhs == *rhs; }))
    {
        return false;
    }

    if (!compareHashMaps(
            Baselines,
            rhs.Baselines,
            [](const TVector<TSharedVector<float>>& lhs, const TVector<TSharedVector<float>>& rhs) {
                if (lhs.size() != rhs.size()) {
                    return false;
                }
                for (auto i : xrange(lhs.size())) {
                    if (*(lhs[i]) != *(rhs[i])) {
                        return false;
                    }
                }
                return true;
            }
        ))
    {
        return false;
    }

    if (!compareHashMaps(
            GroupInfos,
            rhs.GroupInfos,
            [](const auto& lhs, const auto& rhs) { return *lhs == *rhs; }))
    {
        return false;
    }

    return true;
}


void TProcessedTargetData::Check(const TObjectsGrouping& objectsGrouping) const {
    const ui32 objectCount = objectsGrouping.GetObjectCount();

    for (const auto& [name, classCount] : TargetsClassCount) {
        CB_ENSURE_INTERNAL(classCount > 0, "Class count for target " << name << " must be non-0");
        CB_ENSURE_INTERNAL(Targets.contains(name), "No data in Targets with name " << name);
    }

    for (const auto& [name, targetsData] : Targets) {
        CheckTarget(targetsData, objectCount);
    }
    for (const auto& [name, weightsData] : Weights) {
        CheckDataSize(weightsData->GetSize(), objectCount, "Weights " + name);
    }
    for (const auto& [name, baselineData] : Baselines) {
        try {
            CB_ENSURE_INTERNAL(!baselineData.empty(), "empty");
            auto* classCount = MapFindPtr(TargetsClassCount, name);
            CheckBaseline(baselineData, objectCount, classCount ? TMaybe<ui32>(*classCount) : Nothing());
        } catch (TCatBoostException& e) {
            throw TCatBoostException() << "Baseline data " << name << ": " << e.what();
        }
    }
    for (const auto& [name, groupInfosData] : GroupInfos) {
        CheckGroupInfo(*groupInfosData, objectsGrouping, false);
    }
}


template <class TSharedDataPtr>
static void LoadWithCache(
    const TSerializationTargetSingleTypeDataCache<TSharedDataPtr>& cachePart,
    IBinSaver* binSaver,
    THashMap<TString, TSharedDataPtr>* data
) {
    ui32 dataCount = 0;
    LoadMulti(binSaver, &dataCount);
    for (ui32 dataIdx : xrange(dataCount)) {
        Y_UNUSED(dataIdx);

        TString name;
        ui64 id;
        LoadMulti(binSaver, &name, &id);
        data->emplace(name, cachePart.at(id));
    }
}

void TProcessedTargetData::Load(IBinSaver* binSaver) {
    TSerializationTargetDataCache cache;
    LoadMulti(binSaver, &cache);

    LoadMulti(binSaver, &TargetsClassCount);
    LoadWithCache(cache.Weights, binSaver, &Weights);

    ui32 targetCount = 0;
    LoadMulti(binSaver, &targetCount);
    for (ui32 targetIdx : xrange(targetCount)) {
        Y_UNUSED(targetIdx);

        TString name;
        ui32 dimensionCount;
        LoadMulti(binSaver, &name, &dimensionCount);

        TVector<TSharedVector<float>> target;
        target.reserve(dimensionCount);
        for (ui32 dimensionIdx : xrange(dimensionCount)) {
            Y_UNUSED(dimensionIdx);

            ui64 id;
            LoadMulti(binSaver, &id);
            target.push_back(cache.Targets.at(id));
        }

        Targets.emplace(name, std::move(target));
    }

    ui32 baselineCount = 0;
    LoadMulti(binSaver, &baselineCount);
    for (ui32 baselineIdx : xrange(baselineCount)) {
        Y_UNUSED(baselineIdx);

        TString name;
        ui32 dimensionCount;
        LoadMulti(binSaver, &name, &dimensionCount);

        TVector<TSharedVector<float>> baseline;
        baseline.reserve(dimensionCount);
        for (ui32 dimensionIdx : xrange(dimensionCount)) {
            Y_UNUSED(dimensionIdx);

            ui64 id;
            LoadMulti(binSaver, &id);
            baseline.push_back(cache.Baselines.at(id));
        }

        Baselines.emplace(name, std::move(baseline));
    }

    LoadWithCache(cache.GroupInfos, binSaver, &GroupInfos);
}


template <class TSharedDataPtr>
inline void AddToCacheAndSaveId(
    const TSharedDataPtr sharedData,
    IBinSaver* binSaver,
    TSerializationTargetSingleTypeDataCache<TSharedDataPtr>* cache
) {
    const ui64 id = reinterpret_cast<ui64>(sharedData.Get());
    SaveMulti(binSaver, id);
    if (id) {
        cache->emplace(id, sharedData);
    }
}

template <class TSharedDataPtr>
static void SaveWithCache(
    const THashMap<TString, TSharedDataPtr>& data,
    IBinSaver* binSaver,
    TSerializationTargetSingleTypeDataCache<TSharedDataPtr>* cachePart
) {
    SaveMulti(binSaver, SafeIntegerCast<ui32>(data.size()));
    for (const auto& [name, dataPart] : data) {
        SaveMulti(binSaver, name);
        AddToCacheAndSaveId(dataPart, binSaver, cachePart);
    }
}

void TProcessedTargetData::Save(IBinSaver* binSaver) const {
    TSerializationTargetDataCache cache;

    /* For serialization we have get serialized target data with ids first before cache is filled
     * For deserialization we get cache filled first and then create target data from it
     *  so we need to change the order of these parts while saving
     */
    TBuffer serializedTargetDataWithIds;

    {
        TBufferOutput out(serializedTargetDataWithIds);
        TYaStreamOutput out2(out);
        IBinSaver targetDataWithIdsBinSaver(out2, false);

        SaveMulti(&targetDataWithIdsBinSaver, TargetsClassCount);
        SaveWithCache(Weights, &targetDataWithIdsBinSaver, &cache.Weights);

        {
            SaveMulti(&targetDataWithIdsBinSaver, SafeIntegerCast<ui32>(Targets.size()));
            for (const auto& [name, dataPart] : Targets) {
                const ui32 dimensionCount = dataPart.size();
                SaveMulti(&targetDataWithIdsBinSaver, name, dimensionCount);
                for (const auto& oneTarget : dataPart) {
                    AddToCacheAndSaveId(oneTarget, &targetDataWithIdsBinSaver, &cache.Targets);
                }
            }
        }

        {
            SaveMulti(&targetDataWithIdsBinSaver, SafeIntegerCast<ui32>(Baselines.size()));
            for (const auto& [name, dataPart] : Baselines) {
                const ui32 dimensionCount = dataPart.size();
                SaveMulti(&targetDataWithIdsBinSaver, name, dimensionCount);
                for (const auto& oneBaseline : dataPart) {
                    AddToCacheAndSaveId(oneBaseline, &targetDataWithIdsBinSaver, &cache.Baselines);
                }
            }
        }

        SaveWithCache(GroupInfos, &targetDataWithIdsBinSaver, &cache.GroupInfos);
    }

    SaveMulti(binSaver, cache);

    SaveArrayData(
        TConstArrayRef<ui8>((ui8*)serializedTargetDataWithIds.Data(), serializedTargetDataWithIds.Size()),
        binSaver
    );
}


TTargetDataProvider::TTargetDataProvider(
    TObjectsGroupingPtr objectsGrouping,
    TProcessedTargetData&& processedTargetData,
    bool skipCheck
) {
    if (!skipCheck) {
        processedTargetData.Check(*objectsGrouping);
    }
    ObjectsGrouping = std::move(objectsGrouping);
    Data = std::move(processedTargetData);

    for (const auto& [name, baselineData] : Data.Baselines) {
        TVector<TConstArrayRef<float>> baselineView(baselineData.size());
        for (auto i : xrange(baselineData.size())) {
            baselineView[i] = *(baselineData[i]);
        }
        BaselineViews.emplace(name, std::move(baselineView));
    }

    for (const auto& [name, targetData] : Data.Targets) {
        TVector<TConstArrayRef<float>> targetView(targetData.size());
        for (auto i : xrange(targetData.size())) {
            targetView[i] = *(targetData[i]);
        }
        TargetViews.emplace(name, std::move(targetView));
    }
}

bool TTargetDataProvider::operator==(const TTargetDataProvider& rhs) const {
    return (*ObjectsGrouping == *rhs.ObjectsGrouping) && (Data == rhs.Data);
}


static void GetObjectsFloatDataSubsetImpl(
    const TSharedVector<float> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor,
    TSharedVector<float>* dstSubset
) {
    *dstSubset = MakeAtomicShared<TVector<float>>(
        NCB::GetSubset<float>(*src, objectsGroupingSubset.GetObjectsIndexing(), localExecutor)
    );
}


static void GetObjectWeightsSubsetImpl(
    const TSharedWeights<float> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor,
    TSharedWeights<float>* dstSubset
) {
    *dstSubset = MakeIntrusive<TWeights<float>>(
        src->GetSubset(objectsGroupingSubset.GetObjectsIndexing(), localExecutor)
    );
}


void NCB::GetGroupInfosSubset(
    TConstArrayRef<TQueryInfo> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor,
    TVector<TQueryInfo>* dstSubset
) {
    const TObjectsGrouping& dstSubsetGrouping = *(objectsGroupingSubset.GetSubsetGrouping());

    // resize, not yresize because TQueryInfo is not POD type
    dstSubset->resize(dstSubsetGrouping.GetGroupCount());

    if (dstSubsetGrouping.GetGroupCount() != 0) {
        const auto& subsetObjectsIndexing = objectsGroupingSubset.GetObjectsIndexing();

        TConstArrayRef<ui32> indexedSubset;
        TVector<ui32> indexedSubsetStorage;
        if (std::holds_alternative<TIndexedSubset<ui32>>(subsetObjectsIndexing)) {
            indexedSubset = subsetObjectsIndexing.Get<TIndexedSubset<ui32>>();
        } else {
            indexedSubsetStorage.yresize(subsetObjectsIndexing.Size());
            subsetObjectsIndexing.ParallelForEach(
                [&](ui32 idx, ui32 srcIdx) { indexedSubsetStorage[idx] = srcIdx; },
                localExecutor
            );
            indexedSubset = indexedSubsetStorage;
        }


        // CB_ENSURE inside is ok, groups must be nontrivial if there is groupInfo data in some targets
        TConstArrayRef<TGroupBounds> dstSubsetGroupBounds = dstSubsetGrouping.GetNonTrivialGroups();

        objectsGroupingSubset.GetGroupsIndexing().ParallelForEach(
            [&] (ui32 dstGroupIdx, ui32 srcGroupIdx) {
                const auto& srcGroupData = src[srcGroupIdx];
                auto& dstGroupData = (*dstSubset)[dstGroupIdx];
                ((TGroupBounds&)dstGroupData) = dstSubsetGroupBounds[dstGroupIdx];

                dstGroupData.Weight = srcGroupData.Weight;

                auto getSrcIdxInGroup = [&](ui32 dstIdxInGroup) {
                    return indexedSubset[dstGroupData.Begin + dstIdxInGroup] - srcGroupData.Begin;
                };

                if (!srcGroupData.SubgroupId.empty()) {
                    dstGroupData.SubgroupId.yresize(dstGroupData.GetSize());
                    for (auto dstIdxInGroup : xrange(dstGroupData.GetSize())) {
                        dstGroupData.SubgroupId[dstIdxInGroup] =
                            srcGroupData.SubgroupId[getSrcIdxInGroup(dstIdxInGroup)];
                    }
                }
                if (!srcGroupData.Competitors.empty()) {
                    // srcIdxInGroup -> dstIdxInGroup
                    TVector<ui32> invertedGroupPermutation;
                    invertedGroupPermutation.yresize(dstGroupData.GetSize());
                    for (auto dstIdxInGroup : xrange(dstGroupData.GetSize())) {
                        invertedGroupPermutation[getSrcIdxInGroup(dstIdxInGroup)] = dstIdxInGroup;
                    }

                    dstGroupData.Competitors.resize(dstGroupData.GetSize());
                    for (auto dstIdxInGroup : xrange(dstGroupData.GetSize())) {
                        auto& dstCompetitors = dstGroupData.Competitors[dstIdxInGroup];
                        const auto& srcCompetitors = srcGroupData.Competitors[
                            getSrcIdxInGroup(dstIdxInGroup)
                        ];
                        dstCompetitors.yresize(srcCompetitors.size());
                        for (auto competitorIdx : xrange(dstCompetitors.size())) {
                            auto& dstCompetitor = dstCompetitors[competitorIdx];
                            const auto& srcCompetitor = srcCompetitors[competitorIdx];
                            dstCompetitor.Id = invertedGroupPermutation[srcCompetitor.Id];
                            dstCompetitor.Weight = srcCompetitor.Weight;
                            dstCompetitor.SampleWeight = srcCompetitor.SampleWeight;
                        }
                    }
                }
            },
            localExecutor
        );
    }
}


void GetGroupInfosSubsetImpl(
    const TSharedVector<TQueryInfo> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor,
    TSharedVector<TQueryInfo>* dstSubset
) {
    TVector<TQueryInfo> dstSubsetData;
    GetGroupInfosSubset(*src, objectsGroupingSubset, localExecutor, &dstSubsetData);
    *dstSubset = MakeAtomicShared<TVector<TQueryInfo>>(
        std::move(dstSubsetData)
    );
}


/*
 * Subsets are created only once for each shared data for efficiency
 *
 * Subset creation works in 3 stages:
 *
 *   1) all data gathered from maps in TTargetDataProvider to keys in TSubsetTargetDataCache
 *   2) subsets for all cached data are created in parallel
 *   3) data maps in subset TTargetDataProvider are assigned from TSubsetTargetDataCache
 *      with the same keys as in source TTargetDataProvider
 *
 * TTargetDataProvider::GetSubset below does all these stages
 */

template <class TSharedDataPtr>
using TSrcToSubsetDataCache = TTargetSingleTypeDataCache<TSharedDataPtr, TSharedDataPtr>;

struct TSubsetTargetDataCache {
    TSrcToSubsetDataCache<TSharedVector<float>> Targets;

    TSrcToSubsetDataCache<TSharedWeights<float>> Weights;

    // multidim baselines are stored as separate pointers for simplicity
    TSrcToSubsetDataCache<TSharedVector<float>> Baselines;
    TSrcToSubsetDataCache<TSharedVector<TQueryInfo>> GroupInfos;
};


// arguments are (srcPtr, objectsGroupingSubset, localExecutor, dstSubsetPtr)
template <class TSharedDataPtr>
using TGetSubsetFunction = std::function<
        void (const TSharedDataPtr, const TObjectsGroupingSubset&, NPar::ILocalExecutor*, TSharedDataPtr*)
    >;


// getSubsetFunction
template <class TSharedDataPtr>
static void FillSubsetTargetDataCacheSubType(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    TGetSubsetFunction<TSharedDataPtr>&& getSubsetFunction,
    NPar::ILocalExecutor* localExecutor,
    TSrcToSubsetDataCache<TSharedDataPtr>* cache // access is exclusive to this function
) {
    // (srcPtr, dstPtr)
    using TSrcDstPair = std::pair<TSharedDataPtr, TSharedDataPtr*>;

    // it's ok to store ptrs to hash map values and update them as long as no keys are changed
    // we need to copy data to vector here to run parallel tasks on it
    TVector<TSrcDstPair> tasksData;
    tasksData.reserve(cache->size());

    for (auto& srcToDstPtr : *cache) {
        CB_ENSURE(
            !srcToDstPtr.second.Get(),
            "destination in TSrcDataToSubsetData has been updated prematurely"
        );
        tasksData.emplace_back(srcToDstPtr.first, &(srcToDstPtr.second));
    }

    localExecutor->ExecRangeWithThrow(
        [&] (int idx) {
            getSubsetFunction(
                tasksData[idx].first,
                objectsGroupingSubset,
                localExecutor,
                tasksData[idx].second
            );
        },
        0,
        SafeIntegerCast<int>(tasksData.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}


static void FillSubsetTargetDataCache(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor,
    TSubsetTargetDataCache* subsetTargetDataCache
) {
    TVector<std::function<void()>> tasks;

    tasks.emplace_back([&] () {
        FillSubsetTargetDataCacheSubType<TSharedWeights<float>>(
            objectsGroupingSubset,
            GetObjectWeightsSubsetImpl,
            localExecutor,
            &(subsetTargetDataCache->Weights)
        );
    });

    tasks.emplace_back([&] () {
        FillSubsetTargetDataCacheSubType<TSharedVector<float>>(
            objectsGroupingSubset,
            GetObjectsFloatDataSubsetImpl,
            localExecutor,
            &(subsetTargetDataCache->Targets)
        );
    });

    tasks.emplace_back([&] () {
        FillSubsetTargetDataCacheSubType<TSharedVector<float>>(
            objectsGroupingSubset,
            GetObjectsFloatDataSubsetImpl,
            localExecutor,
            &(subsetTargetDataCache->Baselines)
        );
    });

    tasks.emplace_back([&] () {
        FillSubsetTargetDataCacheSubType<TSharedVector<TQueryInfo>>(
            objectsGroupingSubset,
            GetGroupInfosSubsetImpl,
            localExecutor,
            &(subsetTargetDataCache->GroupInfos)
        );
    });

    ExecuteTasksInParallel(&tasks, localExecutor);
}


TIntrusivePtr<TTargetDataProvider> TTargetDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::ILocalExecutor* localExecutor
) const {
    TSubsetTargetDataCache subsetTargetDataCache;

    for (const auto& [name, targetsData] : Data.Targets) {
        for (const auto& oneTarget : targetsData) {
            subsetTargetDataCache.Targets.emplace(oneTarget, TSharedVector<float>());
        }
    }
    for (const auto& [name, weightsData] : Data.Weights) {
        subsetTargetDataCache.Weights.emplace(weightsData, TSharedWeights<float>());
    }
    for (const auto& [name, baselineData] : Data.Baselines) {
        for (const auto& oneBaseline : baselineData) {
            subsetTargetDataCache.Baselines.emplace(oneBaseline, TSharedVector<float>());
        }
    }
    for (const auto& [name, groupInfosData] : Data.GroupInfos) {
        subsetTargetDataCache.GroupInfos.emplace(groupInfosData, TSharedVector<TQueryInfo>());
    }


    FillSubsetTargetDataCache(objectsGroupingSubset, localExecutor, &subsetTargetDataCache);


    TProcessedTargetData subsetData;

    subsetData.TargetsClassCount = Data.TargetsClassCount;

    for (const auto& [name, targetsData] : Data.Targets) {
        TVector<TSharedVector<float>> subsetTargetData;

        for (const auto& oneTarget : targetsData) {
            subsetTargetData.emplace_back(subsetTargetDataCache.Targets.at(oneTarget));
        }

        subsetData.Targets.emplace(name, std::move(subsetTargetData));
    }
    for (const auto& [name, weightsData] : Data.Weights) {
        subsetData.Weights.emplace(name, subsetTargetDataCache.Weights.at(weightsData));
    }
    for (const auto& [name, baselineData] : Data.Baselines) {
        TVector<TSharedVector<float>> subsetBaselineData;

        for (const auto& oneBaseline : baselineData) {
            subsetBaselineData.emplace_back(subsetTargetDataCache.Baselines.at(oneBaseline));
        }

        subsetData.Baselines.emplace(name, std::move(subsetBaselineData));
    }
    for (const auto& [name, groupInfosData] : Data.GroupInfos) {
        subsetData.GroupInfos.emplace(name, subsetTargetDataCache.GroupInfos.at(groupInfosData));
    }


    return MakeIntrusive<TTargetDataProvider>(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetData)
    );
}


void TTargetSerialization::Load(
    TObjectsGroupingPtr objectsGrouping,
    IBinSaver* binSaver,
    TTargetDataProviderPtr* targetDataProvider
) {
    TProcessedTargetData processedTargetData;
    processedTargetData.Load(binSaver);
    *targetDataProvider = MakeIntrusive<TTargetDataProvider>(objectsGrouping, std::move(processedTargetData));
}


void TTargetSerialization::SaveNonSharedPart(
    const TTargetDataProvider& targetDataProvider,
    IBinSaver* binSaver
) {
    targetDataProvider.SaveDataNonSharedPart(binSaver);
}
