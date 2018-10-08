#include "target.h"

#include "util.h"

#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/stream/output.h>
#include <util/string/builder.h>

#include <functional>
#include <utility>


using namespace NCB;


static void CheckRawTarget(TMaybeData<TConstArrayRef<TString>> rawTarget, ui32 objectCount) {
    if (rawTarget) {
        auto targetData = *rawTarget;
        CheckDataSize(targetData.size(), (size_t)objectCount, "Target", false);

        for (auto i : xrange(targetData.size())) {
            CB_ENSURE(!targetData[i].empty(), "Target[" << i << "] is empty");
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

static void CheckMaybeEmptyBaseline(TSharedVector<float> baseline, ui32 objectCount) {
    if (baseline) {
        CheckOneBaseline(*baseline, 0, objectCount);
    }
}


static void CheckBaseline(const TVector<TSharedVector<float>>& baseline, ui32 objectCount) {
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
    for (auto pairIdx : xrange(pairs.size())) {
        const auto& pair = pairs[pairIdx];
        try {
            CB_ENSURE(pair.WinnerId != pair.LoserId, "WinnerId is equal to LoserId");
            CB_ENSURE(pair.Weight >= 0.0f, "Weight is negative");

            ui32 winnerIdGroupIdx = objectsGrouping.GetGroupIdxForObject(pair.WinnerId);
            ui32 loserIdGroupIdx = objectsGrouping.GetGroupIdxForObject(pair.LoserId);

            CB_ENSURE(
                winnerIdGroupIdx == loserIdGroupIdx,
                "winner id group #" << winnerIdGroupIdx << " is not equal to loser id group #"
                << loserIdGroupIdx
            );
        } catch (const TCatboostException& e) {
            // throw, not ythrow to avoid duplication of line info
            throw TCatboostException() << "Pair #" << pairIdx << ' ' << HumanReadableDescription(pair) << ": "
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

// local definition because there's no universal way to print NCB::TIndexRange
static TString HumanReadable(TGroupBounds groupBounds) {
    return TStringBuilder() << '[' << groupBounds.Begin << ',' << groupBounds.End << ')';
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
                "bounds " << HumanReadable(groupInfo)
                << " are not equal to grouping's corresponding group bounds: "
                << HumanReadable(objectsGrouping.GetGroup(i))
            );
            CheckOneGroupInfo(groupInfo);
            if (!groupInfo.Competitors.empty()) {
                hasPairData = true;
            }
        } catch (const TCatboostException& e) {
            // throw, not ythrow to avoid duplication of line info
            throw TCatboostException() << "groupInfo[" << i << "]: " << e.what();
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


bool TRawTargetData::operator==(const TRawTargetData& rhs) const {
    return (Target == rhs.Target) && (Baseline == rhs.Baseline) && (Weights == rhs.Weights) &&
        (GroupWeights == rhs.GroupWeights) && EqualAsMultiSets(Pairs, rhs.Pairs);
}


void TRawTargetData::Check(
    const TObjectsGrouping& objectsGrouping,
    NPar::TLocalExecutor* localExecutor
) const {
    ui32 objectCount = objectsGrouping.GetObjectCount();

    // Weights's values have been already checked in it's constructor
    CheckDataSize(Weights.GetSize(), objectCount, "Weights");


    TVector<std::function<void()>> tasks;

    tasks.emplace_back(
        [&, this]() {
            CheckRawTarget(Target, objectCount);
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

    tasks.emplace_back(
        [&, this]() {
            CheckPairs(Pairs, objectsGrouping);
        }
    );

    ExecuteTasksInParallel(&tasks, localExecutor);
}


void TRawTargetData::PrepareForInitialization(const TDataMetaInfo& metaInfo, ui32 objectCount) {
    NCB::PrepareForInitialization(metaInfo.HasTarget, objectCount, &Target);

    Baseline.resize(metaInfo.BaselineCount);
    for (auto& dim : Baseline) {
        dim.yresize(objectCount);
    }

    // if weights are not trivial reset at the end of building
    // here they are set to trivial to clear previous buffers
    SetTrivialWeights(objectCount);

    Pairs.clear();
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


static void GetMultidimBaselineSubset(
    const TVector<TVector<float>>& src,
    const TArraySubsetIndexing<ui32>& subset,
    NPar::TLocalExecutor* localExecutor,
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

static void GetPairsSubset(
    // assumes pairs and objectsGrouping consistency has already been checked
    TConstArrayRef<TPair> pairs,
    const TObjectsGrouping& objectsGrouping,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    TVector<TPair>* result
) {
    if (HoldsAlternative<TFullSubset<ui32>>(objectsGroupingSubset.GetObjectsIndexing())) {
        Assign(pairs, result);
        return;
    }

    THashMap<ui32, ui32> srcToDstGroupIndices;
    objectsGroupingSubset.GetGroupsIndexing().ForEach(
        [&srcToDstGroupIndices] (ui32 idx, ui32 srcIdx) { srcToDstGroupIndices[srcIdx] = idx; }
    );

    // CB_ENSURE inside is ok, groups must be nontrivial if there is pairs data
    TConstArrayRef<TGroupBounds> srcGroupsBounds = objectsGrouping.GetNonTrivialGroups();
    TConstArrayRef<TGroupBounds> subsetGroupsBounds =
        objectsGroupingSubset.GetSubsetGrouping()->GetNonTrivialGroups();

    result->clear();
    for (const auto& pair : pairs) {
        ui32 srcGroupIdx = objectsGrouping.GetGroupIdxForObject(pair.WinnerId);
        TGroupBounds srcGroupBounds = srcGroupsBounds[srcGroupIdx];
        auto* subsetGroupIdx = MapFindPtr(srcToDstGroupIndices, srcGroupIdx);
        if (subsetGroupIdx) {
            TGroupBounds subsetGroupBounds = subsetGroupsBounds[*subsetGroupIdx];
            result->emplace_back(
                subsetGroupBounds.Begin + (pair.WinnerId - srcGroupBounds.Begin),
                subsetGroupBounds.Begin + (pair.LoserId - srcGroupBounds.Begin),
                pair.Weight
            );
        }
    }
}


TRawTargetDataProvider TRawTargetDataProvider::GetSubset(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) const {
    const TArraySubsetIndexing<ui32>& objectsSubsetIndexing = objectsGroupingSubset.GetObjectsIndexing();

    TRawTargetData subsetData;

    TVector<std::function<void()>> tasks;

    tasks.emplace_back(
        [&, this]() {
            subsetData.Target = GetSubsetOfMaybeEmpty<TString>(
                GetTarget(),
                objectsSubsetIndexing,
                localExecutor
            );
        }
    );

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

    if (!Data.Pairs.empty()) {
        tasks.emplace_back(
            [&, this]() {
                GetPairsSubset(Data.Pairs, *ObjectsGrouping, objectsGroupingSubset, &subsetData.Pairs);
            }
        );
    }

    ExecuteTasksInParallel(&tasks, localExecutor);

    return TRawTargetDataProvider(
        objectsGroupingSubset.GetSubsetGrouping(),
        std::move(subsetData),
        true,
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


template <>
void Out<NCB::TTargetDataSpecification>(
    IOutputStream& out,
    const NCB::TTargetDataSpecification& targetDataSpecification
) {
    out << '(' << targetDataSpecification.Type << ',' << targetDataSpecification.Description << ')';
}



void GetObjectsFloatDataSubsetImpl(
    const TSharedVector<float> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor,
    TSharedVector<float>* dstSubset
) {
    *dstSubset = MakeAtomicShared<TVector<float>>(
        NCB::GetSubset<float>(*src, objectsGroupingSubset.GetObjectsIndexing(), localExecutor)
    );
}


void GetObjectWeightsSubsetImpl(
    const TSharedWeights<float> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor,
    TSharedWeights<float>* dstSubset
) {
    *dstSubset = MakeIntrusive<TWeights<float>>(
        src->GetSubset(objectsGroupingSubset.GetObjectsIndexing(), localExecutor)
    );
}

void GetGroupInfosSubsetImpl(
    const TSharedVector<TQueryInfo> src,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor,
    TSharedVector<TQueryInfo>* dstSubset
) {
    const TObjectsGrouping& dstSubsetGrouping = *(objectsGroupingSubset.GetSubsetGrouping());

    TVector<TQueryInfo> dstSubsetData;

    // resize, not yresize because TQueryInfo is not POD type
    dstSubsetData.resize(dstSubsetGrouping.GetGroupCount());

    if (dstSubsetGrouping.GetGroupCount() != 0) {
        const TVector<TQueryInfo>& srcData = *src;

        // CB_ENSURE inside is ok, groups must be nontrivial if there is groupInfo data in some targets
        TConstArrayRef<TGroupBounds> dstSubsetGroupBounds = dstSubsetGrouping.GetNonTrivialGroups();

        objectsGroupingSubset.GetGroupsIndexing().ParallelForEach(
            [&] (ui32 dstGroupIdx, ui32 srcGroupIdx) {
                const auto& srcGroupData = srcData[srcGroupIdx];
                auto& dstGroupData = dstSubsetData[dstGroupIdx];
                ((TGroupBounds&)dstGroupData) = dstSubsetGroupBounds[dstGroupIdx];

                dstGroupData.Weight = srcGroupData.Weight;
                dstGroupData.SubgroupId = srcGroupData.SubgroupId;
                dstGroupData.Competitors = srcGroupData.Competitors;
            },
            localExecutor
        );
    }

    *dstSubset = MakeAtomicShared<TVector<TQueryInfo>>(
        std::move(dstSubsetData)
    );
}


// arguments are (srcPtr, objectsGroupingSubset, localExecutor, dstSubsetPtr)
template <class TSharedDataPtr>
using TGetSubsetFunction = std::function<
        void (const TSharedDataPtr, const TObjectsGroupingSubset&, NPar::TLocalExecutor*, TSharedDataPtr*)
    >;


// getSubsetFunction
template <class TSharedDataPtr>
static void FillSubsetTargetDataCacheSubType(
    const TObjectsGroupingSubset& objectsGroupingSubset,
    TGetSubsetFunction<TSharedDataPtr>&& getSubsetFunction,
    NPar::TLocalExecutor* localExecutor,
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
    NPar::TLocalExecutor* localExecutor,
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


TTargetDataProviders NCB::GetSubsets(
    const TTargetDataProviders& srcTargetDataProviders,
    const TObjectsGroupingSubset& objectsGroupingSubset,
    NPar::TLocalExecutor* localExecutor
) {
    TSubsetTargetDataCache subsetTargetDataCache;

    for (const auto& specAndDataProvider : srcTargetDataProviders) {
        specAndDataProvider.second->GetSourceDataForSubsetCreation(&subsetTargetDataCache);
    }


    FillSubsetTargetDataCache(objectsGroupingSubset, localExecutor, &subsetTargetDataCache);


    TObjectsGroupingPtr objectsGrouping = objectsGroupingSubset.GetSubsetGrouping();

    TTargetDataProviders result;

    for (const auto& specAndDataProvider : srcTargetDataProviders) {
        result.emplace(
            specAndDataProvider.first,
            specAndDataProvider.second->GetSubset(objectsGrouping, subsetTargetDataCache)
        );
    }

    return result;
}

TBinClassTarget::TBinClassTarget(
    const TString& description,
    TObjectsGroupingPtr objectsGrouping,
    TSharedVector<float> target,
    TSharedWeights<float> weights,
    TSharedVector<float> baseline,
    bool skipCheck
)
    : TTargetDataProvider(
        TTargetDataSpecification(ETargetType::BinClass, description),
        std::move(objectsGrouping)
      )
{
    if (!skipCheck) {
        CheckDataSize(target->size(), (size_t)GetObjectCount(), "target");
        CheckDataSize(weights->GetSize(), GetObjectCount(), "weights");
        CheckMaybeEmptyBaseline(baseline, GetObjectCount());
    }
    Target = std::move(target);
    Weights = std::move(weights);
    Baseline = std::move(baseline);
}


void TBinClassTarget::GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const {
    subsetTargetDataCache->Targets.emplace(Target, TSharedVector<float>());
    subsetTargetDataCache->Weights.emplace(Weights, TSharedWeights<float>());
    if (Baseline) {
        subsetTargetDataCache->Baselines.emplace(Baseline, TSharedVector<float>());
    }
}

TTargetDataProviderPtr TBinClassTarget::GetSubset(
    TObjectsGroupingPtr objectsGrouping,
    const TSubsetTargetDataCache& subsetTargetDataCache
) const {
    return MakeIntrusive<TBinClassTarget>(
        GetSpecification().Description,
        std::move(objectsGrouping),
        subsetTargetDataCache.Targets.at(Target),
        subsetTargetDataCache.Weights.at(Weights),

        // reuse empty vector
        Baseline ? subsetTargetDataCache.Baselines.at(Baseline) : Baseline,
        true
    );
}


TMultiClassTarget::TMultiClassTarget(
    const TString& description,
    TObjectsGroupingPtr objectsGrouping,
    TSharedVector<float> target,
    TSharedWeights<float> weights,
    TVector<TSharedVector<float>>&& baseline,
    bool skipCheck
)
    : TTargetDataProvider(
        TTargetDataSpecification(ETargetType::MultiClass, description),
        std::move(objectsGrouping)
      )
{
    if (!skipCheck) {
        CheckDataSize(target->size(), (size_t)GetObjectCount(), "target");
        CheckDataSize(weights->GetSize(), GetObjectCount(), "weights");
        CheckBaseline(baseline, GetObjectCount());
    }
    Target = std::move(target);
    Weights = std::move(weights);
    Baseline = std::move(baseline);

    BaselineView.resize(Baseline.size());
    for (auto i : xrange(Baseline.size())) {
        BaselineView[i] = *(Baseline[i]);
    }
}



void TMultiClassTarget::GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const {
    subsetTargetDataCache->Targets.emplace(Target, TSharedVector<float>());
    subsetTargetDataCache->Weights.emplace(Weights, TSharedWeights<float>());
    for (const auto& oneBaseline : Baseline) {
        subsetTargetDataCache->Baselines.emplace(oneBaseline, TSharedVector<float>());
    }
}


TTargetDataProviderPtr TMultiClassTarget::GetSubset(
    TObjectsGroupingPtr objectsGrouping,
    const TSubsetTargetDataCache& subsetTargetDataCache
) const {
    TVector<TSharedVector<float>> subsetBaseline;
    for (const auto& oneBaseline : Baseline) {
        subsetBaseline.emplace_back(subsetTargetDataCache.Baselines.at(oneBaseline));
    }

    return MakeIntrusive<TMultiClassTarget>(
        GetSpecification().Description,
        std::move(objectsGrouping),
        subsetTargetDataCache.Targets.at(Target),
        subsetTargetDataCache.Weights.at(Weights),
        std::move(subsetBaseline),
        true
    );
}


TRegressionTarget::TRegressionTarget(
    const TString& description,
    TObjectsGroupingPtr objectsGrouping,
    TSharedVector<float> target,
    TSharedWeights<float> weights,
    TSharedVector<float> baseline,
    bool skipCheck
)
    : TTargetDataProvider(
        TTargetDataSpecification(ETargetType::Regression, description),
        std::move(objectsGrouping)
      )
{
    if (!skipCheck) {
        CheckDataSize(target->size(), (size_t)GetObjectCount(), "target");
        CheckDataSize(weights->GetSize(), GetObjectCount(), "weights");
        CheckMaybeEmptyBaseline(baseline, GetObjectCount());
    }
    Target = std::move(target);
    Weights = std::move(weights);
    Baseline = std::move(baseline);
}

void TRegressionTarget::GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const {
    subsetTargetDataCache->Targets.emplace(Target, TSharedVector<float>());
    subsetTargetDataCache->Weights.emplace(Weights, TSharedWeights<float>());
    if (Baseline) {
        subsetTargetDataCache->Baselines.emplace(Baseline, TSharedVector<float>());
    }
}

TTargetDataProviderPtr TRegressionTarget::GetSubset(
    TObjectsGroupingPtr objectsGrouping,
    const TSubsetTargetDataCache& subsetTargetDataCache
) const {
    return MakeIntrusive<TRegressionTarget>(
        GetSpecification().Description,
        std::move(objectsGrouping),
        subsetTargetDataCache.Targets.at(Target),
        subsetTargetDataCache.Weights.at(Weights),

        // reuse empty vector
        Baseline ? subsetTargetDataCache.Baselines.at(Baseline) : Baseline,
        true
    );
}


TGroupwiseRankingTarget::TGroupwiseRankingTarget(
    const TString& description,
    TObjectsGroupingPtr objectsGrouping,
    TSharedVector<float> target,
    TSharedWeights<float> weights,
    TSharedVector<float> baseline,
    TSharedVector<TQueryInfo> groupInfo,
    bool skipCheck
)
    : TTargetDataProvider(
        TTargetDataSpecification(ETargetType::GroupwiseRanking, description),
        std::move(objectsGrouping)
      )
{
    if (!skipCheck) {
        CheckDataSize(target->size(), (size_t)GetObjectCount(), "target");
        CheckDataSize(weights->GetSize(), GetObjectCount(), "weights");
        CheckMaybeEmptyBaseline(baseline, GetObjectCount());
        CheckGroupInfo(*groupInfo, *ObjectsGrouping, false);
    }
    Target = std::move(target);
    Weights = std::move(weights);
    Baseline = std::move(baseline);
    GroupInfo = std::move(groupInfo);
}

void TGroupwiseRankingTarget::GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const {
    subsetTargetDataCache->Targets.emplace(Target, TSharedVector<float>());
    subsetTargetDataCache->Weights.emplace(Weights, TSharedWeights<float>());
    if (Baseline) {
        subsetTargetDataCache->Baselines.emplace(Baseline, TSharedVector<float>());
    }
    subsetTargetDataCache->GroupInfos.emplace(GroupInfo, TSharedVector<TQueryInfo>());
}

TTargetDataProviderPtr TGroupwiseRankingTarget::GetSubset(
    TObjectsGroupingPtr objectsGrouping,
    const TSubsetTargetDataCache& subsetTargetDataCache
) const {
    return MakeIntrusive<TGroupwiseRankingTarget>(
        GetSpecification().Description,
        std::move(objectsGrouping),
        subsetTargetDataCache.Targets.at(Target),
        subsetTargetDataCache.Weights.at(Weights),

        // reuse empty vector
        Baseline ? subsetTargetDataCache.Baselines.at(Baseline) : Baseline,
        subsetTargetDataCache.GroupInfos.at(GroupInfo),
        true
    );
}


TGroupPairwiseRankingTarget::TGroupPairwiseRankingTarget(
    const TString& description,
    TObjectsGroupingPtr objectsGrouping,
    TSharedVector<float> baseline,
    TSharedVector<TQueryInfo> groupInfo,
    bool skipCheck
)
    : TTargetDataProvider(
        TTargetDataSpecification(ETargetType::GroupPairwiseRanking, description),
        std::move(objectsGrouping)
      )
{
    if (!skipCheck) {
        CheckMaybeEmptyBaseline(baseline, GetObjectCount());
        CheckGroupInfo(*groupInfo, *ObjectsGrouping, true);
    }
    Baseline = std::move(baseline);
    GroupInfo = std::move(groupInfo);
}


void TGroupPairwiseRankingTarget::GetSourceDataForSubsetCreation(TSubsetTargetDataCache* subsetTargetDataCache) const {
    if (Baseline) {
        subsetTargetDataCache->Baselines.emplace(Baseline, TSharedVector<float>());
    }
    subsetTargetDataCache->GroupInfos.emplace(GroupInfo, TSharedVector<TQueryInfo>());
}

TTargetDataProviderPtr TGroupPairwiseRankingTarget::GetSubset(
    TObjectsGroupingPtr objectsGrouping,
    const TSubsetTargetDataCache& subsetTargetDataCache
) const {
    return MakeIntrusive<TGroupPairwiseRankingTarget>(
        GetSpecification().Description,
        std::move(objectsGrouping),

        // reuse empty vector
        Baseline ? subsetTargetDataCache.Baselines.at(Baseline) : Baseline,
        subsetTargetDataCache.GroupInfos.at(GroupInfo),
        true
    );
}


TConstArrayRef<float> NCB::GetTarget(const TTargetDataProviders& targetDataProviders) {
    for (const auto& specAndDataProvider : targetDataProviders) {
        switch (specAndDataProvider.first.Type) {

#define GET_FIELD_FROM_TYPE(targetType) \
            case ETargetType::targetType: \
                return dynamic_cast<T##targetType##Target&>(*specAndDataProvider.second).GetTarget();

            GET_FIELD_FROM_TYPE(BinClass);
            GET_FIELD_FROM_TYPE(MultiClass);
            GET_FIELD_FROM_TYPE(Regression);
            GET_FIELD_FROM_TYPE(GroupwiseRanking);

#undef GET_FIELD_FROM_TYPE

            default:
                ;
        }
    }

    CB_ENSURE_INTERNAL(false, "no Target data in targetDataProviders");
    return {}; // to keep compiler happy
}

TConstArrayRef<float> NCB::GetWeights(const TTargetDataProviders& targetDataProviders) {
    for (const auto& specAndDataProvider : targetDataProviders) {
        switch (specAndDataProvider.first.Type) {

#define GET_FIELD_FROM_TYPE(targetType) \
            case ETargetType::targetType: { \
                auto& weights = \
                    dynamic_cast<T##targetType##Target&>(*specAndDataProvider.second).GetWeights(); \
                return weights.IsTrivial() ? TConstArrayRef<float>() : weights.GetNonTrivialData(); \
            }

            GET_FIELD_FROM_TYPE(BinClass);
            GET_FIELD_FROM_TYPE(MultiClass);
            GET_FIELD_FROM_TYPE(Regression);
            GET_FIELD_FROM_TYPE(GroupwiseRanking);

#undef GET_FIELD_FROM_TYPE

            default:
                ;
        }
    }
    CATBOOST_WARNING_LOG << "no Weights data in targetDataProviders";
    return {};
}

TVector<TConstArrayRef<float>> NCB::GetBaseline(const TTargetDataProviders& targetDataProviders) {
    // only one of nonMultiClassBaseline or multiClassBaseline can be non-empty

    TVector<TConstArrayRef<float>> nonMultiClassBaseline;
    TVector<TConstArrayRef<float>> multiClassBaseline;

    for (const auto& specAndDataProvider : targetDataProviders) {
        switch (specAndDataProvider.first.Type) {

#define GET_ONE_BASELINE_FROM_TYPE(targetType) \
            case ETargetType::targetType: { \
                    if (nonMultiClassBaseline.empty()) {\
                        auto baseline = \
                            dynamic_cast<T##targetType##Target&>(*specAndDataProvider.second).GetBaseline(); \
                        if (baseline) { \
                            nonMultiClassBaseline.push_back(*baseline); \
                        } \
                    } \
                } \
                break;

            GET_ONE_BASELINE_FROM_TYPE(BinClass);

            case ETargetType::MultiClass: {
                    auto baselineView =
                        dynamic_cast<TMultiClassTarget&>(*specAndDataProvider.second).GetBaseline();
                    if (baselineView) {
                        Assign(*baselineView, &multiClassBaseline);
                    }
                }
                break;

            GET_ONE_BASELINE_FROM_TYPE(Regression);
            GET_ONE_BASELINE_FROM_TYPE(GroupwiseRanking);
            GET_ONE_BASELINE_FROM_TYPE(GroupPairwiseRanking);

#undef GET_ONE_BASELINE_FROM_TYPE
        }
    }
    CB_ENSURE_INTERNAL(
        nonMultiClassBaseline.empty() || multiClassBaseline.empty(),
        "Old GetBaseline is compatible with at most one non-empty baseline type "
        " - either non-multiclass or multiclass"
    );
    return !multiClassBaseline.empty() ? multiClassBaseline : nonMultiClassBaseline;
}

TConstArrayRef<TQueryInfo> NCB::GetGroupInfo(const TTargetDataProviders& targetDataProviders) {
    for (const auto& specAndDataProvider : targetDataProviders) {
        switch (specAndDataProvider.first.Type) {

#define GET_FIELD_FROM_TYPE(targetType) \
            case ETargetType::targetType: \
                return dynamic_cast<T##targetType##Target&>(*specAndDataProvider.second).GetGroupInfo();

            GET_FIELD_FROM_TYPE(GroupwiseRanking);
            GET_FIELD_FROM_TYPE(GroupPairwiseRanking);

#undef GET_FIELD_FROM_TYPE

            default:
                ;
        }
    }
    return {};
}




