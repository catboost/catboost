#include "greedy_tensor_search.h"

#include "estimated_features.h"
#include "feature_penalties_calcer.h"
#include "fold.h"
#include "helpers.h"
#include "index_calcer.h"
#include "leafwise_scoring.h"
#include "learn_context.h"
#include "monotonic_constraint_utils.h"
#include "nonsymmetric_index_calcer.h"
#include "scoring.h"
#include "split.h"
#include "tensor_search_helpers.h"
#include "tree_print.h"

#include <catboost/libs/data/feature_estimators.h>
#include <catboost/libs/data/feature_index.h>
#include <catboost/libs/data/packed_binary_features.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>
#include <catboost/private/libs/distributed/master.h>

#include <library/cpp/fast_log/fast_log.h>

#include <util/generic/cast.h>
#include <util/generic/queue.h>
#include <util/generic/scope.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/system/mem_info.h>


using namespace NCB;


constexpr size_t MAX_ONLINE_CTR_FEATURES = 50;

namespace {
    struct TSplitLeafCandidate {
        TIndexType Leaf;
        double Gain;
        TCandidateInfo BestCandidate;

        TSplitLeafCandidate(TIndexType leaf, double gain, const TCandidateInfo& bestCandidate)
            : Leaf(leaf)
            , Gain(gain)
            , BestCandidate(bestCandidate)
        {}

        bool operator<(const TSplitLeafCandidate& other) const {
            return Gain < other.Gain;
        }
    };
}

void TrimOnlineCTRcache(const TVector<TFold*>& folds) {
    for (auto& fold : folds) {
        fold->TrimOnlineCTR(MAX_ONLINE_CTR_FEATURES);
    }
}

static double CalcDerivativesStDevFromZeroOrderedBoosting(
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor
) {
    double sum2 = 0;
    size_t count = 0;
    for (const auto& bt : fold.BodyTailArr) {
        for (const auto& perDimensionWeightedDerivatives : bt.WeightedDerivatives) {
            sum2 += L2NormSquared<double>(
                MakeArrayRef(perDimensionWeightedDerivatives.data() + bt.BodyFinish, bt.TailFinish - bt.BodyFinish),
                localExecutor
            );
        }

        count += bt.TailFinish - bt.BodyFinish;
    }

    Y_ASSERT(count > 0);
    return sqrt(sum2 / count);
}

static double CalcDerivativesStDevFromZeroPlainBoosting(
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor
) {
    Y_ASSERT(fold.BodyTailArr.size() == 1);
    Y_ASSERT(fold.BodyTailArr.front().WeightedDerivatives.size() > 0);

    const auto& weightedDerivatives = fold.BodyTailArr.front().WeightedDerivatives;

    double sum2 = 0;
    for (const auto& perDimensionWeightedDerivatives : weightedDerivatives) {
        sum2 += L2NormSquared<double>(perDimensionWeightedDerivatives, localExecutor);
    }

    return sqrt(sum2 / weightedDerivatives.front().size());
}

static double CalcDerivativesStDevFromZero(
    const TFold& fold,
    const EBoostingType boosting,
    NPar::ILocalExecutor* localExecutor
) {
    switch (boosting) {
        case EBoostingType::Ordered:
            return CalcDerivativesStDevFromZeroOrderedBoosting(fold, localExecutor);
        case EBoostingType::Plain:
            return CalcDerivativesStDevFromZeroPlainBoosting(fold, localExecutor);
        default:
            CB_ENSURE(false, "Unexpected boosting type");
    }
    Y_UNREACHABLE();
}

static double CalcDerivativesStDevFromZeroMultiplier(int learnSampleCount, double modelLength) {
    double modelExpLength = log(static_cast<double>(learnSampleCount));
    double modelLeft = exp(modelExpLength - modelLength);
    return modelLeft / (1.0 + modelLeft);
}


inline static void MarkFeatureAsIncluded(
    const TPackedBinaryIndex& packedBinaryIndex,
    TVector<TBinaryFeaturesPack>* perBinaryPackMasks) {

    (*perBinaryPackMasks)[packedBinaryIndex.PackIdx]
        |= TBinaryFeaturesPack(1) << packedBinaryIndex.BitIdx;
}

inline static void MarkFeatureAsExcluded(
    const TPackedBinaryIndex& packedBinaryIndex,
    TVector<TBinaryFeaturesPack>* perBinaryPackMasks) {

    (*perBinaryPackMasks)[packedBinaryIndex.PackIdx]
        &= ~(TBinaryFeaturesPack(1) << packedBinaryIndex.BitIdx);
}


static void AddFloatFeatures(
    bool isEstimated,
    bool isOnlineEstimated,
    const TQuantizedObjectsDataProvider& learnObjectsData,
    TCandidateList* candList) {

    learnObjectsData.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Float>(
        [&](TFloatFeatureIdx floatFeatureIdx) {
            TSplitCandidate splitCandidate;
            splitCandidate.FeatureIdx = (int)*floatFeatureIdx;
            splitCandidate.Type = isEstimated ? ESplitType::EstimatedFeature : ESplitType::FloatFeature;
            splitCandidate.IsOnlineEstimatedFeature = isOnlineEstimated;

            TSplitEnsemble splitEnsemble(std::move(splitCandidate), isEstimated, isOnlineEstimated);
            TCandidateInfo candidate;
            candidate.SplitEnsemble = std::move(splitEnsemble);
            candList->emplace_back(std::move(candidate));
        }
    );
}


static void AddOneHotFeatures(
    const TQuantizedObjectsDataProvider& learnObjectsData,
    TLearnContext* ctx,
    TCandidateList* candList) {

    const auto& quantizedFeaturesInfo = *learnObjectsData.GetQuantizedFeaturesInfo();
    const ui32 oneHotMaxSize = ctx->Params.CatFeatureParams.Get().OneHotMaxSize;

    learnObjectsData.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&](TCatFeatureIdx catFeatureIdx) {
            auto onLearnOnlyCount = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly;
            if ((onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1)) {
                return;
            }

            TSplitCandidate splitCandidate;
            splitCandidate.FeatureIdx = (int)*catFeatureIdx;
            splitCandidate.Type = ESplitType::OneHotFeature;

            TSplitEnsemble splitEnsemble(std::move(splitCandidate));

            TCandidateInfo candidate;
            candidate.SplitEnsemble = std::move(splitEnsemble);
            candList->emplace_back(std::move(candidate));
        }
    );
}


static void CompressCandidates(
    bool isEstimated,
    bool isOnlineEstimated,
    const TQuantizedObjectsDataProvider& learnObjectsData,
    TCandidatesContext* candidatesContext) {

    auto& candList = candidatesContext->CandidateList;
    auto& selectedFeaturesInBundles = candidatesContext->SelectedFeaturesInBundles;
    auto& perBinaryPackMasks = candidatesContext->PerBinaryPackMasks;
    auto& selectedFeaturesInGroups = candidatesContext->SelectedFeaturesInGroups;

    selectedFeaturesInBundles.assign(learnObjectsData.GetExclusiveFeatureBundlesSize(), TVector<ui32>());
    perBinaryPackMasks.assign(learnObjectsData.GetBinaryFeaturesPacksSize(), TBinaryFeaturesPack(0));
    selectedFeaturesInGroups.assign(learnObjectsData.GetFeaturesGroupsSize(), TVector<ui32>());

    TCandidateList updatedCandList;
    updatedCandList.reserve(candList.size());

    for (auto& candSubList : candList) {
        const auto& splitEnsemble = candSubList.Candidates[0].SplitEnsemble;
        Y_ASSERT(splitEnsemble.Type == ESplitEnsembleType::OneFeature);
        const auto& splitCandidate = splitEnsemble.SplitCandidate;

        TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex;
        TMaybe<TPackedBinaryIndex> maybePackedBinaryIndex;
        TMaybe<TFeaturesGroupIndex> maybeFeaturesGroupIndex;

        if ((splitCandidate.Type == ESplitType::FloatFeature) ||
            (splitCandidate.Type == ESplitType::EstimatedFeature))
        {
            auto floatFeatureIdx = TFloatFeatureIdx(splitCandidate.FeatureIdx);
            maybeExclusiveBundleIndex = learnObjectsData.GetFeatureToExclusiveBundleIndex(floatFeatureIdx);
            maybePackedBinaryIndex = learnObjectsData.GetFeatureToPackedBinaryIndex(floatFeatureIdx);
            maybeFeaturesGroupIndex = learnObjectsData.GetFeatureToFeaturesGroupIndex(floatFeatureIdx);
        } else {
            auto catFeatureIdx = TCatFeatureIdx(splitCandidate.FeatureIdx);
            maybeExclusiveBundleIndex = learnObjectsData.GetFeatureToExclusiveBundleIndex(catFeatureIdx);
            maybePackedBinaryIndex = learnObjectsData.GetFeatureToPackedBinaryIndex(catFeatureIdx);
        }
        CB_ENSURE_INTERNAL(
            maybeExclusiveBundleIndex.Defined()
                + maybePackedBinaryIndex.Defined()
                + maybeFeaturesGroupIndex.Defined() <= 1,
            "Feature #"
            << learnObjectsData.GetFeaturesLayout()->GetExternalFeatureIdx(
                splitCandidate.FeatureIdx,
                (splitCandidate.Type == ESplitType::FloatFeature) ?
                    EFeatureType::Float :
                    EFeatureType::Categorical
            )
            << " is mis-included into more than one aggregated column");

        if (maybeExclusiveBundleIndex) {
            selectedFeaturesInBundles[maybeExclusiveBundleIndex->BundleIdx].push_back(
                maybeExclusiveBundleIndex->InBundleIdx);
        } else if (maybePackedBinaryIndex) {
            MarkFeatureAsIncluded(*maybePackedBinaryIndex, &perBinaryPackMasks);
        } else if (maybeFeaturesGroupIndex) {
            selectedFeaturesInGroups[maybeFeaturesGroupIndex->GroupIdx].push_back(
                maybeFeaturesGroupIndex->InGroupIdx
            );
        } else {
            updatedCandList.push_back(std::move(candSubList));
        }
    }

    for (auto bundleIdx : xrange(SafeIntegerCast<ui32>(selectedFeaturesInBundles.size()))) {
        auto& bundle = selectedFeaturesInBundles[bundleIdx];
        if (bundle.empty()) {
            continue;
        }

        Sort(bundle);

        TCandidateInfo candidate;
        candidate.SplitEnsemble = TSplitEnsemble{TExclusiveFeaturesBundleRef{bundleIdx}};
        updatedCandList.emplace_back(std::move(candidate));
    }


    for (auto packIdx : xrange(SafeIntegerCast<ui32>(perBinaryPackMasks.size()))) {
        if (!perBinaryPackMasks[packIdx]) {
            continue;
        }

        TCandidateInfo candidate;
        candidate.SplitEnsemble = TSplitEnsemble(
            TBinarySplitsPackRef{packIdx},
            isEstimated,
            isOnlineEstimated
        );
        updatedCandList.emplace_back(std::move(candidate));
    }

    for (auto groupIdx : xrange(SafeIntegerCast<ui32>(selectedFeaturesInGroups.size()))) {
        auto& group = selectedFeaturesInGroups[groupIdx];
        if (group.empty()) {
            continue;
        }

        Sort(group);

        TCandidateInfo candidate;
        candidate.SplitEnsemble = TSplitEnsemble{TFeaturesGroupRef{groupIdx}};
        updatedCandList.emplace_back(std::move(candidate));
    }

    candList = std::move(updatedCandList);
}


static void SelectCandidatesAndCleanupStatsFromPrevTree(
    TLearnContext* ctx,
    TCandidatesContext* candidatesContext,
    TBucketStatsCache* statsFromPrevTree) {

    auto& candList = candidatesContext->CandidateList;
    auto& selectedFeaturesInBundles = candidatesContext->SelectedFeaturesInBundles;
    auto& perBinaryPackMasks = candidatesContext->PerBinaryPackMasks;
    auto& selectedFeaturesInGroups = candidatesContext->SelectedFeaturesInGroups;

    TCandidateList updatedCandList;
    updatedCandList.reserve(candList.size());

    const double rsm = ctx->Params.ObliviousTreeOptions->Rsm;
    auto& rand = ctx->LearnProgress->Rand;

    for (auto& candSubList : candList) {
        const auto& splitEnsemble = candSubList.Candidates[0].SplitEnsemble;

        bool addCandSubListToResult;

        switch (splitEnsemble.Type) {
            case ESplitEnsembleType::OneFeature:
                addCandSubListToResult = rand.GenRandReal1() <= rsm;
                break;
            case ESplitEnsembleType::BinarySplits:
                {
                    const ui32 packIdx = splitEnsemble.BinarySplitsPackRef.PackIdx;
                    TBinaryFeaturesPack& perPackMask = perBinaryPackMasks[packIdx];
                    for (size_t idxInPack : xrange(sizeof(TBinaryFeaturesPack) * CHAR_BIT)) {
                        if ((perPackMask >> idxInPack) & 1) {
                            const bool addToCandidates = rand.GenRandReal1() <= rsm;
                            if (!addToCandidates) {
                                MarkFeatureAsExcluded(
                                    TPackedBinaryIndex(packIdx, idxInPack),
                                    &perBinaryPackMasks);
                            }
                        }
                    }
                    addCandSubListToResult = perPackMask != TBinaryFeaturesPack(0);
                }
                break;
            case ESplitEnsembleType::ExclusiveBundle:
                {
                    TVector<ui32>& selectedFeaturesInBundle
                        = selectedFeaturesInBundles[splitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx];

                    TVector<ui32> filteredFeaturesInBundle;
                    filteredFeaturesInBundle.reserve(selectedFeaturesInBundle.size());
                    for (auto inBundleIdx : selectedFeaturesInBundle) {
                        const bool addToCandidates = rand.GenRandReal1() <= rsm;
                        if (addToCandidates) {
                            filteredFeaturesInBundle.push_back(inBundleIdx);
                        }
                    }
                    selectedFeaturesInBundle = std::move(filteredFeaturesInBundle);
                    addCandSubListToResult = !selectedFeaturesInBundle.empty();
                }
                break;
            case ESplitEnsembleType::FeaturesGroup:
                {
                    TVector<ui32>& selectedFeaturesInGroup
                        = selectedFeaturesInGroups[splitEnsemble.FeaturesGroupRef.GroupIdx];

                    TVector<ui32> filteredFeaturesInGroup;
                    filteredFeaturesInGroup.reserve(selectedFeaturesInGroup.size());
                    for (auto inGroupIdx : selectedFeaturesInGroup) {
                        const bool addToCandidates = rand.GenRandReal1() <= rsm;
                        if (addToCandidates) {
                            filteredFeaturesInGroup.push_back(inGroupIdx);
                        }
                    }
                    selectedFeaturesInGroup = std::move(filteredFeaturesInGroup);
                    addCandSubListToResult = !selectedFeaturesInGroup.empty();
                }
                break;
        }

        if (addCandSubListToResult) {
            updatedCandList.push_back(std::move(candSubList));
        } else if (ctx->UseTreeLevelCaching()) {
            statsFromPrevTree->Stats.erase(splitEnsemble);
        }
    }

    candList = std::move(updatedCandList);
}


static void AddCtrsToCandList(
    const TFold& fold,
    const TLearnContext& ctx,
    const TProjection& proj,
    TCandidateList* candList) {

    TCandidatesInfoList ctrSplits;
    const auto& ctrsHelper = ctx.CtrsHelper;
    const auto& ctrInfo = ctrsHelper.GetCtrInfo(proj);

    for (ui32 ctrIdx = 0; ctrIdx < ctrInfo.size(); ++ctrIdx) {
        const ui32 targetClassesCount = fold.TargetClassesCount[ctrInfo[ctrIdx].TargetClassifierIdx];
        int targetBorderCount = GetTargetBorderCount(ctrInfo[ctrIdx], targetClassesCount);
        const auto& priors = ctrInfo[ctrIdx].Priors;
        int priorsCount = priors.size();
        Y_ASSERT(targetBorderCount < 256);
        for (int border = 0; border < targetBorderCount; ++border) {
            for (int prior = 0; prior < priorsCount; ++prior) {
                TSplitCandidate splitCandidate;
                splitCandidate.Type = ESplitType::OnlineCtr;
                splitCandidate.Ctr = TCtr(proj, ctrIdx, border, prior, ctrInfo[ctrIdx].BorderCount);

                TCandidateInfo candidate;
                candidate.SplitEnsemble = TSplitEnsemble(std::move(splitCandidate));
                ctrSplits.Candidates.emplace_back(candidate);
            }
        }
    }

    candList->push_back(ctrSplits);
}

static void DropStatsForProjection(
    const TFold& fold,
    const TLearnContext& ctx,
    const TProjection& proj,
    TBucketStatsCache* statsFromPrevTree) {

    const auto& ctrsHelper = ctx.CtrsHelper;
    const auto& ctrsInfo = ctrsHelper.GetCtrInfo(proj);
    for (int ctrIdx = 0 ; ctrIdx < ctrsInfo.ysize(); ++ctrIdx) {
        const auto& ctrMeta = ctrsInfo[ctrIdx];
        const ui32 targetClasses = fold.TargetClassesCount[ctrMeta.TargetClassifierIdx];
        int targetBorderCount = GetTargetBorderCount(ctrMeta, targetClasses);

        for (int border = 0; border < targetBorderCount; ++border) {
            int priorsCount = ctrMeta.Priors.ysize();
            for (int prior = 0; prior < priorsCount; ++prior) {
                TSplitCandidate splitCandidate;
                splitCandidate.Type = ESplitType::OnlineCtr;
                splitCandidate.Ctr = TCtr(proj, ctrIdx, border, prior, ctrMeta.BorderCount);
                statsFromPrevTree->Stats.erase(TSplitEnsemble(std::move(splitCandidate)));
            }
        }
    }
}

static void AddSimpleCtrs(
    const TQuantizedObjectsDataProvider& learnObjectsData,
    TFold* fold,
    TLearnContext* ctx,
    TBucketStatsCache* statsFromPrevTree,
    TCandidateList* candList) {

    const auto& quantizedFeaturesInfo = *learnObjectsData.GetQuantizedFeaturesInfo();
    const ui32 oneHotMaxSize = ctx->Params.CatFeatureParams.Get().OneHotMaxSize;

    learnObjectsData.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&](TCatFeatureIdx catFeatureIdx) {
            if (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly <= oneHotMaxSize) {
                return;
            }

            TProjection proj;
            proj.AddCatFeature((int)*catFeatureIdx);

            if (ctx->LearnProgress->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
                if (ctx->UseTreeLevelCaching()) {
                    DropStatsForProjection(*fold, *ctx, proj, statsFromPrevTree);
                }
                return;
            }
            AddCtrsToCandList(*fold, *ctx, proj, candList);
            auto* ownedCtrs = fold->GetOwnedCtrs(proj);
            if (ownedCtrs) {
                ownedCtrs->EnsureProjectionInData(proj);
            }
        }
    );
}

static void AddTreeCtrs(
    const TQuantizedObjectsDataProvider& learnObjectsData,
    const TSplitTree& currentTree,
    TFold* fold,
    TLearnContext* ctx,
    TBucketStatsCache* statsFromPrevTree,
    TCandidateList* candList) {

    const auto& quantizedFeaturesInfo = *learnObjectsData.GetQuantizedFeaturesInfo();
    const auto& featuresLayout = *learnObjectsData.GetFeaturesLayout();
    const ui32 oneHotMaxSize = ctx->Params.CatFeatureParams.Get().OneHotMaxSize;

    using TSeenProjHash = THashSet<TProjection>;
    TSeenProjHash seenProj;

    // greedy construction
    TProjection binAndOneHotFeaturesTree;
    binAndOneHotFeaturesTree.BinFeatures = currentTree.GetBinFeatures();
    binAndOneHotFeaturesTree.OneHotFeatures = currentTree.GetOneHotFeatures();
    seenProj.insert(binAndOneHotFeaturesTree);

    for (const auto& ctr : currentTree.GetUsedCtrs()) {
        seenProj.insert(ctr.Projection);
    }

    TSeenProjHash addedProjHash;
    for (const auto& baseProj : seenProj) {
        if (baseProj.IsEmpty()) {
            continue;
        }
        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&](TCatFeatureIdx catFeatureIdx) {
                const bool isOneHot =
                    (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly <= oneHotMaxSize);
                if (isOneHot || (ctx->LearnProgress->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm)) {
                    return;
                }

                TProjection proj = baseProj;
                proj.AddCatFeature((int)*catFeatureIdx);

                if (proj.IsRedundant() ||
                    proj.GetFullProjectionLength() > ctx->Params.CatFeatureParams->MaxTensorComplexity)
                {
                    return;
                }

                if (addedProjHash.contains(proj)) {
                    return;
                }

                addedProjHash.insert(proj);

                AddCtrsToCandList(*fold, *ctx, proj, candList);
                auto* ownedCtrs = fold->GetOwnedCtrs(proj);
                if (ownedCtrs) {
                    ownedCtrs->EnsureProjectionInData(proj);
                }
            }
        );
    }
    if (ctx->UseTreeLevelCaching()) {
        THashSet<TSplitEnsemble> candidatesToErase;
        for (auto& [splitEnsemble, value] : statsFromPrevTree->Stats) {
            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                if (!addedProjHash.contains(splitEnsemble.SplitCandidate.Ctr.Projection)) {
                    candidatesToErase.insert(splitEnsemble);
                }
            }
        }
        for (const auto& splitEnsemble : candidatesToErase) {
            statsFromPrevTree->Stats.erase(splitEnsemble);
        }
    }
}


static void ForEachCtrCandidate(
    const std::function<void(TCandidatesInfoList*)>& callback,
    TCandidateList* candList) {

    for (auto& candSubList : *candList) {
        if (candSubList.Candidates[0].SplitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
            callback(&candSubList);
        }
    }
}


static void SelectCtrsToDropAfterCalc(
    size_t memoryLimit,
    int sampleCount,
    int threadCount,

    // note: if ctrs are in precomputed storage this function should return false
    const std::function<bool(const TProjection&)>& isInCache,
    TCandidateList* candList) {

    size_t maxMemoryForOneCtr = 0;
    size_t fullNeededMemoryForCtrs = 0;

    ForEachCtrCandidate(
        [&] (TCandidatesInfoList* candSubList) {
            if (isInCache(candSubList->Candidates[0].SplitEnsemble.SplitCandidate.Ctr.Projection)) {
                const size_t neededMem = sampleCount * candSubList->Candidates.size();
                maxMemoryForOneCtr = Max<size_t>(neededMem, maxMemoryForOneCtr);
                fullNeededMemoryForCtrs += neededMem;
            }
        },
        candList);

    if (!fullNeededMemoryForCtrs) {
        return;
    }
    auto currentMemoryUsage = NMemInfo::GetMemInfo().RSS;
    if (fullNeededMemoryForCtrs + currentMemoryUsage > memoryLimit) {
        CATBOOST_DEBUG_LOG << "Need more memory than allowed, will drop some ctrs after score calculation"
            << Endl;
        const float GB = (ui64)1024 * 1024 * 1024;
        CATBOOST_DEBUG_LOG << "current rss: " << currentMemoryUsage / GB << " full needed memory: "
            << fullNeededMemoryForCtrs / GB << Endl;
        size_t currentNonDroppableMemory = currentMemoryUsage;
        size_t maxMemForOtherThreadsApprox = (ui64)(threadCount - 1) * maxMemoryForOneCtr;

        ForEachCtrCandidate(
            [&] (TCandidatesInfoList* candSubList) {
                if (isInCache(candSubList->Candidates[0].SplitEnsemble.SplitCandidate.Ctr.Projection)) {
                    const size_t neededMem = sampleCount * candSubList->Candidates.size();
                    if (currentNonDroppableMemory + neededMem + maxMemForOtherThreadsApprox <= memoryLimit) {
                        candSubList->ShouldDropCtrAfterCalc = false;
                        currentNonDroppableMemory += neededMem;
                    } else {
                        candSubList->ShouldDropCtrAfterCalc = true;
                    }
                } else {
                    candSubList->ShouldDropCtrAfterCalc = false;
                }
            },
            candList);
    }
}


static void CalcBestScore(
    const TTrainingDataProviders& data,
    const TSplitTree& currentTree,
    ui64 randSeed,
    double scoreStDev,
    TVector<TCandidatesContext>* candidatesContexts,
    TFold* fold,
    TLearnContext* ctx) {

    auto scoreDistribution = GetScoreDistribution(ctx->Params.ObliviousTreeOptions->RandomScoreType);

    const TFlatPairsInfo pairs = UnpackPairsFromQueries(fold->LearnQueriesInfo);
    const auto& monotonicConstraints = ctx->Params.ObliviousTreeOptions->MonotoneConstraints.Get();
    const TVector<int> currTreeMonotonicConstraints = (
        monotonicConstraints.empty()
        ? TVector<int>()
        : GetTreeMonotoneConstraints(currentTree, monotonicConstraints)
    );

    TVector<std::pair<size_t, size_t>> tasks; // vector of (contextIdx, candId)

    for (auto contextIdx : xrange(candidatesContexts->size())) {
        TCandidatesContext& candidatesContext = (*candidatesContexts)[contextIdx];
        for (auto candId : xrange(candidatesContext.CandidateList.size())) {
            tasks.emplace_back(contextIdx, candId);
        }
    }

    ctx->LocalExecutor->ExecRange(
        [&] (int taskIdx) {
            TCandidatesContext& candidatesContext = (*candidatesContexts)[tasks[taskIdx].first];
            TCandidateList& candList = candidatesContext.CandidateList;

            auto& candidate = candList[tasks[taskIdx].second];

            const auto& splitEnsemble = candidate.Candidates[0].SplitEnsemble;

            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                auto* ownedCtr = fold->GetOwnedCtrs(proj);
                if (ownedCtr && ownedCtr->Data.at(proj).Feature.empty()) {
                    ComputeOnlineCTRs(data, *fold, proj, ctx, ownedCtr);
                }
            }
            TVector<TVector<double>> allScores(candidate.Candidates.size());
            ctx->LocalExecutor->ExecRange(
                [&](int oneCandidate) {
                    THolder<IScoreCalcer> scoreCalcer;
                    if (IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction())) {
                        scoreCalcer.Reset(new TPairwiseScoreCalcer);
                    } else {
                        scoreCalcer = MakePointwiseScoreCalcer(
                            ctx->Params.ObliviousTreeOptions->ScoreFunction
                        );
                    }

                    CalcStatsAndScores(
                        *candidatesContext.LearnData,
                        fold->GetAllCtrs(),
                        ctx->SampledDocs,
                        ctx->SmallestSplitSideDocs,
                        fold,
                        pairs,
                        ctx->Params,
                        candidate.Candidates[oneCandidate],
                        currentTree.GetDepth(),
                        ctx->UseTreeLevelCaching(),
                        currTreeMonotonicConstraints,
                        monotonicConstraints,
                        ctx->LocalExecutor,
                        &ctx->PrevTreeLevelStats,
                        /*stats3d*/nullptr,
                        /*pairwiseStats*/nullptr,
                        scoreCalcer.Get());
                    scoreCalcer->GetScores().swap(allScores[oneCandidate]);
                },
                0,
                candidate.Candidates.ysize(),
                NPar::TLocalExecutor::WAIT_COMPLETE);

            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr) && candidate.ShouldDropCtrAfterCalc) {
                fold->ClearCtrDataForProjectionIfOwned(splitEnsemble.SplitCandidate.Ctr.Projection);
            }

            SetBestScore(
                randSeed + taskIdx,
                allScores,
                scoreDistribution,
                scoreStDev,
                candidatesContext,
                &candidate.Candidates);

            PenalizeBestSplits(
                xrange(ctx->SampledDocs.LeavesCount),
                *ctx,
                data,
                *fold,
                candidatesContext.OneHotMaxSize,
                &candidate.Candidates
            );
        },
        0,
        tasks.ysize(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void DoBootstrap(
    const TVector<TIndexType>& indices,
    TFold* fold,
    TLearnContext* ctx,
    ui32 leavesCount = 0) {

    if (!ctx->Params.SystemOptions->IsSingleHost()) {
        MapBootstrap(ctx);
    } else {
        Bootstrap(
            ctx->Params,
            !ctx->LearnProgress->EstimatedFeaturesContext.OfflineEstimatedFeaturesLayout.empty(),
            indices,
            ctx->LearnProgress->LeafValues,
            fold,
            &ctx->SampledDocs,
            ctx->LocalExecutor,
            &ctx->LearnProgress->Rand,
            IsLeafwiseScoringApplicable(ctx->Params),
            leavesCount);
        if (ctx->Params.BoostingOptions->Langevin) {
            for (auto& bodyTail : fold->BodyTailArr) {
                AddLangevinNoiseToDerivatives(
                    ctx->Params.BoostingOptions->DiffusionTemperature,
                    ctx->Params.BoostingOptions->LearningRate,
                    ctx->LearnProgress->Rand.GenRand(),
                    &bodyTail.WeightedDerivatives,
                    ctx->LocalExecutor
                );
            }
        }
    }
}

static void CalcBestScoreLeafwise(
    const TTrainingDataProviders& data,
    const TVector<TIndexType>& leafs,
    const TStatsForSubtractionTrick& statsForSubtractionTrick,
    ui64 randSeed,
    double scoreStDev,
    TVector<TCandidatesContext>* candidatesContexts, // [dataset]
    TFold* fold,
    TLearnContext* ctx) {

    auto scoreDistribution = GetScoreDistribution(ctx->Params.ObliviousTreeOptions->RandomScoreType);

    TVector<std::pair<size_t, size_t>> tasks; // vector of (contextIdx, candId)

    for (auto contextIdx : xrange(candidatesContexts->size())) {
        TCandidatesContext& candidatesContext = (*candidatesContexts)[contextIdx];
        for (auto candId : xrange(candidatesContext.CandidateList.size())) {
            tasks.emplace_back(contextIdx, candId);
        }
    }

    ctx->LocalExecutor->ExecRange(
        [&] (int taskIdx) {
            TCandidatesContext& candidatesContext = (*candidatesContexts)[tasks[taskIdx].first];
            TCandidateList& candList = candidatesContext.CandidateList;

            auto& candidate = candList[tasks[taskIdx].second];

            const auto& splitEnsemble = candidate.Candidates[0].SplitEnsemble;

            // Calc online ctr if needed
            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                auto* ownedCtr = fold->GetOwnedCtrs(proj);
                if (ownedCtr && ownedCtr->Data.at(proj).Feature.empty()) {
                    ComputeOnlineCTRs(data, *fold, proj, ctx, ownedCtr);
                }
            }
            const int maxBucketCount = statsForSubtractionTrick.GetMaxBucketCount();
            const int maxSplitEnsembles = statsForSubtractionTrick.GetMaxSplitEnsembles();
            const size_t statsSize = maxBucketCount * maxSplitEnsembles;

            const auto candidateScores = CalcScoresForOneCandidate(
                *candidatesContext.LearnData,
                candidate,
                ctx->SampledDocs,
                *fold,
                leafs,
                statsForSubtractionTrick.MakeSlice(taskIdx, statsSize),
                ctx);

            SetBestScore(
                randSeed + taskIdx,
                candidateScores,
                scoreDistribution,
                scoreStDev,
                candidatesContext,
                &candidate.Candidates);

            PenalizeBestSplits(
                leafs,
                *ctx,
                data,
                *fold,
                candidatesContext.OneHotMaxSize,
                &candidate.Candidates
            );

            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr) && candidate.ShouldDropCtrAfterCalc) {
                fold->ClearCtrDataForProjectionIfOwned(splitEnsemble.SplitCandidate.Ctr.Projection);
            }
        },
        0,
        tasks.ysize(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

static double CalcScoreStDev(
    ui32 learnSampleCount,
    double modelLength,
    const TFold& fold,
    TLearnContext* ctx) {

    const double derivativesStDevFromZero = ctx->Params.SystemOptions->IsSingleHost()
        ? CalcDerivativesStDevFromZero(fold, ctx->Params.BoostingOptions->BoostingType, ctx->LocalExecutor)
        : MapCalcDerivativesStDevFromZero(learnSampleCount, ctx);

    return ctx->Params.ObliviousTreeOptions->RandomStrength
        * derivativesStDevFromZero
        * (ctx->Params.ObliviousTreeOptions->RandomScoreType == ERandomScoreType::NormalWithModelSizeDecrease ?
            CalcDerivativesStDevFromZeroMultiplier(learnSampleCount, modelLength)
            : 1.0
        );
}

static void CalcScores(
    const TTrainingDataProviders& data,
    const TSplitTree& currentSplitTree,
    const double scoreStDev,
    TVector<TCandidatesContext>* candidatesContexts, // [dataset]
    TFold* fold,
    TLearnContext* ctx) {

    if (!ctx->Params.SystemOptions->IsSingleHost()) {
        if (IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction())) {
            MapRemotePairwiseCalcScore(scoreStDev, candidatesContexts, ctx);
        } else {
            MapRemoteCalcScore(scoreStDev, candidatesContexts, ctx);
        }
    } else {
        const ui64 randSeed = ctx->LearnProgress->Rand.GenRand();
        if (IsLeafwiseScoringApplicable(ctx->Params)) {
            CalcBestScoreLeafwise(
                data,
                xrange(ctx->SampledDocs.LeavesCount),
                /*statsForSubtractionTrick*/ TStatsForSubtractionTrick{},
                randSeed,
                scoreStDev,
                candidatesContexts,
                fold,
                ctx);
        } else {
            CalcBestScore(
                data,
                currentSplitTree,
                randSeed,
                scoreStDev,
                candidatesContexts,
                fold,
                ctx);
        }
    }
}

static double GetCatFeatureWeight(
    const TCandidateInfo& candidate,
    const TLearnContext& ctx,
    const TFold& fold,
    size_t maxFeatureValueCount
) {
    const auto& splitEnsemble = candidate.SplitEnsemble;
    if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
        TProjection projection = splitEnsemble.SplitCandidate.Ctr.Projection;
        ECtrType ctrType =
            ctx.CtrsHelper.GetCtrInfo(projection)[splitEnsemble.SplitCandidate.Ctr.CtrIdx].Type;

        if (!ctx.LearnProgress->UsedCtrSplits.contains(std::make_pair(ctrType, projection))) {

            const auto& uniqValuesCounts =
                fold.GetCtrs(projection).GetUniqValuesCounts(projection);

            return pow(
                1 + (uniqValuesCounts.GetUniqueValueCountForType(ctrType) /
                        static_cast<double>(maxFeatureValueCount)),
                -ctx.Params.ObliviousTreeOptions->ModelSizeReg.Get());
        }
    }
    return 1.0;
}

static void SelectBestCandidate(
    const TTrainingDataProviders& trainingData,
    const TLearnContext& ctx,
    TConstArrayRef<TCandidatesContext> candidatesContexts,
    size_t maxFeatureValueCount,
    const TFold& fold,
    double scoreBeforeSplit,
    double* bestScore,
    const TCandidateInfo** bestSplitCandidate
) {
    const TFeaturesLayout& layout = *ctx.Layout;
    const auto oneHotMaxSize = ctx.Params.CatFeatureParams->OneHotMaxSize;
    const auto& featurePenaltiesOptions = ctx.Params.ObliviousTreeOptions->FeaturePenalties.Get();
    const NCatboostOptions::TPerFeaturePenalty& featureWeights = featurePenaltiesOptions.FeatureWeights;
    double bestGain = -std::numeric_limits<double>::infinity();
    for (const auto& candidatesContext : candidatesContexts) {
        for (const auto& subList : candidatesContext.CandidateList) {
            for (const auto& candidate : subList.Candidates) {
                double score = candidate.BestScore.GetInstance(ctx.LearnProgress->Rand);
                score *= GetCatFeatureWeight(candidate, ctx, fold, maxFeatureValueCount);

                double gain = score - scoreBeforeSplit;
                const auto bestSplit = candidate.GetBestSplit(trainingData, fold, oneHotMaxSize);
                gain *= GetSplitFeatureWeight(
                    bestSplit,
                    ctx.LearnProgress->EstimatedFeaturesContext,
                    layout,
                    featureWeights);
                if (gain > bestGain) {
                    *bestScore = score;
                    *bestSplitCandidate = &candidate;
                    bestGain = gain;
                }
            }
        }
    }
}


static TCandidatesContext SelectDatasetFeaturesForScoring(
    bool isEstimated,
    bool isOnlineEstimated,
    TQuantizedObjectsDataProviderPtr learnData, // can be nullptr if estimated data is absent
    ui32 testSampleCount,
    const TMaybe<TSplitTree>& currentSplitTree,
    TFold* fold,
    TLearnContext* ctx
) {
    TCandidatesContext candidatesContext;
    if (!learnData) {
        return candidatesContext;
    }

    candidatesContext.LearnData = learnData;
    candidatesContext.OneHotMaxSize = ctx->Params.CatFeatureParams->OneHotMaxSize;
    candidatesContext.BundlesMetaData = learnData->GetExclusiveFeatureBundlesMetaData();
    candidatesContext.FeaturesGroupsMetaData = learnData->GetFeaturesGroupsMetaData();

    AddFloatFeatures(isEstimated, isOnlineEstimated, *learnData, &candidatesContext.CandidateList);
    AddOneHotFeatures(*learnData, ctx, &candidatesContext.CandidateList);
    CompressCandidates(isEstimated, isOnlineEstimated, *learnData, &candidatesContext);
    SelectCandidatesAndCleanupStatsFromPrevTree(ctx, &candidatesContext, &ctx->PrevTreeLevelStats);

    if (!isEstimated) {
        AddSimpleCtrs(*learnData, fold, ctx, &ctx->PrevTreeLevelStats, &candidatesContext.CandidateList);

        if (currentSplitTree.Defined()) {
            AddTreeCtrs(
                *learnData,
                currentSplitTree.GetRef(),
                fold,
                ctx,
                &ctx->PrevTreeLevelStats,
                &candidatesContext.CandidateList);
        }

        const auto isInCache =
            [fold](const TProjection& proj) -> bool {
                auto* ownedCtr = fold->GetOwnedCtrs(proj);
                return ownedCtr && !ownedCtr->Data[proj].Feature.empty();
            };
        const auto cpuUsedRamLimit = ParseMemorySizeDescription(ctx->Params.SystemOptions->CpuUsedRamLimit.Get());
        const ui32 learnSampleCount = learnData->GetObjectCount();
        SelectCtrsToDropAfterCalc(
            cpuUsedRamLimit,
            learnSampleCount + testSampleCount,
            ctx->Params.SystemOptions->NumThreads,
            isInCache,
            &candidatesContext.CandidateList);
    }

    return candidatesContext;
}


// returns vector ot per-dataset (main, online estimated, offline estimated) candidates contexts
static TVector<TCandidatesContext> SelectFeaturesForScoring(
    const TTrainingDataProviders& data,
    const TMaybe<TSplitTree>& currentSplitTree,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TCandidatesContext> result;
    result.push_back(
        SelectDatasetFeaturesForScoring(
            /*isEstimated*/ false,
            /*isOnlineEstimated*/ false,
            data.Learn->ObjectsData,
            data.GetTestSampleCount(),
            currentSplitTree,
            fold,
            ctx));

    result.push_back(
        SelectDatasetFeaturesForScoring(
            /*isEstimated*/ true,
            /*isOnlineEstimated*/ false,
            data.EstimatedObjectsData.Learn,
            /*testSampleCount*/ 0, // unused
            currentSplitTree,
            fold,
            ctx));

    result.push_back(
        SelectDatasetFeaturesForScoring(
            /*isEstimated*/ true,
            /*isOnlineEstimated*/ true,
            fold->GetOnlineEstimatedFeatures().Learn,
            /*testSampleCount*/ 0, // unused
            currentSplitTree,
            fold,
            ctx));

    return result;
}

static size_t CalcMaxFeatureValueCount(
    const TFold& fold,
    TConstArrayRef<TCandidatesContext> candidatesContexts) {

    i32 maxFeatureValueCount = 1;

    for (const auto& candidatesContext : candidatesContexts) {
        for (const auto& candidate : candidatesContext.CandidateList) {
            const auto& splitEnsemble = candidate.Candidates[0].SplitEnsemble;
            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                maxFeatureValueCount = Max(
                    maxFeatureValueCount,
                    fold.GetCtrs(proj).GetUniqValuesCounts(proj).GetMaxUniqueValueCount());
            }
        }
    }
    return SafeIntegerCast<size_t>(maxFeatureValueCount);
}

static void ProcessCtrSplit(
    const TTrainingDataProviders& data,
    const TSplit& bestSplit,
    TFold* fold,
    TLearnContext* ctx) {

    const auto& ctr = bestSplit.Ctr;

    const ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(ctr.Projection)[ctr.CtrIdx].Type;
    ctx->LearnProgress->UsedCtrSplits.insert(std::make_pair(ctrType, ctr.Projection));

    const auto& proj = bestSplit.Ctr.Projection;
    auto* ownedCtr = fold->GetOwnedCtrs(proj);
    if (ownedCtr && ownedCtr->Data[proj].Feature.empty()) {
        ComputeOnlineCTRs(data, *fold, proj, ctx, ownedCtr);
        if (ctx->UseTreeLevelCaching()) {
            DropStatsForProjection(*fold, *ctx, proj, &ctx->PrevTreeLevelStats);
        }
    }
}

static void MarkFeaturesAsUsed(
    const TSplit& split,
    const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
    const TFeaturesLayout& layout,
    TVector<bool>* usedFeatures
) {
    const auto markFeatureAsUsed = [&layout, usedFeatures](const int internalFeatureIndex, const EFeatureType type) {
        const auto externalFeatureIndex = layout.GetExternalFeatureIdx(internalFeatureIndex, type);
        (*usedFeatures)[externalFeatureIndex] = true;
    };

    split.IterateOverUsedFeatures(estimatedFeaturesContext, markFeatureAsUsed);
}

static void MarkFeaturesAsUsedPerObject(
    const TSplit& split,
    const TIndexedSubset<ui32>& docsSubset,
    const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
    const TFeaturesLayout& layout,
    TMap<ui32, TVector<bool>>* usedFeatures
) {
    const auto markFeatureAsUsed = [&docsSubset, &layout, usedFeatures](const int internalFeatureIndex, const EFeatureType type) {
        const auto externalFeatureIndex = layout.GetExternalFeatureIdx(internalFeatureIndex, type);
        auto it = usedFeatures->find(externalFeatureIndex);
        if (it != usedFeatures->end()) { //if we want to keep this feature usage by objects
            TArrayRef<bool> perObjectUsage(it->second);
            for (const auto idx : docsSubset) {
                perObjectUsage[idx] = true;
            }
        }
    };

    split.IterateOverUsedFeatures(estimatedFeaturesContext, markFeatureAsUsed);
}

static void MarkFeaturesAsUsed(
    const TSplit& split,
    const TMaybe<TIndexedSubset<ui32>>& docsSubset, //if empty, apply for all objects
    const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
    const TFeaturesLayout& layout,
    TVector<bool>* usedFeatures,
    TMap<ui32, TVector<bool>>* usedFeaturesPerObject
) {
    MarkFeaturesAsUsed(
        split,
        estimatedFeaturesContext,
        layout,
        usedFeatures
    );
    if (docsSubset.Defined()) {
        MarkFeaturesAsUsedPerObject(
            split,
            docsSubset.GetRef(),
            estimatedFeaturesContext,
            layout,
            usedFeaturesPerObject
        );
    }
}

static TSplitTree GreedyTensorSearchOblivious(
    const TTrainingDataProviders& data,
    double modelLength,
    TProfileInfo& profile,
    TVector<TIndexType>* indices,
    TFold* fold,
    TLearnContext* ctx) {

    TSplitTree currentSplitTree;

    CATBOOST_INFO_LOG << "\n";

    const bool useLeafwiseScoring = IsLeafwiseScoringApplicable(ctx->Params);
    const bool isSamplingPerTree = IsSamplingPerTree(ctx->Params.ObliviousTreeOptions);
    const ui32 learnSampleCount = data.Learn->ObjectsData->GetObjectCount();
    const double scoreStDev = CalcScoreStDev(learnSampleCount, modelLength, *fold, ctx);
    double scoreBeforeSplit = 0.0;

    for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
        TVector<TCandidatesContext> candidatesContexts
            = SelectFeaturesForScoring(data, currentSplitTree, fold, ctx);
        CheckInterrupted(); // check after long-lasting operation

        if (!isSamplingPerTree) {  // sampling per tree level
            DoBootstrap(*indices, fold, ctx, /* leavesCount */ 1u << curDepth);
        }
        profile.AddOperation(TStringBuilder() << "Bootstrap, depth " << curDepth);

        CalcScores(data, currentSplitTree, scoreStDev, &candidatesContexts, fold, ctx);

        const size_t maxFeatureValueCount = CalcMaxFeatureValueCount(*fold, candidatesContexts);

        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Calc scores " << curDepth);

        double bestScore = MINIMAL_SCORE;
        const TCandidateInfo* bestSplitCandidate = nullptr;
        SelectBestCandidate(data, *ctx, candidatesContexts, maxFeatureValueCount, *fold, scoreBeforeSplit, &bestScore, &bestSplitCandidate);
        fold->DropEmptyCTRs();
        if (bestScore == MINIMAL_SCORE) {
            break;
        }
        Y_ASSERT(bestSplitCandidate != nullptr);
        scoreBeforeSplit = bestScore;

        const TSplit bestSplit = bestSplitCandidate->GetBestSplit(
            data,
            *fold,
            candidatesContexts[0].OneHotMaxSize);

        if (bestSplit.Type == ESplitType::OnlineCtr) {
            ProcessCtrSplit(data, bestSplit, fold, ctx);
        }

        MarkFeaturesAsUsed(
            bestSplit,
            /*docsSubset*/ Nothing(),
            ctx->LearnProgress->EstimatedFeaturesContext,
            *ctx->Layout,
            &ctx->LearnProgress->UsedFeatures,
            &ctx->LearnProgress->UsedFeaturesPerObject
        );

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            SetPermutedIndices(
                bestSplit,
                data,
                curDepth + 1,
                *fold,
                *indices,
                ctx->LocalExecutor);
            if (isSamplingPerTree) {
                if (useLeafwiseScoring) {
                    ctx->SampledDocs.UpdateIndicesInLeafwiseSortedFold(*indices, ctx->LocalExecutor);
                } else {
                    ctx->SampledDocs.UpdateIndices(*indices, ctx->LocalExecutor);
                }
                if (ctx->UseTreeLevelCaching() && !useLeafwiseScoring) {
                    ctx->SmallestSplitSideDocs.SelectSmallestSplitSide(
                        curDepth + 1,
                        ctx->SampledDocs,
                        ctx->LocalExecutor);
                }
            }
        } else {
            MapSetIndices(bestSplit, ctx);
        }
        currentSplitTree.AddSplit(bestSplit);
        CATBOOST_INFO_LOG << BuildDescription(*ctx->Layout, bestSplit) << " score " << bestScore << "\n";

        profile.AddOperation(TStringBuilder() << "Select best split " << curDepth);

        int redundantIdx = -1;
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            redundantIdx = GetRedundantSplitIdx(GetIsLeafEmpty(curDepth + 1, *indices, ctx->LocalExecutor));
        } else {
            redundantIdx = MapGetRedundantSplitIdx(ctx);
        }
        if (redundantIdx != -1) {
            currentSplitTree.DeleteSplit(redundantIdx);
            CATBOOST_INFO_LOG << "  tensor " << redundantIdx << " is redundant, remove it and stop\n";
            break;
        }
    }
    return currentSplitTree;
}


static void SplitDocsSubset(
    const TIndexedSubset<ui32>& subsetToSplit,
    TConstArrayRef<TIndexType> indices,
    TIndexType leftChildIdx,
    TIndexedSubset<ui32>* leftChildSubset,
    TIndexedSubset<ui32>* rightChildSubset) {

    // TODO(ilyzhin) it can be parallel
    for (auto doc : subsetToSplit) {
        if (indices[doc] == leftChildIdx) {
            leftChildSubset->push_back(doc);
        } else{
            rightChildSubset->push_back(doc);
        }
    }
}

static TNonSymmetricTreeStructure GreedyTensorSearchLossguide(
    const TTrainingDataProviders& data,
    double modelLength,
    TProfileInfo& profile,
    TVector<TIndexType>* indices,
    TFold* fold,
    TLearnContext* ctx) {

    Y_ASSERT(IsSamplingPerTree(ctx->Params.ObliviousTreeOptions));

    TNonSymmetricTreeStructure currentStructure;

    const ui32 learnSampleCount = data.Learn->ObjectsData->GetObjectCount();
    TArrayRef<TIndexType> indicesRef(*indices);

    const double scoreStDev = CalcScoreStDev(learnSampleCount, modelLength, *fold, ctx);

    TPriorityQueue<TSplitLeafCandidate> queue;
    TVector<ui32> leafDepth(ctx->Params.ObliviousTreeOptions->MaxLeaves);
    const auto findBestCandidate = [&](TIndexType leaf) {
        Y_DEFER { profile.AddOperation(TStringBuilder() << "Find best candidate for leaf " << leaf); };
        const auto leafBounds = ctx->SampledDocs.LeavesBounds[leaf];
        const bool needSplit = leafDepth[leaf] < ctx->Params.ObliviousTreeOptions->MaxDepth
            && leafBounds.GetSize() >= ctx->Params.ObliviousTreeOptions->MinDataInLeaf;
        if (!needSplit) {
            return;
        }
        auto candidatesContexts = SelectFeaturesForScoring(data, {}, fold, ctx);
        CalcBestScoreLeafwise(
            data,
            {leaf},
            TStatsForSubtractionTrick{},
            ctx->LearnProgress->Rand.GenRand(),
            scoreStDev,
            &candidatesContexts,
            fold,
            ctx);
        const size_t maxFeatureValueCount = CalcMaxFeatureValueCount(*fold, candidatesContexts);
        CheckInterrupted(); // check after long-lasting operation

        double bestScore = MINIMAL_SCORE;
        const TCandidateInfo* bestSplitCandidate = nullptr;
        const double scoreBeforeSplit = CalcScoreWithoutSplit(leaf, *fold, *ctx);
        SelectBestCandidate(data, *ctx, candidatesContexts, maxFeatureValueCount, *fold, scoreBeforeSplit, &bestScore, &bestSplitCandidate);
        fold->DropEmptyCTRs();
        if (bestSplitCandidate == nullptr) {
            return;
        }
        const double gain = bestScore - scoreBeforeSplit;
        CATBOOST_DEBUG_LOG << "Best gain for leaf #" << leaf << " = " << gain << Endl;
        if (gain < 1e-9) {
            return;
        }
        queue.emplace(leaf, gain, *bestSplitCandidate);
    };
    findBestCandidate(0);

    TVector<TIndexedSubset<ui32>> subsetsForLeafs(ctx->Params.ObliviousTreeOptions->MaxLeaves);
    subsetsForLeafs[0] = xrange(learnSampleCount).operator TIndexedSubset<ui32>();

    while (!queue.empty() && currentStructure.GetLeafCount() < ctx->Params.ObliviousTreeOptions->MaxLeaves) {
        /*
         * There is a problem with feature penalties calculation.
         * We consider a feature unused until we extracted a split with it from the queue.
         * Before that, all new splits are calculated with penalty for that feature.
         * And when first split by this feature is extracted from the queue, all other
         * splits in the queue should be recalculated without penalty for that feature.
         * However, we don't do it for performance and code simplicity.
         */
        const TSplitLeafCandidate curSplitLeaf = queue.top();
        queue.pop();

        const TSplit bestSplit = curSplitLeaf.BestCandidate.GetBestSplit(
            data,
            *fold,
            ctx->Params.CatFeatureParams->OneHotMaxSize);
        if (bestSplit.Type == ESplitType::OnlineCtr) {
            ProcessCtrSplit(data, bestSplit, fold, ctx);
        }

        const TIndexType splittedNodeIdx = curSplitLeaf.Leaf;
        MarkFeaturesAsUsed(
            bestSplit,
            subsetsForLeafs[splittedNodeIdx],
            ctx->LearnProgress->EstimatedFeaturesContext,
            *ctx->Layout,
            &ctx->LearnProgress->UsedFeatures,
            &ctx->LearnProgress->UsedFeaturesPerObject
        );

        const auto& node = currentStructure.AddSplit(bestSplit, curSplitLeaf.Leaf);
        const TIndexType leftChildIdx = ~node.Left;
        const TIndexType rightChildIdx = ~node.Right;
        UpdateIndices(
            node,
            data,
            subsetsForLeafs[splittedNodeIdx],
            *fold,
            ctx->LocalExecutor,
            indicesRef
        );

        Y_ASSERT(leftChildIdx == splittedNodeIdx);
        TIndexedSubset<ui32> leftChildSubset, rightChildSubset;
        SplitDocsSubset(subsetsForLeafs[splittedNodeIdx], indicesRef, leftChildIdx, &leftChildSubset, &rightChildSubset);
        subsetsForLeafs[leftChildIdx] = std::move(leftChildSubset);
        subsetsForLeafs[rightChildIdx] = std::move(rightChildSubset);

        ctx->SampledDocs.UpdateIndicesInLeafwiseSortedFoldForSingleLeaf(
            splittedNodeIdx,
            leftChildIdx,
            rightChildIdx,
            *indices,
            ctx->LocalExecutor);
        const int newDepth = leafDepth[splittedNodeIdx] + 1;
        leafDepth[leftChildIdx] = newDepth;
        leafDepth[rightChildIdx] = newDepth;
        CATBOOST_DEBUG_LOG << "For node #" << splittedNodeIdx << " built childs: " << leftChildIdx << " and " << rightChildIdx
            << "with split: " << BuildDescription(*ctx->Layout, bestSplit) << " and score = " << curSplitLeaf.Gain << Endl;
        findBestCandidate(leftChildIdx);
        findBestCandidate(rightChildIdx);
    }

    return currentStructure;
}

namespace {
    struct TSubtractTrickInfo {
        const TTrainingDataProviders* Data;
        TVector<TCandidatesContext>* CandidatesContexts;
        TFold* Fold;
        TLearnContext* Ctx;
        TQueue<TVector<TBucketStats>>* ParentsQueue;
        double ScoreStDev;
        int MaxBucketCount;
        ui64 MaxSplitEnsembles;
        ui64 StatsSize;
        size_t MaxFeatureValueCount;

        TSubtractTrickInfo(
            const TTrainingDataProviders* data,
            TVector<TCandidatesContext>* candidatesContexts,
            TFold* fold,
            TLearnContext* ctx,
            TQueue<TVector<TBucketStats>>* parentsQueue,
            double scoresStDev)
            : Data(data)
            , CandidatesContexts(candidatesContexts)
            , Fold(fold)
            , Ctx(ctx)
            , ParentsQueue(parentsQueue)
            , ScoreStDev(scoresStDev)
        {
            size_t nFeatures = 0;
            MaxBucketCount = 0;
            MaxSplitEnsembles = 0;
            for (auto contextIdx : xrange(CandidatesContexts->size())) {
                TCandidatesContext& candidatesContext = (*CandidatesContexts)[contextIdx];
                nFeatures += candidatesContext.CandidateList.size();
                for (auto candId : xrange(candidatesContext.CandidateList.size())) {
                    MaxSplitEnsembles = std::max(MaxSplitEnsembles, (ui64)candidatesContext.CandidateList[candId].Candidates.size());
                    for (auto id : xrange(candidatesContext.CandidateList[candId].Candidates.size())) {
                        const int currentBucketCount = GetBucketCount(
                            candidatesContext.CandidateList[candId].Candidates[id].SplitEnsemble,
                            *(candidatesContext.LearnData->GetQuantizedFeaturesInfo()),
                            candidatesContext.LearnData->GetPackedBinaryFeaturesSize(),
                            candidatesContext.LearnData->GetExclusiveFeatureBundlesMetaData(),
                            candidatesContext.LearnData->GetFeaturesGroupsMetaData()
                        );
                        MaxBucketCount = std::max(MaxBucketCount, currentBucketCount);
                    }
                }
            }
            MaxFeatureValueCount = CalcMaxFeatureValueCount(*fold, *CandidatesContexts);
            // for MultiClassClassification or MultiTarget multiply by approxDimensionion
            StatsSize = MaxBucketCount * nFeatures * MaxSplitEnsembles;
        }
        void ParentsQueuePop() {
            if (!ParentsQueue->empty()) {
                ParentsQueue->pop();
            }
        }
    };
}

inline static void ConditionalPushToParentsQueue(
    const double gain,
    const TCandidateInfo* bestSplitCandidate,
    TVector<TBucketStats>&& stats,
    TQueue<TVector<TBucketStats>>* parentsQueue) {

    if (!(gain < 1e-9) && bestSplitCandidate != nullptr && stats.size() != 0) {
        parentsQueue->push(std::move(stats));
    }
}

inline static void CalcBestScoreAndCandidate (
    const TSubtractTrickInfo& subTrickInfo,
    const TIndexType id,
    const TStatsForSubtractionTrick& statsForSubtractionTrick,
    double* gainLocal,
    const TCandidateInfo** bestSplitCandidateLocal,
    TSplit* bestSplitLocal) {

    CalcBestScoreLeafwise(
        *subTrickInfo.Data,
        {id},
        statsForSubtractionTrick,
        subTrickInfo.Ctx->LearnProgress->Rand.GenRand(),
        subTrickInfo.ScoreStDev,
        subTrickInfo.CandidatesContexts,
        subTrickInfo.Fold,
        subTrickInfo.Ctx);
    double bestScoreLocal = MINIMAL_SCORE;
    double scoreBeforeSplitLocal = CalcScoreWithoutSplit(id, *subTrickInfo.Fold, *subTrickInfo.Ctx);
    SelectBestCandidate(
        *subTrickInfo.Data,
        *subTrickInfo.Ctx,
        *subTrickInfo.CandidatesContexts,
        subTrickInfo.MaxFeatureValueCount,
        *subTrickInfo.Fold,
        scoreBeforeSplitLocal,
        &bestScoreLocal,
        bestSplitCandidateLocal);
    if (*bestSplitCandidateLocal) {
        *bestSplitLocal = (*bestSplitCandidateLocal)->GetBestSplit(
            *subTrickInfo.Data,
            *subTrickInfo.Fold,
            subTrickInfo.Ctx->Params.CatFeatureParams->OneHotMaxSize);
    }
    *gainLocal = bestScoreLocal - scoreBeforeSplitLocal;
}

static TVector<TBucketStats> CalculateStats(
    const TSubtractTrickInfo& subTrickInfo,
    const TIndexType smallId,
    double* gain,
    const TCandidateInfo** bestSplitCandidate,
    TSplit* bestSplit) {

    TVector<TBucketStats> smallStats;
    // TODO(ShvetsKS, espetrov) speedup memory allocation to enable subtraction trick for multiclass, mnist dataset
    if (subTrickInfo.Fold->GetApproxDimension() == 1) {
        smallStats.yresize(subTrickInfo.StatsSize);
    }
    const TArrayRef<TBucketStats> emptyStats;
    const TStatsForSubtractionTrick statsForSubtractionTrickSmall(
        smallStats,
        emptyStats,
        emptyStats,
        subTrickInfo.MaxBucketCount,
        subTrickInfo.MaxSplitEnsembles);
    CalcBestScoreAndCandidate(subTrickInfo, smallId, statsForSubtractionTrickSmall, gain, bestSplitCandidate, bestSplit);

    return smallStats;
}

static TVector<TBucketStats> CalculateWithSubtractTrick(
    const TSubtractTrickInfo& subTrickInfo,
    const TIndexType largeId,
    const TArrayRef<TBucketStats> smallStats,
    double* gain,
    const TCandidateInfo** bestSplitCandidate,
    TSplit* bestSplit) {

    TVector<TBucketStats> largeStats;
    CB_ENSURE(subTrickInfo.Fold->GetApproxDimension() == 1, "Subtraction trick is not implemented for MultiClass and MultiRegression");
    largeStats.yresize(subTrickInfo.StatsSize);
    CB_ENSURE(!subTrickInfo.ParentsQueue->empty());
    TStatsForSubtractionTrick statsForSubtractionTrickLarge(
        largeStats,
        subTrickInfo.ParentsQueue->front(),
        smallStats,
        subTrickInfo.MaxBucketCount,
        subTrickInfo.MaxSplitEnsembles);
    CalcBestScoreAndCandidate(subTrickInfo, largeId, statsForSubtractionTrickLarge, gain, bestSplitCandidate, bestSplit);

    return largeStats;
}

static TNonSymmetricTreeStructure GreedyTensorSearchDepthwise(
    const TTrainingDataProviders& data,
    double modelLength,
    TProfileInfo& profile,
    TVector<TIndexType>* indices,
    TFold* fold,
    TLearnContext* ctx) {

    TNonSymmetricTreeStructure currentStructure;

    const ui32 learnSampleCount = data.Learn->ObjectsData->GetObjectCount();
    TArrayRef<TIndexType> indicesRef(*indices);

    const double scoreStDev = CalcScoreStDev(learnSampleCount, modelLength, *fold, ctx);

    TVector<TIndexedSubset<ui32>> subsetsForLeafs(1 << ctx->Params.ObliviousTreeOptions->MaxDepth);
    subsetsForLeafs[0].yresize(learnSampleCount);
    std::iota(subsetsForLeafs[0].data(), subsetsForLeafs[0].data() + learnSampleCount, 0);

    const bool isSamplingPerTree = IsSamplingPerTree(ctx->Params.ObliviousTreeOptions);
    const bool isMultiClassOrMultiRegression = fold->GetApproxDimension() != 1;
    const bool isSimpleRsm = ctx->Params.ObliviousTreeOptions->Rsm == 1.0f;
    const bool isSubtractTrickAllowed = !isMultiClassOrMultiRegression && isSimpleRsm;

    TVector<TIndexType> curLevelLeafs = {0};

    TQueue<TVector<TBucketStats>> parentsQueue;

    for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
        TVector<TCandidatesContext> candidatesContexts = SelectFeaturesForScoring(data, {}, fold, ctx);

        CheckInterrupted(); // check after long-lasting operation

        if (!isSamplingPerTree) {  // sampling per tree level
            DoBootstrap(*indices, fold, ctx, currentStructure.GetLeafCount());
        }
        profile.AddOperation(TStringBuilder() << "Bootstrap, depth " << curDepth);

        TVector<TIndexType> splittedLeafs;
        TVector<TIndexType> nextLevelLeafs;

        TSubtractTrickInfo subTrickInfo(
            &data,
            &candidatesContexts,
            fold,
            ctx,
            &parentsQueue,
            scoreStDev
        );

        TSplit bestSplitNext;
        const TCandidateInfo* bestSplitCandidateNext = nullptr;
        double nextGain = 0;
        bool isStatsCalculated = false;

        if (curDepth != 0) {
            CB_ENSURE(curLevelLeafs.size() % 2 == 0);
        }

        for (size_t id = 0; id < curLevelLeafs.size(); ++id) {
            const auto& leafBounds = ctx->SampledDocs.LeavesBounds[curLevelLeafs[id]];
            const ui32 leafBoundsSize = leafBounds.GetSize();
            const ui32 nextleafBoundsSize = (id == (curLevelLeafs.size() - 1)) ? 0 : ctx->SampledDocs.LeavesBounds[curLevelLeafs[id + 1]].GetSize();
            const bool isNextLeafConsidered = nextleafBoundsSize >= ctx->Params.ObliviousTreeOptions->MinDataInLeaf;
            const bool isEvenId = id % 2 == 0;
            if (leafBoundsSize < ctx->Params.ObliviousTreeOptions->MinDataInLeaf) {
                continue;
            }

            const TCandidateInfo* bestSplitCandidate = nullptr;
            double gain = 0;
            TSplit bestSplit;

            if (!isEvenId && isStatsCalculated) {
                gain = nextGain;
                bestSplit = bestSplitNext;
                bestSplitCandidate = bestSplitCandidateNext;
                isStatsCalculated = false;
            } else if (isEvenId && (leafBoundsSize <= nextleafBoundsSize) && isNextLeafConsidered && isSubtractTrickAllowed) {
                TVector<TBucketStats> smallStats = CalculateStats(
                    subTrickInfo,
                    curLevelLeafs[id],
                    &gain,
                    &bestSplitCandidate,
                    &bestSplit);
                TVector<TBucketStats> largeStats = CalculateWithSubtractTrick(
                    subTrickInfo,
                    curLevelLeafs[id + 1],
                    smallStats,
                    &nextGain,
                    &bestSplitCandidateNext,
                    &bestSplitNext);
                subTrickInfo.ParentsQueuePop();
                ConditionalPushToParentsQueue(gain, bestSplitCandidate, std::move(smallStats), &parentsQueue);
                ConditionalPushToParentsQueue(nextGain, bestSplitCandidateNext, std::move(largeStats), &parentsQueue);
                isStatsCalculated = true;
            } else if (isEvenId && (leafBoundsSize > nextleafBoundsSize) && isNextLeafConsidered && isSubtractTrickAllowed) {
                TVector<TBucketStats> smallStats = CalculateStats(
                    subTrickInfo,
                    curLevelLeafs[id + 1],
                    &nextGain,
                    &bestSplitCandidateNext,
                    &bestSplitNext);
                TVector<TBucketStats> largeStats = CalculateWithSubtractTrick(
                    subTrickInfo,
                    curLevelLeafs[id],
                    smallStats,
                    &gain,
                    &bestSplitCandidate,
                    &bestSplit);
                subTrickInfo.ParentsQueuePop();
                ConditionalPushToParentsQueue(gain, bestSplitCandidate, std::move(largeStats), &parentsQueue);
                ConditionalPushToParentsQueue(nextGain, bestSplitCandidateNext, std::move(smallStats), &parentsQueue);
                isStatsCalculated = true;
            } else {
                TVector<TBucketStats> stats = CalculateStats(
                    subTrickInfo,
                    curLevelLeafs[id],
                    &gain,
                    &bestSplitCandidate,
                    &bestSplit);
                subTrickInfo.ParentsQueuePop();
                ConditionalPushToParentsQueue(gain, bestSplitCandidate, std::move(stats), &parentsQueue);
                isStatsCalculated = false;
            }

            if (bestSplitCandidate == nullptr) {
                continue;
            }
            if (gain < 1e-9) {
                continue;
            }
            if (bestSplit.Type == ESplitType::OnlineCtr) {
                ProcessCtrSplit(data, bestSplit, fold, ctx);
            }

            MarkFeaturesAsUsed(
                bestSplit,
                subsetsForLeafs[curLevelLeafs[id]],
                ctx->LearnProgress->EstimatedFeaturesContext,
                *ctx->Layout,
                &ctx->LearnProgress->UsedFeatures,
                &ctx->LearnProgress->UsedFeaturesPerObject
            );

            const auto& node = currentStructure.AddSplit(bestSplit, curLevelLeafs[id]);

            const TIndexType leftChildIdx = ~node.Left;
            const TIndexType rightChildIdx = ~node.Right;
            splittedLeafs.push_back(curLevelLeafs[id]);
            nextLevelLeafs.push_back(leftChildIdx);
            nextLevelLeafs.push_back(rightChildIdx);

            UpdateIndicesWithSplit(
                node,
                data,
                subsetsForLeafs[curLevelLeafs[id]],
                *fold,
                ctx->LocalExecutor,
                indicesRef,
                &subsetsForLeafs[leftChildIdx],
                &subsetsForLeafs[rightChildIdx]
            );
        }

        if (isSamplingPerTree) {
            ctx->SampledDocs.UpdateIndicesInLeafwiseSortedFold(
                splittedLeafs,
                nextLevelLeafs,
                *indices,
                ctx->LocalExecutor);
        }
        curLevelLeafs = std::move(nextLevelLeafs);

        fold->DropEmptyCTRs();
        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Build level " << curDepth);
    }

    return currentStructure;
}


void GreedyTensorSearch(
    const TTrainingDataProviders& data,
    double modelLength,
    TProfileInfo& profile,
    TFold* fold,
    TLearnContext* ctx,
    std::variant<TSplitTree, TNonSymmetricTreeStructure>* resTreeStructure) {

    TrimOnlineCTRcache({fold});

    ui32 learnSampleCount = data.Learn->ObjectsData->GetObjectCount();
    TVector<TIndexType> indices(learnSampleCount); // always for all documents

    if (!ctx->Params.SystemOptions->IsSingleHost()) {
        MapTensorSearchStart(ctx);
    }

    if (IsSamplingPerTree(ctx->Params.ObliviousTreeOptions)) {
        DoBootstrap(indices, fold, ctx, /* leavesCount */ 1);
        if (ctx->UseTreeLevelCaching()) {
            ctx->PrevTreeLevelStats.GarbageCollect();
        }
    }

    const auto growPolicy = ctx->Params.ObliviousTreeOptions.Get().GrowPolicy;
    switch(growPolicy) {
        case EGrowPolicy::SymmetricTree:
            *resTreeStructure = GreedyTensorSearchOblivious(
                data,
                modelLength,
                profile,
                &indices,
                fold,
                ctx);
            break;
        case EGrowPolicy::Lossguide:
            *resTreeStructure = GreedyTensorSearchLossguide(
                data,
                modelLength,
                profile,
                &indices,
                fold,
                ctx);
            break;
        case EGrowPolicy::Depthwise:
            *resTreeStructure = GreedyTensorSearchDepthwise(
                data,
                modelLength,
                profile,
                &indices,
                fold,
                ctx);
            break;
        default:
            CB_ENSURE(false, "GrowPolicy " << growPolicy << " is unimplemented for CPU.");
    }
}
