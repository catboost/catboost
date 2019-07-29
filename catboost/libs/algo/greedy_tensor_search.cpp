#include "greedy_tensor_search.h"

#include "fold.h"
#include "helpers.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "score_calcer.h"
#include "split.h"
#include "tensor_search_helpers.h"
#include "tree_print.h"
#include "monotonic_constraint_utils.h"

#include <catboost/libs/data_new/feature_index.h>
#include <catboost/libs/data_new/packed_binary_features.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/logging/profile_info.h>

#include <library/fast_log/fast_log.h>

#include <util/generic/cast.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/system/mem_info.h>


using namespace NCB;


constexpr size_t MAX_ONLINE_CTR_FEATURES = 50;

void TrimOnlineCTRcache(const TVector<TFold*>& folds) {
    for (auto& fold : folds) {
        fold->TrimOnlineCTR(MAX_ONLINE_CTR_FEATURES);
    }
}

static double CalcDerivativesStDevFromZeroOrderedBoosting(
    const TFold& fold,
    NPar::TLocalExecutor* localExecutor
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
    NPar::TLocalExecutor* localExecutor
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
    NPar::TLocalExecutor* localExecutor
) {
    switch (boosting) {
        case EBoostingType::Ordered:
            return CalcDerivativesStDevFromZeroOrderedBoosting(fold, localExecutor);
        case EBoostingType::Plain:
            return CalcDerivativesStDevFromZeroPlainBoosting(fold, localExecutor);
    }
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
    const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
    TCandidateList* candList) {

    learnObjectsData.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Float>(
        [&](TFloatFeatureIdx floatFeatureIdx) {
            TSplitCandidate splitCandidate;
            splitCandidate.FeatureIdx = (int)*floatFeatureIdx;
            splitCandidate.Type = ESplitType::FloatFeature;

            TSplitEnsemble splitEnsemble(std::move(splitCandidate));
            TCandidateInfo candidate;
            candidate.SplitEnsemble = std::move(splitEnsemble);
            candList->emplace_back(TCandidatesInfoList(candidate));
        }
    );
}


static void AddOneHotFeatures(
    const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
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
            candList->emplace_back(TCandidatesInfoList(candidate));
        }
    );
}


static void CompressCandidates(
    const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
    TCandidatesContext* candidatesContext) {

    auto& candList = candidatesContext->CandidateList;
    auto& selectedFeaturesInBundles = candidatesContext->SelectedFeaturesInBundles;
    auto& perBinaryPackMasks = candidatesContext->PerBinaryPackMasks;

    selectedFeaturesInBundles.assign(learnObjectsData.GetExclusiveFeatureBundlesSize(), TVector<ui32>());
    perBinaryPackMasks.assign(learnObjectsData.GetBinaryFeaturesPacksSize(), TBinaryFeaturesPack(0));

    TCandidateList updatedCandList;
    updatedCandList.reserve(candList.size());

    for (auto& candSubList : candList) {
        const auto& splitEnsemble = candSubList.Candidates[0].SplitEnsemble;
        Y_ASSERT(splitEnsemble.Type == ESplitEnsembleType::OneFeature);
        const auto& splitCandidate = splitEnsemble.SplitCandidate;

        TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex;
        TMaybe<TPackedBinaryIndex> maybePackedBinaryIndex;

        if (splitCandidate.Type == ESplitType::FloatFeature) {
            auto floatFeatureIdx = TFloatFeatureIdx(splitCandidate.FeatureIdx);
            maybeExclusiveBundleIndex = learnObjectsData.GetFeatureToExclusiveBundleIndex(floatFeatureIdx);
            maybePackedBinaryIndex = learnObjectsData.GetFeatureToPackedBinaryIndex(floatFeatureIdx);
        } else {
            auto catFeatureIdx = TCatFeatureIdx(splitCandidate.FeatureIdx);
            maybeExclusiveBundleIndex = learnObjectsData.GetFeatureToExclusiveBundleIndex(catFeatureIdx);
            maybePackedBinaryIndex = learnObjectsData.GetFeatureToPackedBinaryIndex(catFeatureIdx);
        }
        CB_ENSURE_INTERNAL(
            !maybeExclusiveBundleIndex || !maybePackedBinaryIndex,
            "Feature #" << learnObjectsData.GetQuantizedFeaturesInfo()
                ->GetFeaturesLayout()->GetExternalFeatureIdx(
                    splitCandidate.FeatureIdx,
                    (splitCandidate.Type == ESplitType::FloatFeature) ?
                        EFeatureType::Float :
                        EFeatureType::Categorical
                )
            << " is both in an exclusive bundle and in a binary pack");

        if (maybeExclusiveBundleIndex) {
            selectedFeaturesInBundles[maybeExclusiveBundleIndex->BundleIdx].push_back(
                maybeExclusiveBundleIndex->InBundleIdx);
        } else if (maybePackedBinaryIndex) {
            MarkFeatureAsIncluded(*maybePackedBinaryIndex, &perBinaryPackMasks);
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
        updatedCandList.emplace_back(TCandidatesInfoList(candidate));
    }


    for (auto packIdx : xrange(SafeIntegerCast<ui32>(perBinaryPackMasks.size()))) {
        TCandidateInfo candidate;
        candidate.SplitEnsemble = TSplitEnsemble{TBinarySplitsPackRef{packIdx}};
        updatedCandList.emplace_back(TCandidatesInfoList(candidate));
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

    TCandidateList updatedCandList;
    updatedCandList.reserve(candList.size());

    for (auto& candSubList : candList) {
        const auto& splitEnsemble = candSubList.Candidates[0].SplitEnsemble;

        bool addCandSubListToResult;

        switch (splitEnsemble.Type) {
            case ESplitEnsembleType::OneFeature:
                addCandSubListToResult
                    = ctx->LearnProgress->Rand.GenRandReal1() <= ctx->Params.ObliviousTreeOptions->Rsm;
                break;
            case ESplitEnsembleType::BinarySplits:
                {
                    const ui32 packIdx = splitEnsemble.BinarySplitsPackRef.PackIdx;
                    TBinaryFeaturesPack& perPackMask = perBinaryPackMasks[packIdx];
                    for (size_t idxInPack : xrange(sizeof(TBinaryFeaturesPack) * CHAR_BIT)) {
                        if ((perPackMask >> idxInPack) & 1) {
                            const bool addToCandidates
                                = ctx->LearnProgress->Rand.GenRandReal1()
                                    <= ctx->Params.ObliviousTreeOptions->Rsm;
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
                        const bool addToCandidates
                            = ctx->LearnProgress->Rand.GenRandReal1() <= ctx->Params.ObliviousTreeOptions->Rsm;
                        if (addToCandidates) {
                            filteredFeaturesInBundle.push_back(inBundleIdx);
                        }
                    }
                    selectedFeaturesInBundle = std::move(filteredFeaturesInBundle);
                    addCandSubListToResult = !selectedFeaturesInBundle.empty();
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
    const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
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
            fold->GetCtrRef(proj);
        }
    );
}

static void AddTreeCtrs(
    const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
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

    for (const auto& ctrSplit : currentTree.GetCtrSplits()) {
        seenProj.insert(ctrSplit.Projection);
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
                fold->GetCtrRef(proj);
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

    auto currentMemoryUsage = NMemInfo::GetMemInfo().RSS;
    if (fullNeededMemoryForCtrs + currentMemoryUsage > memoryLimit) {
        CATBOOST_DEBUG_LOG << "Needed more memory then allowed, will drop some ctrs after score calculation"
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
    const TTrainingForCPUDataProviders& data,
    const TSplitTree& currentTree,
    ui64 randSeed,
    double scoreStDev,
    TCandidatesContext* candidatesContext,
    TFold* fold,
    TLearnContext* ctx) {

    const TFlatPairsInfo pairs = UnpackPairsFromQueries(fold->LearnQueriesInfo);
    TCandidateList& candList = candidatesContext->CandidateList;
    const auto& monotonicConstraints = ctx->Params.ObliviousTreeOptions->MonotoneConstraints.Get();
    const TVector<int> currTreeMonotonicConstraints = (
        monotonicConstraints.empty()
        ? TVector<int>()
        : GetTreeMonotoneConstraints(currentTree, monotonicConstraints)
    );
    ctx->LocalExecutor->ExecRange(
        [&](int id) {
            auto& candidate = candList[id];

            const auto& splitEnsemble = candidate.Candidates[0].SplitEnsemble;

            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                if (fold->GetCtrRef(proj).Feature.empty()) {
                    ComputeOnlineCTRs(
                        data,
                        *fold,
                        proj,
                        ctx,
                        &fold->GetCtrRef(proj));
                }
            }
            TVector<TVector<double>> allScores(candidate.Candidates.size());
            ctx->LocalExecutor->ExecRange(
                [&](int oneCandidate) {
                    const auto& candidateInfo = candidate.Candidates[oneCandidate];
                    const auto& splitEnsemble = candidateInfo.SplitEnsemble;

                    if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                        const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                        Y_ASSERT(!fold->GetCtrRef(proj).Feature.empty());
                    }
                    TVector<TScoreBin> scoreBins;
                    CalcStatsAndScores(
                        *data.Learn->ObjectsData,
                        fold->GetAllCtrs(),
                        ctx->SampledDocs,
                        ctx->SmallestSplitSideDocs,
                        fold,
                        pairs,
                        ctx->Params,
                        candidateInfo,
                        currentTree.GetDepth(),
                        ctx->UseTreeLevelCaching(),
                        currTreeMonotonicConstraints,
                        monotonicConstraints,
                        ctx->LocalExecutor,
                        &ctx->PrevTreeLevelStats,
                        /*stats3d*/nullptr,
                        /*pairwiseStats*/nullptr,
                        &scoreBins);
                    allScores[oneCandidate] = GetScores(scoreBins);
                },
                NPar::TLocalExecutor::TExecRangeParams(0, candidate.Candidates.ysize()),
                NPar::TLocalExecutor::WAIT_COMPLETE);

            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr) && candidate.ShouldDropCtrAfterCalc) {
                fold->GetCtrRef(splitEnsemble.SplitCandidate.Ctr.Projection).Feature.clear();
            }

            SetBestScore(
                randSeed + id,
                allScores,
                scoreStDev,
                *candidatesContext,
                &candidate.Candidates);
        },
        0,
        candList.ysize(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

void GreedyTensorSearch(
    const TTrainingForCPUDataProviders& data,
    double modelLength,
    TProfileInfo& profile,
    TFold* fold,
    TLearnContext* ctx,
    TSplitTree* resSplitTree) {

    TSplitTree currentSplitTree;
    TrimOnlineCTRcache({fold});

    ui32 learnSampleCount = data.Learn->ObjectsData->GetObjectCount();
    ui32 testSampleCount = data.GetTestSampleCount();
    TVector<TIndexType> indices(learnSampleCount); // always for all documents
    CATBOOST_INFO_LOG << "\n";

    if (!ctx->Params.SystemOptions->IsSingleHost()) {
        MapTensorSearchStart(ctx);
    }

    const bool isSamplingPerTree = IsSamplingPerTree(ctx->Params.ObliviousTreeOptions);
    if (isSamplingPerTree) {
        if (!ctx->Params.SystemOptions->IsSingleHost()) {
            MapBootstrap(ctx);
        } else {
            Bootstrap(ctx->Params, indices, fold, &ctx->SampledDocs, ctx->LocalExecutor, &ctx->LearnProgress->Rand);
        }
        if (ctx->UseTreeLevelCaching()) {
            ctx->PrevTreeLevelStats.GarbageCollect();
        }
    }
    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());

    for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
        TCandidatesContext candidatesContext;
        candidatesContext.OneHotMaxSize = ctx->Params.CatFeatureParams->OneHotMaxSize;
        candidatesContext.BundlesMetaData = data.Learn->ObjectsData->GetExclusiveFeatureBundlesMetaData();

        AddFloatFeatures(*data.Learn->ObjectsData, &candidatesContext.CandidateList);
        AddOneHotFeatures(*data.Learn->ObjectsData, ctx, &candidatesContext.CandidateList);
        CompressCandidates(*data.Learn->ObjectsData, &candidatesContext);
        SelectCandidatesAndCleanupStatsFromPrevTree(ctx, &candidatesContext, &ctx->PrevTreeLevelStats);

        AddSimpleCtrs(
            *data.Learn->ObjectsData,
            fold,
            ctx,
            &ctx->PrevTreeLevelStats,
            &candidatesContext.CandidateList);
        AddTreeCtrs(
            *data.Learn->ObjectsData,
            currentSplitTree,
            fold,
            ctx,
            &ctx->PrevTreeLevelStats,
            &candidatesContext.CandidateList);

        auto isInCache =
            [&fold](const TProjection& proj) -> bool { return fold->GetCtrRef(proj).Feature.empty(); };
        auto cpuUsedRamLimit = ParseMemorySizeDescription(ctx->Params.SystemOptions->CpuUsedRamLimit.Get());
        SelectCtrsToDropAfterCalc(
            cpuUsedRamLimit,
            learnSampleCount + testSampleCount,
            ctx->Params.SystemOptions->NumThreads,
            isInCache,
            &candidatesContext.CandidateList);

        CheckInterrupted(); // check after long-lasting operation
        if (!isSamplingPerTree) {
            if (!ctx->Params.SystemOptions->IsSingleHost()) {
                MapBootstrap(ctx);
            } else {
                Bootstrap(
                    ctx->Params,
                    indices,
                    fold,
                    &ctx->SampledDocs,
                    ctx->LocalExecutor,
                    &ctx->LearnProgress->Rand);
            }
        }
        profile.AddOperation(TStringBuilder() << "Bootstrap, depth " << curDepth);

        const auto scoreStDev =
            ctx->Params.ObliviousTreeOptions->RandomStrength
            * CalcDerivativesStDevFromZero(*fold, ctx->Params.BoostingOptions->BoostingType, ctx->LocalExecutor)
            * CalcDerivativesStDevFromZeroMultiplier(learnSampleCount, modelLength);
        if (!ctx->Params.SystemOptions->IsSingleHost()) {
            if (isPairwiseScoring) {
                MapRemotePairwiseCalcScore(scoreStDev, &candidatesContext, ctx);
            } else {
                MapRemoteCalcScore(scoreStDev, &candidatesContext, ctx);
            }
        } else {
            const ui64 randSeed = ctx->LearnProgress->Rand.GenRand();
            CalcBestScore(
                data,
                currentSplitTree,
                randSeed,
                scoreStDev,
                &candidatesContext,
                fold,
                ctx);
        }

        size_t maxFeatureValueCount = 1;
        for (const auto& candidate : candidatesContext.CandidateList) {
            const auto& splitEnsemble = candidate.Candidates[0].SplitEnsemble;
            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                maxFeatureValueCount = Max(
                    maxFeatureValueCount,
                    fold->GetCtrRef(proj).GetMaxUniqueValueCount());
            }
        }

        fold->DropEmptyCTRs();
        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Calc scores " << curDepth);

        const TCandidateInfo* bestSplitCandidate = nullptr;
        double bestScore = MINIMAL_SCORE;
        for (const auto& subList : candidatesContext.CandidateList) {
            for (const auto& candidate : subList.Candidates) {
                double score = candidate.BestScore.GetInstance(ctx->LearnProgress->Rand);
                // CATBOOST_INFO_LOG << BuildDescription(ctx->Layout, candidate.SplitCandidate) << " = "
                //     << score << "\t";

                const auto& splitEnsemble = candidate.SplitEnsemble;
                if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                    TProjection projection = splitEnsemble.SplitCandidate.Ctr.Projection;
                    ECtrType ctrType =
                        ctx->CtrsHelper.GetCtrInfo(projection)[splitEnsemble.SplitCandidate.Ctr.CtrIdx].Type;

                    if (!ctx->LearnProgress->UsedCtrSplits.contains(std::make_pair(ctrType, projection)) &&
                        score != MINIMAL_SCORE)
                    {
                        score *= pow(
                            1 + (fold->GetCtrRef(projection).GetUniqueValueCountForType(ctrType) /
                                static_cast<double>(maxFeatureValueCount)),
                            -ctx->Params.ObliviousTreeOptions->ModelSizeReg.Get());
                    }
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestSplitCandidate = &candidate;
                }
            }
        }
        // CATBOOST_INFO_LOG << Endl;
        if (bestScore == MINIMAL_SCORE) {
            break;
        }
        Y_ASSERT(bestSplitCandidate != nullptr);

        const auto& bestSplitEnsemble = bestSplitCandidate->SplitEnsemble;
        if (bestSplitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
            const auto& ctr = bestSplitEnsemble.SplitCandidate.Ctr;

            ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(ctr.Projection)[ctr.CtrIdx].Type;
            ctx->LearnProgress->UsedCtrSplits.insert(std::make_pair(ctrType, ctr.Projection));
        }
        TSplit bestSplit = bestSplitCandidate->GetBestSplit(
            *data.Learn->ObjectsData,
            candidatesContext.OneHotMaxSize);

        if (bestSplit.Type == ESplitType::OnlineCtr) {
            const auto& proj = bestSplit.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(data, *fold, proj, ctx, &fold->GetCtrRef(proj));
                if (ctx->UseTreeLevelCaching()) {
                    DropStatsForProjection(*fold, *ctx, proj, &ctx->PrevTreeLevelStats);
                }
            }
        }

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            SetPermutedIndices(
                bestSplit,
                *data.Learn->ObjectsData,
                curDepth + 1,
                *fold,
                &indices,
                ctx->LocalExecutor);
            if (isSamplingPerTree) {
                ctx->SampledDocs.UpdateIndices(indices, ctx->LocalExecutor);
                if (ctx->UseTreeLevelCaching()) {
                    ctx->SmallestSplitSideDocs.SelectSmallestSplitSide(
                        curDepth + 1,
                        ctx->SampledDocs,
                        ctx->LocalExecutor);
                }
            }
        } else {
            Y_ASSERT(bestSplit.Type != ESplitType::OnlineCtr);
            MapSetIndices(bestSplit, ctx);
        }
        currentSplitTree.AddSplit(bestSplit);
        CATBOOST_INFO_LOG << BuildDescription(*ctx->Layout, bestSplit) << " score " << bestScore << "\n";

        profile.AddOperation(TStringBuilder() << "Select best split " << curDepth);

        int redundantIdx = -1;
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            redundantIdx = GetRedundantSplitIdx(GetIsLeafEmpty(curDepth + 1, indices));
        } else {
            redundantIdx = MapGetRedundantSplitIdx(ctx);
        }
        if (redundantIdx != -1) {
            currentSplitTree.DeleteSplit(redundantIdx);
            CATBOOST_INFO_LOG << "  tensor " << redundantIdx << " is redundant, remove it and stop\n";
            break;
        }
    }
    *resSplitTree = std::move(currentSplitTree);
}
