#include "greedy_tensor_search.h"
#include "helpers.h"
#include "index_calcer.h"
#include "score_calcer.h"
#include "tensor_search_helpers.h"
#include "tree_print.h"

#include <catboost/libs/data_new/feature_index.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>

#include <library/dot_product/dot_product.h>
#include <library/fast_log/fast_log.h>

#include <util/string/builder.h>
#include <util/system/mem_info.h>


using namespace NCB;


constexpr size_t MAX_ONLINE_CTR_FEATURES = 50;

void TrimOnlineCTRcache(const TVector<TFold*>& folds) {
    for (auto& fold : folds) {
        fold->TrimOnlineCTR(MAX_ONLINE_CTR_FEATURES);
    }
}

static double CalcDerivativesStDevFromZeroOrderedBoosting(const TFold& fold) {
    double sum2 = 0;
    size_t count = 0;
    for (const auto& bt : fold.BodyTailArr) {
        for (const auto& perDimensionWeightedDerivatives : bt.WeightedDerivatives) {
            // TODO(yazevnul): replace with `L2NormSquared` when it's implemented
            sum2 += DotProduct(
                perDimensionWeightedDerivatives.data() + bt.BodyFinish,
                perDimensionWeightedDerivatives.data() + bt.BodyFinish,
                bt.TailFinish - bt.BodyFinish);
        }

        count += bt.TailFinish - bt.BodyFinish;
    }

    Y_ASSERT(count > 0);
    return sqrt(sum2 / count);
}

static double CalcDerivativesStDevFromZeroPlainBoosting(const TFold& fold) {
    Y_ASSERT(fold.BodyTailArr.size() == 1);
    Y_ASSERT(fold.BodyTailArr.front().WeightedDerivatives.size() > 0);

    const auto& weightedDerivatives = fold.BodyTailArr.front().WeightedDerivatives;

    double sum2 = 0;
    for (const auto& perDimensionWeightedDerivatives : weightedDerivatives) {
        sum2 += DotProduct(
            perDimensionWeightedDerivatives.data(),
            perDimensionWeightedDerivatives.data(),
            perDimensionWeightedDerivatives.size());
    }

    return sqrt(sum2 / weightedDerivatives.front().size());
}

static double CalcDerivativesStDevFromZero(const TFold& fold, const EBoostingType boosting) {
    switch (boosting) {
        case EBoostingType::Ordered:
            return CalcDerivativesStDevFromZeroOrderedBoosting(fold);
        case EBoostingType::Plain:
            return CalcDerivativesStDevFromZeroPlainBoosting(fold);
    }
}

static double CalcDerivativesStDevFromZeroMultiplier(int learnSampleCount, double modelLength) {
    double modelExpLength = log(static_cast<double>(learnSampleCount));
    double modelLeft = exp(modelExpLength - modelLength);
    return modelLeft / (1.0 + modelLeft);
}

static void AddFloatFeatures(const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
                             TLearnContext* ctx,
                             TBucketStatsCache* statsFromPrevTree,
                             TCandidateList* candList) {
    learnObjectsData.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Float>(
        [&](TFloatFeatureIdx floatFeatureIdx) {
            TCandidateInfo split;
            split.SplitCandidate.FeatureIdx = (int)*floatFeatureIdx;
            split.SplitCandidate.Type = ESplitType::FloatFeature;

            if (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
                if (ctx->UseTreeLevelCaching()) {
                    statsFromPrevTree->Stats.erase(split.SplitCandidate);
                }
                return;
            }
            candList->emplace_back(TCandidatesInfoList(split));
        }
    );
}

static void AddOneHotFeatures(const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
                              TLearnContext* ctx,
                              TBucketStatsCache* statsFromPrevTree,
                              TCandidateList* candList) {
    const auto& quantizedFeaturesInfo = *learnObjectsData.GetQuantizedFeaturesInfo();
    const ui32 oneHotMaxSize = ctx->Params.CatFeatureParams.Get().OneHotMaxSize;

    learnObjectsData.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&](TCatFeatureIdx catFeatureIdx) {
            auto onLearnOnlyCount = quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly;
            if ((onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1)) {
                return;
            }

            TCandidateInfo split;
            split.SplitCandidate.FeatureIdx = (int)*catFeatureIdx;
            split.SplitCandidate.Type = ESplitType::OneHotFeature;
            if (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
                if (ctx->UseTreeLevelCaching()) {
                    statsFromPrevTree->Stats.erase(split.SplitCandidate);
                }
                return;
            }

            candList->emplace_back(TCandidatesInfoList(split));
        }
    );
}

static void AddCtrsToCandList(const TFold& fold,
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
                TCandidateInfo split;
                split.SplitCandidate.Type = ESplitType::OnlineCtr;
                split.SplitCandidate.Ctr = TCtr(proj, ctrIdx, border, prior, ctrInfo[ctrIdx].BorderCount);
                ctrSplits.Candidates.emplace_back(split);
            }
        }
    }

    candList->push_back(ctrSplits);
}
static void DropStatsForProjection(const TFold& fold,
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
                TCandidateInfo split;
                split.SplitCandidate.Type = ESplitType::OnlineCtr;
                split.SplitCandidate.Ctr = TCtr(proj, ctrIdx, border, prior, ctrMeta.BorderCount);
                statsFromPrevTree->Stats.erase(split.SplitCandidate);
            }
        }
    }
}
static void AddSimpleCtrs(const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
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

            if (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
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

static void AddTreeCtrs(const TQuantizedForCPUObjectsDataProvider& learnObjectsData,
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
                if (isOneHot || (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm)) {
                    return;
                }

                TProjection proj = baseProj;
                proj.AddCatFeature((int)*catFeatureIdx);

                if (proj.IsRedundant() || proj.GetFullProjectionLength() > ctx->Params.CatFeatureParams->MaxTensorComplexity) {
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
        THashSet<TSplitCandidate> candidatesToErase;
        for (auto& splitCandidate : statsFromPrevTree->Stats) {
            if (splitCandidate.first.Type == ESplitType::OnlineCtr) {
                if (!addedProjHash.contains(splitCandidate.first.Ctr.Projection)) {
                    candidatesToErase.insert(splitCandidate.first);
                }
            }
        }
        for (const auto& splitCandidate : candidatesToErase) {
            statsFromPrevTree->Stats.erase(splitCandidate);
        }
    }
}

static void SelectCtrsToDropAfterCalc(size_t memoryLimit,
                                      int sampleCount,
                                      int threadCount,
                                      const std::function<bool(const TProjection&)>& IsInCache,
                                      TCandidateList* candList) {
    size_t maxMemoryForOneCtr = 0;
    size_t fullNeededMemoryForCtrs = 0;
    for (auto& candSubList : *candList) {
        const auto firstSubCandidate = candSubList.Candidates[0].SplitCandidate;
        if (firstSubCandidate.Type != ESplitType::OnlineCtr ||!IsInCache(firstSubCandidate.Ctr.Projection)) {
            candSubList.ShouldDropCtrAfterCalc = false;
            continue;
        }
        const size_t neededMem = sampleCount * candSubList.Candidates.size();
        maxMemoryForOneCtr = Max<size_t>(neededMem, maxMemoryForOneCtr);
        fullNeededMemoryForCtrs += neededMem;
    }

    auto currentMemoryUsage = NMemInfo::GetMemInfo().RSS;
    if (fullNeededMemoryForCtrs + currentMemoryUsage > memoryLimit) {
        CATBOOST_DEBUG_LOG << "Needed more memory then allowed, will drop some ctrs after score calculation" << Endl;
        const float GB = (ui64)1024 * 1024 * 1024;
        CATBOOST_DEBUG_LOG << "current rss: " << currentMemoryUsage / GB << " full needed memory: " << fullNeededMemoryForCtrs / GB << Endl;
        size_t currentNonDroppableMemory = currentMemoryUsage;
        size_t maxMemForOtherThreadsApprox = (ui64)(threadCount - 1) * maxMemoryForOneCtr;
        for (auto& candSubList : *candList) {
            const auto firstSubCandidate = candSubList.Candidates[0].SplitCandidate;
            if (firstSubCandidate.Type != ESplitType::OnlineCtr || !IsInCache(firstSubCandidate.Ctr.Projection)) {
                candSubList.ShouldDropCtrAfterCalc = false;
                continue;
            }
            candSubList.ShouldDropCtrAfterCalc = true;
            const size_t neededMem = sampleCount * candSubList.Candidates.size();
            if (currentNonDroppableMemory + neededMem + maxMemForOtherThreadsApprox <= memoryLimit) {
                candSubList.ShouldDropCtrAfterCalc = false;
                currentNonDroppableMemory += neededMem;
            }
        }
    }
}

static void CalcBestScore(const TTrainingForCPUDataProviders& data,
        const TVector<int>& splitCounts,
        int currentDepth,
        ui64 randSeed,
        double scoreStDev,
        TCandidateList* candidateList,
        TFold* fold,
        TLearnContext* ctx) {
    CB_ENSURE(static_cast<ui32>(ctx->LocalExecutor->GetThreadCount()) == ctx->Params.SystemOptions->NumThreads - 1);
    const TFlatPairsInfo pairs = UnpackPairsFromQueries(fold->LearnQueriesInfo);
    TCandidateList& candList = *candidateList;
    ctx->LocalExecutor->ExecRange([&](int id) {
        auto& candidate = candList[id];
        if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr) {
            const auto& proj = candidate.Candidates[0].SplitCandidate.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(data,
                                  *fold,
                                  proj,
                                  ctx,
                                  &fold->GetCtrRef(proj));
            }
        }
        TVector<TVector<double>> allScores(candidate.Candidates.size());
        ctx->LocalExecutor->ExecRange([&](int oneCandidate) {
            if (candidate.Candidates[oneCandidate].SplitCandidate.Type == ESplitType::OnlineCtr) {
                const auto& proj = candidate.Candidates[oneCandidate].SplitCandidate.Ctr.Projection;
                Y_ASSERT(!fold->GetCtrRef(proj).Feature.empty());
            }
            TVector<TScoreBin> scoreBins;
            CalcStatsAndScores(*data.Learn->ObjectsData,
                               splitCounts,
                               fold->GetAllCtrs(),
                               ctx->SampledDocs,
                               ctx->SmallestSplitSideDocs,
                               fold,
                               pairs,
                               ctx->Params,
                               candidate.Candidates[oneCandidate].SplitCandidate,
                               currentDepth,
                               ctx->UseTreeLevelCaching(),
                               ctx->LocalExecutor,
                               &ctx->PrevTreeLevelStats,
                               /*stats3d*/nullptr,
                               /*pairwiseStats*/nullptr,
                               &scoreBins);
            allScores[oneCandidate] = GetScores(scoreBins);
        }, NPar::TLocalExecutor::TExecRangeParams(0, candidate.Candidates.ysize())
         , NPar::TLocalExecutor::WAIT_COMPLETE);
        if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr && candidate.ShouldDropCtrAfterCalc) {
            fold->GetCtrRef(candidate.Candidates[0].SplitCandidate.Ctr.Projection).Feature.clear();
        }
        SetBestScore(randSeed + id, allScores, scoreStDev, &candidate.Candidates);
    }, 0, candList.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

void GreedyTensorSearch(const TTrainingForCPUDataProviders& data,
                        const TVector<int>& splitCounts,
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
            Bootstrap(ctx->Params, indices, fold, &ctx->SampledDocs, ctx->LocalExecutor, &ctx->Rand);
        }
        if (ctx->UseTreeLevelCaching()) {
            ctx->PrevTreeLevelStats.GarbageCollect();
        }
    }
    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());

    for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
        TCandidateList candList;
        AddFloatFeatures(*data.Learn->ObjectsData, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddOneHotFeatures(*data.Learn->ObjectsData, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddSimpleCtrs(*data.Learn->ObjectsData, fold, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddTreeCtrs(*data.Learn->ObjectsData, currentSplitTree, fold, ctx, &ctx->PrevTreeLevelStats, &candList);

        auto IsInCache = [&fold](const TProjection& proj) -> bool {return fold->GetCtrRef(proj).Feature.empty();};
        auto cpuUsedRamLimit = ParseMemorySizeDescription(ctx->Params.SystemOptions->CpuUsedRamLimit.Get());
        SelectCtrsToDropAfterCalc(cpuUsedRamLimit, learnSampleCount + testSampleCount, ctx->Params.SystemOptions->NumThreads, IsInCache, &candList);

        CheckInterrupted(); // check after long-lasting operation
        if (!isSamplingPerTree) {
            if (!ctx->Params.SystemOptions->IsSingleHost()) {
                MapBootstrap(ctx);
            } else {
                Bootstrap(ctx->Params, indices, fold, &ctx->SampledDocs, ctx->LocalExecutor, &ctx->Rand);
            }
        }
        profile.AddOperation(TStringBuilder() << "Bootstrap, depth " << curDepth);

        const auto scoreStDev =
            ctx->Params.ObliviousTreeOptions->RandomStrength
            * CalcDerivativesStDevFromZero(*fold, ctx->Params.BoostingOptions->BoostingType)
            * CalcDerivativesStDevFromZeroMultiplier(learnSampleCount, modelLength);
        if (!ctx->Params.SystemOptions->IsSingleHost()) {
            if (isPairwiseScoring) {
                MapRemotePairwiseCalcScore(scoreStDev, &candList, ctx);
            } else {
                MapRemoteCalcScore(scoreStDev, &candList, ctx);
            }
        } else {
            const ui64 randSeed = ctx->Rand.GenRand();
            CalcBestScore(data, splitCounts, currentSplitTree.GetDepth(), randSeed, scoreStDev, &candList, fold, ctx);
        }

        size_t maxFeatureValueCount = 1;
        for (const auto& candidate : candList) {
            const auto& split = candidate.Candidates[0].SplitCandidate;
            if (split.Type == ESplitType::OnlineCtr) {
                const auto& proj = split.Ctr.Projection;
                maxFeatureValueCount = Max(maxFeatureValueCount, fold->GetCtrRef(proj).GetMaxUniqueValueCount());
            }
        }

        fold->DropEmptyCTRs();
        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Calc scores " << curDepth);

        const TCandidateInfo* bestSplitCandidate = nullptr;
        double bestScore = MINIMAL_SCORE;
        for (const auto& subList : candList) {
            for (const auto& candidate : subList.Candidates) {
                double score = candidate.BestScore.GetInstance(ctx->Rand);
                // CATBOOST_INFO_LOG << BuildDescription(ctx->Layout, candidate.SplitCandidate) << " = " << score << "\t";
                TProjection projection = candidate.SplitCandidate.Ctr.Projection;
                ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(projection)[candidate.SplitCandidate.Ctr.CtrIdx].Type;

                if (candidate.SplitCandidate.Type == ESplitType::OnlineCtr &&
                    !ctx->LearnProgress.UsedCtrSplits.contains(std::make_pair(ctrType, projection)) &&
                    score != MINIMAL_SCORE)
                {
                    score *= pow(
                        1 + fold->GetCtrRef(projection).GetUniqueValueCountForType(ctrType) / static_cast<double>(maxFeatureValueCount),
                        -ctx->Params.ObliviousTreeOptions->ModelSizeReg.Get()
                    );
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
        if (bestSplitCandidate->SplitCandidate.Type == ESplitType::OnlineCtr) {
            TProjection projection = bestSplitCandidate->SplitCandidate.Ctr.Projection;
            ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(projection)[bestSplitCandidate->SplitCandidate.Ctr.CtrIdx].Type;

            ctx->LearnProgress.UsedCtrSplits.insert(std::make_pair(ctrType, projection));
        }
        auto bestSplit = TSplit(bestSplitCandidate->SplitCandidate, bestSplitCandidate->BestBinBorderId);
        if (bestSplit.Type == ESplitType::OnlineCtr) {
            const auto& proj = bestSplit.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(data,
                                  *fold,
                                  proj,
                                  ctx,
                                  &fold->GetCtrRef(proj));
                if (ctx->UseTreeLevelCaching()) {
                    DropStatsForProjection(*fold, *ctx, proj, &ctx->PrevTreeLevelStats);
                }
            }
        }

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            SetPermutedIndices(bestSplit, *data.Learn->ObjectsData, curDepth + 1, *fold, &indices, ctx->LocalExecutor);
            if (isSamplingPerTree) {
                ctx->SampledDocs.UpdateIndices(indices, ctx->LocalExecutor);
                if (ctx->UseTreeLevelCaching()) {
                    ctx->SmallestSplitSideDocs.SelectSmallestSplitSide(curDepth + 1, ctx->SampledDocs, ctx->LocalExecutor);
                }
            }
        } else {
            Y_ASSERT(bestSplit.Type != ESplitType::OnlineCtr);
            MapSetIndices(*bestSplitCandidate, ctx);
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
