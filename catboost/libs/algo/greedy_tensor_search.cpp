#include "greedy_tensor_search.h"
#include "index_calcer.h"
#include "score_calcer.h"
#include "tensor_search_helpers.h"
#include "tree_print.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/helpers/interrupt.h>

#include <library/fast_log/fast_log.h>

#include <util/string/builder.h>
#include <util/system/mem_info.h>

constexpr size_t MAX_ONLINE_CTR_FEATURES = 50;

void TrimOnlineCTRcache(const TVector<TFold*>& folds) {
    for (auto& fold : folds) {
        fold->TrimOnlineCTR(MAX_ONLINE_CTR_FEATURES);
    }
}

static void AddFloatFeatures(const TDataset& learnData,
                             TLearnContext* ctx,
                             TBucketStatsCache* statsFromPrevTree,
                             TCandidateList* candList) {
    for (int f = 0; f < learnData.AllFeatures.FloatHistograms.ysize(); ++f) {
        if (learnData.AllFeatures.FloatHistograms[f].empty()) {
            continue;
        }
        TCandidateInfo split;
        split.SplitCandidate.FeatureIdx = f;
        split.SplitCandidate.Type = ESplitType::FloatFeature;

        if (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
            statsFromPrevTree->Stats.erase(split.SplitCandidate);
            continue;
        }
        candList->emplace_back(TCandidatesInfoList(split));
    }
}

static void AddOneHotFeatures(const TDataset& learnData,
                              TLearnContext* ctx,
                              TBucketStatsCache* statsFromPrevTree,
                              TCandidateList* candList) {
    for (int cf = 0; cf < learnData.AllFeatures.CatFeaturesRemapped.ysize(); ++cf) {
        if (learnData.AllFeatures.CatFeaturesRemapped[cf].empty() ||
            !learnData.AllFeatures.IsOneHot[cf]) {
            continue;
        }

        TCandidateInfo split;
        split.SplitCandidate.FeatureIdx = cf;
        split.SplitCandidate.Type = ESplitType::OneHotFeature;
        if (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
            statsFromPrevTree->Stats.erase(split.SplitCandidate);
            continue;
        }

        candList->emplace_back(TCandidatesInfoList(split));
    }
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
static void AddSimpleCtrs(const TDataset& learnData,
                          TFold* fold,
                          TLearnContext* ctx,
                          TBucketStatsCache* statsFromPrevTree,
                          TCandidateList* candList) {
    for (int cf = 0; cf < learnData.AllFeatures.CatFeaturesRemapped.ysize(); ++cf) {
        if (learnData.AllFeatures.CatFeaturesRemapped[cf].empty() ||
            learnData.AllFeatures.IsOneHot[cf]) {
            continue;
        }

        TProjection proj;
        proj.AddCatFeature(cf);

        if (ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
            DropStatsForProjection(*fold, *ctx, proj, statsFromPrevTree);
            continue;
        }
        AddCtrsToCandList(*fold, *ctx, proj, candList);
        fold->GetCtrRef(proj);
    }
}

static void AddTreeCtrs(const TDataset& learnData,
                        const TSplitTree& currentTree,
                        TFold* fold,
                        TLearnContext* ctx,
                        TBucketStatsCache* statsFromPrevTree,
                        TCandidateList* candList) {
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
        for (int cf = 0; cf < learnData.AllFeatures.CatFeaturesRemapped.ysize(); ++cf) {
            if (learnData.AllFeatures.CatFeaturesRemapped[cf].empty() ||
                learnData.AllFeatures.IsOneHot[cf] ||
                ctx->Rand.GenRandReal1() > ctx->Params.ObliviousTreeOptions->Rsm) {
                continue;
            }

            TProjection proj = baseProj;
            proj.AddCatFeature(cf);

            if (proj.IsRedundant() || proj.GetFullProjectionLength() > ctx->Params.CatFeatureParams->MaxTensorComplexity) {
                continue;
            }

            if (addedProjHash.has(proj)) {
                continue;
            }

            addedProjHash.insert(proj);

            AddCtrsToCandList(*fold, *ctx, proj, candList);
            fold->GetCtrRef(proj);
        }
    }
    THashSet<TSplitCandidate> candidatesToErase;
    for (auto& splitCandidate : statsFromPrevTree->Stats) {
        if (splitCandidate.first.Type == ESplitType::OnlineCtr) {
            if (!addedProjHash.has(splitCandidate.first.Ctr.Projection)) {
                candidatesToErase.insert(splitCandidate.first);
            }
        }
    }
    for (const auto& splitCandidate : candidatesToErase) {
        statsFromPrevTree->Stats.erase(splitCandidate);
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
        MATRIXNET_DEBUG_LOG << "Needed more memory then allowed, will drop some ctrs after score calculation" << Endl;
        const float GB = (ui64)1024 * 1024 * 1024;
        MATRIXNET_DEBUG_LOG << "current rss " << currentMemoryUsage / GB << fullNeededMemoryForCtrs / GB << Endl;
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

static void CalcBestScore(const TDataset& learnData,
        const TDataset* testData,
        const TVector<int>& splitCounts,
        int currentDepth,
        ui64 randSeed,
        double scoreStDev,
        TCandidateList* candidateList,
        TFold* fold,
        TLearnContext* ctx) {
    CB_ENSURE(static_cast<ui32>(ctx->LocalExecutor.GetThreadCount()) == ctx->Params.SystemOptions->NumThreads - 1);

    TCandidateList& candList = *candidateList;
    ctx->LocalExecutor.ExecRange([&](int id) {
        auto& candidate = candList[id];
        if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr) {
            const auto& proj = candidate.Candidates[0].SplitCandidate.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(learnData,
                                  testData,
                                  *fold,
                                  proj,
                                  ctx,
                                  &fold->GetCtrRef(proj));
            }
        }
        TVector<TVector<double>> allScores(candidate.Candidates.size());
        ctx->LocalExecutor.ExecRange([&](int oneCandidate) {
            if (candidate.Candidates[oneCandidate].SplitCandidate.Type == ESplitType::OnlineCtr) {
                const auto& proj = candidate.Candidates[oneCandidate].SplitCandidate.Ctr.Projection;
                Y_ASSERT(!fold->GetCtrRef(proj).Feature.empty());
            }
            allScores[oneCandidate] = GetScores(CalcScore(learnData.AllFeatures,
                                        splitCounts,
                                        fold->GetAllCtrs(),
                                        ctx->SampledDocs,
                                        ctx->SmallestSplitSideDocs,
                                        ctx->Params,
                                        candidate.Candidates[oneCandidate].SplitCandidate,
                                        currentDepth,
                                        &ctx->PrevTreeLevelStats));
        }, NPar::TLocalExecutor::TExecRangeParams(0, candidate.Candidates.ysize())
         , NPar::TLocalExecutor::WAIT_COMPLETE);
        if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr && candidate.ShouldDropCtrAfterCalc) {
            fold->GetCtrRef(candidate.Candidates[0].SplitCandidate.Ctr.Projection).Feature.clear();
        }
        SetBestScore(randSeed + id, allScores, scoreStDev, &candidate.Candidates);
    }, 0, candList.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

void GreedyTensorSearch(const TDataset& learnData,
                        const TDataset* testData,
                        const TVector<int>& splitCounts,
                        double modelLength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TSplitTree* resSplitTree) {
    TSplitTree currentSplitTree;
    TrimOnlineCTRcache({fold});

    int learnSampleCount = learnData.GetSampleCount();
    int testSampleCount = testData ? testData->GetSampleCount() : 0;
    TVector<TIndexType> indices(learnSampleCount); // always for all documents
    MATRIXNET_INFO_LOG << "\n";

    if (!ctx->Params.SystemOptions->IsSingleHost()) {
        MapTensorSearchStart(ctx);
    }

    const bool isSamplingPerTree = IsSamplingPerTree(ctx->Params.ObliviousTreeOptions);
    if (isSamplingPerTree) {
        if (!ctx->Params.SystemOptions->IsSingleHost()) {
            MapBootstrap(ctx);
        } else {
            Bootstrap(ctx->Params, indices, fold, &ctx->SampledDocs, &ctx->LocalExecutor, &ctx->Rand);
        }
        ctx->PrevTreeLevelStats.GarbageCollect();
    }

    for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
        TCandidateList candList;
        AddFloatFeatures(learnData, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddOneHotFeatures(learnData, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddSimpleCtrs(learnData, fold, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddTreeCtrs(learnData, currentSplitTree, fold, ctx, &ctx->PrevTreeLevelStats, &candList);

        auto IsInCache = [&fold](const TProjection& proj) -> bool {return fold->GetCtrRef(proj).Feature.empty();};
        SelectCtrsToDropAfterCalc(ctx->Params.SystemOptions->CpuUsedRamLimit, learnSampleCount + testSampleCount, ctx->Params.SystemOptions->NumThreads, IsInCache, &candList);

        CheckInterrupted(); // check after long-lasting operation
        if (!isSamplingPerTree) {
            if (!ctx->Params.SystemOptions->IsSingleHost()) {
                MapBootstrap(ctx);
            } else {
                Bootstrap(ctx->Params, indices, fold, &ctx->SampledDocs, &ctx->LocalExecutor, &ctx->Rand);
            }
        }
        profile.AddOperation(TStringBuilder() << "Bootstrap, depth " << curDepth);

        const double scoreStDev = ctx->Params.ObliviousTreeOptions->RandomStrength * CalcScoreStDev(*fold) * CalcScoreStDevMult(learnSampleCount, modelLength);
        if (!ctx->Params.SystemOptions->IsSingleHost()) {
            MapRemoteCalcScore(scoreStDev, currentSplitTree.GetDepth(), &candList, ctx);
        } else {
            const ui64 randSeed = ctx->Rand.GenRand();
            CalcBestScore(learnData, testData, splitCounts, currentSplitTree.GetDepth(), randSeed, scoreStDev, &candList, fold, ctx);
        }

        size_t maxFeatureValueCount = 1;
        for (const auto& candidate : candList) {
            const auto& split = candidate.Candidates[0].SplitCandidate;
            if (split.Type == ESplitType::OnlineCtr) {
                const auto& proj = split.Ctr.Projection;
                maxFeatureValueCount = Max(maxFeatureValueCount, fold->GetCtrRef(proj).FeatureValueCount);
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
                // MATRIXNET_INFO_LOG << BuildDescription(ctx->Layout, candidate.SplitCandidate) << " = " << score << "\t";
                TProjection projection = candidate.SplitCandidate.Ctr.Projection;
                ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(projection)[candidate.SplitCandidate.Ctr.CtrIdx].Type;

                if (candidate.SplitCandidate.Type == ESplitType::OnlineCtr &&
                    !ctx->LearnProgress.UsedCtrSplits.has(std::make_pair(ctrType, projection)) &&
                    score != MINIMAL_SCORE)
                {
                    score *= pow(1 + fold->GetCtrRef(projection).FeatureValueCount / static_cast<double>(maxFeatureValueCount),
                                 -ctx->Params.ObliviousTreeOptions->ModelSizeReg.Get());
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestSplitCandidate = &candidate;
                }
            }
        }
        // MATRIXNET_INFO_LOG << Endl;
        if (bestScore == MINIMAL_SCORE) {
            break;
        }
        if (bestSplitCandidate->SplitCandidate.Type == ESplitType::OnlineCtr) {
            TProjection projection = bestSplitCandidate->SplitCandidate.Ctr.Projection;
            ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(projection)[bestSplitCandidate->SplitCandidate.Ctr.CtrIdx].Type;

            ctx->LearnProgress.UsedCtrSplits.insert(std::make_pair(ctrType, projection));
        }
        auto bestSplit = TSplit(bestSplitCandidate->SplitCandidate, bestSplitCandidate->BestBinBorderId);
        if (bestSplit.Type == ESplitType::OnlineCtr) {
            const auto& proj = bestSplit.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(learnData,
                                  testData,
                                  *fold,
                                  proj,
                                  ctx,
                                  &fold->GetCtrRef(proj));
                DropStatsForProjection(*fold, *ctx, proj, &ctx->PrevTreeLevelStats);
            }
        }

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            SetPermutedIndices(bestSplit, learnData.AllFeatures, curDepth + 1, *fold, &indices, &ctx->LocalExecutor);
            if (isSamplingPerTree) {
                ctx->SampledDocs.UpdateIndices(indices, &ctx->LocalExecutor);
                ctx->SmallestSplitSideDocs.SelectSmallestSplitSide(curDepth + 1, ctx->SampledDocs, &ctx->LocalExecutor);
            }
        } else {
            Y_ASSERT(bestSplit.Type != ESplitType::OnlineCtr);
            MapSetIndices(*bestSplitCandidate, ctx);
        }
        currentSplitTree.AddSplit(bestSplit);
        MATRIXNET_INFO_LOG << BuildDescription(ctx->Layout, bestSplit);
        MATRIXNET_INFO_LOG << " score " << bestScore << "\n";


        profile.AddOperation(TStringBuilder() << "Select best split " << curDepth);

        int redundantIdx = -1;
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            redundantIdx = GetRedundantSplitIdx(GetIsLeafEmpty(curDepth + 1, indices));
        } else {
            redundantIdx = MapGetRedundantSplitIdx(ctx);
        }
        if (redundantIdx != -1) {
            currentSplitTree.DeleteSplit(redundantIdx);
            MATRIXNET_INFO_LOG << "  tensor " << redundantIdx << " is redundant, remove it and stop\n";
            break;
        }
    }
    *resSplitTree = std::move(currentSplitTree);
}
