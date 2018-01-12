#include "greedy_tensor_search.h"
#include "index_calcer.h"
#include "score_calcer.h"
#include "tree_print.h"


#include <catboost/libs/logging/profile_info.h>

#include <catboost/libs/helpers/interrupt.h>

#include <library/dot_product/dot_product.h>
#include <library/fast_log/fast_log.h>

#include <util/string/builder.h>
#include <util/system/mem_info.h>

constexpr size_t MAX_ONLINE_CTR_FEATURES = 50;

void TrimOnlineCTRcache(const TVector<TFold*>& folds) {
    for (auto& fold : folds) {
        fold->TrimOnlineCTR(MAX_ONLINE_CTR_FEATURES);
    }
}

static void GenerateRandomWeights(int learnSampleCount,
                                  TLearnContext* ctx,
                                  TFold* fold) {
    const float baggingTemperature = ctx->Params.ObliviousTreeOptions->BootstrapConfig->GetBaggingTemperature();
    if (baggingTemperature == 0) {
        return;
    }
    TVector<float> sampleWeights;
    sampleWeights.yresize(learnSampleCount);

    const ui64 randSeed = ctx->Rand.GenRand();
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, learnSampleCount);
    blockParams.SetBlockSize(10000);
    ctx->LocalExecutor.ExecRange([&](int blockIdx) {
        TFastRng64 rand(randSeed + blockIdx);
        rand.Advance(10); // reduce correlation between RNGs in different threads
        float* sampleWeightsData = sampleWeights.data();
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&rand, sampleWeightsData, baggingTemperature](int i) {
            const float w = -FastLogf(rand.GenRandReal1() + 1e-100);
            sampleWeightsData[i] = powf(w, baggingTemperature);
        })(blockIdx);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    TFold& ff = *fold;
    ff.AssignPermuted(sampleWeights, &ff.SampleWeights);
}


static void CalcWeightedData(int learnSampleCount,
                                TLearnContext* ctx,
                                TFold* fold) {
    TFold& ff = *fold;
    if (!ff.LearnWeights.empty()) {
        for (int i = 0; i < learnSampleCount; ++i) {
            ff.SampleWeights[i] *= ff.LearnWeights[i];
        }
    }

    const int approxDimension = ff.GetApproxDimension();
    for (TFold::TBodyTail& bt : ff.BodyTailArr) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            double* weightedDerData = bt.WeightedDer[dim].data();
            const double* derData = bt.Derivatives[dim].data();
            const float* sampleWeightsData = ff.SampleWeights.data();
            ctx->LocalExecutor.ExecRange([=](int z) {
                weightedDerData[z] = derData[z] * sampleWeightsData[z];
            }, NPar::TLocalExecutor::TExecRangeParams(bt.BodyFinish, bt.TailFinish).SetBlockSize(4000)
             , NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
}


struct TCandidateInfo {
    TSplitCandidate SplitCandidate;
    TRandomScore BestScore;
    int BestBinBorderId;
    bool ShouldDropAfterScoreCalc;
};

struct TCandidatesInfoList {
    TCandidatesInfoList() = default;
    explicit TCandidatesInfoList(const TCandidateInfo& oneCandidate) {
        Candidates.emplace_back(oneCandidate);
    }
    // All candidates here are either float or one-hot, or have the same
    // projection.
    // TODO(annaveronika): put projection out, because currently it's not clear.
    TVector<TCandidateInfo> Candidates;
    bool ShouldDropCtrAfterCalc = false;
    size_t ResultingCtrTableSize = 0;
};

using TCandidateList = TVector<TCandidatesInfoList>;

static void AddFloatFeatures(const TTrainData& data,
                             TLearnContext* ctx,
                             TBucketStatsCache* statsFromPrevTree,
                             TCandidateList* candList) {
    for (int f = 0; f < data.AllFeatures.FloatHistograms.ysize(); ++f) {
        if (data.AllFeatures.FloatHistograms[f].empty()) {
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

static void AddOneHotFeatures(const TTrainData& data,
                              TLearnContext* ctx,
                              TBucketStatsCache* statsFromPrevTree,
                              TCandidateList* candList) {
    for (int cf = 0; cf < data.AllFeatures.CatFeatures.ysize(); ++cf) {
        if (data.AllFeatures.CatFeatures[cf].empty() ||
            !data.AllFeatures.IsOneHot[cf]) {
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
static void AddSimpleCtrs(const TTrainData& data,
                          TFold* fold,
                          TLearnContext* ctx,
                          TBucketStatsCache* statsFromPrevTree,
                          TCandidateList* candList) {
    for (int cf = 0; cf < data.AllFeatures.CatFeatures.ysize(); ++cf) {
        if (data.AllFeatures.CatFeatures[cf].empty() ||
            data.AllFeatures.IsOneHot[cf]) {
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

static void AddTreeCtrs(const TTrainData& data,
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
        for (int cf = 0; cf < data.AllFeatures.CatFeatures.ysize(); ++cf) {
            if (data.AllFeatures.CatFeatures[cf].empty() ||
                data.AllFeatures.IsOneHot[cf] ||
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

static double CalcScoreStDev(const TFold& ff) {
    double sum2 = 0, totalSum2Count = 0;
    for (const TFold::TBodyTail& bt : ff.BodyTailArr) {
        for (int dim = 0; dim < bt.Derivatives.ysize(); ++dim) {
            sum2 += DotProduct(bt.Derivatives[dim].data() + bt.BodyFinish, bt.Derivatives[dim].data() + bt.BodyFinish, bt.TailFinish - bt.BodyFinish);
        }
        totalSum2Count += bt.TailFinish - bt.BodyFinish;
    }
    return sqrt(sum2 / Max(totalSum2Count, DBL_EPSILON));
}

static double CalcScoreStDevMult(int learnSampleCount, double modelLength) {
    double modelExpLength = log(learnSampleCount * 1.0);
    double modelLeft = exp(modelExpLength - modelLength);
    return modelLeft / (1 + modelLeft);
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

static void Bootstrap(const NCatboostOptions::TOption<NCatboostOptions::TBootstrapConfig>& samplingConfig,
               const TVector<TIndexType>& indices,
               int learnSampleCount,
               TFold* fold,
               TLearnContext* ctx) {
    float takenFraction = 1;
    switch (samplingConfig->GetBootstrapType()) {
        case EBootstrapType::Bernoulli:
            takenFraction = samplingConfig->GetTakenFraction();
            break;
        case EBootstrapType::Bayesian:
            GenerateRandomWeights(learnSampleCount, ctx, fold);
            break;
        case EBootstrapType::No:
            break;
        default:
            CB_ENSURE(false, "Not supported bootstrap type on CPU: " << samplingConfig->GetBootstrapType());
    }
    CalcWeightedData(learnSampleCount, ctx, fold);
    ctx->SampledDocs.Sample(*fold, indices, takenFraction, &ctx->Rand, &ctx->LocalExecutor);
}

void GreedyTensorSearch(const TTrainData& data,
                        const TVector<int>& splitCounts,
                        double modelLength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TSplitTree* resSplitTree) {
    TSplitTree currentSplitTree;
    TrimOnlineCTRcache({fold});

    TVector<TIndexType> indices(data.LearnSampleCount); // always for all documents
    MATRIXNET_INFO_LOG << "\n";

    const bool isSamplingPerTree = IsSamplingPerTree(ctx->Params.ObliviousTreeOptions.Get());
    const auto& samplingConfig = ctx->Params.ObliviousTreeOptions->BootstrapConfig;
    if (isSamplingPerTree) {
        Bootstrap(samplingConfig, indices, data.LearnSampleCount, fold, ctx);
        ctx->PrevTreeLevelStats.GarbageCollect();
    }

    for (ui32 curDepth = 0; curDepth < ctx->Params.ObliviousTreeOptions->MaxDepth; ++curDepth) {
        TCandidateList candList;
        AddFloatFeatures(data, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddOneHotFeatures(data, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddSimpleCtrs(data, fold, ctx, &ctx->PrevTreeLevelStats, &candList);
        AddTreeCtrs(data, currentSplitTree, fold, ctx, &ctx->PrevTreeLevelStats, &candList);

        auto IsInCache = [&fold](const TProjection& proj) -> bool {return fold->GetCtrRef(proj).Feature.empty();};
        SelectCtrsToDropAfterCalc(ctx->Params.SystemOptions->CpuUsedRamLimit, data.GetSampleCount(), ctx->Params.SystemOptions->NumThreads, IsInCache, &candList);

        CheckInterrupted(); // check after long-lasting operation
        if (!isSamplingPerTree) {
            Bootstrap(samplingConfig, indices, data.LearnSampleCount, fold, ctx);
        }
        profile.AddOperation(TStringBuilder() << "Bootstrap, depth " << curDepth);
        double scoreStDev = ctx->Params.ObliviousTreeOptions->RandomStrength * CalcScoreStDev(*fold) * CalcScoreStDevMult(data.LearnSampleCount, modelLength);

        TVector<size_t> candLeafCount(candList.ysize(), 1);
        const ui64 randSeed = ctx->Rand.GenRand();
        CB_ENSURE(static_cast<ui32>(ctx->LocalExecutor.GetThreadCount()) == ctx->Params.SystemOptions->NumThreads - 1);
        ctx->LocalExecutor.ExecRange([&](int id) {
            auto& candidate = candList[id];
            if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr) {
                const auto& proj = candidate.Candidates[0].SplitCandidate.Ctr.Projection;
                if (fold->GetCtrRef(proj).Feature.empty()) {
                    ComputeOnlineCTRs(data,
                                      *fold,
                                      proj,
                                      ctx,
                                      &fold->GetCtrRef(proj),
                                      &candidate.ResultingCtrTableSize);
                    candLeafCount[id] = candidate.ResultingCtrTableSize;
                }
            }
            TVector<TVector<double>> allScores(candidate.Candidates.size());
            ctx->LocalExecutor.ExecRange([&](int oneCandidate) {
                if (candidate.Candidates[oneCandidate].SplitCandidate.Type == ESplitType::OnlineCtr) {
                    const auto& proj = candidate.Candidates[oneCandidate].SplitCandidate.Ctr.Projection;
                    Y_ASSERT(!fold->GetCtrRef(proj).Feature.empty());
                }
                allScores[oneCandidate] = CalcScore(data.AllFeatures,
                                                    splitCounts,
                                                    fold->GetAllCtrs(),
                                                    ctx->SampledDocs,
                                                    ctx->SmallestSplitSideDocs,
                                                    ctx->Params,
                                                    candidate.Candidates[oneCandidate].SplitCandidate,
                                                    currentSplitTree.GetDepth(),
                                                    &ctx->PrevTreeLevelStats);
            }, NPar::TLocalExecutor::TExecRangeParams(0, candidate.Candidates.ysize())
             , NPar::TLocalExecutor::WAIT_COMPLETE);
            if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr && candidate.ShouldDropCtrAfterCalc) {
                fold->GetCtrRef(candidate.Candidates[0].SplitCandidate.Ctr.Projection).Feature.clear();
            }
            TFastRng64 rand(randSeed + id);
            rand.Advance(10); // reduce correlation between RNGs in different threads
            for (size_t i = 0; i < allScores.size(); ++i) {
                double bestScoreInstance = MINIMAL_SCORE;
                auto& splitInfo = candidate.Candidates[i];
                const auto& scores = allScores[i];
                for (int binFeatureIdx = 0; binFeatureIdx < scores.ysize(); ++binFeatureIdx) {
                    const double score = scores[binFeatureIdx];
                    const double scoreInstance = TRandomScore(score, scoreStDev).GetInstance(rand);
                    if (scoreInstance > bestScoreInstance) {
                        bestScoreInstance = scoreInstance;
                        splitInfo.BestScore = TRandomScore(score, scoreStDev);
                        splitInfo.BestBinBorderId = binFeatureIdx;
                    }
                }
            }
        }, 0, candList.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        size_t maxLeafCount = 1;
        for (size_t leafCount : candLeafCount) {
            maxLeafCount = Max(maxLeafCount, leafCount);
        }
        fold->DropEmptyCTRs();
        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Calc scores " << curDepth);

        const TCandidateInfo* bestSplitCandidate = nullptr;
        double bestScore = MINIMAL_SCORE;
        for (const auto& subList : candList) {
            for (const auto& candidate : subList.Candidates) {
                double score = candidate.BestScore.GetInstance(ctx->Rand);
                TProjection projection = candidate.SplitCandidate.Ctr.Projection;
                ECtrType ctrType = ctx->CtrsHelper.GetCtrInfo(projection)[candidate.SplitCandidate.Ctr.CtrIdx].Type;

                if (!ctx->LearnProgress.UsedCtrSplits.has(std::make_pair(ctrType, projection)) && score != MINIMAL_SCORE) {
                    score *= pow(1 / (1 + subList.ResultingCtrTableSize / static_cast<double>(maxLeafCount)), ctx->Params.ObliviousTreeOptions->ModelSizeReg.Get());
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestSplitCandidate = &candidate;
                }
            }
        }
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
                size_t totalLeafCount;
                ComputeOnlineCTRs(data,
                                  *fold,
                                  proj,
                                  ctx,
                                  &fold->GetCtrRef(proj),
                                  &totalLeafCount);
                DropStatsForProjection(*fold, *ctx, proj, &ctx->PrevTreeLevelStats);
            }
        } else if (bestSplit.Type == ESplitType::OneHotFeature) {
            bestSplit.BinBorder = data.AllFeatures.OneHotValues[bestSplit.FeatureIdx][bestSplit.BinBorder];
        }
        SetPermutedIndices(bestSplit, data.AllFeatures, curDepth + 1, *fold, &indices, ctx);
        if (isSamplingPerTree) {
            ctx->SampledDocs.UpdateIndices(indices, &ctx->LocalExecutor);
            ctx->SmallestSplitSideDocs.SelectSmallestSplitSide(curDepth + 1, ctx->SampledDocs, &ctx->LocalExecutor);
        }
        currentSplitTree.AddSplit(bestSplit);
        MATRIXNET_INFO_LOG << BuildDescription(ctx->Layout, bestSplit);
        MATRIXNET_INFO_LOG << " score " << bestScore << "\n";


        profile.AddOperation(TStringBuilder() << "Select best split " << curDepth);

        int redundantIdx = GetRedundantSplitIdx(curDepth + 1, indices);
        if (redundantIdx != -1) {
            DeleteSplit(curDepth + 1, redundantIdx, &currentSplitTree, &indices);
            MATRIXNET_INFO_LOG << "  tensor " << redundantIdx << " is redundant, remove it and stop\n";
            break;
        }
    }
    *resSplitTree = std::move(currentSplitTree);
}
