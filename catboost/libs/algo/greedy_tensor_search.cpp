#include "greedy_tensor_search.h"
#include "error_functions.h"
#include "index_calcer.h"
#include "tree_print.h"
#include "interrupt.h"
#include <util/string/builder.h>
#include <util/system/mem_info.h>

constexpr size_t MAX_ONLINE_CTR_FEATURES = 50;

void TrimOnlineCTRcache(const yvector<TFold*>& folds) {
    for (auto& fold : folds) {
        fold->TrimOnlineCTR(MAX_ONLINE_CTR_FEATURES);
    }
}

static void AssignRandomWeights(int learnSampleCount,
                                TLearnContext* ctx,
                                TFold* fold) {
    yvector<float> sampleWeights(learnSampleCount);

    const ui64 randSeed = ctx->Rand.GenRand();
    NPar::TLocalExecutor::TBlockParams blockParams(0, learnSampleCount);
    blockParams.SetBlockSize(10000);
    ctx->LocalExecutor.ExecRange([&] (int blockIdx) {
        TFastRng64 rand(randSeed + blockIdx);
        rand.Advance(10); // reduce correlation between RNGs in different threads
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&] (int i) {
            float w = -log(rand.GenRandReal1() + 1e-100);
            sampleWeights[i] = powf(w, ctx->Params.BaggingTemperature);
        })(blockIdx);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    TFold& ff = *fold;
    ff.AssignPermuted(sampleWeights, &ff.SampleWeights);
    if (!ff.LearnWeights.empty()) {
        for (int i = 0; i < learnSampleCount; ++i) {
            ff.SampleWeights[i] *= ff.LearnWeights[i];
        }
    }

    const int approxDimension = ff.GetApproxDimension();
    for (int mixTailId = 0; mixTailId < ff.MixTailArr.ysize(); ++mixTailId) {
        TFold::TMixTail& mt = ff.MixTailArr[mixTailId];
        for (int dim = 0; dim < approxDimension; ++dim) {
            double* weightedDerData = mt.WeightedDer[dim].data();
            const double* derData = mt.Derivatives[dim].data();
            const float* sampleWeightsData = ff.SampleWeights.data();
            ctx->LocalExecutor.ExecRange([=](int z) {
                weightedDerData[z] = derData[z] * sampleWeightsData[z];
            }, NPar::TLocalExecutor::TBlockParams(mt.MixCount, mt.TailFinish).SetBlockSize(4000).WaitCompletion());
        }
    }
}

struct TCandidateInfo {
    TSplitCandidate SplitCandidate;
    TSplit BestSplit;
    TRandomScore BestScore;
    bool ShouldDropAfterScoreCalc;
};

struct TCandidatesInfoList {
    TCandidatesInfoList() = default;
    explicit TCandidatesInfoList(const TCandidateInfo& oneCandidate) {
        Candidates.emplace_back(oneCandidate);
    }
    yvector<TCandidateInfo> Candidates;
    bool ShouldDropCtrAfterCalc = false;
};

using TCandidateList = yvector<TCandidatesInfoList>;

static void AddFloatFeatures(const TTrainData& data,
                             TLearnContext* ctx,
                             TCandidateList* candList) {
    for (int f = 0; f < data.AllFeatures.FloatHistograms.ysize(); ++f) {
        if (data.AllFeatures.FloatHistograms[f].empty() || ctx->Rand.GenRandReal1() > ctx->Params.Rsm) {
            continue;
        }
        TCandidateInfo split;
        split.SplitCandidate.FeatureIdx = f;
        split.SplitCandidate.Type = ESplitType::FloatFeature;

        candList->emplace_back(TCandidatesInfoList(split));
    }
}

static void AddOneHotFeatures(const TTrainData& data,
                              TLearnContext* ctx,
                              TCandidateList* candList) {
    for (int cf = 0; cf < data.AllFeatures.CatFeatures.ysize(); ++cf) {
        if (data.AllFeatures.CatFeatures[cf].empty() ||
            !data.AllFeatures.IsOneHot[cf] ||
            ctx->Rand.GenRandReal1() > ctx->Params.Rsm) {
            continue;
        }

        TCandidateInfo split;
        split.SplitCandidate.FeatureIdx = cf;
        split.SplitCandidate.Type = ESplitType::OneHotFeature;

        candList->emplace_back(TCandidatesInfoList(split));
    }
}

static void AddCtrsToCandList(const TFold& fold,
                              const TLearnContext& ctx,
                              const TProjection& proj,
                              TCandidateList* candList) {
    TCandidatesInfoList ctrSplits;
    int priorsCount = ctx.Priors.GetPriors(proj).ysize();
    for (int ctrIdx = 0; ctrIdx < ctx.Params.CtrParams.Ctrs.ysize(); ++ctrIdx) {
        ECtrType ctrType = ctx.Params.CtrParams.Ctrs[ctrIdx].CtrType;
        int borderCount = GetCtrBorderCount(fold.TargetClassesCount[ctrIdx], ctrType);
        for (int border = 0; border < borderCount; ++border) {
            for (int prior = 0; prior < priorsCount; ++prior) {
                TCandidateInfo split;
                split.SplitCandidate.Type = ESplitType::OnlineCtr;
                split.SplitCandidate.Ctr = TCtr(proj, ctrIdx, border, prior);
                ctrSplits.Candidates.emplace_back(split);
            }
        }
    }

    candList->push_back(ctrSplits);
}
static void AddSimpleCtrs(const TTrainData& data,
                          TFold* fold,
                          TLearnContext* ctx,
                          TCandidateList* candList) {
    for (int cf = 0; cf < data.AllFeatures.CatFeatures.ysize(); ++cf) {
        if (data.AllFeatures.CatFeatures[cf].empty() ||
            data.AllFeatures.IsOneHot[cf] ||
            ctx->Rand.GenRandReal1() > ctx->Params.Rsm) {
            continue;
        }

        TProjection proj;
        proj.AddCatFeature(cf);

        AddCtrsToCandList(*fold, *ctx, proj, candList);
        fold->GetCtrRef(proj);
    }
}

static void AddTreeCtrs(const TTrainData& data,
                        const TTensorStructure3& currentTree,
                        TFold* fold,
                        TLearnContext* ctx,
                        TCandidateList* candList) {
    using TSeenProjHash = yhash_set<TProjection, TProjHash>;
    TSeenProjHash seenProj;

    // greedy construction
    TProjection binAndOneHotFeaturesTree;
    binAndOneHotFeaturesTree.BinFeatures = GetBinFeatures(currentTree);
    binAndOneHotFeaturesTree.OneHotFeatures = GetOneHotFeatures(currentTree);
    seenProj.insert(binAndOneHotFeaturesTree);

    for (const auto& ctrSplit : GetCtrSplits(currentTree)) {
        seenProj.insert(ctrSplit.Ctr.Projection);
    }

    TSeenProjHash addedProjHash;
    for (const auto& baseProj : seenProj) {
        if (baseProj.IsEmpty()) {
            continue;
        }
        for (int cf = 0; cf < data.AllFeatures.CatFeatures.ysize(); ++cf) {
            if (data.AllFeatures.CatFeatures[cf].empty() ||
                data.AllFeatures.IsOneHot[cf] ||
                ctx->Rand.GenRandReal1() > ctx->Params.Rsm) {
                continue;
            }

            TProjection proj = baseProj;
            proj.AddCatFeature(cf);

            if (proj.IsRedundant() || proj.CatFeatures.ysize() > ctx->Params.CtrParams.MaxCtrComplexity) {
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
}

static double CalcScoreStDev(const TFold& ff) {
    double sum2 = 0, totalSum2Count = 0;
    for (int mixTailId = 0; mixTailId < ff.MixTailArr.ysize(); ++mixTailId) {
        const TFold::TMixTail& mt = ff.MixTailArr[mixTailId];
        for (int dim = 0; dim < mt.Derivatives.ysize(); ++dim) {
            for (int z = mt.MixCount; z < mt.TailFinish; ++z) {
                sum2 += Sqr(mt.Derivatives[dim][z]);
            }
        }
        totalSum2Count += mt.TailFinish - mt.MixCount;
    }
    return sqrt(sum2 / Max(totalSum2Count, DBL_EPSILON));
}

static double CalcScoreStDevMult(int learnSampleCount, double modelLength) {
    double modelExpLength = log(learnSampleCount * 1.0);
    double modelLeft = exp(modelExpLength - modelLength);
    return modelLeft / (1 + modelLeft);
}

void GreedyTensorSearch(const TTrainData& data,
                        const yvector<int>& splitCounts,
                        double modelLength,
                        float l2Regularizer,
                        float randomStrength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TTensorStructure3* resTree) {
    TTensorStructure3 currentTree;

    TrimOnlineCTRcache({fold});

    yvector<TIndexType> indices(data.LearnSampleCount);
    if (ctx->Params.PrintTrees) {
        MATRIXNET_INFO_LOG << "\n";
    }
    for (int curDepth = 0; curDepth < ctx->Params.Depth; ++curDepth) {
        auto currentMemoryUsage = NMemInfo::GetMemInfo().RSS;
        TCandidateList candList;
        AddFloatFeatures(data, ctx, &candList);
        AddOneHotFeatures(data, ctx, &candList);
        AddSimpleCtrs(data, fold, ctx, &candList);
        AddTreeCtrs(data, currentTree, fold, ctx, &candList);

        size_t maxMemoryForOneCtr = 0;
        size_t fullNeededMemoryForCtrs = 0;
        for (auto& candSubList : candList) {
            const auto firstSubCandidate = candSubList.Candidates[0].SplitCandidate;
            if (firstSubCandidate.Type != ESplitType::OnlineCtr
                || !fold->GetCtrRef(firstSubCandidate.Ctr.Projection).Feature.empty()) {
                candSubList.ShouldDropCtrAfterCalc = false;
                continue;
            }
            const size_t neededMem = data.GetSampleCount() * candSubList.Candidates.size();
            maxMemoryForOneCtr = Max<size_t>(neededMem, maxMemoryForOneCtr);
            fullNeededMemoryForCtrs += neededMem;
        }

        if (fullNeededMemoryForCtrs + currentMemoryUsage > ctx->Params.UsedRAMLimit) {
            MATRIXNET_DEBUG_LOG << "Needed more memory then allowed, will drop some ctrs after score calculation" << Endl;
            const float GB = (ui64)1024 * 1024 * 1024;
            MATRIXNET_DEBUG_LOG << "current rss " << currentMemoryUsage / GB <<  fullNeededMemoryForCtrs / GB << Endl;
            size_t currentNonDroppableMemory = currentMemoryUsage;
            size_t maxMemForOtherThreadsApprox = (ui64)(ctx->Params.ThreadCount - 1)  * maxMemoryForOneCtr;
            for (auto& candSubList : candList) {
                const auto firstSubCandidate = candSubList.Candidates[0].SplitCandidate;
                if (firstSubCandidate.Type != ESplitType::OnlineCtr
                    || !fold->GetCtrRef(firstSubCandidate.Ctr.Projection).Feature.empty()) {
                    candSubList.ShouldDropCtrAfterCalc = false;
                    continue;
                }
                candSubList.ShouldDropCtrAfterCalc = true;
                const size_t neededMem = data.GetSampleCount() * candSubList.Candidates.size();
                if (currentNonDroppableMemory + neededMem + maxMemForOtherThreadsApprox <= ctx->Params.UsedRAMLimit) {
                    candSubList.ShouldDropCtrAfterCalc = false;
                    currentNonDroppableMemory += neededMem;
                }
            }
        }

        CheckInterrupted(); // check after long-lasting operation
        AssignRandomWeights(data.LearnSampleCount, ctx, fold);
        profile.AddOperation(TStringBuilder() << "AssignRandomWeights, depth " << curDepth);
        double scoreStDev = randomStrength * CalcScoreStDev(*fold) * CalcScoreStDevMult(data.LearnSampleCount, modelLength);

        const ui64 randSeed = ctx->Rand.GenRand();
        ctx->LocalExecutor.ExecRange([&] (int id) {
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
            yvector<yvector<double>> allScores(candidate.Candidates.size());
            ctx->LocalExecutor.ExecRange([&](int oneCandidate) {
                if (candidate.Candidates[oneCandidate].SplitCandidate.Type == ESplitType::OnlineCtr) {
                    const auto& proj = candidate.Candidates[oneCandidate].SplitCandidate.Ctr.Projection;
                    CB_ENSURE(!fold->GetCtrRef(proj).Feature.empty());
                }
                allScores[oneCandidate] = CalcScore(data.AllFeatures,
                    splitCounts,
                    *fold,
                    indices,
                    candidate.Candidates[oneCandidate].SplitCandidate,
                    currentTree.GetDepth(),
                    ctx->Params.CtrParams.CtrBorderCount,
                    l2Regularizer);
            }, NPar::TLocalExecutor::TBlockParams(0, candidate.Candidates.ysize()).SetBlockSize(1).WaitCompletion().HighPriority());
            if (candidate.Candidates[0].SplitCandidate.Type == ESplitType::OnlineCtr && candidate.ShouldDropCtrAfterCalc) {
                fold->GetCtrRef(candidate.Candidates[0].SplitCandidate.Ctr.Projection).Feature.clear();
            }
            TFastRng64 rand(randSeed + id);
            rand.Advance(10); // reduce correlation between RNGs in different threads
            for (size_t i = 0; i < allScores.size(); ++i) {
                double bestScoreInstance = MINIMAL_SCORE;
                auto& splitInfo = candidate.Candidates[i];
                const auto& splitCandidate = splitInfo.SplitCandidate;
                const auto& scores = allScores[i];
                for (int binFeatureIdx = 0; binFeatureIdx < scores.ysize(); ++binFeatureIdx) {
                    const double score = scores[binFeatureIdx];
                    const double scoreInstance = TRandomScore(score, scoreStDev).GetInstance(rand);
                    if (scoreInstance > bestScoreInstance) {
                        bestScoreInstance = scoreInstance;
                        splitInfo.BestScore = TRandomScore(score, scoreStDev);
                        if (splitCandidate.Type == ESplitType::OnlineCtr) {
                            splitInfo.BestSplit = TSplit(TCtrSplit(splitCandidate.Ctr, binFeatureIdx));
                        } else if (splitCandidate.Type == ESplitType::FloatFeature) {
                            splitInfo.BestSplit = TSplit(ESplitType::FloatFeature, splitCandidate.FeatureIdx, binFeatureIdx);
                        } else {
                            Y_ASSERT(splitCandidate.Type == ESplitType::OneHotFeature);
                            splitInfo.BestSplit = TSplit(ESplitType::OneHotFeature,
                                                         splitCandidate.FeatureIdx,
                                                         data.AllFeatures.OneHotValues[splitCandidate.FeatureIdx][binFeatureIdx]);
                        }
                    }
                }
            }
        }, 0, candList.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        fold->DropEmptyCTRs();
        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Calc scores " << curDepth);

        TSplit bestSplit;
        double bestScore = MINIMAL_SCORE;
        for (const auto& subList : candList) {
            for (const auto& candidate : subList.Candidates) {
                const double score = candidate.BestScore.GetInstance(ctx->Rand);
                if (score > bestScore) {
                    bestScore = score;
                    bestSplit = candidate.BestSplit;
                }
            }
        }
        if (bestScore == MINIMAL_SCORE) {
            break;
        }
        if (bestSplit.Type == ESplitType::OnlineCtr) {
            const auto& proj = bestSplit.OnlineCtr.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(data,
                    *fold,
                    proj,
                    ctx,
                    &fold->GetCtrRef(proj));
            }
        }
        SetPermutedIndices(bestSplit, data.AllFeatures, curDepth + 1, *fold, &indices, ctx);
        currentTree.Add(bestSplit);

        if (ctx->Params.PrintTrees) {
            MATRIXNET_INFO_LOG << BuildDescription(ctx->Layout, bestSplit);
            MATRIXNET_INFO_LOG << " score " << bestScore << "\n";
        }

        profile.AddOperation(TStringBuilder() << "Select best split " << curDepth);

        int redundantIdx = GetRedundantSplitIdx(curDepth + 1, indices);
        if (redundantIdx != -1) {
            DeleteSplit(curDepth + 1, redundantIdx, &currentTree, &indices);
            MATRIXNET_DEBUG_LOG << "  tesor " << redundantIdx << " is redundant, remove it and stop\n";
            break;
        }
    }
    *resTree = currentTree;
}
