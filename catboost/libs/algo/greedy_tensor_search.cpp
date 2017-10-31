#include "greedy_tensor_search.h"
#include "error_functions.h"
#include "index_calcer.h"
#include "tree_print.h"
#include "interrupt.h"
#include <library/dot_product/dot_product.h>
#include <library/fast_log/fast_log.h>
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
    ctx->LocalExecutor.ExecRange([&](int blockIdx) {
        TFastRng64 rand(randSeed + blockIdx);
        rand.Advance(10); // reduce correlation between RNGs in different threads
        const float baggingTemperature = ctx->Params.BaggingTemperature;
        float* sampleWeightsData = sampleWeights.data();
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&rand, sampleWeightsData, baggingTemperature](int i) {
            const float w = -FastLogf(rand.GenRandReal1() + 1e-100);
            sampleWeightsData[i] = powf(w, baggingTemperature);
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
    for (TFold::TBodyTail& bt : ff.BodyTailArr) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            double* weightedDerData = bt.WeightedDer[dim].data();
            const double* derData = bt.Derivatives[dim].data();
            const float* sampleWeightsData = ff.SampleWeights.data();
            ctx->LocalExecutor.ExecRange([=](int z) {
                weightedDerData[z] = derData[z] * sampleWeightsData[z];
            }, NPar::TLocalExecutor::TBlockParams(bt.BodyFinish, bt.TailFinish).SetBlockSize(4000).WaitCompletion());
        }
    }
}


struct TCandidateInfo {
    TSplitCandidate SplitCandidate;
    TRandomScore BestScore;
    int BestBinBorderId;
    bool ShouldDropAfterScoreCalc;
    TModelSplit GetModelSplit(const TLearnContext& ctx, const TTrainData& data) const {
        TModelSplit split;
        split.Type = SplitCandidate.Type;
        if (SplitCandidate.Type == ESplitType::FloatFeature) {
            split.BinFeature.FloatFeature = SplitCandidate.FeatureIdx;
            split.BinFeature.SplitIdx = BestBinBorderId;
        } else if (SplitCandidate.Type == ESplitType::OneHotFeature) {
            split.OneHotFeature.CatFeatureIdx = SplitCandidate.FeatureIdx;
            split.OneHotFeature.Value = data.AllFeatures.OneHotValues[SplitCandidate.FeatureIdx][BestBinBorderId];
        } else {
            Y_ASSERT(SplitCandidate.Type == ESplitType::OnlineCtr);
            split.OnlineCtr.Ctr.Projection = SplitCandidate.Ctr.Projection;
            const yvector<float>& priors = ctx.Priors.GetPriors(split.OnlineCtr.Ctr.Projection);
            yvector<float> shift;
            yvector<float> norm;
            CalcNormalization(priors, &shift, &norm);
            split.OnlineCtr.Ctr.CtrType = ctx.Params.CtrParams.Ctrs[SplitCandidate.Ctr.CtrIdx].CtrType;
            split.OnlineCtr.Ctr.TargetBorderClassifierIdx = SplitCandidate.Ctr.CtrIdx;
            split.OnlineCtr.Ctr.TargetBorderIdx = SplitCandidate.Ctr.TargetBorderIdx;
            split.OnlineCtr.Ctr.PriorNum = priors[SplitCandidate.Ctr.PriorIdx];
            split.OnlineCtr.Ctr.PriorDenom = 1.0f;
            split.OnlineCtr.Ctr.Shift = shift[SplitCandidate.Ctr.PriorIdx];
            split.OnlineCtr.Ctr.Scale = ctx.Params.CtrParams.CtrBorderCount / norm[SplitCandidate.Ctr.PriorIdx];
            split.OnlineCtr.Border = (float)BestBinBorderId + 0.999999f; // hack to emulate ui8 rounding :)
        }
        return split;
    }
};

struct TCandidatesInfoList {
    TCandidatesInfoList() = default;
    explicit TCandidatesInfoList(const TCandidateInfo& oneCandidate) {
        Candidates.emplace_back(oneCandidate);
    }
    // All candidates here are either float or one-hot, or have the same
    // projection.
    // TODO(annaveronika): put projection out, because currently it's not clear.
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
    using TSeenProjHash = yhash_set<TProjection>;
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

            if (proj.IsRedundant() || proj.GetFullProjectionLength() > (size_t)ctx->Params.CtrParams.MaxCtrComplexity) {
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

void GreedyTensorSearch(const TTrainData& data,
                        const yvector<int>& splitCounts,
                        double modelLength,
                        float l2Regularizer,
                        float randomStrength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TTensorStructure3* resTree,
                        yvector<TSplit>* resSplitTree) {
    yvector<TSplit> currentSplitTree;
    TTensorStructure3 currentTree;

    TrimOnlineCTRcache({fold});

    yvector<TIndexType> indices(data.LearnSampleCount);
    if (ctx->Params.PrintTrees) {
        MATRIXNET_INFO_LOG << "\n";
    }
    for (int curDepth = 0; curDepth < ctx->Params.Depth; ++curDepth) {
        TCandidateList candList;
        AddFloatFeatures(data, ctx, &candList);
        AddOneHotFeatures(data, ctx, &candList);
        AddSimpleCtrs(data, fold, ctx, &candList);
        AddTreeCtrs(data, currentTree, fold, ctx, &candList);

        auto IsInCache = [&fold](const TProjection& proj) -> bool {return fold->GetCtrRef(proj).Feature.empty();};
        SelectCtrsToDropAfterCalc(ctx->Params.UsedRAMLimit, data.GetSampleCount(), ctx->Params.ThreadCount, IsInCache, &candList);

        CheckInterrupted(); // check after long-lasting operation
        AssignRandomWeights(data.LearnSampleCount, ctx, fold);
        profile.AddOperation(TStringBuilder() << "AssignRandomWeights, depth " << curDepth);
        double scoreStDev = randomStrength * CalcScoreStDev(*fold) * CalcScoreStDevMult(data.LearnSampleCount, modelLength);

        const ui64 randSeed = ctx->Rand.GenRand();
        ctx->LocalExecutor.ExecRange([&](int id) {
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
                    Y_ASSERT(!fold->GetCtrRef(proj).Feature.empty());
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
        fold->DropEmptyCTRs();
        CheckInterrupted(); // check after long-lasting operation
        profile.AddOperation(TStringBuilder() << "Calc scores " << curDepth);

        const TCandidateInfo* bestSplitCandidate = nullptr;
        double bestScore = MINIMAL_SCORE;
        for (const auto& subList : candList) {
            for (const auto& candidate : subList.Candidates) {
                const double score = candidate.BestScore.GetInstance(ctx->Rand);
                if (score > bestScore) {
                    bestScore = score;
                    bestSplitCandidate = &candidate;
                }
            }
        }
        if (bestScore == MINIMAL_SCORE) {
            break;
        }
        auto bestSplit = TSplit(bestSplitCandidate->SplitCandidate, bestSplitCandidate->BestBinBorderId);
        auto modelSplit = bestSplitCandidate->GetModelSplit(*ctx, data);
        if (bestSplit.Type == ESplitType::OnlineCtr) {
            const auto& proj = bestSplit.Ctr.Projection;
            if (fold->GetCtrRef(proj).Feature.empty()) {
                ComputeOnlineCTRs(data,
                                  *fold,
                                  proj,
                                  ctx,
                                  &fold->GetCtrRef(proj));
            }
        }
        SetPermutedIndices(bestSplit, data.AllFeatures, curDepth + 1, *fold, &indices, ctx);
        currentTree.Add(modelSplit);
        currentSplitTree.push_back(bestSplit);
        if (ctx->Params.PrintTrees) {
            MATRIXNET_INFO_LOG << BuildDescription(ctx->Layout, modelSplit);
            MATRIXNET_INFO_LOG << " score " << bestScore << "\n";
        }

        profile.AddOperation(TStringBuilder() << "Select best split " << curDepth);

        int redundantIdx = GetRedundantSplitIdx(curDepth + 1, indices);
        if (redundantIdx != -1) {
            DeleteSplit(curDepth + 1, redundantIdx, &currentSplitTree, &currentTree, &indices);
            if (ctx->Params.PrintTrees) {
                MATRIXNET_INFO_LOG << "  tensor " << redundantIdx << " is redundant, remove it and stop\n";
            }
            break;
        }
    }
    *resTree = std::move(currentTree);
    *resSplitTree = std::move(currentSplitTree);
}
