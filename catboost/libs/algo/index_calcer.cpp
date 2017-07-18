#include "index_calcer.h"
#include "score_calcer.h"
#include "online_ctr.h"
#include <catboost/libs/model/model.h>
#include <catboost/libs/helpers/dense_hash.h>

static bool GetFloatFeatureSplit(const TSplit& split, const TAllFeatures& features, int idxOriginal) {
    const TBinFeature& bf = split.BinFeature;
    bool ans = IsTrueHistogram(features.FloatHistograms[bf.FloatFeature][idxOriginal], bf.SplitIdx);
    return ans;
}

static bool GetOneHotFeatureSplit(const TSplit& split, const TAllFeatures& features, int idxOriginal) {
    const TOneHotFeature& ohf = split.OneHotFeature;
    bool ans = IsTrueOneHotFeature(features.CatFeatures[ohf.CatFeatureIdx][idxOriginal], ohf.Value);
    return ans;
}

static bool GetCtrSplit(const TSplit& split, int idxPermuted, const TOnlineCTR& ctr) {
    ui8 ctrValue = ctr.Feature[split.OnlineCtr.Ctr.CtrTypeIdx]
                              [split.OnlineCtr.Ctr.TargetBorderIdx]
                              [split.OnlineCtr.Ctr.PriorIdx]
                              [idxPermuted];
    return ctrValue > split.OnlineCtr.Border;
}

void SetPermutedIndices(const TSplit& split,
                        const TAllFeatures& features,
                        int curDepth,
                        const TFold& fold,
                        yvector<TIndexType>* indices,
                        TLearnContext* ctx) {
    CB_ENSURE(curDepth > 0);
    if (split.Type == ESplitType::FloatFeature) {
        ctx->LocalExecutor.ExecRange([&] (int i) {
            const int idxOriginal = fold.LearnPermutation[i];
            const int splitTrue = GetFloatFeatureSplit(split, features, idxOriginal);
            (*indices)[i] |= splitTrue << (curDepth - 1);
        }, NPar::TLocalExecutor::TBlockParams(0, indices->ysize()).SetBlockSize(1000).WaitCompletion());
    } else if (split.Type == ESplitType::OnlineCtr) {
        auto& ctr = fold.GetCtr(split.OnlineCtr.Ctr.Projection);
        for (int i = 0; i < indices->ysize(); ++i) {
            const int splitTrue = GetCtrSplit(split, i, ctr);
            (*indices)[i] |= splitTrue << (curDepth - 1);
        }
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        ctx->LocalExecutor.ExecRange([&](int i) {
            const int idxOriginal = fold.LearnPermutation[i];
            const int splitTrue = GetOneHotFeatureSplit(split, features, idxOriginal);
            (*indices)[i] |= splitTrue << (curDepth - 1);
        }, NPar::TLocalExecutor::TBlockParams(0, indices->ysize()).SetBlockSize(1000).WaitCompletion());
    }
}

int GetRedundantSplitIdx(int curDepth, const yvector<TIndexType>& indices) {
    yvector<bool> isEmpty(1 << curDepth, true);
    for (const auto& idx : indices) {
        isEmpty[idx] = false;
    }

    for (int splitIdx = 0; splitIdx < curDepth; ++splitIdx) {
        bool isRedundantSplit = true;
        for (int idx = 0; idx < (1 << curDepth); ++idx) {
            if (idx & (1 << splitIdx)) {
                continue;
            }
            if (!isEmpty[idx] && !isEmpty[idx ^ (1 << splitIdx)]) {
                isRedundantSplit = false;
                break;
            }
        }
        if (isRedundantSplit) {
            return splitIdx;
        }
    }

    return -1;
}

void DeleteSplit(int curDepth, int redundantIdx, TTensorStructure3* tree, yvector<TIndexType>* indices) {
    if (redundantIdx != curDepth - 1) {
        DoSwap(tree->SelectedSplits[redundantIdx], tree->SelectedSplits.back());
        for (auto& idx : *indices) {
            bool isTrueBack = (idx >> (curDepth - 1)) & 1;
            bool isTrueRedundant = (idx >> redundantIdx) & 1;
            idx ^= isTrueRedundant << redundantIdx;
            idx ^= isTrueBack << redundantIdx;
        }
    }

    tree->SelectedSplits.pop_back();
    for (auto& idx : *indices) {
        idx &= (1 << (curDepth - 1)) - 1;
    }
}

static inline ui8 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinFeature.SplitIdx;
}

static inline const yvector<ui8>& GetFloatHistogram(const TSplit& split, const TAllFeatures& features) {
    return features.FloatHistograms[split.BinFeature.FloatFeature];
}

static inline const yvector<int>& GetCatFeatures(const TSplit& split, const TAllFeatures& features) {
    return features.CatFeatures[split.OneHotFeature.CatFeatureIdx];
}

yvector<TIndexType> BuildIndices(const TFold& fold,
    const TTensorStructure3& tree,
    const TTrainData& data,
    NPar::TLocalExecutor* localExecutor) {
    yvector<TIndexType> indices(fold.EffectiveDocCount);
    yvector<const TOnlineCTR*> onlineCtrs(tree.SelectedSplits.ysize());
    for (int splitIdx = 0; splitIdx < tree.SelectedSplits.ysize(); ++splitIdx) {
        const auto& split = tree.SelectedSplits[splitIdx];
        if (split.Type == ESplitType::OnlineCtr) {
            onlineCtrs[splitIdx] = &fold.GetCtr(split.OnlineCtr.Ctr.Projection);
        }
    }

    NPar::TLocalExecutor::TBlockParams blockParams(0, fold.EffectiveDocCount);
    blockParams.SetBlockSize(1000).WaitCompletion();

    localExecutor->ExecRange([&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.SelectedSplits.ysize(); ++splitIdx) {
            const auto& split = tree.SelectedSplits[splitIdx];
            if (split.Type == ESplitType::FloatFeature) {
                const ui8 featureSplitIdx = GetFeatureSplitIdx(split);
                const ui8* floatHistogramData = GetFloatHistogram(split, data.AllFeatures).data();
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int doc) {
                    const int idxOriginal = doc < data.LearnSampleCount ? fold.LearnPermutation[doc] : doc;
                    const int splitTrue = IsTrueHistogram(floatHistogramData[idxOriginal], featureSplitIdx);
                    indices[doc] |= splitTrue << splitIdx;
                })(blockIdx);
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int doc) {
                    const int splitTrue = GetCtrSplit(split, doc, splitOnlineCtr);
                    indices[doc] |= splitTrue << splitIdx;
                })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                const int featureSplitValue = split.OneHotFeature.Value;
                const int* featureValueData = GetCatFeatures(split, data.AllFeatures).data();
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int doc) {
                    const int idxOriginal = doc < data.LearnSampleCount ? fold.LearnPermutation[doc] : doc;
                    const int splitTrue = IsTrueOneHotFeature(featureValueData[idxOriginal], featureSplitValue);
                    indices[doc] |= splitTrue << splitIdx;
                })(blockIdx);
            }
        }
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    return indices;
}

int GetDocCount(const TAllFeatures& features) {
    for (int i = 0; i < features.FloatHistograms.ysize(); ++i) {
        if (!features.FloatHistograms[i].empty())
            return features.FloatHistograms[i].ysize();
    }
    for (int i = 0; i < features.CatFeatures.ysize(); ++i) {
        if (!features.CatFeatures[i].empty())
            return features.CatFeatures[i].ysize();
    }
    return 0;
}

yvector<TIndexType> BuildIndices(const TTensorStructure3& tree,
                                 const TFullModel& model,
                                 const TAllFeatures& features,
                                 const TCommonContext& ctx) {
    int samplesCount = GetDocCount(features);
    yvector<TIndexType> indices(samplesCount);
    const int splitCount = tree.SelectedSplits.ysize();
    yvector<ui64> ctrHashes;
    yvector<float> shift;
    yvector<float> norm;

    for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
        const auto& split = tree.SelectedSplits[splitIdx];

        if (split.Type == ESplitType::FloatFeature) {
            for (int doc = 0; doc < samplesCount; ++doc) {
                indices[doc] |= GetFloatFeatureSplit(split, features, doc) << splitIdx;
            }
        } else if (split.Type == ESplitType::OneHotFeature) {
            for (int doc = 0; doc < samplesCount; ++doc) {
                indices[doc] |= GetOneHotFeatureSplit(split, features, doc) << splitIdx;
            }
        } else {
            Y_ASSERT(split.Type == ESplitType::OnlineCtr);
            const TCtr& ctr = split.OnlineCtr.Ctr;
            const ECtrType ctrType = ctx.Params.CtrParams.Ctrs[ctr.CtrTypeIdx].CtrType;
            const auto& learnCtr = model.CtrCalcerData.LearnCtrs.find(ctr)->second;
            ctrHashes.resize(samplesCount);
            const auto& projection = split.OnlineCtr.Ctr.Projection;
            CalcHashes(projection, features, samplesCount, yvector<int>(), &ctrHashes);
            CalcNormalization(ctx.Priors.GetPriors(projection), &shift, &norm);
            const float prior = ctx.Priors.GetPriors(projection)[ctr.PriorIdx];
            const float priorShift = shift[ctr.PriorIdx];
            const float priorNorm = norm[ctr.PriorIdx];
            const int borderCount = ctx.Params.CtrParams.CtrBorderCount;

            if (ctrType == ECtrType::MeanValue) {
                for (int doc = 0; doc < samplesCount; ++doc) {
                    int goodCount = 0;
                    int totalCount = 0;
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    if (idx != TCtrValueTable::UnknownHash) {
                        const TCtrMeanHistory& ctrMeanHistory = learnCtr.CtrMean[idx];
                        goodCount = ctrMeanHistory.Sum;
                        totalCount = ctrMeanHistory.Count;
                    }
                    const ui8 ctrValue = CalcCTR(goodCount, totalCount, prior, priorShift, priorNorm, borderCount);
                    indices[doc] |= (ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            } else if (IsCounter(ctrType)) {
                NTensor2::TDenseHash<> additionalHash;
                additionalHash.Init(2 * samplesCount);
                yvector<int> additionalCtrTotal(2 * samplesCount);
                int denominator = learnCtr.CounterDenominator;
                for (int doc = 0; doc < samplesCount; ++doc) {
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    const ui64 elemId = additionalHash.AddIndex(ctrHashes[doc]);
                    ++additionalCtrTotal[elemId];
                    int currentBucket = additionalCtrTotal[elemId];
                    if (idx != TCtrValueTable::UnknownHash) {
                        currentBucket += learnCtr.CtrTotal[idx];
                    }
                    if (ctrType == ECtrType::CounterTotal) {
                        ++denominator;
                    } else {
                        denominator = Max(denominator, currentBucket);
                    }
                    const ui8 ctrValue = CalcCTR(currentBucket, denominator, prior, priorShift, priorNorm, borderCount);
                    indices[doc] |= (ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            } else if (ctrType == ECtrType::Buckets) {
                for (int doc = 0; doc < samplesCount; ++doc) {
                    int goodCount = 0;
                    int totalCount = 0;
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    if (idx != TCtrValueTable::UnknownHash) {
                        const yvector<int>& ctrHistory = learnCtr.Ctr[idx];
                        const int targetClassesCount = ctrHistory.ysize();
                        goodCount = ctrHistory[ctr.TargetBorderIdx];
                        for (int classId = 0; classId < targetClassesCount; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                    }
                    const ui8 ctrValue = CalcCTR(goodCount, totalCount, prior, priorShift, priorNorm, borderCount);
                    indices[doc] |= (ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            } else {
                for (int doc = 0; doc < samplesCount; ++doc) {
                    int goodCount = 0;
                    int totalCount = 0;
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    if (idx != TCtrValueTable::UnknownHash) {
                        const yvector<int>& ctrHistory = learnCtr.Ctr[idx];
                        const int targetClassesCount = ctrHistory.ysize();
                        for (int classId = 0; classId < ctr.TargetBorderIdx + 1; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                        for (int classId = ctr.TargetBorderIdx + 1; classId < targetClassesCount; ++classId) {
                            goodCount += ctrHistory[classId];
                        }
                        totalCount += goodCount;
                    }
                    const ui8 ctrValue = CalcCTR(goodCount, totalCount, prior, priorShift, priorNorm, borderCount);
                    indices[doc] |= (ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            }
        }
    }

    return indices;
}
