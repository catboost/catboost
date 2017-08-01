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

static inline ui8 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinFeature.SplitIdx;
}

static inline const yvector<ui8>& GetFloatHistogram(const TSplit& split, const TAllFeatures& features) {
    return features.FloatHistograms[split.BinFeature.FloatFeature];
}

static inline const yvector<int>& GetCatFeatures(const TSplit& split, const TAllFeatures& features) {
    return features.CatFeatures[split.OneHotFeature.CatFeatureIdx];
}

template <typename TCount, bool (*CmpOp)(TCount, TCount), int vectorWidth>
void BuildIndicesKernel(const int* permutation, const TCount* histogram, TCount value, int level, TIndexType* indices) {
    Y_ASSERT(vectorWidth == 4);
    const int perm0 = permutation[0];
    const int perm1 = permutation[1];
    const int perm2 = permutation[2];
    const int perm3 = permutation[3];
    const TCount hist0 = histogram[perm0];
    const TCount hist1 = histogram[perm1];
    const TCount hist2 = histogram[perm2];
    const TCount hist3 = histogram[perm3];
    const TIndexType idx0 = indices[0];
    const TIndexType idx1 = indices[1];
    const TIndexType idx2 = indices[2];
    const TIndexType idx3 = indices[3];
    indices[0] = idx0 + CmpOp(hist0, value) * level;
    indices[1] = idx1 + CmpOp(hist1, value) * level;
    indices[2] = idx2 + CmpOp(hist2, value) * level;
    indices[3] = idx3 + CmpOp(hist3, value) * level;
}

template <typename TCount, bool (*CmpOp)(TCount, TCount)>
void OfflineCtrBlock(const NPar::TLocalExecutor::TBlockParams& params,
    int blockIdx,
    const TFold& fold,
    const TCount* histogram,
    TCount value,
    int level,
    TIndexType* indices) {
    const int* permutation = fold.LearnPermutation.data();
    const int blockStart = blockIdx * params.GetBlockSize();
    const int nextBlockStart = Min<ui64>(blockStart + params.GetBlockSize(), params.LastId);
    constexpr int vectorWidth = 4;
    int doc;
    for (doc = blockStart; doc + vectorWidth <= nextBlockStart; doc += vectorWidth) {
        BuildIndicesKernel<TCount, CmpOp, vectorWidth>(permutation + doc, histogram, value, level, indices + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        const int idxOriginal = permutation[doc];
        indices[doc] += CmpOp(histogram[idxOriginal], value) * level;
    }
}

void SetPermutedIndices(const TSplit& split,
                        const TAllFeatures& features,
                        int curDepth,
                        const TFold& fold,
                        yvector<TIndexType>* indices,
                        TLearnContext* ctx) {
    CB_ENSURE(curDepth > 0);

    const int blockSize = 1000;
    NPar::TLocalExecutor::TBlockParams blockParams(0, indices->ysize());
    blockParams.SetBlockSize(blockSize).WaitCompletion();

    const int splitWeight = 1 << (curDepth - 1);
    TIndexType* indicesData = indices->data();
    if (split.Type == ESplitType::FloatFeature) {
        ctx->LocalExecutor.ExecRange([&] (int blockIdx) {
            OfflineCtrBlock<ui8, IsTrueHistogram>(blockParams, blockIdx, fold, GetFloatHistogram(split, features).data(),
                GetFeatureSplitIdx(split), splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    } else if (split.Type == ESplitType::OnlineCtr) {
        auto& ctr = fold.GetCtr(split.OnlineCtr.Ctr.Projection);
        ctx->LocalExecutor.ExecRange([&] (int i) {
            indicesData[i] += GetCtrSplit(split, i, ctr) * splitWeight;
        }, blockParams);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        ctx->LocalExecutor.ExecRange([&] (int blockIdx) {
            OfflineCtrBlock<int, IsTrueOneHotFeature>(blockParams, blockIdx, fold, GetCatFeatures(split, features).data(),
                split.OneHotFeature.Value, splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
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

    const int blockSize = 1000;
    NPar::TLocalExecutor::TBlockParams learnBlockParams(0, data.LearnSampleCount);
    learnBlockParams.SetBlockSize(blockSize).WaitCompletion();

    localExecutor->ExecRange([&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.SelectedSplits.ysize(); ++splitIdx) {
            const auto& split = tree.SelectedSplits[splitIdx];
            const int splitWeight = 1 << splitIdx;
            if (split.Type == ESplitType::FloatFeature) {
                OfflineCtrBlock<ui8, IsTrueHistogram>(learnBlockParams, blockIdx, fold, GetFloatHistogram(split, data.AllFeatures).data(),
                    GetFeatureSplitIdx(split), splitWeight, indices.data());
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(learnBlockParams, [&](int doc) {
                    indices[doc] += GetCtrSplit(split, doc, splitOnlineCtr) * splitWeight;
                })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                OfflineCtrBlock<int, IsTrueOneHotFeature>(learnBlockParams, blockIdx, fold, GetCatFeatures(split, data.AllFeatures).data(),
                    split.OneHotFeature.Value, splitWeight, indices.data());
            }
        }
    }, 0, learnBlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    NPar::TLocalExecutor::TBlockParams tailBlockParams(data.LearnSampleCount, fold.EffectiveDocCount);
    tailBlockParams.SetBlockSize(blockSize).WaitCompletion();

    localExecutor->ExecRange([&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.SelectedSplits.ysize(); ++splitIdx) {
            const auto& split = tree.SelectedSplits[splitIdx];
            const int splitWeight = 1 << splitIdx;
            if (split.Type == ESplitType::FloatFeature) {
                const ui8 featureSplitIdx = GetFeatureSplitIdx(split);
                const ui8* floatHistogramData = GetFloatHistogram(split, data.AllFeatures).data();
                NPar::TLocalExecutor::BlockedLoopBody(tailBlockParams, [&](int doc) {
                    indices[doc] += IsTrueHistogram(floatHistogramData[doc], featureSplitIdx) * splitWeight;
                })(blockIdx);
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(tailBlockParams, [&](int doc) {
                    indices[doc] += GetCtrSplit(split, doc, splitOnlineCtr) * splitWeight;
                })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                const int featureSplitValue = split.OneHotFeature.Value;
                const int* featureValueData = GetCatFeatures(split, data.AllFeatures).data();
                NPar::TLocalExecutor::BlockedLoopBody(tailBlockParams, [&](int doc) {
                    indices[doc] += IsTrueOneHotFeature(featureValueData[doc], featureSplitValue) * splitWeight;
                })(blockIdx);
            }
        }
    }, 0, tailBlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

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
            } else if (ctrType == ECtrType::Counter) {
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
                    denominator = Max(denominator, currentBucket);
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
