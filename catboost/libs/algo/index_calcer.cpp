#include "index_calcer.h"
#include "score_calcer.h"
#include "online_ctr.h"
#include "tree_print.h"
#include <catboost/libs/model/model.h>
#include <catboost/libs/helpers/dense_hash.h>

static bool GetFloatFeatureSplit(const TModelSplit& split, const TAllFeatures& features, int idxOriginal) {
    const auto& bf = split.BinFeature;
    bool ans = IsTrueHistogram(features.FloatHistograms[bf.FloatFeature][idxOriginal], bf.SplitIdx);
    return ans;
}

static bool GetOneHotFeatureSplit(const TModelSplit& split, const TAllFeatures& features, int idxOriginal) {
    const TOneHotFeature& ohf = split.OneHotFeature;
    bool ans = IsTrueOneHotFeature(features.CatFeatures[ohf.CatFeatureIdx][idxOriginal], ohf.Value);
    return ans;
}

static bool GetCtrSplit(const TSplit& split, int idxPermuted, const TOnlineCTR& ctr) {
    ui8 ctrValue = ctr.Feature[split.Ctr.CtrIdx]
                              [split.Ctr.TargetBorderIdx]
                              [split.Ctr.PriorIdx]
                              [idxPermuted];
    return ctrValue > split.BinBorder;
}

static inline ui8 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinBorder;
}

static inline const yvector<ui8>& GetFloatHistogram(const TSplit& split, const TAllFeatures& features) {
    return features.FloatHistograms[split.FeatureIdx];
}

static inline const yvector<int>& GetRemappedCatFeatures(const TSplit& split, const TAllFeatures& features) {
    return features.CatFeaturesRemapped[split.FeatureIdx];
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
        ctx->LocalExecutor.ExecRange([&](int blockIdx) {
            OfflineCtrBlock<ui8, IsTrueHistogram>(blockParams, blockIdx, fold, GetFloatHistogram(split, features).data(),
                                                  GetFeatureSplitIdx(split), splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    } else if (split.Type == ESplitType::OnlineCtr) {
        auto& ctr = fold.GetCtr(split.Ctr.Projection);
        ctx->LocalExecutor.ExecRange([&] (int i) {
            indicesData[i] += GetCtrSplit(split, i, ctr) * splitWeight;
        }, blockParams);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        ctx->LocalExecutor.ExecRange([&] (int blockIdx) {
            OfflineCtrBlock<int, IsTrueOneHotFeature>(blockParams, blockIdx, fold, GetRemappedCatFeatures(split, features).data(),
                                                      split.BinBorder, splitWeight, indicesData);
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

void DeleteSplit(int curDepth, int redundantIdx, yvector<TSplit>* tree, TTensorStructure3* tensor3, yvector<TIndexType>* indices) {
    if (redundantIdx != curDepth - 1) {
        DoSwap(tensor3->SelectedSplits[redundantIdx], tensor3->SelectedSplits.back());
        DoSwap(tree->at(redundantIdx), tree->back());
        for (auto& idx : *indices) {
            bool isTrueBack = (idx >> (curDepth - 1)) & 1;
            bool isTrueRedundant = (idx >> redundantIdx) & 1;
            idx ^= isTrueRedundant << redundantIdx;
            idx ^= isTrueBack << redundantIdx;
        }
    }

    tree->pop_back();
    tensor3->SelectedSplits.pop_back();
    for (auto& idx : *indices) {
        idx &= (1 << (curDepth - 1)) - 1;
    }
}

yvector<TIndexType> BuildIndices(const TFold& fold,
    const yvector<TSplit>& tree,
    const TTrainData& data,
    NPar::TLocalExecutor* localExecutor) {
    yvector<TIndexType> indices(fold.EffectiveDocCount);
    yvector<const TOnlineCTR*> onlineCtrs(tree.ysize());
    for (int splitIdx = 0; splitIdx < tree.ysize(); ++splitIdx) {
        const auto& split = tree[splitIdx];
        if (split.Type == ESplitType::OnlineCtr) {
            onlineCtrs[splitIdx] = &fold.GetCtr(split.Ctr.Projection);
        }
    }

    const int blockSize = 1000;
    NPar::TLocalExecutor::TBlockParams learnBlockParams(0, data.LearnSampleCount);
    learnBlockParams.SetBlockSize(blockSize).WaitCompletion();

    localExecutor->ExecRange([&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.ysize(); ++splitIdx) {
            const auto& split = tree[splitIdx];
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
                OfflineCtrBlock<int, IsTrueOneHotFeature>(learnBlockParams, blockIdx, fold, GetRemappedCatFeatures(split, data.AllFeatures).data(),
                    split.BinBorder, splitWeight, indices.data());
            }
        }
    }, 0, learnBlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    NPar::TLocalExecutor::TBlockParams tailBlockParams(data.LearnSampleCount, fold.EffectiveDocCount);
    tailBlockParams.SetBlockSize(blockSize).WaitCompletion();

    localExecutor->ExecRange([&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.ysize(); ++splitIdx) {
            const auto& split = tree[splitIdx];
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
                const int featureSplitValue = split.BinBorder;
                const int* featureValueData = GetRemappedCatFeatures(split, data.AllFeatures).data();
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
                                 const TCommonContext& ) {
    int samplesCount = GetDocCount(features);
    yvector<TIndexType> indices(samplesCount);
    const int splitCount = tree.SelectedSplits.ysize();
    yvector<ui64> ctrHashes;

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
            const auto& ctr = split.OnlineCtr.Ctr;
            const ECtrType ctrType = ctr.CtrType;
            const auto& learnCtr = model.CtrCalcerData.LearnCtrs.find(ctr)->second;
            ctrHashes.resize(samplesCount);
            const auto& projection = split.OnlineCtr.Ctr.Projection;
            CalcHashes(projection, features, samplesCount, yvector<int>(), &ctrHashes);

            if (ctrType == ECtrType::BinarizedTargetMeanValue || ctrType == ECtrType::FloatTargetMeanValue) {
                auto ctrMean = learnCtr.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
                for (int doc = 0; doc < samplesCount; ++doc) {
                    float targetStatisticSum = 0;
                    int totalCount = 0;
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    if (idx != TCtrValueTable::UnknownHash) {
                        const TCtrMeanHistory& ctrMeanHistory = ctrMean[idx];
                        targetStatisticSum = ctrMeanHistory.Sum;
                        totalCount = ctrMeanHistory.Count;
                    }
                    const float ctrValue = ctr.Calc(targetStatisticSum, totalCount);
                    indices[doc] |= ((int)(ctrValue > split.OnlineCtr.Border)) << splitIdx;
                }
            } else if (ctrType == ECtrType::FeatureFreq || ctrType == ECtrType::Counter) {
                int denominator = learnCtr.CounterDenominator;
                NArrayRef::TConstArrayRef<int> ctrTotal = learnCtr.GetTypedArrayRefForBlobData<int>();
                for (int doc = 0; doc < samplesCount; ++doc) {
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    int currentBucket = 0;
                    if (idx != TCtrValueTable::UnknownHash) {
                        currentBucket = ctrTotal[idx];
                    }
                    const float ctrValue = ctr.Calc(currentBucket, denominator);
                    indices[doc] |= (int)(ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            } else if (ctrType == ECtrType::Buckets) {
                auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int targetClassesCount = learnCtr.TargetClassesCount;
                for (int doc = 0; doc < samplesCount; ++doc) {
                    int goodCount = 0;
                    int totalCount = 0;
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    if (idx != TCtrValueTable::UnknownHash) {
                        auto ctrHistory = MakeArrayRef(ctrIntArray.data() + idx * targetClassesCount, targetClassesCount);
                        goodCount = ctrHistory[ctr.TargetBorderIdx];
                        for (int classId = 0; classId < targetClassesCount; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                    }
                    const float ctrValue = ctr.Calc(goodCount, totalCount);
                    indices[doc] |= (int)(ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            } else if (ctrType == ECtrType::Borders) {
                auto ctrIntArray = learnCtr.GetTypedArrayRefForBlobData<int>();
                const int targetClassesCount = learnCtr.TargetClassesCount;
                for (int doc = 0; doc < samplesCount; ++doc) {
                    int goodCount = 0;
                    int totalCount = 0;
                    const auto idx = learnCtr.ResolveHashToIndex(ctrHashes[doc]);
                    if (idx != TCtrValueTable::UnknownHash) {
                        auto ctrHistory = MakeArrayRef(ctrIntArray.data() + idx * targetClassesCount, targetClassesCount);
                        for (int classId = 0; classId < ctr.TargetBorderIdx + 1; ++classId) {
                            totalCount += ctrHistory[classId];
                        }
                        for (int classId = ctr.TargetBorderIdx + 1; classId < targetClassesCount; ++classId) {
                            goodCount += ctrHistory[classId];
                        }
                        totalCount += goodCount;
                    }
                    const float ctrValue = ctr.Calc(goodCount, totalCount);
                    indices[doc] |= (int)(ctrValue > split.OnlineCtr.Border) << splitIdx;
                }
            } else {
                ythrow TCatboostException() << "Unsupported ctr type " << ctrType;
            }
        }
    }
    return indices;
}
