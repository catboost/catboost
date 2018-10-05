#include "index_calcer.h"
#include "score_calcer.h"
#include "online_ctr.h"
#include "tree_print.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/helpers/dense_hash.h>
#include <catboost/libs/model/model_pool_compatibility.h>


static bool GetCtrSplit(const TSplit& split, int idxPermuted,
                        const TOnlineCTR& ctr) {

    ui8 ctrValue = ctr.Feature[split.Ctr.CtrIdx]
                              [split.Ctr.TargetBorderIdx]
                              [split.Ctr.PriorIdx]
                              [idxPermuted];
    return ctrValue > split.BinBorder;
}

static inline ui8 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinBorder;
}

static inline const TVector<ui8>& GetFloatHistogram(const TSplit& split, const TAllFeatures& features) {
    return features.FloatHistograms[split.FeatureIdx];
}

static inline const TVector<int>& GetRemappedCatFeatures(const TSplit& split, const TAllFeatures& features) {
    return features.CatFeaturesRemapped[split.FeatureIdx];
}

template <typename TCount, bool (*CmpOp)(TCount, TCount), int vectorWidth>
void BuildIndicesKernel(const size_t* permutation, const TCount* histogram, TCount value, int level, TIndexType* indices) {
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
void OfflineCtrBlock(const NPar::TLocalExecutor::TExecRangeParams& params,
                     int blockIdx,
                     const TFold& fold,
                     const TCount* histogram,
                     TCount value,
                     int level,
                     TIndexType* indices) {
    const size_t* permutation = fold.LearnPermutation.data();
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
                        TVector<TIndexType>* indices,
                        NPar::TLocalExecutor* localExecutor) {
    CB_ENSURE(curDepth > 0);

    const int blockSize = 1000;
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, indices->ysize());
    blockParams.SetBlockSize(blockSize);

    const int splitWeight = 1 << (curDepth - 1);
    TIndexType* indicesData = indices->data();
    if (split.Type == ESplitType::FloatFeature) {
        localExecutor->ExecRange([&](int blockIdx) {
            OfflineCtrBlock<ui8, IsTrueHistogram>(blockParams, blockIdx, fold, GetFloatHistogram(split, features).data(),
                                                  GetFeatureSplitIdx(split), splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    } else if (split.Type == ESplitType::OnlineCtr) {
        auto& ctr = fold.GetCtr(split.Ctr.Projection);
        localExecutor->ExecRange([&] (int i) {
            indicesData[i] += GetCtrSplit(split, i, ctr) * splitWeight;
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        localExecutor->ExecRange([&] (int blockIdx) {
            OfflineCtrBlock<int, IsTrueOneHotFeature>(blockParams, blockIdx, fold, GetRemappedCatFeatures(split, features).data(),
                                                      split.BinBorder, splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

TVector<bool> GetIsLeafEmpty(int curDepth, const TVector<TIndexType>& indices) {
    TVector<bool> isLeafEmpty(1 << curDepth, true);
    for (const auto& idx : indices) {
        isLeafEmpty[idx] = false;
    }
    return isLeafEmpty;
}

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty) {
    const int leafCount = isLeafEmpty.ysize();
    for (int splitIdx = 0; (1 << splitIdx) < leafCount; ++splitIdx) {
        bool isRedundantSplit = true;
        for (int idx = 0; idx < leafCount; ++idx) {
            if (idx & (1 << splitIdx)) {
                continue;
            }
            if (!isLeafEmpty[idx] && !isLeafEmpty[idx ^ (1 << splitIdx)]) {
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

// Get OnlineCTRs associated with a fold
static TVector<const TOnlineCTR*> GetOnlineCtrs(const TFold& fold, const TSplitTree& tree) {
    TVector<const TOnlineCTR*> onlineCtrs(tree.GetDepth());
    for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
        const auto& split = tree.Splits[splitIdx];
        if (split.Type == ESplitType::OnlineCtr) {
            onlineCtrs[splitIdx] = &fold.GetCtr(split.Ctr.Projection);
        }
    }
    return onlineCtrs;
}

static void BuildIndicesForLearn(const TSplitTree& tree,
                                 const TDataset& learnData,
                                 int learnSampleCount,
                                 const TVector<const TOnlineCTR*>& onlineCtrs,
                                 const TFold& fold,
                                 NPar::TLocalExecutor* localExecutor,
                                 TIndexType* indices) {
    const int blockSize = 1000;
    NPar::TLocalExecutor::TExecRangeParams learnBlockParams(0, learnSampleCount);
    learnBlockParams.SetBlockSize(blockSize);

    auto updateLearnIndex = [&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
            const auto& split = tree.Splits[splitIdx];
            const int splitWeight = 1 << splitIdx;
            if (split.Type == ESplitType::FloatFeature) {
                OfflineCtrBlock<ui8, IsTrueHistogram>(learnBlockParams, blockIdx, fold,
                    GetFloatHistogram(split, learnData.AllFeatures).data(),
                    GetFeatureSplitIdx(split), splitWeight, indices);
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(learnBlockParams, [&](int doc) {
                    indices[doc] += GetCtrSplit(split, doc, splitOnlineCtr) * splitWeight;
                })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                OfflineCtrBlock<int, IsTrueOneHotFeature>(learnBlockParams, blockIdx, fold,
                    GetRemappedCatFeatures(split, learnData.AllFeatures).data(),
                    split.BinBorder, splitWeight, indices);
            }
        }
    };

    localExecutor->ExecRange(updateLearnIndex, 0, learnBlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void BuildIndicesForTest(const TSplitTree& tree,
                                const TDataset& testData,
                                int tailSampleCount,
                                const TVector<const TOnlineCTR*>& onlineCtrs,
                                int docOffset,
                                NPar::TLocalExecutor* localExecutor,
                                TIndexType* indices) {
    const int blockSize = 1000;
    NPar::TLocalExecutor::TExecRangeParams tailBlockParams(0, tailSampleCount);
    tailBlockParams.SetBlockSize(blockSize);

    auto updateTailIndex = [&](int blockIdx) {
        TIndexType* tailIndices = indices;
        for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
            const auto& split = tree.Splits[splitIdx];
            const int splitWeight = 1 << splitIdx;
            if (split.Type == ESplitType::FloatFeature) {
                const ui8 featureSplitIdx = GetFeatureSplitIdx(split);
                const ui8* floatHistogramData = GetFloatHistogram(split, testData.AllFeatures).data();
                NPar::TLocalExecutor::BlockedLoopBody(tailBlockParams, [&](int doc) {
                    tailIndices[doc] += IsTrueHistogram(floatHistogramData[doc], featureSplitIdx) * splitWeight;
                })(blockIdx);
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(tailBlockParams, [&](int doc) {
                    tailIndices[doc] += GetCtrSplit(split, doc + docOffset, splitOnlineCtr) * splitWeight;
                })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                const int featureSplitValue = split.BinBorder;
                const int* featureValueData = GetRemappedCatFeatures(split, testData.AllFeatures).data();
                NPar::TLocalExecutor::BlockedLoopBody(tailBlockParams, [&](int doc) {
                    tailIndices[doc] += IsTrueOneHotFeature(featureValueData[doc], featureSplitValue) * splitWeight;
                })(blockIdx);
            }
        }
    };

    localExecutor->ExecRange(updateTailIndex, 0, tailBlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

TVector<TIndexType> BuildIndices(const TFold& fold,
                                 const TSplitTree& tree,
                                 const TDataset& learnData,
                                 const TDatasetPtrs& testDataPtrs,
                                 NPar::TLocalExecutor* localExecutor) {
    int learnSampleCount = learnData.GetSampleCount();
    int tailSampleCount = GetSampleCount(testDataPtrs);

    const TVector<const TOnlineCTR*>& onlineCtrs = GetOnlineCtrs(fold, tree);

    TVector<TIndexType> indices(learnSampleCount + tailSampleCount);

    BuildIndicesForLearn(tree, learnData, learnSampleCount, onlineCtrs, fold, localExecutor, indices.begin());
    int docOffset = learnSampleCount;
    for (int testIdx = 0; testIdx < testDataPtrs.ysize(); ++testIdx) {
        const TDataset* testData = testDataPtrs[testIdx];
        BuildIndicesForTest(tree, *testData, testData->GetSampleCount(), onlineCtrs, docOffset, localExecutor, indices.begin() + docOffset);
        docOffset += testData->GetSampleCount();
    }
    return indices;
}

void BinarizeFeatures(const TFullModel& model,
                      const TPool& pool,
                      size_t start,
                      size_t end,
                      TVector<ui8>* result) {
    CB_ENSURE(!pool.IsQuantized(), "Not supported for quantized pools");
    THashMap<int, int> columnReorderMap;
    CheckModelAndPoolCompatibility(model, pool, &columnReorderMap);
    auto docCount = end - start;
    result->resize(model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() * docCount);
    TVector<int> transposedHash(docCount * model.GetUsedCatFeaturesCount());
    TVector<float> ctrs(model.ObliviousTrees.GetUsedModelCtrs().size() * docCount);

    TVector<TConstArrayRef<float>> repackedFeatures;
    if (columnReorderMap.empty()) {
        for (int i = 0; i < pool.Docs.GetEffectiveFactorCount(); ++i) {
            repackedFeatures.emplace_back(MakeArrayRef(pool.Docs.Factors[i].data() + start, docCount));
        }
    } else {
        for (size_t i = 0; i < model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(); ++i) {
            repackedFeatures.emplace_back(MakeArrayRef(pool.Docs.Factors[columnReorderMap[i]].data() + start, docCount));
        }
    }

    BinarizeFeatures(model,
        [&repackedFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return repackedFeatures[floatFeature.FlatFeatureIndex][index];
        },
        [&repackedFeatures](const TCatFeature& catFeature, size_t index) -> int {
            return ConvertFloatCatFeatureToIntHash(repackedFeatures[catFeature.FlatFeatureIndex][index]);
        },
        0,
        docCount,
        *result,
        transposedHash,
        ctrs);
}

TVector<ui8> BinarizeFeatures(const TFullModel& model, const TPool& pool, size_t start, size_t end) {
    TVector<ui8> result;
    BinarizeFeatures(model, pool, start, end, &result);
    return result;
}

TVector<ui8> BinarizeFeatures(const TFullModel& model, const TPool& pool) {
    return BinarizeFeatures(model, pool, /*start*/0, pool.Docs.GetDocCount());
}

TVector<TIndexType> BuildIndicesForBinTree(const TFullModel& model, const TVector<ui8>& binarizedFeatures, size_t treeId) {
    if (model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() == 0) {
        return TVector<TIndexType>();
    }

    auto docCount = binarizedFeatures.size() / model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount();
    TVector<TIndexType> indexesVec(docCount);
    const auto* treeSplitsCurPtr =
        model.ObliviousTrees.GetRepackedBins().data() +
        model.ObliviousTrees.TreeStartOffsets[treeId];
    CalcIndexes(!model.ObliviousTrees.OneHotFeatures.empty(), binarizedFeatures.data(), docCount, indexesVec.data(), treeSplitsCurPtr, model.ObliviousTrees.TreeSizes[treeId]);
    return indexesVec;
}
