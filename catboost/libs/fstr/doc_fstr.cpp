#include "doc_fstr.h"
#include "feature_str.h"
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/model/split.h>
#include <catboost/libs/algo/features_layout.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/index_calcer.h>

static bool SplitHasFeature(const int feature, const TSplit& split, const TFeaturesLayout& layout) {
    const EFeatureType featureType = layout.GetFeatureType(feature);
    const int internalIdx = layout.GetInternalFeatureIdx(feature);
    if (split.Type == ESplitType::FloatFeature) {
        return split.BinFeature.FloatFeature == internalIdx && featureType == EFeatureType::Float;
    } else if (split.Type == ESplitType::OneHotFeature) {
        return split.OneHotFeature.CatFeatureIdx == internalIdx && featureType == EFeatureType::Categorical;
    } else {
        Y_ASSERT(split.Type == ESplitType::OnlineCtr);
        const TProjection proj = split.OnlineCtr.Ctr.Projection;
        if (featureType == EFeatureType::Categorical) {
            for (int featureIdx : proj.CatFeatures) {
                if (featureIdx == internalIdx) {
                    return true;
                }
            }
            for (const auto& oneHotFeature : proj.OneHotFeatures) {
                if (oneHotFeature.CatFeatureIdx == internalIdx) {
                    return true;
                }
            }
        } else {
            for (const auto& binFeature : proj.BinFeatures) {
                if (binFeature.FloatFeature == internalIdx) {
                    return true;
                }
            }
        }
    }
    return false;
}

yvector<yvector<TIndexType>> BuildIndicesWithoutFeature(const TTensorStructure3& tree,
                                                        const TFullModel& model,
                                                        const TAllFeatures& features,
                                                        const int ignoredFeatureIdx,
                                                        const TCommonContext& ctx) {
    yvector<TIndexType> indicesSource = BuildIndices(tree, model, features, ctx);
    int samplesCount = indicesSource.ysize();
    yvector<yvector<TIndexType>> indices(samplesCount, yvector<TIndexType>(1));
    for (int i = 0; i < indicesSource.ysize(); ++i) {
        indices[i][0] = indicesSource[i];
    }

    const int splitCount = tree.SelectedSplits.ysize();

    for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
        const auto& split = tree.SelectedSplits[splitIdx];
        if (SplitHasFeature(ignoredFeatureIdx, split, ctx.Layout)) {
            for (int doc = 0; doc < samplesCount; ++doc) {
                int indicesCount = indices[doc].ysize();
                for (int i = 0; i < indicesCount; ++i) {
                    indices[doc].push_back(indices[doc][i] ^ (1 << splitIdx));
                }
            }
        }
    }

    return indices;
}

static yvector<yvector<double>> CalcFeatureImportancesForDocuments(const TFullModel& model,
                                                                  const TAllFeatures& features,
                                                                  const yvector<yvector<yvector<double>>>& approx,
                                                                  TCommonContext* ctx) {
    const int approxDimension = model.LeafValues[0].ysize();
    const int docCount = approx[0][0].ysize();
    const int featureCount = features.CatFeatures.ysize() + features.FloatHistograms.ysize();

    const TTreeFunction CalcFeatureImportanceForTree = [&](const TAllFeatures& features,
                                                           const TFullModel& model,
                                                           int treeIdx,
                                                           const TCommonContext& ctx,
                                                           yvector<yvector<double>>* resultPtr) { // [docId][featureId]
        yvector<yvector<double>>& result = *resultPtr;
        for (int featureId = 0; featureId < featureCount; ++featureId) {
            yvector<yvector<TIndexType>> indices = BuildIndicesWithoutFeature(model.TreeStruct[treeIdx],
                                                                              model,
                                                                              features,
                                                                              featureId,
                                                                              ctx);
            for (int dim = 0; dim < approxDimension; ++dim) {
                for (int doc = 0; doc < docCount; ++doc) {
                    double leafValue = 0;
                    for (int leafIdx = 0; leafIdx < indices[doc].ysize(); ++leafIdx) {
                        leafValue += model.LeafValues[treeIdx][dim][indices[doc][leafIdx]];
                    }
                    leafValue /= static_cast<double>(indices[doc].ysize());
                    result[featureId][doc] += Abs(approx[treeIdx][dim][doc] - leafValue);
                }
            }
        }
    };
    return MapFunctionToTrees(model, features, 0, 0, CalcFeatureImportanceForTree, featureCount, ctx);
}

yvector<yvector<double>> CalcFeatureImportancesForDocuments(const TFullModel& model,
                                                            const TPool& pool,
                                                            const int threadCount) {
    CB_ENSURE(!pool.Docs.empty(), "Pool should not be empty");
    CB_ENSURE(!model.TreeStruct.empty(), "Model is empty. Did you fit the model?");
    int featureCount = pool.Docs[0].Factors.ysize();
    NJson::TJsonValue jsonParams = ReadTJsonValue(model.ParamsJson);
    jsonParams.InsertValue("thread_count", threadCount);
    TCommonContext ctx(jsonParams, Nothing(), Nothing(), featureCount, pool.CatFeatures, pool.FeatureId);

    TAllFeatures allFeatures;
    PrepareAllFeatures(pool.Docs, ctx.CatFeatures, model.Borders, yvector<int>(), LearnNotSet, ctx.Params.OneHotMaxSize, ctx.LocalExecutor, &allFeatures);

    const int approxDimension = model.LeafValues[0].ysize();
    const int docCount = pool.Docs.ysize();

    int treeCount = model.TreeStruct.ysize();
    yvector<yvector<yvector<double>>> approx(treeCount,
                                             yvector<yvector<double>>(approxDimension, yvector<double>(docCount))); // [tree][dim][docIdx]
    for (int treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        CalcApproxForTree(allFeatures, model, treeIdx, ctx, &approx[treeIdx]);
    }
    yvector<yvector<double>> result = CalcFeatureImportancesForDocuments(model, allFeatures, approx, &ctx);

    return result;
}
