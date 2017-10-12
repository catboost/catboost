#include "calc_fstr.h"
#include "index_calcer.h"
#include "full_features.h"
#include "learn_context.h"

#include <catboost/libs/fstr/feature_str.h>
#include <catboost/libs/fstr/doc_fstr.h>

#include <util/generic/xrange.h>
#include <util/generic/set.h>
#include <util/generic/maybe.h>


static TFeature GetFeature(const TModelSplit& split) {
    TFeature result;
    result.Type = split.Type;
    switch(result.Type) {
        case ESplitType::FloatFeature:
            result.FeatureIdx = split.BinFeature.FloatFeature;
            break;
        case ESplitType::OneHotFeature:
            result.FeatureIdx = split.OneHotFeature.CatFeatureIdx;
            break;
        case ESplitType::OnlineCtr:
            result.Ctr = split.OnlineCtr.Ctr;
            break;
    }
    return result;
}

struct TFeatureHash {
    size_t operator()(const TFeature& f) const {
        return f.GetHash();
    }
};

static yvector<TMxTree> BuildTrees(const yhash<TFeature, int, TFeatureHash>& featureToIdx,
                                   const yvector<TTensorStructure3>& treeStruct,
                                   const yvector<yvector<yvector<double>>>& leafValues) {
    yvector<TMxTree> trees(treeStruct.ysize());

    for (int treeIdx = 0; treeIdx < trees.ysize(); ++treeIdx) {
        auto& tree = trees[treeIdx];
        const int approxDimension = leafValues[treeIdx].ysize();
        const int leafCount = leafValues[treeIdx][0].ysize();

        tree.Leafs.resize(leafCount);
        for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            tree.Leafs[leafIdx].Vals.resize(approxDimension);
        }

        for (int dim = 0; dim < approxDimension; ++dim) {
            for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                tree.Leafs[leafIdx].Vals[dim] = leafValues[treeIdx][dim][leafIdx];
            }
        }

        for (const auto& split : treeStruct[treeIdx].SelectedSplits) {
            auto f = GetFeature(split);
            int fIdx = featureToIdx.find(f)->second;
            tree.SrcFeatures.push_back(fIdx);
        }
    }
    return trees;
}

static yvector<yvector<ui64>> CollectLeavesStatistics(const TPool& pool, const TFullModel& model,
                                                      const TAllFeatures& features,
                                                      const TCommonContext& ctx) {
    const size_t treeCount = model.TreeStruct.size();
    yvector<yvector<ui64>> leavesStatistics(treeCount, yvector<ui64>{});
    for (size_t index = 0; index < treeCount; ++index) {
        leavesStatistics[index].resize(model.LeafValues[index][0].size());
    }

    const int documentsCount = pool.Docs.GetDocCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        yvector<TIndexType> indices = BuildIndices(
            model.TreeStruct[treeIdx],
            model,
            features,
            ctx);

        if (indices.empty()) {
            continue;
        }
        for (int doc = 0; doc < documentsCount; ++doc) {
            const TIndexType valueIndex = indices[doc];
            ++leavesStatistics[treeIdx][valueIndex];
        }
    }
    return leavesStatistics;
}

yvector<std::pair<double, TFeature>> CalcFeatureEffect(const TFullModel& model, const TPool& pool, int threadCount/*= 1*/) {
    CB_ENSURE(pool.Docs.GetDocCount() != 0, "Pool should not be empty");
    if (model.TreeStruct.empty()) {
        return yvector<std::pair<double, TFeature>>();
    }
    int featureCount = pool.Docs.GetFactorsCount();
    NJson::TJsonValue jsonParams = ReadTJsonValue(model.ModelInfo.at("params"));
    jsonParams.InsertValue("thread_count", threadCount);
    TCommonContext ctx(jsonParams, Nothing(), Nothing(), featureCount, pool.CatFeatures, pool.FeatureId);

    TAllFeatures allFeatures;
    PrepareAllFeatures(ctx.CatFeatures, model.Borders, model.HasNans, yvector<int>(), LearnNotSet, ctx.Params.OneHotMaxSize, ctx.Params.NanMode, /*clear learn pool*/ false, ctx.LocalExecutor, &pool.Docs, &allFeatures);


    CB_ENSURE(!model.TreeStruct.empty(), "model should not be empty");
    CB_ENSURE(allFeatures.CatFeatures.ysize() + allFeatures.FloatHistograms.ysize() > 0, "no features in pool");
    yhash<TFeature, int, TFeatureHash> featureToIdx;
    yvector<TFeature> features;
    for (const auto& tree : model.TreeStruct) {
        for (const auto& split : tree.SelectedSplits) {
            TFeature feature = GetFeature(split);
            if (featureToIdx.has(feature))
                continue;
            int featureIdx = featureToIdx.ysize();
            featureToIdx[feature] = featureIdx;
            features.push_back(feature);
        }
    }

    yvector<TMxTree> trees = BuildTrees(featureToIdx,
                                        model.TreeStruct,
                                        model.LeafValues);
    yvector<yvector<ui64>> leavesStatistics = CollectLeavesStatistics(pool, model, allFeatures, ctx);
    yvector<double> effect = CalcEffect(trees, leavesStatistics);
    yvector<std::pair<double, int>> effectWithFeature;
    for (int i = 0; i < effect.ysize(); ++i) {
        effectWithFeature.emplace_back(effect[i], i);
    }
    Sort(effectWithFeature.begin(), effectWithFeature.end(), std::greater<std::pair<double, int>>());

    yvector<std::pair<double, TFeature>> result;
    for (int i = 0; i < effectWithFeature.ysize(); ++i) {
        result.emplace_back(effectWithFeature[i].first, features[effectWithFeature[i].second]);
    }
    return result;
}

yvector<TFeatureEffect> CalcRegularFeatureEffect(const yvector<std::pair<double, TFeature>>& internalEffect,
                                                 int catFeaturesCount, int floatFeaturesCount) {
    yvector<double> catFeatureEffect(catFeaturesCount);
    yvector<double> floatFeatureEffect(floatFeaturesCount);

    for (const auto& effectWithSplit : internalEffect) {
        TFeature feature = effectWithSplit.second;
        switch (feature.Type) {
            case ESplitType::FloatFeature:
                floatFeatureEffect[feature.FeatureIdx] += effectWithSplit.first;
                break;
            case ESplitType::OneHotFeature:
                catFeatureEffect[feature.FeatureIdx] += effectWithSplit.first;
                break;
            case ESplitType::OnlineCtr:
                TProjection proj = feature.Ctr.Projection;
                int featuresInSplit = proj.BinFeatures.ysize() + proj.CatFeatures.ysize() + proj.OneHotFeatures.ysize();
                double addEffect = effectWithSplit.first / featuresInSplit;
                for (const auto& binFeature : proj.BinFeatures) {
                    floatFeatureEffect[binFeature.FloatFeature] += addEffect;
                }
                for (auto catIndex : proj.CatFeatures) {
                    catFeatureEffect[catIndex] += addEffect;
                }
                for (auto oneHotFeature : proj.OneHotFeatures) {
                    catFeatureEffect[oneHotFeature.CatFeatureIdx] += addEffect;
                }
                break;
        }
    }

    double totalCat = Accumulate(catFeatureEffect.begin(), catFeatureEffect.end(), 0.0);
    double totalFloat = Accumulate(floatFeatureEffect.begin(), floatFeatureEffect.end(), 0.0);
    double total = totalCat + totalFloat;

    yvector<TFeatureEffect> regularFeatureEffect;
    for (int i = 0; i < catFeatureEffect.ysize(); ++i) {
        if (catFeatureEffect[i] > 0) {
            regularFeatureEffect.push_back(TFeatureEffect(catFeatureEffect[i] / total * 100, EFeatureType::Categorical, i));
        }
    }
    for (int i = 0; i < floatFeatureEffect.ysize(); ++i) {
        if (floatFeatureEffect[i] > 0) {
            regularFeatureEffect.push_back(TFeatureEffect(floatFeatureEffect[i] / total * 100, EFeatureType::Float, i));
        }
    }

    Sort(regularFeatureEffect.rbegin(), regularFeatureEffect.rend(), [](const TFeatureEffect& left, const TFeatureEffect& right) {
        return left.Score < right.Score;
    });
    return regularFeatureEffect;
}

yvector<double> CalcRegularFeatureEffect(const TFullModel& model, const TPool& pool, int threadCount/*= 1*/) {
    int featureCount = pool.Docs.GetFactorsCount();
    CB_ENSURE(featureCount == model.FeatureCount, "train and test datasets should have the same feature count");
    int catFeaturesCount = pool.CatFeatures.ysize();
    int floatFeaturesCount = featureCount - catFeaturesCount;
    TFeaturesLayout layout(featureCount, pool.CatFeatures, pool.FeatureId);

    yvector<TFeatureEffect> regularEffect = CalcRegularFeatureEffect(CalcFeatureEffect(model, pool, threadCount),
                                                                     catFeaturesCount, floatFeaturesCount);

    yvector<double> effect(featureCount);
    for (const auto& featureEffect : regularEffect) {
        int featureIdx = layout.GetFeature(featureEffect.Feature.Index, featureEffect.Feature.Type);
        Y_ASSERT(featureIdx < featureCount);
        effect[featureIdx] = featureEffect.Score;
    }

    return effect;
}

yvector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model) {
    if (model.TreeStruct.empty()) {
        return yvector<TInternalFeatureInteraction>();
    }

    yhash<TFeature, int, TFeatureHash> featureToIdx;
    yvector<TFeature> features;
    for (const auto& tree : model.TreeStruct) {
        for (const auto& split : tree.SelectedSplits) {
            TFeature feature = GetFeature(split);
            if (featureToIdx.has(feature))
                continue;
            int featureIdx = featureToIdx.ysize();
            featureToIdx[feature] = featureIdx;
            features.push_back(feature);
        }
    }

    yvector<TMxTree> trees = BuildTrees(featureToIdx,
                                        model.TreeStruct,
                                        model.LeafValues);
    yvector<TFeaturePairInteractionInfo> pairwiseEffect = CalcMostInteractingFeatures(trees);
    yvector<TInternalFeatureInteraction> result;
    result.reserve(pairwiseEffect.size());
    for (const auto& efffect : pairwiseEffect) {
        result.emplace_back(efffect.Score, features[efffect.Feature1], features[efffect.Feature2]);
    }
    return result;
}

yvector<TFeatureInteraction> CalcFeatureInteraction(const yvector<TInternalFeatureInteraction>& internalFeatureInteraction,
                                                          const TFeaturesLayout& layout) {
    yhash<std::pair<int, int>, double> sumInteraction;
    double totalEffect = 0;

    for (const auto& effectWithFeaturePair : internalFeatureInteraction) {
        yvector<TFeature> features{effectWithFeaturePair.FirstFeature, effectWithFeaturePair.SecondFeature};

        yvector<yvector<int>> internalToRegular;
        for (const auto& internalFeature : features) {
            yvector<int> regularFeatures;
            if (internalFeature.Type == ESplitType::FloatFeature) {
                regularFeatures.push_back(layout.GetFeature(internalFeature.FeatureIdx, EFeatureType::Float));
            } else {
                TProjection proj = internalFeature.Ctr.Projection;
                for (auto& binFeature : proj.BinFeatures) {
                    regularFeatures.push_back(layout.GetFeature(binFeature.FloatFeature, EFeatureType::Float));
                }
                for (auto catFeature : proj.CatFeatures) {
                    regularFeatures.push_back(layout.GetFeature(catFeature, EFeatureType::Categorical));
                }
            }
            internalToRegular.push_back(regularFeatures);
        }

        double effect = effectWithFeaturePair.Score;
        for (int f0 : internalToRegular[0]) {
            for (int f1 : internalToRegular[1]) {
                if (f0 == f1) {
                    continue;
                }
                if (f1 < f0) {
                    DoSwap(f0, f1);
                }
                sumInteraction[std::make_pair(f0, f1)] += effect / (internalToRegular[0].ysize() * internalToRegular[1].ysize());
            }
        }
        totalEffect += effect;
    }

    yvector<TFeatureInteraction> regularFeatureEffect;
    for (const auto& pairInteraction : sumInteraction) {
        int f0 = pairInteraction.first.first;
        int f1 = pairInteraction.first.second;
        regularFeatureEffect.push_back(
            TFeatureInteraction(sumInteraction[pairInteraction.first] / totalEffect * 100, layout.GetFeatureType(f0),
                                layout.GetInternalFeatureIdx(f0),
                                layout.GetFeatureType(f1), layout.GetInternalFeatureIdx(f1)));
    }

    Sort(regularFeatureEffect.rbegin(), regularFeatureEffect.rend(), [](const TFeatureInteraction& left, const TFeatureInteraction& right) {
        return left.Score < right.Score;
    });
    return regularFeatureEffect;
}

TString TFeature::BuildDescription(const TFeaturesLayout& layout) const {
    TStringBuilder result;
    if (Type == ESplitType::OnlineCtr) {
        result << ::BuildDescription(layout, Ctr.Projection);
        result << " prior_num=" << Ctr.PriorNum;
        result << " prior_denom=" << Ctr.PriorDenom;
        result << " targetborder=" << Ctr.TargetBorderIdx;
        result << " type=" << Ctr.CtrType;
    } else if (Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, FeatureIdx, EFeatureType::Float);
    } else {
        Y_ASSERT(Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, FeatureIdx, EFeatureType::Categorical);
    }
    return result;
}

yvector<yvector<double>> CalcFstr(const TFullModel& model, const TPool& pool, int threadCount){
    yvector<double> regularEffect = CalcRegularFeatureEffect(model, pool, threadCount);
    yvector<yvector<double>> result;
    for (const auto& value : regularEffect){
        yvector<double> vec = {value};
        result.push_back(vec);
    }
    return result;
}

yvector<yvector<double>> CalcInteraction(const TFullModel& model, const TPool& pool){
    int featureCount = pool.Docs.GetFactorsCount();
    TFeaturesLayout layout(featureCount, pool.CatFeatures, pool.FeatureId);

    yvector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    yvector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);
    yvector<yvector<double>> result;
    for (const auto& value : interaction){
        int featureIdxFirst = layout.GetFeature(value.FirstFeature.Index, value.FirstFeature.Type);
        int featureIdxSecond = layout.GetFeature(value.SecondFeature.Index, value.SecondFeature.Type);
        yvector<double> vec = {static_cast<double>(featureIdxFirst), static_cast<double>(featureIdxSecond), value.Score};
        result.push_back(vec);
    }
    return result;
}

yvector<yvector<double>> GetFeatureImportances(const TFullModel& model, const TPool& pool, const TString& type, int threadCount){
    CB_ENSURE(pool.Docs.GetDocCount() != 0, "Pool should not be empty");
    EFstrType FstrType = FromString<EFstrType>(type);
    switch (FstrType) {
        case EFstrType::FeatureImportance:
            return CalcFstr(model, pool, threadCount);
        case EFstrType::Interaction:
            return CalcInteraction(model, pool);
        case EFstrType::Doc:
            return CalcFeatureImportancesForDocuments(model, pool, threadCount);
        default:
            Y_UNREACHABLE();
    }
}
