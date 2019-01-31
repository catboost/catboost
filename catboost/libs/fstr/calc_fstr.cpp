#include "calc_fstr.h"

#include "feature_str.h"
#include "shap_values.h"
#include "util.h"

#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/plot.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/target/data_providers.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/xrange.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/system/compiler.h>

#include <functional>


using namespace NCB;


static TFeature GetFeature(const TModelSplit& split) {
    TFeature result;
    result.Type = split.Type;
    switch(result.Type) {
        case ESplitType::FloatFeature:
            result.FeatureIdx = split.FloatFeature.FloatFeature;
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

static TVector<TFeature>  GetCombinationClassFeatures(const TObliviousTrees& forest) {
    NCB::TFeaturesLayout layout(forest.FloatFeatures, forest.CatFeatures);
    TVector<std::pair<TVector<int>, TFeature>> featuresCombinations;

    for (const TFloatFeature& floatFeature : forest.FloatFeatures) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        featuresCombinations.emplace_back();
        featuresCombinations.back().first = { floatFeature.FlatFeatureIndex };
        featuresCombinations.back().second = TFeature(floatFeature);
    }
    for (const TOneHotFeature& oneHotFeature: forest.OneHotFeatures) {
        featuresCombinations.emplace_back();
        featuresCombinations.back().first = {
                (int)layout.GetExternalFeatureIdx(oneHotFeature.CatFeatureIndex,
                                                  EFeatureType::Categorical)
        };
        featuresCombinations.back().second = TFeature(oneHotFeature);
    }
    for (const TCtrFeature& ctrFeature : forest.CtrFeatures) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        featuresCombinations.emplace_back();
        for (int catFeatureIdx : combination.CatFeatures) {
            featuresCombinations.back().first.push_back(
                    layout.GetExternalFeatureIdx(catFeatureIdx, EFeatureType::Categorical));
        }
        featuresCombinations.back().second = TFeature(ctrFeature);
    }
    TVector<int> sortedBinFeatures(featuresCombinations.size());
    Iota(sortedBinFeatures.begin(), sortedBinFeatures.end(), 0);
    Sort(
            sortedBinFeatures.begin(),
            sortedBinFeatures.end(),
            [featuresCombinations](int feature1, int feature2) {
                return featuresCombinations[feature1].first < featuresCombinations[feature2].first;
            }
    );
    TVector<TFeature> combinationClassFeatures;

    for (ui32 featureIdx = 0; featureIdx < featuresCombinations.size(); ++featureIdx) {
        int currentFeature = sortedBinFeatures[featureIdx];
        int previousFeature = featureIdx == 0 ? -1 : sortedBinFeatures[featureIdx - 1];
        if (featureIdx == 0 || featuresCombinations[currentFeature].first != featuresCombinations[previousFeature].first) {
            combinationClassFeatures.push_back(featuresCombinations[currentFeature].second);
        }
    }
    return combinationClassFeatures;
}

struct TFeatureHash {
    size_t operator()(const TFeature& f) const {
        return f.GetHash();
    }
};

static TVector<TMxTree> BuildTrees(
    const THashMap<TFeature, int, TFeatureHash>& featureToIdx,
    const TFullModel& model)
{
    TVector<TMxTree> trees(model.ObliviousTrees.GetTreeCount());
    auto& binFeatures = model.ObliviousTrees.GetBinFeatures();
    for (int treeIdx = 0; treeIdx < trees.ysize(); ++treeIdx) {
        auto& tree = trees[treeIdx];
        const int leafCount = (1uLL << model.ObliviousTrees.TreeSizes[treeIdx]);

        tree.Leaves.resize(leafCount);
        for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            tree.Leaves[leafIdx].Vals.resize(model.ObliviousTrees.ApproxDimension);
        }
        auto firstTreeLeafPtr = model.ObliviousTrees.GetFirstLeafPtrForTree(treeIdx);
        for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            for (int dim = 0; dim < model.ObliviousTrees.ApproxDimension; ++dim) {
                tree.Leaves[leafIdx].Vals[dim] = firstTreeLeafPtr[leafIdx
                    * model.ObliviousTrees.ApproxDimension + dim];
            }
        }
        auto treeSplitsStart = model.ObliviousTrees.TreeStartOffsets[treeIdx];
        auto treeSplitsStop = treeSplitsStart + model.ObliviousTrees.TreeSizes[treeIdx];
        for (auto splitIdx = treeSplitsStart; splitIdx < treeSplitsStop; ++splitIdx) {
            auto feature = GetFeature(binFeatures[model.ObliviousTrees.TreeSplits[splitIdx]]);
            tree.SrcFeatures.push_back(featureToIdx.at(feature));
        }
    }
    return trees;
}

TVector<TMxTree> BuildMatrixnetTrees(const TFullModel& model, TVector<TFeature>* features) {
    THashMap<TFeature, int, TFeatureHash> featureToIdx;
    const auto& modelBinFeatures = model.ObliviousTrees.GetBinFeatures();
    for (auto binSplit : model.ObliviousTrees.TreeSplits) {
        TFeature feature = GetFeature(modelBinFeatures[binSplit]);
        if (featureToIdx.contains(feature)) {
            continue;
        }
        int featureIdx = featureToIdx.ysize();
        featureToIdx[feature] = featureIdx;
        features->push_back(feature);
    }

    return BuildTrees(featureToIdx, model);
}

static const TDataProvider GetSubset(
        const TDataProvider& dataset,
        ui32 maxDocumentCount,
        NPar::TLocalExecutor* localExecutor
) {
    auto documentCount = dataset.ObjectsData->GetObjectCount();

    if (documentCount > maxDocumentCount) {
        ui32 foldCount = documentCount / maxDocumentCount;

        TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets;

        testSubsets = NCB::Split(*dataset.ObjectsGrouping, foldCount, /*oldCvStyleSplit*/ true);

        auto subset = dataset.GetSubset(
                GetSubset(
                        dataset.ObjectsGrouping,
                        std::move(testSubsets[0]),
                        NCB::EObjectsOrder::Ordered
                ),
                localExecutor
        );
        return *subset.Get();
    } else {
        return dataset;
    }
}

static TVector<std::pair<double, TFeature>> CalcFeatureEffectAverageChange(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    NPar::TLocalExecutor* localExecutor)
{
    if (model.GetTreeCount() == 0) {
        return TVector<std::pair<double, TFeature>>();
    }

    // use only if model.ObliviousTrees.LeafWeights is empty
    TVector<TVector<double>> leavesStatisticsOnPool;
    if (model.ObliviousTrees.LeafWeights.empty()) {
        CB_ENSURE(dataset, "CalcFeatureEffect requires either non-empty LeafWeights in model"
            " or provided dataset");
        CB_ENSURE(dataset->GetObjectCount() != 0, "no docs in pool");
        CB_ENSURE(dataset->MetaInfo.GetFeatureCount() > 0, "no features in pool");

        leavesStatisticsOnPool = CollectLeavesStatistics(*dataset, model, localExecutor);
    }

    TVector<TFeature> features;
    TVector<TMxTree> trees = BuildMatrixnetTrees(model, &features);

    TVector<double> effect = CalcEffect(
        trees,
        model.ObliviousTrees.LeafWeights.empty() ? leavesStatisticsOnPool : model.ObliviousTrees.LeafWeights);

    TVector<std::pair<double, int>> effectWithFeature;
    for (int i = 0; i < effect.ysize(); ++i) {
        effectWithFeature.emplace_back(effect[i], i);
    }
    Sort(effectWithFeature.begin(), effectWithFeature.end(), std::greater<std::pair<double, int>>());

    TVector<std::pair<double, TFeature>> result;
    for (int i = 0; i < effectWithFeature.ysize(); ++i) {
        result.emplace_back(effectWithFeature[i].first, features[effectWithFeature[i].second]);
    }
    return result;
}

static bool TryGetLossDescription(const TFullModel& model, NCatboostOptions::TLossDescription& lossDescription) {
    if (!(model.ModelInfo.contains("loss_function") ||  model.ModelInfo.contains("params") && ReadTJsonValue(model.ModelInfo.at("params")).Has("loss_function"))) {
        return false;
    }
    if (model.ModelInfo.contains("loss_function")) {
        lossDescription.Load(ReadTJsonValue(model.ModelInfo.at("loss_function")));
    } else {
        lossDescription.Load(ReadTJsonValue(model.ModelInfo.at("params"))["loss_function"]);
    }
    return true;
}

static TVector<std::pair<double, TFeature>> CalcFeatureEffectLossChange(
        const TFullModel& model,
        const TDataProvider& dataProvider,
        NPar::TLocalExecutor* localExecutor)
{
    auto combinationClassFeatures = GetCombinationClassFeatures(model.ObliviousTrees);
    int featuresCount = combinationClassFeatures.size();
    if (featuresCount == 0) {
        TVector<std::pair<double, TFeature>> result;
        return result;
    }

    NCatboostOptions::TLossDescription lossDescription;
    CB_ENSURE(TryGetLossDescription(model, lossDescription));
    ui32 documentCount = dataProvider.ObjectsData->GetObjectCount();
    ui32 maxDocuments = Min(documentCount, Max(ui32(1e5), ui32(1e9 / dataProvider.ObjectsData->GetFeaturesLayout()->GetExternalFeatureCount())));

    const auto dataset = GetSubset(dataProvider, maxDocuments, localExecutor);

    documentCount = dataset.ObjectsData->GetObjectCount();
    const TObjectsDataProvider& objectsData = *dataset.ObjectsData;
    const auto* rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData);
    CB_ENSURE(rawObjectsData, "Quantized dataProviders are not supported yet");

    TRestorableFastRng64 rand(0);
    auto targetData = CreateModelCompatibleProcessedDataProvider(dataset, {lossDescription}, model, &rand, localExecutor).TargetData;
    TShapPreparedTrees preparedTrees = PrepareTrees(model, &dataset, 0, localExecutor, true);

    TVector<TMetricHolder> scores(featuresCount + 1);

    TConstArrayRef<TQueryInfo> queriesInfo = GetGroupInfo(targetData);
    TVector<TVector<double>> approx = ApplyModelMulti(model, objectsData, EPredictionType::RawFormulaVal, 0, documentCount,
                                                      localExecutor);
    THolder<IMetric> metric = std::move(CreateMetricFromDescription(lossDescription, documentCount)[0]);
    ui32 blockCount = queriesInfo.empty() ? documentCount : queriesInfo.size();
    scores.back().Add(
            metric->Eval(approx, GetTarget(targetData), GetWeights(targetData), queriesInfo, 0, blockCount, *localExecutor)
    );
    ui32 blockSize = Min(ui32(10000), ui32(1e6) / (featuresCount * approx.ysize())); // shapValues[blockSize][featuresCount][dim] double
    int approDimension = model.ObliviousTrees.ApproxDimension;
    for (ui32 queryBegin = 0; queryBegin < blockCount; queryBegin += blockSize) {
        ui32 queryEnd = Min(blockCount, queryBegin + blockSize);
        ui32 begin, end;
        if (queriesInfo.empty()) {
            begin = queryBegin;
            end = queryEnd;
        } else {
            begin = queriesInfo[queryBegin].Begin;
            end = queriesInfo[queryEnd - 1].End;
        }
        TVector<TVector<TVector<double>>> shapValues;
        CalcShapValuesInternalForFeature(preparedTrees, model, dataset, 0, begin, end, featuresCount,
                                         rawObjectsData, &shapValues, localExecutor);

        for (int featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
            NPar::TLocalExecutor::TExecRangeParams blockParams(begin, end);
            blockParams.SetBlockCountToThreadCount();
            localExecutor->ExecRange([&](ui32 docIdx) {
                for (int dimensionIdx = 0; dimensionIdx < approDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] -= shapValues[docIdx - begin][featureIdx][dimensionIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
            scores[featureIdx].Add(
                    metric->Eval(approx, GetTarget(targetData), GetWeights(targetData), queriesInfo, queryBegin, queryEnd,
                                 *localExecutor)
            );
            localExecutor->ExecRange([&](ui32 docIdx) {
                for (int dimensionIdx = 0; dimensionIdx < approDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] += shapValues[docIdx - begin][featureIdx][dimensionIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }

    TVector<std::pair<double, int>> featureScore(featuresCount);
    for (int idx = 0; idx < featuresCount; ++idx) {
        featureScore[idx].first = metric->GetFinalError(scores[idx]) - metric->GetFinalError(scores.back());
        featureScore[idx].second = idx;
    }
    Sort(featureScore.begin(), featureScore.end(), std::greater<std::pair<double, int>>());
    TVector<std::pair<double, TFeature>> result;

    for (const auto& score: featureScore) {
        result.emplace_back();
        result.back().first = score.first;
        result.back().second = combinationClassFeatures[score.second];
    }
    return result;
}

TVector<std::pair<double, TFeature>> CalcFeatureEffect(
        const TFullModel& model,
        const TDataProviderPtr dataset,
        EFstrType type,
        NPar::TLocalExecutor* localExecutor)
{
     if (type == EFstrType::LossFunctionChange) {
         CB_ENSURE(dataset, "dataset is not provided");

         const TObjectsDataProvider& objectsData = *(dataset->ObjectsData);
         const auto* rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData);
         CB_ENSURE(rawObjectsData); // Not supported for quantized pools
         NCatboostOptions::TLossDescription lossDescription;
         CB_ENSURE(TryGetLossDescription(model, lossDescription));
         NCatboostOptions::TLossDescription loss_description;
         loss_description.Load(ReadTJsonValue(model.ModelInfo.at("params"))["loss_function"]);
         CB_ENSURE(IsGroupwiseMetric(loss_description.LossFunction), "Use loss change fstr only for groupwise metric");

         return CalcFeatureEffectLossChange(model, *dataset, localExecutor);
     } else {
         return CalcFeatureEffectAverageChange(model, dataset, localExecutor);
     }
}

TVector<TFeatureEffect> CalcRegularFeatureEffect(
    const TVector<std::pair<double, TFeature>>& internalEffect,
    int catFeaturesCount,
    int floatFeaturesCount)
{
    TVector<double> catFeatureEffect(catFeaturesCount);
    TVector<double> floatFeatureEffect(floatFeaturesCount);

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
                auto& proj = feature.Ctr.Base.Projection;
                int featuresInSplit = proj.BinFeatures.ysize() + proj.CatFeatures.ysize()
                    + proj.OneHotFeatures.ysize();
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

    TVector<TFeatureEffect> regularFeatureEffect;
    for (int i = 0; i < catFeatureEffect.ysize(); ++i) {
        regularFeatureEffect.push_back(
            TFeatureEffect(catFeatureEffect[i], EFeatureType::Categorical, i));
    }
    for (int i = 0; i < floatFeatureEffect.ysize(); ++i) {
        regularFeatureEffect.push_back(
            TFeatureEffect(floatFeatureEffect[i], EFeatureType::Float, i));
    }

    Sort(
        regularFeatureEffect.rbegin(),
        regularFeatureEffect.rend(),
        [](const TFeatureEffect& left, const TFeatureEffect& right) {
            return left.Score < right.Score ||
                (left.Score == right.Score && left.Feature.Index > right.Feature.Index);
        });
    return regularFeatureEffect;
}

TVector<double> CalcRegularFeatureEffect(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    EFstrType type,
    NPar::TLocalExecutor* localExecutor)
{
    const NCB::TFeaturesLayout layout(
        model.ObliviousTrees.FloatFeatures,
        model.ObliviousTrees.CatFeatures);

    TVector<TFeatureEffect> regularEffect = CalcRegularFeatureEffect(
        CalcFeatureEffect(model, dataset, type, localExecutor),
        model.GetNumCatFeatures(),
        model.GetNumFloatFeatures());

    TVector<double> effect(layout.GetExternalFeatureCount());
    for (const auto& featureEffect : regularEffect) {
        int externalFeatureIdx = layout.GetExternalFeatureIdx(
            featureEffect.Feature.Index,
            featureEffect.Feature.Type);
        effect[externalFeatureIdx] = featureEffect.Score;
    }

    return effect;
}

TVector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model) {
    if (model.GetTreeCount() == 0) {
        return TVector<TInternalFeatureInteraction>();
    }

    TVector<TFeature> features;
    TVector<TMxTree> trees = BuildMatrixnetTrees(model, &features);

    TVector<TFeaturePairInteractionInfo> pairwiseEffect = CalcMostInteractingFeatures(trees);
    TVector<TInternalFeatureInteraction> result;
    result.reserve(pairwiseEffect.size());
    for (const auto& efffect : pairwiseEffect) {
        result.emplace_back(efffect.Score, features[efffect.Feature1], features[efffect.Feature2]);
    }
    return result;
}

TVector<TFeatureInteraction> CalcFeatureInteraction(
    const TVector<TInternalFeatureInteraction>& internalFeatureInteraction,
    const NCB::TFeaturesLayout& layout)
{
    THashMap<std::pair<int, int>, double> sumInteraction;
    double totalEffect = 0;

    for (const auto& effectWithFeaturePair : internalFeatureInteraction) {
        TVector<TFeature> features{effectWithFeaturePair.FirstFeature, effectWithFeaturePair.SecondFeature};

        TVector<TVector<int>> internalToRegular;
        for (const auto& internalFeature : features) {
            TVector<int> regularFeatures;
            if (internalFeature.Type == ESplitType::FloatFeature) {
                regularFeatures.push_back(
                    layout.GetExternalFeatureIdx(internalFeature.FeatureIdx, EFeatureType::Float));
            } else {
                auto proj = internalFeature.Ctr.Base.Projection;
                for (auto& binFeature : proj.BinFeatures) {
                    regularFeatures.push_back(
                        layout.GetExternalFeatureIdx(binFeature.FloatFeature, EFeatureType::Float));
                }
                for (auto catFeature : proj.CatFeatures) {
                    regularFeatures.push_back(
                        layout.GetExternalFeatureIdx(catFeature, EFeatureType::Categorical));
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
                sumInteraction[std::make_pair(f0, f1)] += effect
                    / (internalToRegular[0].ysize() * internalToRegular[1].ysize());
            }
        }
        totalEffect += effect;
    }

    TVector<TFeatureInteraction> regularFeatureEffect;
    for (const auto& pairInteraction : sumInteraction) {
        int f0 = pairInteraction.first.first;
        int f1 = pairInteraction.first.second;
        regularFeatureEffect.push_back(
            TFeatureInteraction(
                sumInteraction[pairInteraction.first] / totalEffect * 100,
                layout.GetExternalFeatureType(f0),
                layout.GetInternalFeatureIdx(f0),
                layout.GetExternalFeatureType(f1),
                layout.GetInternalFeatureIdx(f1)));
    }

    Sort(
        regularFeatureEffect.rbegin(),
        regularFeatureEffect.rend(),
        [](const TFeatureInteraction& left, const TFeatureInteraction& right) {
            return left.Score < right.Score;
        });
    return regularFeatureEffect;
}

TString TFeature::BuildDescription(const TFeaturesLayout& layout) const {
    TStringBuilder result;
    if (Type == ESplitType::OnlineCtr) {
        result << "{";
        int feature_count = 0;
        auto proj = Ctr.Base.Projection;

        for (const int featureIdx : proj.CatFeatures) {
            if (feature_count++ > 0) {
                result << ", ";
            }
            result << BuildFeatureDescription(layout, featureIdx, EFeatureType::Categorical);
        }

        for (const auto& feature : proj.BinFeatures) {
            if (feature_count++ > 0) {
                result << ", ";
            }
            result << BuildFeatureDescription(layout, feature.FloatFeature, EFeatureType::Float);
        }

        for (const TOneHotSplit& feature : proj.OneHotFeatures) {
            if (feature_count++ > 0) {
                result << ", ";
            }
            result << BuildFeatureDescription(layout, feature.CatFeatureIdx, EFeatureType::Categorical);
        }
        result << "}";
        result << " prior_num=" << Ctr.PriorNum;
        result << " prior_denom=" << Ctr.PriorDenom;
        result << " targetborder=" << Ctr.TargetBorderIdx;
        result << " type=" << Ctr.Base.CtrType;
    } else if (Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, FeatureIdx, EFeatureType::Float);
    } else {
        Y_ASSERT(Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, FeatureIdx, EFeatureType::Categorical);
    }
    return result;
}

static TVector<TVector<double>> CalcFstr(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    EFstrType type,
    NPar::TLocalExecutor* localExecutor)
{
    CB_ENSURE(
        !model.ObliviousTrees.LeafWeights.empty() || (dataset != nullptr),
        "CalcFstr requires either non-empty LeafWeights in model or provided dataset");

    TVector<double> regularEffect = CalcRegularFeatureEffect(model, dataset, type, localExecutor);
    TVector<TVector<double>> result;
    for (const auto& value : regularEffect){
        TVector<double> vec = {value};
        result.push_back(vec);
    }
    return result;
}

TVector<TVector<double>> CalcInteraction(const TFullModel& model) {
    const TFeaturesLayout layout(
        model.ObliviousTrees.FloatFeatures,
        model.ObliviousTrees.CatFeatures);

    TVector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    TVector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);
    TVector<TVector<double>> result;
    for (const auto& value : interaction){
        int featureIdxFirst = layout.GetExternalFeatureIdx(value.FirstFeature.Index, value.FirstFeature.Type);
        int featureIdxSecond = layout.GetExternalFeatureIdx(
            value.SecondFeature.Index,
            value.SecondFeature.Type);
        TVector<double> vec = {
            static_cast<double>(featureIdxFirst),
            static_cast<double>(featureIdxSecond),
            value.Score
        };
        result.push_back(vec);
    }
    return result;
}


static bool AllFeatureIdsEmpty(TConstArrayRef<TFeatureMetaInfo> featuresMetaInfo) {
    return AllOf(
        featuresMetaInfo.begin(),
        featuresMetaInfo.end(),
        [](const auto& featureMetaInfo) { return featureMetaInfo.Name.empty(); }
    );
}


TVector<TVector<double>> GetFeatureImportances(
    const TString& type,
    const TFullModel& model,
    const TDataProviderPtr dataset, // can be nullptr
    int threadCount,
    int logPeriod)
{
    TSetLoggingVerbose inThisScope;

    EFstrType fstrType = FromString<EFstrType>(type);

    switch (fstrType) {
        case EFstrType::PredictionValuesChange: {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount - 1);

            return CalcFstr(model, dataset, fstrType, &localExecutor);
        }
        case EFstrType::Interaction:
            return CalcInteraction(model);
        case EFstrType::ShapValues: {
            CB_ENSURE(dataset, "dataset is not provided");

            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(threadCount - 1);

            return CalcShapValues(model, *dataset, logPeriod, &localExecutor);
        }
        default:
            Y_UNREACHABLE();
    }
}

TVector<TVector<TVector<double>>> GetFeatureImportancesMulti(
    const TString& type,
    const TFullModel& model,
    const TDataProviderPtr dataset,
    int threadCount,
    int logPeriod)
{
    TSetLoggingVerbose inThisScope;

    EFstrType fstrType = FromString<EFstrType>(type);

    CB_ENSURE(fstrType == EFstrType::ShapValues, "Only shap values can provide multi approxes.");

    CB_ENSURE(dataset, "dataset is not provided");

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    return CalcShapValuesMulti(model, *dataset, logPeriod, &localExecutor);
}

TVector<TString> GetMaybeGeneratedModelFeatureIds(const TFullModel& model, const TDataProviderPtr dataset) {
    const NCB::TFeaturesLayout modelFeaturesLayout(
        model.ObliviousTrees.FloatFeatures,
        model.ObliviousTrees.CatFeatures);
    TVector<TString> modelFeatureIds;
    if (AllFeatureIdsEmpty(modelFeaturesLayout.GetExternalFeaturesMetaInfo())) {
        if (dataset) {
            const auto& datasetFeaturesLayout = *dataset->MetaInfo.FeaturesLayout;
            const auto datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();
            if (!AllFeatureIdsEmpty(datasetFeaturesMetaInfo)) {
                CB_ENSURE(
                    datasetFeaturesMetaInfo.size() >= (size_t)modelFeaturesLayout.GetExternalFeatureCount(),
                    "dataset has less features than the model"
                );
                for (auto i : xrange(modelFeaturesLayout.GetExternalFeatureCount())) {
                    modelFeatureIds.push_back(datasetFeaturesMetaInfo[i].Name);
                }
            }
        } else {
            modelFeatureIds.resize(modelFeaturesLayout.GetExternalFeatureCount());
        }
    } else {
        modelFeatureIds = modelFeaturesLayout.GetExternalFeatureIds();
    }
    for (size_t i = 0; i < modelFeatureIds.size(); ++i) {
        if (modelFeatureIds[i].empty()) {
            modelFeatureIds[i] = ToString(i);
        }
    }
    return modelFeatureIds;
}
