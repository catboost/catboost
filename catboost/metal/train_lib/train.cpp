// Native adapter for the CUDA algorithms translated to Metal. Pool preparation,
// quantization and final model construction remain in CatBoost's shared code.
#include "progress.h"
#include "snapshot.h"
#include "categorical.h"
#include "initialization.h"
#include "multiclass.h"
#include "multioutput.h"
#include "permutations.h"
#include "ordered.h"
#include "query.h"
#include "query_cross_entropy.h"
#include "greedy_model.h"

#include <catboost/libs/train_lib/train_model.h>
#include <catboost/metal/native/metal_trainer.h>
#include <catboost/metal/native/metal_multiclass.h>

#include <catboost/libs/data/objects.h>
#include <catboost/libs/data/target.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/target/classification_target_helper.h>

#include <util/generic/algorithm.h>
#include <util/system/hp_timer.h>

#include <cmath>
#include <limits>

namespace NCB {
namespace {

    void SetMetalDefaultsAndValidate(NCatboostOptions::TCatBoostOptions* options) {
        auto& boosting = options->BoostingOptions.Get();
        auto& tree = options->ObliviousTreeOptions.Get();
        const auto objective = options->LossFunctionDescription->GetLossFunction();
        const bool multiclass = IsMultiClassOnlyMetric(objective);
        const bool multioutput = IsMetalMultiOutput(objective);
        const bool vectorBackend = multiclass || multioutput;
        const bool yeti = objective == ELossFunction::YetiRank;
        const bool yetiPair = objective == ELossFunction::YetiRankPairwise;
        const bool qce = objective == ELossFunction::QueryCrossEntropy;
        const bool querywise = objective == ELossFunction::QueryRMSE || objective == ELossFunction::QuerySoftMax || yeti || yetiPair || qce;
        const bool coupled = objective == ELossFunction::PairLogitPairwise;
        const bool pairwise = objective == ELossFunction::PairLogit || coupled;
        const bool greedy = tree.GrowPolicy != EGrowPolicy::SymmetricTree;
        boosting.BoostingType.SetDefault(EBoostingType::Plain);
        const bool ordered = boosting.BoostingType == EBoostingType::Ordered;
        boosting.DataPartitionType.SetDefault(ordered ? EDataPartitionType::FeatureParallel : EDataPartitionType::DocParallel);
        boosting.PermutationCount.SetDefault(4);
        if (ordered) {
            CB_ENSURE(!vectorBackend, "Native Metal Ordered currently supports scalar objectives only");
            CB_ENSURE(tree.LeavesEstimationMethod != ELeavesEstimation::Exact,
                      "Native Metal Ordered supports Newton or Gradient leaves; GPU Ordered does not support Exact");
            CB_ENSURE(tree.ScoreFunction == EScoreFunction::Cosine || tree.ScoreFunction == EScoreFunction::NewtonCosine,
                      "Native Metal Ordered supports Cosine and NewtonCosine scores");
        }
        if (querywise || pairwise) {
            CB_ENSURE(!ordered, "Native Metal query and pairwise objectives currently require Plain boosting");
            CB_ENSURE(tree.LeavesEstimationMethod != ELeavesEstimation::Exact,
                      "Native Metal query and pairwise objectives support Newton or Gradient leaves");
        }
        if (coupled || qce || yetiPair) {
            CB_ENSURE(tree.MaxDepth <= 8, "Metal full-matrix objectives support depth <= 8 like CUDA");
            CB_ENSURE(tree.BootstrapConfig->GetBootstrapType() != EBootstrapType::MVS,
                "Metal PairLogitPairwise does not support MVS bootstrap");
        }
        if (yetiPair) CB_ENSURE(tree.BootstrapConfig->GetBootstrapType() != EBootstrapType::Poisson,
            "Metal YetiRankPairwise does not support Poisson bootstrap");
        if (qce) {
            CB_ENSURE((tree.LeavesEstimationMethod == ELeavesEstimation::Newton || tree.LeavesEstimationMethod == ELeavesEstimation::Simple) && tree.ScoreFunction != EScoreFunction::L2,
                "Metal QueryCrossEntropy requires Newton/Simple leaves and a Newton structure score like CUDA");
            const auto sampling = tree.BootstrapConfig->GetBootstrapType();
            CB_ENSURE(sampling == EBootstrapType::No || sampling == EBootstrapType::Bernoulli,
                "Metal QueryCrossEntropy supports No or Bernoulli query bootstrap only");
            for (const auto& item : options->LossFunctionDescription->GetLossParamsMap()) {
                CB_ENSURE(item.first == "alpha" || item.first == "raw_values_scale",
                    "Metal QueryCrossEntropy supports alpha and raw_values_scale parameters");
            }
        }
        options->CatFeatureParams->MaxTensorComplexity.SetDefault(1);
        if (yeti || yetiPair) {
            NCatboostOptions::TLossDescription metric;
            metric.Load(LossDescriptionToJson("PFound"));
            options->MetricOptions->EvalMetric.SetDefault(metric);
            CB_ENSURE((yetiPair || tree.LeavesEstimationMethod == ELeavesEstimation::Newton) &&
                tree.LeavesEstimationBacktrackingType == ELeavesEstimationStepBacktracking::No,
                "Metal YetiRank requires Newton leaves; YetiRank objectives do not support backtracking");
            const auto& lossParams = options->LossFunctionDescription->GetLossParamsMap();
            for (const auto& item : lossParams) {
                CB_ENSURE(item.first == "permutations" || item.first == "decay" || item.first == "mode",
                    "Metal YetiRank supports classic permutations and decay parameters only");
                if (item.first == "mode") CB_ENSURE(item.second == "Classic", "Metal YetiRank requires mode=Classic");
            }
        } else options->MetricOptions->EvalMetric.SetDefault(options->LossFunctionDescription.Get());

        CB_ENSURE(objective == ELossFunction::RMSE || objective == ELossFunction::Logloss ||
                  objective == ELossFunction::CrossEntropy || objective == ELossFunction::Poisson ||
                  objective == ELossFunction::Huber || objective == ELossFunction::Expectile ||
                  objective == ELossFunction::Lq || objective == ELossFunction::Tweedie ||
                  objective == ELossFunction::LogLinQuantile || objective == ELossFunction::Quantile ||
                  objective == ELossFunction::MAE || objective == ELossFunction::MAPE || vectorBackend || querywise || pairwise,
                  "Metal supports RMSE, Logloss, CrossEntropy, Poisson, Huber, Expectile, Lq, "
                  "Tweedie, LogLinQuantile, Quantile, MAE, MAPE, MultiClass, MultiClassOneVsAll, "
                  "QueryRMSE, QuerySoftMax, PairLogit, PairLogitPairwise, QueryCrossEntropy, YetiRank, MultiRMSE, RMSEWithUncertainty, MultiLogloss, and MultiCrossEntropy objectives");
        if (vectorBackend) {
            CB_ENSURE(tree.BootstrapConfig->GetBootstrapType() != EBootstrapType::MVS,
                      "Metal multidimensional training does not support MVS bootstrap");
        }
        CB_ENSURE(ordered ? boosting.DataPartitionType == EDataPartitionType::FeatureParallel :
                  boosting.BoostingType == EBoostingType::Plain && boosting.DataPartitionType == EDataPartitionType::DocParallel,
                  "Metal supports Plain/DocParallel or Ordered/FeatureParallel training");
        if (greedy) {
            CB_ENSURE(!ordered, "Metal non-symmetric training requires Plain boosting");
            CB_ENSURE((!querywise || objective == ELossFunction::QueryRMSE || objective == ELossFunction::QuerySoftMax) &&
                (!pairwise || objective == ELossFunction::PairLogit) && objective != ELossFunction::Lq &&
                (!multioutput || objective == ELossFunction::RMSEWithUncertainty),
                "Metal greedy training supports the eleven CUDA-registered scalar objectives, MultiClass, MultiClassOneVsAll, RMSEWithUncertainty, QueryRMSE, QuerySoftMax and PairLogit; Lq and remaining ranking are unsupported");
            CB_ENSURE(tree.GrowPolicy == EGrowPolicy::Depthwise || tree.GrowPolicy == EGrowPolicy::Lossguide ||
                tree.GrowPolicy == EGrowPolicy::Region, "Unsupported Metal grow policy");
            CB_ENSURE(tree.BootstrapConfig->GetBootstrapType() != EBootstrapType::MVS,
                "Metal greedy training does not support MVS bootstrap");
            CB_ENSURE(tree.MaxLeaves > 0 && tree.MaxLeaves <= 65536, "Metal greedy max_leaves must be in [1,65536]");
            if (tree.GrowPolicy == EGrowPolicy::Depthwise) CB_ENSURE(tree.MaxDepth <= 16, "Metal Depthwise depth is limited to 16");
            if (tree.GrowPolicy == EGrowPolicy::Region) CB_ENSURE(tree.MaxDepth <= 65535, "Metal Region depth is limited to 65535");
        } else {
            CB_ENSURE(tree.MaxDepth <= 16, "Metal supports symmetric tree depth up to 16");
        }
        CB_ENSURE(tree.BootstrapConfig->GetSamplingUnit() == ESamplingUnit::Object || yetiPair,
                  "Metal currently supports object bootstrap sampling only");
        CB_ENSURE(tree.LeavesEstimationMethod == ELeavesEstimation::Newton ||
                  tree.LeavesEstimationMethod == ELeavesEstimation::Gradient ||
                  tree.LeavesEstimationMethod == ELeavesEstimation::Exact ||
                  ((coupled || qce || yetiPair) && tree.LeavesEstimationMethod == ELeavesEstimation::Simple),
                  "Metal supports Newton/Gradient/Exact and full-matrix Simple leaf estimation");
        if (tree.LeavesEstimationMethod == ELeavesEstimation::Simple)
            CB_ENSURE(tree.MaxDepth > 0 && tree.LeavesEstimationIterations == 1,
                "Metal Simple leaves require depth 1..8 and one estimation iteration");
        CB_ENSURE(tree.LeavesEstimationBacktrackingType == ELeavesEstimationStepBacktracking::No ||
                  tree.LeavesEstimationBacktrackingType == ELeavesEstimationStepBacktracking::AnyImprovement ||
                  tree.LeavesEstimationBacktrackingType == ELeavesEstimationStepBacktracking::Armijo,
                  "Metal supports No, AnyImprovement, and Armijo leaf estimation backtracking");
        CB_ENSURE(tree.ScoreFunction == EScoreFunction::L2 || tree.ScoreFunction == EScoreFunction::Cosine ||
                  tree.ScoreFunction == EScoreFunction::SolarL2 || tree.ScoreFunction == EScoreFunction::LOOL2 ||
                  tree.ScoreFunction == EScoreFunction::SatL2 ||
                  (!vectorBackend && (tree.ScoreFunction == EScoreFunction::NewtonL2 ||
                   tree.ScoreFunction == EScoreFunction::NewtonCosine)),
                  "Metal supports L2, Cosine, SolarL2, LOOL2, SatL2; NewtonL2/NewtonCosine for scalar models");
        CB_ENSURE((ordered || !tree.FoldSizeLossNormalization) && !tree.AddRidgeToTargetFunctionFlag &&
                  tree.MetaL2Exponent == 1 && tree.MetaL2Frequency == 0,
                  "Metal does not yet support normalized, ridge-objective, or meta-L2 scores");
        CB_ENSURE(tree.FeaturePenalties->FeatureWeights.Get().empty(),
                  "Metal does not yet support feature penalties");
        CB_ENSURE(tree.FixedBinarySplits.Get().empty(), "Metal does not yet support fixed_binary_splits");
        CB_ENSURE(tree.Rsm == 1.0f, "Metal does not yet support feature subsampling");
        CB_ENSURE(!boosting.Langevin && !boosting.PosteriorSampling.GetUnchecked() &&
                  boosting.ModelShrinkRate.GetUnchecked() == 0,
                  "Metal does not yet support Langevin, posterior sampling, or model shrinkage");
    }

    ui32 MetalObjective(ELossFunction objective) {
        switch (objective) {
            case ELossFunction::RMSE: return 0;
            case ELossFunction::Logloss: return 1;
            case ELossFunction::CrossEntropy: return 2;
            case ELossFunction::Poisson: return 3;
            case ELossFunction::Huber: return 4;
            case ELossFunction::Expectile: return 5;
            case ELossFunction::Lq: return 6;
            case ELossFunction::Tweedie: return 7;
            case ELossFunction::LogLinQuantile: return 8;
            case ELossFunction::Quantile: return 9;
            case ELossFunction::MAE: return 10;
            case ELossFunction::MAPE: return 11;
            case ELossFunction::QueryRMSE: return 12;
            case ELossFunction::QuerySoftMax: return 13;
            case ELossFunction::PairLogit: return 14;
            case ELossFunction::PairLogitPairwise: return 15;
            case ELossFunction::QueryCrossEntropy: return 16;
            case ELossFunction::YetiRank: return 17;
            case ELossFunction::YetiRankPairwise: return 18;
            default: CB_ENSURE(false, "Unsupported Metal objective");
        }
    }

    ui32 MetalBootstrap(EBootstrapType bootstrap) {
        switch (bootstrap) {
            case EBootstrapType::No: return 0;
            case EBootstrapType::Bayesian: return 1;
            case EBootstrapType::Bernoulli: return 2;
            case EBootstrapType::Poisson: return 3;
            case EBootstrapType::MVS: return 4;
            default: CB_ENSURE(false, "Unsupported Metal bootstrap");
        }
    }

    ui32 MetalScore(EScoreFunction score) {
        switch (score) {
            case EScoreFunction::L2: return 0;
            case EScoreFunction::Cosine: return 1;
            case EScoreFunction::NewtonL2: return 2;
            case EScoreFunction::NewtonCosine: return 3;
            case EScoreFunction::SolarL2: return 4;
            case EScoreFunction::LOOL2: return 5;
            case EScoreFunction::SatL2: return 6;
            default: CB_ENSURE(false, "Unsupported Metal score function");
        }
    }

    CBMObjectiveOptions MetalObjectiveOptions(const NCatboostOptions::TCatBoostOptions& options, ui32 objectiveId) {
        CBMObjectiveOptions result = {};
        result.objective = objectiveId;
        const auto method = options.ObliviousTreeOptions->LeavesEstimationMethod.Get();
        result.leaf_estimation_method = method == ELeavesEstimation::Newton ? 0u : method == ELeavesEstimation::Gradient ? 1u : method == ELeavesEstimation::Exact ? 2u : 3u;
        const auto objective = options.LossFunctionDescription->GetLossFunction();
        if (objective == ELossFunction::Huber) result.objective_param = NCatboostOptions::GetHuberParam(options.LossFunctionDescription);
        else if (objective == ELossFunction::Expectile || objective == ELossFunction::LogLinQuantile || objective == ELossFunction::Quantile)
            result.objective_param = NCatboostOptions::GetAlpha(options.LossFunctionDescription);
        else if (objective == ELossFunction::Lq) result.objective_param = NCatboostOptions::GetLqParam(options.LossFunctionDescription);
        else if (objective == ELossFunction::Tweedie) result.objective_param = NCatboostOptions::GetTweedieParam(options.LossFunctionDescription);
        else if (objective == ELossFunction::QueryCrossEntropy) result.objective_param = NCatboostOptions::GetAlphaQueryCrossEntropy(options.LossFunctionDescription);
        return result;
    }

    struct TMetalData {
        ui32 Rows = 0;
        ui32 BinsPerFeature = 1;
        TVector<TFloatFeature> AllFloatFeatures;
        TVector<ui32> FeatureIndices;
        TVector<ui8> Bins;
        TVector<TVector<ui8>> AdditionalPermutationBins;
        TVector<ui32> CandidateFeatures;
        TVector<ui32> CandidateBins;
        TVector<ui8> CandidateTypes;
        TMetalCategoricalData Categorical;

        ui32 FeatureCount() const {
            return FeatureIndices.size() + Categorical.SplitCandidates.size();
        }
    };

    TMetalData PrepareData(const TTrainingDataProvider& data,
                          const NCatboostOptions::TCatBoostOptions& options,
                          NPar::ILocalExecutor* executor) {
        const auto& objects = *data.ObjectsData;
        const auto& layout = *objects.GetFeaturesLayout();
        CB_ENSURE(layout.GetTextFeatureCount() == 0 &&
                  layout.GetEmbeddingFeatureCount() == 0,
                  "Metal text and embedding feature training is not yet implemented");
        TMetalData result;
        result.Rows = data.GetObjectCount();
        CB_ENSURE(result.Rows > 0 && result.Rows <= (1u << 24),
                  "Metal supports between 1 and 16777216 training rows");
        result.AllFloatFeatures = CreateFloatFeatures(layout, *objects.GetQuantizedFeaturesInfo());
        for (ui32 i = 0; i < result.AllFloatFeatures.size(); ++i) {
            const auto& feature = result.AllFloatFeatures[i];
            if (!layout.GetExternalFeatureMetaInfo(feature.Position.FlatIndex).IsAvailable) {
                continue;
            }
            CB_ENSURE(feature.Borders.size() <= 255,
                      "Metal requires at most 255 borders per numeric feature");
            const auto holder = objects.GetFloatFeature(feature.Position.Index);
            CB_ENSURE(holder, "An available numeric feature has no quantized values");
            const auto values = (*holder)->ExtractValues<ui16>(executor);
            CB_ENSURE(values.size() == result.Rows, "Quantized feature row count mismatch");
            const ui32 denseFeature = result.FeatureIndices.size();
            result.FeatureIndices.push_back(i);
            result.BinsPerFeature = Max<ui32>(result.BinsPerFeature, feature.Borders.size() + 1);
            CB_ENSURE(ui64(result.Bins.size()) + values.size() <= (1ull << 30),
                      "Metal quantized input exceeds the experimental 1 GiB limit");
            for (ui16 value : values) {
                CB_ENSURE(value <= feature.Borders.size(), "Quantized value is outside feature borders");
                result.Bins.push_back(static_cast<ui8>(value));
            }
            for (ui32 border = 0; border < feature.Borders.size(); ++border) {
                result.CandidateFeatures.push_back(denseFeature);
                result.CandidateBins.push_back(border);
                result.CandidateTypes.push_back(0);
            }
        }
        result.Categorical = PrepareMetalCategoricalPermutations(data, options, executor);
        const auto& categorical = result.Categorical;
        for (const auto& permutation : categorical.AdditionalPermutationBins) {
            auto& bins = result.AdditionalPermutationBins.emplace_back(result.Bins);
            bins.insert(bins.end(), permutation.begin(), permutation.end());
        }
        CB_ENSURE(ui64(result.Bins.size()) + categorical.Bins.size() <= (1ull << 30),
                  "Metal quantized input exceeds the experimental 1 GiB limit");
        result.Bins.insert(result.Bins.end(), categorical.Bins.begin(), categorical.Bins.end());
        result.BinsPerFeature = Max(result.BinsPerFeature, categorical.BinsPerFeature);
        for (ui32 i = 0; i < categorical.CandidateFeatures.size(); ++i) {
            result.CandidateFeatures.push_back(result.FeatureIndices.size() + categorical.CandidateFeatures[i]);
            result.CandidateBins.push_back(categorical.CandidateBins[i]);
            result.CandidateTypes.push_back(categorical.CandidateTypes[i]);
        }
        CB_ENSURE(result.FeatureCount() > 0, "Metal requires at least one available feature");
        return result;
    }

    TFullModel AppendMetalTree(
        const TMetalData& data, ui32 treeDepth, ui32 maxDepth,
        TConstArrayRef<ui32> splitFeatures, TConstArrayRef<ui32> splitBins,
        TConstArrayRef<ui8> splitTypes, TConstArrayRef<float> leaves,
        TConstArrayRef<float> weights, TObliviousTreeBuilder* builder, ui32 approxDimension = 1) {
        CB_ENSURE(treeDepth <= maxDepth, "Metal returned an invalid tree depth");
        TVector<TModelSplit> splits;
        for (ui32 level = 0; level < treeDepth; ++level) {
            CB_ENSURE(splitFeatures[level] < data.FeatureCount(), "Metal returned an invalid feature index");
            if (splitFeatures[level] < data.FeatureIndices.size()) {
                CB_ENSURE(splitTypes[level] == 0, "Metal returned a nonnumeric comparison for a numeric feature");
                const auto& feature = data.AllFloatFeatures[data.FeatureIndices[splitFeatures[level]]];
                CB_ENSURE(splitBins[level] < feature.Borders.size(), "Metal returned an invalid split border");
                splits.emplace_back(TFloatSplit{feature.Position.Index, feature.Borders[splitBins[level]]});
            } else {
                splits.push_back(data.Categorical.GetSplit(splitFeatures[level] - data.FeatureIndices.size(),
                    splitBins[level], splitTypes[level]));
            }
        }
        const ui32 count = 1u << treeDepth;
        TVector<double> leafValues(ui64(count) * approxDimension), leafWeights(count);
        for (ui32 leaf = 0; leaf < count; ++leaf) {
            CB_ENSURE(std::isfinite(weights[leaf]) && weights[leaf] >= 0,
                      "Metal returned invalid leaf weights");
            leafWeights[leaf] = weights[leaf];
        }
        for (ui64 value = 0; value < leafValues.size(); ++value) {
            CB_ENSURE(std::isfinite(leaves[value]), "Metal returned invalid leaf values");
            leafValues[value] = leaves[value];
        }
        builder->AddTree(splits, leafValues, leafWeights);
        TObliviousTreeBuilder singleTreeBuilder(data.AllFloatFeatures, data.Categorical.AllCatFeatures, {}, {}, approxDimension);
        singleTreeBuilder.AddTree(splits, leafValues, leafWeights);
        TFullModel model;
        singleTreeBuilder.Build(model.ModelTrees.GetMutable());
        if (data.Categorical.CtrProvider) model.CtrProvider = data.Categorical.CtrProvider->Clone();
        model.UpdateDynamicData();
        return model;
    }

    TFullModel AppendMetalGreedyTree(const TMetalData& data, const TMetalGreedyTree& tree,
        ui32 depthBound, TNonSymmetricTreeModelBuilder* builder, ui32 approxDimension = 1) {
        auto splitLookup = [&](ui32 denseFeature, ui32 bin, ui32 type) {
            CB_ENSURE(denseFeature < data.FeatureCount(), "Metal greedy returned an invalid feature");
            if (denseFeature < data.FeatureIndices.size()) {
                const auto& feature = data.AllFloatFeatures[data.FeatureIndices[denseFeature]];
                CB_ENSURE(type == 0 && bin < feature.Borders.size(), "Metal greedy returned an invalid numeric split");
                return TModelSplit(TFloatSplit{feature.Position.Index, feature.Borders[bin]});
            }
            return data.Categorical.GetSplit(denseFeature - data.FeatureIndices.size(), bin, type);
        };
        builder->AddTree(MakeMetalGreedyTreeRoot(tree, data.FeatureCount(), splitLookup, depthBound, approxDimension));
        TNonSymmetricTreeModelBuilder single(data.AllFloatFeatures, data.Categorical.AllCatFeatures, {}, {}, approxDimension);
        single.AddTree(MakeMetalGreedyTreeRoot(tree, data.FeatureCount(), splitLookup, depthBound, approxDimension));
        TFullModel model;
        single.Build(model.ModelTrees.GetMutable());
        if (data.Categorical.CtrProvider) model.CtrProvider = data.Categorical.CtrProvider->Clone();
        model.UpdateDynamicData();
        return model;
    }

    ui32 SnapshotDataChecksum(const TTrainingDataProviders& data,
                             TMaybe<TFullModel*> initModel, NPar::ILocalExecutor* executor) {
        // Shared categorical perfect-hash dictionaries can grow when an eval
        // Pool introduces unseen values. Fingerprint the actual row contents,
        // never dictionary entries that are not present in this dataset.
        ui32 checksum = 0;
        auto addTarget = [&](const TTrainingDataProvider& dataset) {
            checksum = UpdateCheckSum(checksum, dataset.GetObjectCount());
            const auto& objects = *dataset.ObjectsData;
            const auto& layout = *objects.GetFeaturesLayout();
            checksum = UpdateCheckSum(checksum, layout.GetExternalFeatureCount());
            const auto floatFeatures = CreateFloatFeatures(layout, *objects.GetQuantizedFeaturesInfo());
            for (const auto& feature : floatFeatures) {
                if (!layout.GetExternalFeatureMetaInfo(feature.Position.FlatIndex).IsAvailable) continue;
                checksum = UpdateCheckSum(checksum, feature.Position.Index, feature.Position.FlatIndex,
                    TStringBuf(feature.FeatureId), feature.Borders, feature.NanValueTreatment);
                const auto holder = objects.GetFloatFeature(feature.Position.Index);
                CB_ENSURE(holder, "Snapshot numeric feature values are missing");
                const auto values = (*holder)->ExtractValues<ui16>(executor);
                checksum = UpdateCheckSum(checksum, values);
            }
            checksum = UpdateCheckSum(checksum, CalcMetalCategoricalChecksum(objects, executor));
            checksum = UpdateCheckSum(checksum, dataset.ObjectsGrouping->IsTrivial());
            if (!dataset.ObjectsGrouping->IsTrivial()) {
                const auto groups = dataset.ObjectsGrouping->GetNonTrivialGroups();
                checksum = UpdateCheckSum(checksum, static_cast<ui32>(groups.size()));
                for (const auto& group : groups) {
                    checksum = UpdateCheckSum(checksum, group.Begin, group.End);
                }
            }
            const auto targets = dataset.TargetData->GetTarget();
            checksum = UpdateCheckSum(checksum, static_cast<ui32>(targets ? targets->size() : 0));
            if (targets) {
                for (const auto& target : *targets) checksum = UpdateCheckSum(checksum, target);
            }
            checksum = UpdateCheckSum(checksum, GetWeights(*dataset.TargetData));
            const auto groupInfo = dataset.TargetData->GetGroupInfo();
            checksum = UpdateCheckSum(checksum, static_cast<ui32>(groupInfo ? groupInfo->size() : 0));
            if (groupInfo) {
                for (const auto& group : *groupInfo) {
                    checksum = UpdateCheckSum(checksum, group.Begin, group.End, group.Weight, group.SubgroupId);
                    checksum = UpdateCheckSum(checksum, static_cast<ui32>(group.Competitors.size()));
                    for (const auto& competitors : group.Competitors) {
                        checksum = UpdateCheckSum(checksum, static_cast<ui32>(competitors.size()));
                        for (const auto& competitor : competitors) {
                            checksum = UpdateCheckSum(checksum, competitor.Id, competitor.Weight, competitor.SampleWeight);
                        }
                    }
                }
            }
            const auto baseline = dataset.TargetData->GetBaseline();
            checksum = UpdateCheckSum(checksum, static_cast<ui32>(baseline ? baseline->size() : 0));
            if (baseline) {
                for (const auto& values : *baseline) checksum = UpdateCheckSum(checksum, values);
            }
        };
        addTarget(*data.Learn);
        for (const auto& test : data.Test) addTarget(*test);
        if (initModel) checksum = UpdateCheckSum(checksum, TStringBuf(SerializeModel(**initModel)));
        return checksum;
    }

    class TMetalModelTrainer final : public IModelTrainer {
    public:
        void TrainModel(
            const TTrainModelInternalOptions& internalOptions,
            const NCatboostOptions::TCatBoostOptions& catboostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            TTrainingDataProviders trainingData,
            TMaybe<TPrecomputedOnlineCtrData> precomputedCtrs,
            const TLabelConverter& labelConverter,
            ITrainingCallbacks* trainingCallbacks,
            ICustomCallbacks* customCallbacks,
            TMaybe<TFullModel*> initModel,
            THolder<TLearnProgress> initLearnProgress,
            TDataProviders initModelApplyCompatiblePools,
            NPar::ILocalExecutor* executor,
            const TMaybe<TRestorableFastRng64*> rand,
            TFullModel* dstModel,
            const TVector<TEvalResult*>& evalResultPtrs,
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
            THolder<TLearnProgress>* dstLearnProgress) const override {
            Y_UNUSED(initLearnProgress);
            Y_UNUSED(rand);
            CB_ENSURE(!internalOptions.CalcMetricsOnly, "Metal cross-validation is not yet supported");
            CB_ENSURE(!objectiveDescriptor && !precomputedCtrs,
                      "Metal does not yet support custom objectives or precomputed CTRs");
            CB_ENSURE(!dstLearnProgress, "Metal does not expose CPU learn-progress state");
            NCatboostOptions::TCatBoostOptions options(catboostOptions);
            const auto baseline = trainingData.Learn->TargetData->GetBaseline();
            if (initModel || baseline) {
                options.BoostingOptions->BoostFromAverage.SetDefault(false);
                CB_ENSURE(!options.BoostingOptions->BoostFromAverage,
                          "boost_from_average cannot be combined with an initial model or baseline");
            }
            SetMetalDefaultsAndValidate(&options);
            const bool ordered = options.BoostingOptions->BoostingType == EBoostingType::Ordered;
            const auto growPolicy = options.ObliviousTreeOptions->GrowPolicy.Get();
            const bool greedy = growPolicy != EGrowPolicy::SymmetricTree;
            const ui32 greedyPolicy = growPolicy == EGrowPolicy::Depthwise ? 0u : growPolicy == EGrowPolicy::Lossguide ? 1u : 2u;
            if (ordered) {
                CB_ENSURE(trainingData.Learn->ObjectsData->GetObjectsGrouping()->GetGroupCount() >= 4,
                          "Metal Ordered requires at least four groups or documents");
            }
            const bool yeti = options.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRank;
            const bool yetiPair = options.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRankPairwise;
            if (yeti) {
                CB_ENSURE(!trainingData.Learn->ObjectsGrouping->IsTrivial(),
                    "Metal YetiRank requires at least one query containing multiple rows");
            }
            const bool coupled = options.LossFunctionDescription->GetLossFunction() == ELossFunction::PairLogitPairwise;
            const bool qce = options.LossFunctionDescription->GetLossFunction() == ELossFunction::QueryCrossEntropy;
            TMetalData data = PrepareData(*trainingData.Learn, options, executor);
            const auto orderedHistories = ordered ? MakeMetalFeatureParallelHistoryOrders(*trainingData.Learn, options) :
                TVector<TVector<ui32>>();
            const ui32 permutationCount = ordered ? orderedHistories.size() : data.Categorical.GetPermutationCount();
            const auto objective = options.LossFunctionDescription->GetLossFunction();
            const bool pairwise = objective == ELossFunction::PairLogit || coupled;
            const bool multioutput = IsMetalMultiOutput(objective);
            const auto target = trainingData.Learn->TargetData->GetTarget();
            CB_ENSURE((target && (target->size() == 1 || (multioutput && !target->empty()))) || (pairwise && !target),
                      "Metal requires scalar targets or matching multioutput target columns");
            const TConstArrayRef<float> targets = target ? (*target)[0] : TConstArrayRef<float>();
            const auto sampleWeights = GetWeights(*trainingData.Learn->TargetData);
            const bool multiclass = IsMultiClassOnlyMetric(objective);
            const bool vectorBackend = multiclass || multioutput;
            const ui32 approxDimension = GetApproxDimension(options, labelConverter, target ? target->size() : 1);
            const ui32 optimizerDimension = vectorBackend ?
                (objective == ELossFunction::MultiClass ? approxDimension - 1 : approxDimension) : 0;
            CB_ENSURE(vectorBackend ? (approxDimension >= 2 && approxDimension <= 64) : approxDimension == 1,
                      "Metal supports scalar training and multidimensional models with 2 to 64 outputs");
            CB_ENSURE(ui64(data.Rows) * approxDimension * sizeof(float) <= (1ull << 30),
                      "Metal prediction cursor exceeds the experimental 1 GiB limit");
            TClassificationTargetHelper classificationTargetHelper(labelConverter, options.DataProcessingOptions.Get());
            const auto baselineColumns = GetMetalBaselineColumns(classificationTargetHelper, approxDimension,
                                                                  initModel ? *initModel : nullptr);
            const ui32 iterations = options.BoostingOptions->IterationCount;
            const ui32 depth = options.ObliviousTreeOptions->MaxDepth;
            const ui32 maxLeaves = greedy ? MetalGreedyLeafCapacity(greedyPolicy, depth,
                options.ObliviousTreeOptions->MaxLeaves) : 1u << depth;
            const ui32 greedyDepthBound = greedy ? Min(depth, maxLeaves - 1) : 0;
            // Normal training retains actual tree sizes. The checkpoint format
            // still stores padded per-tree buffers, so bound that allocation only
            // when snapshots are requested.
            if (outputOptions.SaveSnapshot()) {
                const ui64 leavesCount = ui64(iterations) * maxLeaves;
                CB_ENSURE((greedy ? leavesCount * (52 + 4 * approxDimension) + ui64(iterations) * 24 :
                          leavesCount * 4 * (approxDimension + 1) + ui64(iterations) * depth * 9) +
                          ui64(data.Rows) * 4 * (2 * approxDimension + optimizerDimension) *
                          permutationCount <= (1ull << 29),
                          "Metal snapshot output exceeds the experimental 512 MiB limit");
            }
            float bias = 0;
            if (options.BoostingOptions->BoostFromAverage && !multioutput) {
                bias = CalcMetalInitialBias(options.LossFunctionDescription.Get(), targets, sampleWeights);
            }
            TVector<double> modelBias = multioutput ? MetalMultiOutputBias(*target, sampleWeights,
                approxDimension, objective, options.BoostingOptions->BoostFromAverage) : TVector<double>(approxDimension, bias);
            TVector<float> initialPredictions;
            if (initModel || baseline || multioutput) {
                initialPredictions.assign(ui64(data.Rows) * approxDimension, bias);
                if (multioutput) {
                    for (ui32 row = 0; row < data.Rows; ++row) {
                        for (ui32 dimension = 0; dimension < approxDimension; ++dimension) {
                            initialPredictions[ui64(row) * approxDimension + dimension] = modelBias[dimension];
                        }
                    }
                }
                if (initModel) {
                    CB_ENSURE((*initModel)->GetDimensionsCount() == approxDimension,
                              "Metal initial model approximation dimension differs from training");
                    CB_ENSURE((*initModel)->ModelTrees->GetTextFeatures().empty() &&
                              (*initModel)->ModelTrees->GetEmbeddingFeatures().empty(),
                              "Metal does not yet support initial models containing text or embedding features");
                    CB_ENSURE(initModelApplyCompatiblePools.Learn,
                              "Initial-model-compatible training data is required");
                    const auto initialApprox = ApplyModelMulti(**initModel,
                        *initModelApplyCompatiblePools.Learn->ObjectsData,
                        EPredictionType::InternalRawFormulaVal, 0, (*initModel)->GetTreeCount(), executor);
                    for (ui32 dimension = 0; dimension < approxDimension; ++dimension) {
                        for (ui32 row = 0; row < data.Rows; ++row) {
                            initialPredictions[ui64(row) * approxDimension + dimension] +=
                                static_cast<float>(initialApprox[dimension][row]);
                        }
                    }
                }
                if (baseline) {
                    for (ui32 dimension = 0; dimension < approxDimension; ++dimension) {
                        const ui32 column = baseline->size() == approxDimension ? dimension : baselineColumns[dimension];
                        CB_ENSURE(column < baseline->size() && (*baseline)[column].size() == data.Rows,
                                  "Metal baseline must provide one value per object and public class");
                        for (ui32 row = 0; row < data.Rows; ++row) {
                            initialPredictions[ui64(row) * approxDimension + dimension] += (*baseline)[column][row];
                        }
                    }
                }
            }
            TMetalSnapshot snapshot;
            snapshot.Greedy = greedy;
            snapshot.YetiRank = yeti;
            TString snapshotPath;
            ui32 restoredIterations = 0;
            if (outputOptions.SaveSnapshot()) {
                NJson::TJsonValue jsonOptions;
                options.Save(&jsonOptions);
                snapshot.Params = ToString(jsonOptions);
                snapshot.Checksum = SnapshotDataChecksum(trainingData, initModel, executor);
                if (ordered) {
                    snapshot.Checksum = UpdateCheckSum(snapshot.Checksum, permutationCount);
                    for (const auto& history : orderedHistories) snapshot.Checksum = UpdateCheckSum(snapshot.Checksum, history);
                }
                snapshot.Bias = bias;
                snapshotPath = outputOptions.CreateSnapshotFullPath();
                if (snapshot.Load(snapshotPath, trainingCallbacks)) {
                    if (greedy) snapshot.ValidateGreedy(data.Rows, data.FeatureCount(), greedyPolicy, depth, maxLeaves, iterations, permutationCount, approxDimension, optimizerDimension);
                    else snapshot.Validate(data.Rows, depth, iterations, approxDimension,
                                           permutationCount, optimizerDimension, ordered);
                    restoredIterations = snapshot.Depths.size();
                    initialPredictions = snapshot.Predictions;
                    bias = snapshot.Bias;
                    if (!multioutput) std::fill(modelBias.begin(), modelBias.end(), bias);
                }
            }
            CBMSessionParams params = {};
            params.train = {data.Rows, data.FeatureCount(),
                static_cast<ui32>(data.CandidateFeatures.size()), data.BinsPerFeature,
                iterations - restoredIterations, depth,
                MetalScore(options.ObliviousTreeOptions->ScoreFunction),
                static_cast<float>(options.BoostingOptions->LearningRate),
                static_cast<float>(options.ObliviousTreeOptions->L2Reg), bias};
            const ui32 objectiveId = vectorBackend ? 0 : MetalObjective(objective);
            params.objective = objectiveId;
            params.leaf_estimation_iterations = options.ObliviousTreeOptions->LeavesEstimationIterations;
            const auto backtracking = options.ObliviousTreeOptions->LeavesEstimationBacktrackingType.Get();
            params.leaf_estimation_backtracking = backtracking == ELeavesEstimationStepBacktracking::No ? 0u :
                backtracking == ELeavesEstimationStepBacktracking::AnyImprovement ? 1u : 2u;
            TVector<ui32> splitFeatures(greedy ? 0 : depth), splitBins(greedy ? 0 : depth);
            TVector<ui8> splitTypes(greedy ? 0 : depth);
            TVector<float> leaves(ui64(maxLeaves) * approxDimension), weights(maxLeaves),
                learnCursor(ui64(data.Rows) * approxDimension);
            struct TSession {
                void* Handle = nullptr;
                bool Multiclass = false;
                ~TSession() {
                    if (Multiclass) cbm_multiclass_session_close(Handle);
                    else cbm_session_close(Handle);
                }
            } session;
            session.Multiclass = vectorBackend;
            THolder<TMetalDocParallelPermutations> permutations;
            THolder<TMetalOrderedSession> orderedSession;
            THolder<TMetalGreedySession> greedySession;
            TVector<ui32> ctrUniqueValues(data.FeatureCount(), 0);
            TVector<ui8> usedFeatures(data.FeatureCount(), 0);
            {
                const ui64 maxUnique = ui64(data.Rows) +
                    (trainingData.Test.empty() ? 0 : trainingData.Test.front()->GetObjectCount());
                CB_ENSURE(data.Categorical.CtrUniqueValues.size() == data.Categorical.SplitCandidates.size(),
                          "Metal CTR unique counts differ from the runtime feature count");
                for (ui32 feature = 0; feature < data.Categorical.CtrUniqueValues.size(); ++feature) {
                    const ui32 denseFeature = data.FeatureIndices.size() + feature;
                    ctrUniqueValues[denseFeature] = Min<ui64>(maxUnique, data.Categorical.CtrUniqueValues[feature]);
                    if (initModel && ctrUniqueValues[denseFeature] && !data.Categorical.SplitCandidates[feature].empty()) {
                        const auto& ctr = data.Categorical.SplitCandidates[feature].front().OnlineCtr.Ctr;
                        for (const auto& previous : (*initModel)->ModelTrees->GetCtrFeatures()) {
                            if (previous.Ctr == ctr) usedFeatures[denseFeature] = 1;
                        }
                    }
                }
                if (restoredIterations && !ordered && !greedy) {
                    CB_ENSURE(snapshot.UsedFeatures.size() == data.FeatureCount(),
                              "Metal snapshot feature-use state differs from the prepared features");
                    usedFeatures = snapshot.UsedFeatures;
                }
            }
            THolder<TMetalYetiRandom> yetiRandom;
            if (yeti) {
                yetiRandom = MakeHolder<TMetalYetiRandom>(options.RandomSeed,
                    options.ObliviousTreeOptions->BootstrapConfig->GetBootstrapType() != EBootstrapType::No,
                    params.leaf_estimation_iterations, depth, data.CandidateFeatures.size(), permutationCount);
                if (restoredIterations) yetiRandom->Restore(snapshot.YetiRandom, snapshot.Depths);
            }
            char error[2048] = {};
            if (restoredIterations < iterations) {
                if (greedy && !vectorBackend) {
                    const auto objectiveOptions = MetalObjectiveOptions(options, objectiveId);
                    CBMGreedyTrainParams greedyParams = {
                        data.Rows, data.FeatureCount(), static_cast<ui32>(data.CandidateFeatures.size()), data.BinsPerFeature,
                        iterations - restoredIterations, depth, maxLeaves, options.ObliviousTreeOptions->MinDataInLeaf,
                        greedyPolicy, objectiveId, params.train.score_function, objectiveOptions.leaf_estimation_method,
                        params.leaf_estimation_iterations, 0, 0, 0, params.train.learning_rate, params.train.l2_leaf_reg, bias, 0};
                    TMetalPairData pairs;
                    if (objectiveId == 14) pairs = PrepareMetalPairData(*trainingData.Learn->TargetData, data.Rows);
                    TMetalQueryData query;
                    if (objectiveId == 12 || objectiveId == 13)
                        query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping, options.LossFunctionDescription.Get());
                    greedySession = MakeHolder<TMetalGreedySession>(greedyParams, objectiveOptions,
                        data.Bins, targets, sampleWeights, initialPredictions,
                        data.CandidateFeatures, data.CandidateBins, data.CandidateTypes,
                        query.Offsets.empty() ? nullptr : &query.Options,
                        objectiveId == 14 ? pairs.GroupOffsets : query.Offsets,
                        objectiveId == 14 ? &pairs.Options : nullptr, pairs.Winners, pairs.Losers, pairs.Weights);
                } else if (ordered) {
                    const auto objectiveOptions = MetalObjectiveOptions(options, objectiveId);
                    const auto& boosting = options.BoostingOptions.Get();
                    CBMOrderedParams orderedParams = {
                        data.Rows, data.FeatureCount(), static_cast<ui32>(data.CandidateFeatures.size()),
                        iterations - restoredIterations, depth, objectiveId,
                        options.ObliviousTreeOptions->ScoreFunction == EScoreFunction::NewtonCosine ? 1u : 0u,
                        objectiveOptions.leaf_estimation_method, params.leaf_estimation_iterations,
                        permutationCount, boosting.MinFoldSize,
                        options.ObliviousTreeOptions->FoldSizeLossNormalization ? 1u : 0u,
                        params.train.learning_rate, params.train.l2_leaf_reg, bias,
                        static_cast<float>(boosting.FoldLenMultiplier), objectiveOptions.objective_param, 0, 0, 0};
                    orderedSession = MakeHolder<TMetalOrderedSession>(orderedParams, data.Bins, targets,
                        sampleWeights, initialPredictions, data.CandidateFeatures, data.CandidateBins,
                        options.RandomSeed, *trainingData.Learn->ObjectsData->GetObjectsGrouping(), orderedHistories, data.CandidateTypes, boosting.FoldLenMultiplier.Get(), data.AdditionalPermutationBins);
                    if (outputOptions.SaveSnapshot()) {
                        CB_ENSURE(ui64(iterations) * (ui64(maxLeaves) * 8 + ui64(depth) * 9) +
                                  ui64(data.Rows) * sizeof(float) + orderedSession->GetStateBytes() <= (1ull << 29),
                                  "Metal Ordered snapshot output exceeds the experimental 512 MiB limit");
                    }
                } else if (vectorBackend) {
                    CBMMulticlassParams multiclassParams = {
                        data.Rows, data.FeatureCount(), static_cast<ui32>(data.CandidateFeatures.size()), data.BinsPerFeature,
                        approxDimension, multioutput ? MetalMultiOutputObjective(objective) :
                            objective == ELossFunction::MultiClass ? 0u : 1u,
                        iterations - restoredIterations, depth, params.train.score_function,
                        options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Newton ? 0u : 1u,
                        params.leaf_estimation_iterations, 0, params.train.learning_rate, params.train.l2_leaf_reg, 0, 0};
                    if (greedy) {
                        const CBMVectorGreedyOptions greedyOptions = {greedyPolicy, maxLeaves,
                            options.ObliviousTreeOptions->MinDataInLeaf, 0};
                        TVector<ui32> labels;
                        TVector<float> vectorTargets;
                        if (multioutput) vectorTargets = MetalMultiOutputTargets(*target, data.Rows, approxDimension, objective);
                        else {
                            labels.resize(data.Rows);
                            for (ui32 row = 0; row < data.Rows; ++row) {
                                CB_ENSURE(targets[row] >= 0 && targets[row] < approxDimension && targets[row] == std::floor(targets[row]),
                                    "Metal multiclass labels must be compressed class IDs");
                                labels[row] = targets[row];
                            }
                        }
                        CB_ENSURE(cbm_multiclass_session_create_greedy(&multiclassParams, &greedyOptions, data.Bins.data(),
                            labels.empty() ? nullptr : labels.data(), vectorTargets.empty() ? nullptr : vectorTargets.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(), data.CandidateFeatures.data(),
                            data.CandidateBins.data(), data.CandidateTypes.data(), &session.Handle, error, sizeof(error)) == 0,
                            "Metal vector greedy initialization failed: " << error);
                    } else if (multioutput) {
                        const auto vectorTargets = MetalMultiOutputTargets(*target, data.Rows, approxDimension, objective);
                        CB_ENSURE(cbm_multioutput_session_create(&multiclassParams, data.Bins.data(), vectorTargets.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(), initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal multioutput initialization failed: " << error);
                    } else {
                        TVector<ui32> labels(data.Rows);
                        for (ui32 row = 0; row < data.Rows; ++row) {
                            CB_ENSURE(targets[row] >= 0 && targets[row] < approxDimension &&
                                      targets[row] == std::floor(targets[row]), "Metal multiclass labels must be compressed class IDs");
                            labels[row] = targets[row];
                        }
                        CB_ENSURE(cbm_multiclass_session_create(&multiclassParams, data.Bins.data(), labels.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal multiclass initialization failed: " << error);
                    }
                    CB_ENSURE(cbm_multiclass_session_set_backtracking(session.Handle,
                        params.leaf_estimation_backtracking, error, sizeof(error)) == 0,
                        "Metal multiclass backtracking configuration failed: " << error);
                } else {
                    const auto objectiveOptions = MetalObjectiveOptions(options, objectiveId);
                    if (yeti) {
                        const auto query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping,
                            options.LossFunctionDescription.Get());
                        const int draws = NCatboostOptions::GetYetiRankPermutations(options.LossFunctionDescription.Get());
                        CB_ENSURE(draws >= 1 && draws <= 10000, "Metal YetiRank permutations must be in [1,10000]");
                        for (float target : targets) CB_ENSURE(target >= 0 && target <= 1,
                            "Metal classic YetiRank with PFound requires targets in [0,1]");
                        CBMYetiRankOptions yetiOptions = {query.Options.group_count, static_cast<ui32>(draws),
                            static_cast<float>(NCatboostOptions::GetYetiRankDecay(options.LossFunctionDescription.Get())), 0};
                        CB_ENSURE(cbm_session_create_yeti(&params, &objectiveOptions, &yetiOptions,
                            query.Offsets.data(), data.Bins.data(), targets.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal YetiRank initialization failed: " << error);
                    } else if (yetiPair) {
                        const auto query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping, options.LossFunctionDescription.Get());
                        const int draws = NCatboostOptions::GetYetiRankPermutations(options.LossFunctionDescription.Get());
                        CB_ENSURE(draws >= 1 && draws <= 10000, "Metal YetiRankPairwise permutations must be in [1,10000]");
                        for (float target : targets) CB_ENSURE(target >= 0 && target <= 1,
                            "Metal YetiRankPairwise with PFound requires targets in [0,1]");
                        const CBMYetiRankPairwiseOptions yetiOptions = {query.Options.group_count,static_cast<ui32>(draws),
                            static_cast<float>(NCatboostOptions::GetYetiRankDecay(options.LossFunctionDescription.Get())),
                            options.ObliviousTreeOptions->BootstrapConfig->GetSamplingUnit() == ESamplingUnit::Group ? 1u : 0u};
                        CB_ENSURE(cbm_session_create_yeti_pairwise(&params,&objectiveOptions,&yetiOptions,
                            options.ObliviousTreeOptions->PairwiseNonDiagReg,query.Offsets.data(),data.Bins.data(),targets.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(),data.CandidateBins.data(),data.CandidateTypes.data(),
                            &session.Handle,error,sizeof(error)) == 0,"Metal YetiRankPairwise initialization failed: " << error);
                    } else if (objectiveId == 12 || objectiveId == 13) {
                        const auto query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping,
                            options.LossFunctionDescription.Get());
                        CB_ENSURE(cbm_session_create_query(&params, &objectiveOptions, &query.Options,
                            query.Offsets.data(), data.Bins.data(), targets.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal query initialization failed: " << error);
                    } else if (qce) {
                        const auto query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping, options.LossFunctionDescription.Get());
                        const auto scales = SelectMetalQueryCrossEntropyScales(options.LossFunctionDescription.Get(), targets, query.Offsets);
                        const CBMQueryCrossEntropyOptions qceOptions = {query.Options.group_count,0,0,0};
                        CB_ENSURE(cbm_session_create_query_cross_entropy(&params, &objectiveOptions, &qceOptions,
                            options.ObliviousTreeOptions->PairwiseNonDiagReg, query.Offsets.data(), scales.data(),
                            data.Bins.data(), targets.data(), sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal QueryCrossEntropy initialization failed: " << error);
                    } else if (coupled) {
                        const auto pairs = PrepareMetalPairData(*trainingData.Learn->TargetData, data.Rows);
                        CB_ENSURE(cbm_session_create_pair_matrix(&params, &objectiveOptions, &pairs.Options,
                            options.ObliviousTreeOptions->PairwiseNonDiagReg,
                            pairs.Winners.data(), pairs.Losers.data(), pairs.Weights.data(), pairs.GroupOffsets.data(),
                            data.Bins.data(), sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal PairLogitPairwise initialization failed: " << error);
                    } else if (pairwise) {
                        const auto pairs = PrepareMetalPairData(*trainingData.Learn->TargetData, data.Rows);
                        CB_ENSURE(cbm_session_create_pair(&params, &objectiveOptions, &pairs.Options,
                            pairs.Winners.data(), pairs.Losers.data(), pairs.Weights.data(), pairs.GroupOffsets.data(),
                            data.Bins.data(), initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal PairLogit initialization failed: " << error);
                    } else {
                        CB_ENSURE(cbm_session_create_configured(&params, &objectiveOptions, data.Bins.data(), targets.data(),
                            sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(),
                            data.CandidateFeatures.data(), data.CandidateBins.data(), data.CandidateTypes.data(),
                            &session.Handle, error, sizeof(error)) == 0, "Metal training initialization failed: " << error);
                    }
                }
                const auto& bootstrap = options.ObliviousTreeOptions->BootstrapConfig.Get();
                CBMBootstrapOptions bootstrapOptions = {};
                bootstrapOptions.bootstrap_type = MetalBootstrap(bootstrap.GetBootstrapType());
                const ui64 seed = options.RandomSeed;
                bootstrapOptions.random_seed_low = static_cast<ui32>(seed);
                bootstrapOptions.random_seed_high = static_cast<ui32>(seed >> 32);
                const ui64 absoluteOffset = (initModel ? (*initModel)->GetTreeCount() : 0) + restoredIterations;
                CB_ENSURE(absoluteOffset <= Max<ui32>(), "Metal bootstrap iteration offset exceeds uint32 capacity");
                bootstrapOptions.iteration_offset = absoluteOffset;
                bootstrapOptions.bagging_temperature = bootstrap.GetBaggingTemperature();
                bootstrapOptions.subsample = bootstrap.GetTakenFraction();
                const auto mvsReg = bootstrap.GetMvsReg();
                bootstrapOptions.mvs_reg_is_set = mvsReg.Defined();
                bootstrapOptions.mvs_reg = mvsReg.GetOrElse(0);
                bootstrapOptions.initial_mvs_lambda = snapshot.MvsLambda;
                bootstrapOptions.initial_mvs_lambda_is_set = snapshot.MvsLambdaIsSet;
                CBMScoreNoiseOptions noiseOptions = {};
                noiseOptions.random_strength = options.ObliviousTreeOptions->RandomStrength;
                if (greedySession) {
                    greedySession->SetBootstrap(bootstrapOptions);
                    greedySession->SetScoreNoise(noiseOptions);
                    greedySession->SetBacktracking(params.leaf_estimation_backtracking);
                    if (permutationCount > 1) {
                        TVector<TConstArrayRef<ui8>> permutationBins{MakeConstArrayRef(data.Bins)};
                        for (const auto& bins : data.AdditionalPermutationBins) permutationBins.push_back(MakeConstArrayRef(bins));
                        permutations = MakeHolder<TMetalDocParallelPermutations>(greedySession->GetHandle(),
                            data.Rows, data.FeatureCount(), options.RandomSeed, permutationBins,
                            snapshot.PermutationPredictions, snapshot.PermutationMvsLambdas,
                            snapshot.PermutationMvsValid, 1, true);
                    }
                } else if (ordered) {
                    orderedSession->SetBacktracking(params.leaf_estimation_backtracking);
                    orderedSession->SetBootstrap(bootstrapOptions,
                        options.ObliviousTreeOptions->ObservationsToBootstrap == EObservationsToBootstrap::TestOnly);
                    orderedSession->SetScoreNoise(noiseOptions);
                    if (data.Categorical.CtrProvider)
                        orderedSession->SetFeaturePenalties(ctrUniqueValues, options.ObliviousTreeOptions->ModelSizeReg);
                    if (restoredIterations) {
                        orderedSession->RestoreState({snapshot.OrderedDescriptors, snapshot.OrderedCursors});
                        orderedSession->RestoreRandomState({snapshot.OrderedRandomDrawCount,
                            snapshot.OrderedRandomCompletedIterations, snapshot.OrderedBootstrapInitialized}, restoredIterations);
                    }
                } else {
                    const auto setBootstrap = vectorBackend ? cbm_multiclass_session_set_bootstrap : cbm_session_set_bootstrap;
                    CB_ENSURE(setBootstrap(session.Handle, &bootstrapOptions, error, sizeof(error)) == 0,
                              "Metal bootstrap configuration failed: " << error);
                    const auto setScoreNoise = vectorBackend ? cbm_multiclass_session_set_score_noise : cbm_session_set_score_noise;
                    CB_ENSURE(setScoreNoise(session.Handle, &noiseOptions, error, sizeof(error)) == 0,
                              "Metal split-score randomness configuration failed: " << error);
                    if (!greedy) {
                        CBMFeaturePenaltyOptions penaltyOptions = {};
                        penaltyOptions.model_size_reg = options.ObliviousTreeOptions->ModelSizeReg;
                        const auto setFeaturePenalties = vectorBackend ? cbm_multiclass_session_set_feature_penalties :
                            cbm_session_set_feature_penalties;
                        CB_ENSURE(setFeaturePenalties(session.Handle, &penaltyOptions,
                            ctrUniqueValues.data(), nullptr, usedFeatures.data(), error, sizeof(error)) == 0,
                            "Metal CTR model-size penalty configuration failed: " << error);
                    }
                    TVector<TConstArrayRef<ui8>> permutationBins{MakeConstArrayRef(data.Bins)};
                    for (const auto& bins : data.AdditionalPermutationBins) permutationBins.push_back(MakeConstArrayRef(bins));
                    permutations = MakeHolder<TMetalDocParallelPermutations>(session.Handle, data.Rows, data.FeatureCount(),
                        options.RandomSeed, permutationBins, snapshot.PermutationPredictions,
                        snapshot.PermutationMvsLambdas, snapshot.PermutationMvsValid, approxDimension);
                    if (vectorBackend && restoredIterations) {
                        CB_ENSURE(cbm_multiclass_session_restore_optimization_state(session.Handle,
                            data.Categorical.GetPermutationCount(), snapshot.OptimizationPredictions.data(),
                            error, sizeof(error)) == 0, "Metal multiclass optimizer cursor restoration failed: " << error);
                    }
                }
            }
            TFullModel initialBiasModel;
            const TFullModel* progressInitialModel = initModel ? *initModel : nullptr;
            if (multioutput && options.BoostingOptions->BoostFromAverage) {
                initialBiasModel = MetalMultiOutputBiasModel(modelBias);
                progressInitialModel = &initialBiasModel;
            }
            TMetalTrainingProgress progress(options, outputOptions, trainingData, bias, executor,
                progressInitialModel,
                initModel ? &initModelApplyCompatiblePools : nullptr,
                internalOptions.ForceCalcEvalMetricOnEveryIteration, evalMetricDescriptor, approxDimension, baselineColumns);
            TObliviousTreeBuilder builder(data.AllFloatFeatures, data.Categorical.AllCatFeatures, {}, {}, approxDimension);
            TNonSymmetricTreeModelBuilder greedyBuilder(data.AllFloatFeatures, data.Categorical.AllCatFeatures, {}, {}, approxDimension);
            CBMStepInfo info = {};
            CB_ENSURE(cbm_device_info(info.stats.device_name, sizeof(info.stats.device_name), error, sizeof(error)) == 0,
                      "Metal device is unavailable: " << error);
            bool continueTraining = true;
            for (ui32 tree = 0; tree < restoredIterations; ++tree) {
                auto singleTree = greedy ? AppendMetalGreedyTree(data, snapshot.GreedyTrees.GetTree(tree, approxDimension),
                    greedyDepthBound, &greedyBuilder, approxDimension) : AppendMetalTree(data, snapshot.Depths[tree], depth,
                    MakeConstArrayRef(snapshot.SplitFeatures).Slice(ui64(tree) * depth, depth),
                    MakeConstArrayRef(snapshot.SplitBins).Slice(ui64(tree) * depth, depth),
                    MakeConstArrayRef(snapshot.SplitTypes).Slice(ui64(tree) * depth, depth),
                    MakeConstArrayRef(snapshot.Leaves).Slice(ui64(tree) * maxLeaves * approxDimension, ui64(maxLeaves) * approxDimension),
                    MakeConstArrayRef(snapshot.Weights).Slice(ui64(tree) * maxLeaves, maxLeaves), &builder, approxDimension);
                continueTraining = progress.ReplayIteration(tree, singleTree, snapshot.History);
            }
            if (restoredIterations) progress.RestoreTimeHistory(snapshot.History.TimeHistory);
            THPTimer snapshotTimer;
            for (ui32 tree = restoredIterations; tree < iterations && continueTraining; ++tree) {
                CheckInterrupted();
                progress.StartIteration();
                ui32 treeDepth = 0;
                TMetalGreedyTree greedyTree;
                const ui64 absoluteIteration = (initModel ? (*initModel)->GetTreeCount() : 0) + tree;
                if (greedy && vectorBackend) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration);
                    greedyTree.Nodes.resize(ui64(maxLeaves) * 2 - 1);
                    greedyTree.Values.resize(ui64(maxLeaves) * approxDimension); greedyTree.Weights.resize(maxLeaves);
                    CB_ENSURE(cbm_multiclass_session_step_greedy(session.Handle, &greedyTree.Info, greedyTree.Nodes.data(),
                        greedyTree.Values.data(), greedyTree.Weights.data(), error, sizeof(error)) == 0,
                        "Metal vector greedy iteration " << tree << " failed: " << error);
                    CB_ENSURE(greedyTree.Info.leaf_count && greedyTree.Info.leaf_count <= maxLeaves &&
                        greedyTree.Info.node_count == 2 * greedyTree.Info.leaf_count - 1,
                        "Metal vector greedy returned inconsistent tree dimensions");
                    greedyTree.Nodes.resize(greedyTree.Info.node_count);
                    greedyTree.Values.resize(ui64(greedyTree.Info.leaf_count) * approxDimension);
                    greedyTree.Weights.resize(greedyTree.Info.leaf_count);
                    info.completed_iterations = greedyTree.Info.completed_iterations;
                    info.finished = greedyTree.Info.finished; info.stats = greedyTree.Info.stats;
                    treeDepth = ValidateMetalGreedyTree(greedyTree.Nodes, greedyTree.Values, greedyTree.Weights,
                        data.FeatureCount(), greedyDepthBound, approxDimension).Depth;
                } else if (greedySession) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration);
                    greedyTree = greedySession->Step(absoluteIteration);
                    info.completed_iterations = greedyTree.Info.completed_iterations;
                    info.finished = greedyTree.Info.finished;
                    info.stats = greedyTree.Info.stats;
                    treeDepth = ValidateMetalGreedyTree(greedyTree.Nodes, greedyTree.Values, greedyTree.Weights,
                        data.FeatureCount(), greedyDepthBound).Depth;
                } else if (orderedSession) {
                    orderedSession->Step(absoluteIteration, &info, &treeDepth,
                        splitFeatures, splitBins, splitTypes, leaves, weights);
                } else if (yetiRandom) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration);
                    const uint64_t weakSeed = yetiRandom->Begin();
                    CB_ENSURE(cbm_session_set_yeti_oracle_seeds(session.Handle, 1, &weakSeed, error, sizeof(error)) == 0 &&
                        cbm_session_begin_tree(session.Handle, error, sizeof(error)) == 0,
                        "Metal YetiRank weak target failed: " << error);
                    CBMStructureInfo structure = {};
                    ui32 attempts = 0;
                    do {
                        CB_ENSURE(cbm_session_grow_tree(session.Handle, &structure, error, sizeof(error)) == 0,
                            "Metal YetiRank split search failed: " << error);
                        if (depth && !data.CandidateFeatures.empty()) ++attempts;
                    } while (!structure.finished);
                    const auto seeds = yetiRandom->LeafSeeds(attempts);
                    CB_ENSURE(cbm_session_set_yeti_leaf_seeds(session.Handle, seeds.size(), seeds.data(), error, sizeof(error)) == 0 &&
                        cbm_session_finish_tree(session.Handle, &info, &treeDepth, splitFeatures.data(),
                            splitBins.data(), splitTypes.data(), leaves.data(), weights.data(), error, sizeof(error)) == 0,
                        "Metal YetiRank leaf estimation failed: " << error);
                    yetiRandom->Complete();
                } else {
                    if (permutations) permutations->SelectForIteration(absoluteIteration);
                    const auto step = vectorBackend ? cbm_multiclass_session_step : cbm_session_step;
                    CB_ENSURE(step(session.Handle, &info, &treeDepth, splitFeatures.data(),
                        splitBins.data(), splitTypes.data(), leaves.data(), weights.data(), error, sizeof(error)) == 0,
                        "Metal training iteration " << tree << " failed: " << error);
                }
                auto singleTreeModel = greedy ? AppendMetalGreedyTree(data, greedyTree, greedyDepthBound, &greedyBuilder, approxDimension)
                    : AppendMetalTree(data, treeDepth, depth, splitFeatures,
                    splitBins, splitTypes, leaves, weights, &builder, approxDimension);
                if (greedySession) {
                    greedySession->CopyPredictions(learnCursor);
                } else if (orderedSession) {
                    orderedSession->CopyPredictions(learnCursor);
                } else {
                    const auto copyPredictions = vectorBackend ? cbm_multiclass_session_copy_predictions : cbm_session_copy_predictions;
                    CB_ENSURE(copyPredictions(session.Handle, learnCursor.data(), error, sizeof(error)) == 0,
                              "Metal prediction cursor copy failed: " << error);
                }
                continueTraining = progress.OnIteration(tree, singleTreeModel, learnCursor);
                const auto& history = progress.GetHistory();
                continueTraining = continueTraining &&
                    (!trainingCallbacks || trainingCallbacks->IsContinueTraining(history)) &&
                    (!customCallbacks || customCallbacks->AfterIteration(history));
                if (outputOptions.SaveSnapshot()) {
                    snapshot.Depths.push_back(treeDepth);
                    if (greedy) {
                        snapshot.GreedyTrees.Append(greedyTree, data.FeatureCount(), greedyDepthBound, approxDimension);
                    } else {
                        snapshot.SplitFeatures.insert(snapshot.SplitFeatures.end(), splitFeatures.begin(), splitFeatures.end());
                        snapshot.SplitBins.insert(snapshot.SplitBins.end(), splitBins.begin(), splitBins.end());
                        snapshot.SplitTypes.insert(snapshot.SplitTypes.end(), splitTypes.begin(), splitTypes.end());
                        snapshot.Leaves.insert(snapshot.Leaves.end(), leaves.begin(), leaves.end());
                        snapshot.Weights.insert(snapshot.Weights.end(), weights.begin(), weights.end());
                    }
                    if (!continueTraining || tree + 1 == iterations ||
                        snapshotTimer.Passed() >= outputOptions.GetSnapshotSaveInterval()) {
                        CB_ENSURE(info.completed_iterations == tree + 1 - restoredIterations,
                                  "Metal snapshot iteration count mismatch");
                        snapshot.Predictions = learnCursor;
                        ui32 absoluteIterations = 0;
                        if (greedySession) {
                            const auto bootstrapState = greedySession->GetBootstrapState();
                            absoluteIterations = bootstrapState.AbsoluteIterations;
                            snapshot.MvsLambda = bootstrapState.MvsLambda;
                            snapshot.MvsLambdaIsSet = bootstrapState.MvsLambdaIsSet;
                        } else if (orderedSession) {
                            const auto bootstrapState = orderedSession->GetBootstrapState();
                            absoluteIterations = bootstrapState.AbsoluteIterations;
                            snapshot.MvsLambda = bootstrapState.MvsLambda;
                            snapshot.MvsLambdaIsSet = bootstrapState.MvsLambdaIsSet;
                            const auto state = orderedSession->CopyState();
                            snapshot.OrderedDescriptors = state.Descriptors;
                            snapshot.OrderedCursors = state.Cursors;
                            const auto randomState = orderedSession->GetRandomState();
                            snapshot.OrderedRandomDrawCount = randomState.DrawCount;
                            snapshot.OrderedRandomCompletedIterations = randomState.CompletedIterations;
                            snapshot.OrderedBootstrapInitialized = randomState.BootstrapInitialized;
                        } else {
                            const auto getBootstrapState = vectorBackend ? cbm_multiclass_session_get_bootstrap_state : cbm_session_get_bootstrap_state;
                            CB_ENSURE(getBootstrapState(session.Handle, &absoluteIterations,
                                &snapshot.MvsLambda, &snapshot.MvsLambdaIsSet, error, sizeof(error)) == 0,
                                "Metal bootstrap state extraction failed: " << error);
                        }
                        if (permutations) {
                            const auto state = permutations->CopyState();
                            snapshot.PermutationPredictions = state.Predictions;
                            snapshot.PermutationMvsLambdas = state.MvsLambdas;
                            snapshot.PermutationMvsValid = state.MvsValid;
                            if (vectorBackend) {
                                snapshot.OptimizationPredictions.resize(ui64(data.Rows) * optimizerDimension *
                                    data.Categorical.GetPermutationCount());
                                CB_ENSURE(cbm_multiclass_session_copy_optimization_state(session.Handle,
                                    data.Categorical.GetPermutationCount(), snapshot.OptimizationPredictions.data(),
                                    error, sizeof(error)) == 0, "Metal multiclass optimizer snapshot state extraction failed: " << error);
                            }
                            if (!greedy) {
                                snapshot.UsedFeatures.resize(data.FeatureCount());
                                const auto copyFeatureUse = vectorBackend ? cbm_multiclass_session_copy_feature_penalty_state :
                                    cbm_session_copy_feature_penalty_state;
                                CB_ENSURE(copyFeatureUse(session.Handle, snapshot.UsedFeatures.data(), error, sizeof(error)) == 0,
                                    "Metal feature-use snapshot state extraction failed: " << error);
                            }
                        }
                        if (yetiRandom) snapshot.YetiRandom = yetiRandom->GetState();
                        snapshot.History = history;
                        snapshot.Save(snapshotPath, info.stats.device_name, trainingCallbacks);
                        snapshotTimer.Reset();
                    }
                }
            }
            const auto& history = progress.GetHistory();
            if (metricsAndTimeHistory) *metricsAndTimeHistory = history;
            TFullModel model;
            if (greedy) greedyBuilder.Build(model.ModelTrees.GetMutable());
            else builder.Build(model.ModelTrees.GetMutable());
            model.SetScaleAndBias({1.0, modelBias});
            if (yeti) model.ModelInfo["metal_yeti_centering"] = "all_rows";
            if (yetiPair) model.ModelInfo["metal_yeti_pair_rng"] = permutationCount > 1
                ? "item_iteration_dataset_domains_v2" : "item_iteration_domains_v1";
            if (data.Categorical.CtrProvider) model.CtrProvider = data.Categorical.CtrProvider->Clone();
            model.UpdateDynamicData();
            if (classificationTargetHelper.IsInitialized()) {
                model.ModelInfo["class_params"] = classificationTargetHelper.Serialize();
            }
            if (initModel) {
                model = SumModels({*initModel, &model}, {1.0, 1.0}, {"initModel:", ""});
            }
            progress.Finish(&model, evalResultPtrs);
            if (model.CtrProvider) {
                model.CtrProvider->DropUnusedTables(model.ModelTrees->GetApplyData()->GetUsedModelCtrBases());
            }
            model.ModelInfo["metal_backend"] = "METAL";
            model.ModelInfo["metal_device"] = info.stats.device_name;
            model.ModelInfo["metal_permutations"] = ToString(permutationCount);
            model.ModelInfo["metal_port"] = ordered ? "Native CUDA Ordered/FeatureParallel translation" :
                "Native CUDA Plain/DocParallel translation";
            TCoreModelToFullModelConverter converter(options, outputOptions, classificationTargetHelper,
                0, false, EFinalCtrComputationMode::Skip, EFinalFeatureCalcersComputationMode::Skip);
            converter.WithCoreModelFrom(&model).WithObjectsDataFrom(trainingData.Learn->ObjectsData).WithMetrics(history);
            if (dstModel) {
                converter.Do(true, dstModel, executor, nullptr);
            } else {
                converter.Do(outputOptions.CreateResultModelFullPath(), outputOptions.GetModelFormats(),
                             outputOptions.AddFileFormatExtension(), executor, nullptr);
            }
        }

        void ModelBasedEval(const NCatboostOptions::TCatBoostOptions&,
                            const NCatboostOptions::TOutputFilesOptions&,
                            TTrainingDataProviders,
                            const TLabelConverter&,
                            NPar::ILocalExecutor*) const override {
            CB_ENSURE(false, "Metal model-based feature evaluation is not yet implemented");
        }
    };

    TTrainerFactory::TRegistrator<TMetalModelTrainer> MetalModelTrainerRegistrator(ETaskType::GPU);
}
}
