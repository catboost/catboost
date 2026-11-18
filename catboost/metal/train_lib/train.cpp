// Native adapter for the CUDA algorithms translated to Metal. Pool preparation,
// quantization and final model construction remain in CatBoost's shared code.
#include "progress.h"
#include "fixed_splits.h"
#include "feature_metadata.h"
#include "feature_weights.h"
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
#include "tree_ctr_session.h"
#include "combination.h"
#include "ordered_shape.h"
#include "feature_parallel_yeti_random.h"
#include "meta_l2_context.h"
#include "meta_l2_random.h"
#include "langevin_random.h"
#include "estimated_features.h"
#include "estimated_features_checksum.h"

#include <catboost/libs/train_lib/train_model.h>
#include <catboost/metal/native/metal_trainer.h>
#include <catboost/metal/native/metal_multiclass.h>

#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/data/objects.h>
#include <util/stream/file.h>
#include <util/generic/set.h>
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

    template <uint32_t Event>
    int MetalLangevinTargetSeed(void* context, uint64_t* seed) noexcept {
        return TMetalLangevinRandom::SeedCallback(context, Event, seed);
    }

    template <class TRandom>
    int MetalCombinationLeafSeed(void* context, uint64_t* seed) noexcept {
        try {
            *seed = static_cast<TRandom*>(context)->NextLeafSeed();
            return 0;
        } catch (...) {
            return 1;
        }
    }

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
        const bool featureParallelQuery = objective == ELossFunction::QueryRMSE ||
            objective == ELossFunction::QuerySoftMax || objective == ELossFunction::PairLogit || yeti;
        const bool greedy = tree.GrowPolicy != EGrowPolicy::SymmetricTree;
        const bool combination = objective == ELossFunction::Combination;
        const bool custom = objective == ELossFunction::PythonUserDefinedPerObject;
        if (combination || custom) {
            CB_ENSURE(!greedy, "Metal Combination and custom objectives require symmetric trees");
            CB_ENSURE(tree.LeavesEstimationMethod != ELeavesEstimation::Exact,
                "Metal Combination and custom objectives require Newton, Gradient or Simple leaf estimation");
        }
        if (combination) ParseMetalCombinationComponents(options->LossFunctionDescription.Get());
        boosting.BoostingType.SetDefault(EBoostingType::Plain);
        options->CatFeatureParams->MaxTensorComplexity.SetDefault(1);
        const bool treeCtrs = options->CatFeatureParams->MaxTensorComplexity > 1;
        const bool ordered = boosting.BoostingType == EBoostingType::Ordered;
        boosting.DataPartitionType.SetDefault(ordered || treeCtrs ? EDataPartitionType::FeatureParallel : EDataPartitionType::DocParallel);
        boosting.PermutationCount.SetDefault(4);
        if (treeCtrs) {
            CB_ENSURE(!greedy && !vectorBackend && (!(querywise || pairwise) || featureParallelQuery),
                      "Metal compound CTRs support scalar and registered query objectives with symmetric trees");
            CB_ENSURE(boosting.DataPartitionType == EDataPartitionType::FeatureParallel,
                      "Metal compound CTRs require data_partition='FeatureParallel'");
        }
        if (ordered) {
            CB_ENSURE(!vectorBackend, "Native Metal Ordered currently supports scalar objectives only");
            CB_ENSURE(tree.LeavesEstimationMethod != ELeavesEstimation::Exact,
                      "Native Metal Ordered supports Newton or Gradient leaves; GPU Ordered does not support Exact");
            CB_ENSURE(tree.ScoreFunction == EScoreFunction::Cosine || tree.ScoreFunction == EScoreFunction::NewtonCosine,
                      "Native Metal Ordered supports Cosine and NewtonCosine scores");
        }
        if (querywise || pairwise) {
            CB_ENSURE(!ordered || featureParallelQuery,
                      "Native Metal full-matrix query and pairwise objectives require Plain boosting");
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
        } else if (!custom) options->MetricOptions->EvalMetric.SetDefault(options->LossFunctionDescription.Get());

        CB_ENSURE(objective == ELossFunction::RMSE || objective == ELossFunction::Logloss ||
                  objective == ELossFunction::CrossEntropy || objective == ELossFunction::Poisson ||
                  objective == ELossFunction::Huber || objective == ELossFunction::Expectile ||
                  objective == ELossFunction::Lq || objective == ELossFunction::Tweedie ||
                  objective == ELossFunction::LogLinQuantile || objective == ELossFunction::Quantile ||
                  objective == ELossFunction::MAE || objective == ELossFunction::MAPE || vectorBackend || querywise || pairwise ||
                  combination || custom,
                  "Metal supports RMSE, Logloss, CrossEntropy, Poisson, Huber, Expectile, Lq, "
                  "Tweedie, LogLinQuantile, Quantile, MAE, MAPE, MultiClass, MultiClassOneVsAll, "
                  "QueryRMSE, QuerySoftMax, PairLogit, PairLogitPairwise, QueryCrossEntropy, YetiRank, MultiRMSE, RMSEWithUncertainty, MultiLogloss, and MultiCrossEntropy objectives");
        if (vectorBackend) {
            CB_ENSURE(tree.BootstrapConfig->GetBootstrapType() != EBootstrapType::MVS,
                      "Metal multidimensional training does not support MVS bootstrap");
        }
        CB_ENSURE(ordered ? boosting.DataPartitionType == EDataPartitionType::FeatureParallel :
                  boosting.BoostingType == EBoostingType::Plain &&
                  (boosting.DataPartitionType == EDataPartitionType::DocParallel ||
                   (!greedy && !vectorBackend && !coupled && !qce && !yetiPair)),
                  "Metal FeatureParallel supports symmetric scalar and registered query objectives");
        if (greedy) {
            CB_ENSURE(!ordered, "Metal non-symmetric training requires Plain boosting");
            CB_ENSURE((!querywise || objective == ELossFunction::QueryRMSE || objective == ELossFunction::QuerySoftMax || yeti) &&
                (!pairwise || objective == ELossFunction::PairLogit) && objective != ELossFunction::Lq &&
                (!multioutput || objective == ELossFunction::RMSEWithUncertainty),
                "Metal greedy training supports the eleven CUDA-registered scalar objectives, MultiClass, MultiClassOneVsAll, RMSEWithUncertainty, QueryRMSE, QuerySoftMax, PairLogit and YetiRank; Lq and full-matrix ranking are unsupported");
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
                  (tree.LeavesEstimationMethod == ELeavesEstimation::Simple && !yeti),
                  "Metal supports Newton/Gradient/Exact and registered Simple leaf estimation");
        if (tree.LeavesEstimationMethod == ELeavesEstimation::Simple)
            CB_ENSURE((!(coupled || qce || yetiPair) || tree.MaxDepth > 0) && tree.LeavesEstimationIterations == 1,
                "Metal Simple leaves require one estimation iteration and full-matrix objectives require positive depth");
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
        CB_ENSURE(std::isfinite(tree.MetaL2Exponent.Get()) && std::isfinite(tree.MetaL2Frequency.Get()),
                  "Metal meta-L2 parameters must be finite");
        // CUDA scalar/query symmetric trainers ignore this option; the
        // multiclass-family symmetric trainer uses the greedy template.
        if (!greedy && vectorBackend) CB_ENSURE(tree.FixedBinarySplits.Get().empty(),
            "Fixed splits are not supported for symmetric trees");
        CB_ENSURE(tree.Rsm == 1.0f || coupled || qce || yetiPair,
                  "Metal feature subsampling requires PairLogitPairwise, QueryCrossEntropy or YetiRankPairwise");
        CB_ENSURE(!boosting.PosteriorSampling.GetUnchecked() && boosting.ModelShrinkRate.GetUnchecked() == 0,
                  "Metal does not yet support posterior sampling or model shrinkage");
        CB_ENSURE(!boosting.Langevin || (!coupled && !qce && !yetiPair),
                  "Langevin is not supported for PairLogitPairwise, QueryCrossEntropy or YetiRankPairwise");
        CB_ENSURE(!boosting.Langevin || (std::isfinite(boosting.DiffusionTemperature.Get()) && boosting.DiffusionTemperature >= 0),
                  "Metal diffusion temperature must be finite, nonnegative and representable as float32");
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
            case ELossFunction::Combination: return 19;
            case ELossFunction::PythonUserDefinedPerObject: return 20;
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
        TMetalEstimatedFeatures Estimated;
        NPar::ILocalExecutor* Executor = nullptr;
        const TMetalTreeCtrFeatures* TreeCtrs = nullptr;

        ui32 CategoricalFeatureEnd() const {
            return FeatureIndices.size() + Categorical.SplitCandidates.size();
        }

        ui32 StaticFeatureCount() const {
            return CategoricalFeatureEnd() + Estimated.GetFeatureCount();
        }

        ui32 GetPermutationCount() const { return AdditionalPermutationBins.size() + 1; }
        bool HasPermutationDependentFeatures() const {
            return Categorical.HasPermutationDependentCtrs || Estimated.HasOnlineFeatures;
        }

        ui32 FeatureCount() const {
            return StaticFeatureCount() + (TreeCtrs ? TreeCtrs->GetFeatureCount() : 0);
        }

        TModelSplit GetSplit(ui32 feature, ui32 bin, ui8 type) const {
            CB_ENSURE(feature < FeatureCount(), "Metal returned an invalid feature index");
            if (feature < FeatureIndices.size()) {
                const auto& numeric = AllFloatFeatures[FeatureIndices[feature]];
                CB_ENSURE(type == 0 && bin < numeric.Borders.size(), "Metal returned an invalid numeric split");
                return TModelSplit(TFloatSplit{numeric.Position.Index, numeric.Borders[bin]});
            }
            if (feature < CategoricalFeatureEnd())
                return Categorical.GetSplit(feature - FeatureIndices.size(), bin, type);
            if (feature < StaticFeatureCount()) {
                CB_ENSURE(type == 0, "Metal estimated features require greater-than comparison");
                return Estimated.GetSplit(feature - CategoricalFeatureEnd(), bin);
            }
            CB_ENSURE(type == 0, "Metal tree CTR splits require greater-than comparison");
            return TreeCtrs->GetSplit(feature, bin);
        }
    };

    TMetalData PrepareData(const TTrainingDataProviders& trainingData,
                          const NCatboostOptions::TCatBoostOptions& options,
                          NPar::ILocalExecutor* executor) {
        const auto& data = *trainingData.Learn;
        const auto& objects = *data.ObjectsData;
        const auto& layout = *objects.GetFeaturesLayout();
        TMetalData result;
        result.Executor = executor;
        result.Estimated = PrepareMetalEstimatedFeatures(trainingData, options, executor);
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
        result.Categorical = PrepareMetalCategoricalPermutations(data, options, executor,
            trainingData.Test.empty() ? nullptr : trainingData.Test.front().Get());
        const auto& categorical = result.Categorical;
        const auto& estimated = result.Estimated;
        const ui32 permutationCount = Max(categorical.GetPermutationCount(), estimated.GetPermutationCount());
        const ui64 matrixSize = ui64(result.Rows) * result.StaticFeatureCount();
        CB_ENSURE(matrixSize * permutationCount <= (1ull << 30),
                  "Metal quantized permutation inputs exceed the experimental 1 GiB limit");
        const auto numericBins = result.Bins;
        for (ui32 permutation = 0; permutation < permutationCount; ++permutation) {
            auto& bins = permutation ? result.AdditionalPermutationBins.emplace_back(numericBins) : result.Bins;
            const auto catBins = categorical.GetPermutationBins(categorical.GetPermutationCount() > 1 ? permutation : 0);
            bins.insert(bins.end(), catBins.begin(), catBins.end());
            if (estimated.GetFeatureCount()) {
                const auto& estimatedBins = estimated.BinsByPermutation[estimated.GetPermutationCount() > 1 ? permutation : 0];
                bins.insert(bins.end(), estimatedBins.begin(), estimatedBins.end());
            }
            CB_ENSURE(bins.size() == matrixSize, "Metal permutation feature matrices differ in shape");
        }
        result.BinsPerFeature = Max(result.BinsPerFeature, categorical.BinsPerFeature);
        for (ui32 i = 0; i < categorical.CandidateFeatures.size(); ++i) {
            result.CandidateFeatures.push_back(result.FeatureIndices.size() + categorical.CandidateFeatures[i]);
            result.CandidateBins.push_back(categorical.CandidateBins[i]);
            result.CandidateTypes.push_back(categorical.CandidateTypes[i]);
        }
        result.BinsPerFeature = Max(result.BinsPerFeature, estimated.BinsPerFeature);
        for (ui32 feature = 0; feature < estimated.GetFeatureCount(); ++feature) {
            for (ui32 bin = 0; bin < estimated.Borders[feature].size(); ++bin) {
                result.CandidateFeatures.push_back(result.CategoricalFeatureEnd() + feature);
                result.CandidateBins.push_back(bin);
                result.CandidateTypes.push_back(0);
            }
        }
        CB_ENSURE(result.FeatureCount() > 0, "Metal requires at least one available feature");
        return result;
    }

    TFullModel AppendMetalTree(
        const TMetalData& data, ui32 treeDepth, ui32 maxDepth,
        TConstArrayRef<ui32> splitFeatures, TConstArrayRef<ui32> splitBins,
        TConstArrayRef<ui8> splitTypes, TConstArrayRef<float> leaves,
        TConstArrayRef<float> weights, TObliviousTreeBuilder* builder, ui32 approxDimension = 1,
        bool allowSignedLeafWeights = false) {
        CB_ENSURE(treeDepth <= maxDepth, "Metal returned an invalid tree depth");
        TVector<TModelSplit> splits;
        for (ui32 level = 0; level < treeDepth; ++level) {
            splits.push_back(data.GetSplit(splitFeatures[level], splitBins[level], splitTypes[level]));
        }
        const ui32 count = 1u << treeDepth;
        TVector<double> leafValues(ui64(count) * approxDimension), leafWeights(count);
        for (ui32 leaf = 0; leaf < count; ++leaf) {
            CB_ENSURE(std::isfinite(weights[leaf]) && (allowSignedLeafWeights || weights[leaf] >= 0),
                      "Metal returned invalid leaf weights");
            leafWeights[leaf] = weights[leaf];
        }
        for (ui64 value = 0; value < leafValues.size(); ++value) {
            CB_ENSURE(std::isfinite(leaves[value]), "Metal returned invalid leaf values");
            leafValues[value] = leaves[value];
        }
        if (builder) builder->AddTree(splits, leafValues, leafWeights);
        TObliviousTreeBuilder singleTreeBuilder(data.AllFloatFeatures, data.Categorical.AllCatFeatures,
            data.Estimated.AllTextFeatures, data.Estimated.AllEmbeddingFeatures, approxDimension);
        singleTreeBuilder.AddTree(splits, leafValues, leafWeights);
        TFullModel model;
        singleTreeBuilder.Build(model.ModelTrees.GetMutable());
        if (data.Categorical.CtrProvider) model.CtrProvider = data.Categorical.CtrProvider->Clone();
        model.UpdateDynamicData();
        data.Estimated.FinalizeModel(&model, data.Executor);
        return model;
    }

    TFullModel AppendMetalGreedyTree(const TMetalData& data, const TMetalGreedyTree& tree,
        ui32 depthBound, TNonSymmetricTreeModelBuilder* builder, ui32 approxDimension = 1,
        bool allowSignedLeafWeights = false) {
        auto splitLookup = [&](ui32 denseFeature, ui32 bin, ui32 type) {
            return data.GetSplit(denseFeature, bin, type);
        };
        if (builder) builder->AddTree(MakeMetalGreedyTreeRoot(tree, data.FeatureCount(), splitLookup, depthBound, approxDimension, allowSignedLeafWeights));
        TNonSymmetricTreeModelBuilder single(data.AllFloatFeatures, data.Categorical.AllCatFeatures,
            data.Estimated.AllTextFeatures, data.Estimated.AllEmbeddingFeatures, approxDimension);
        single.AddTree(MakeMetalGreedyTreeRoot(tree, data.FeatureCount(), splitLookup, depthBound, approxDimension, allowSignedLeafWeights));
        TFullModel model;
        single.Build(model.ModelTrees.GetMutable());
        if (data.Categorical.CtrProvider) model.CtrProvider = data.Categorical.CtrProvider->Clone();
        model.UpdateDynamicData();
        data.Estimated.FinalizeModel(&model, data.Executor);
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
        return UpdateMetalEstimatedSourceChecksum(checksum, data, executor);
    }

    // An analysis experiment starts from saved prediction cursors, while its
    // model age, metric history and optimization state start at iteration zero.
    struct TMetalModelBasedRun {
        bool LoadBaseline = false;
        ui32 PermutationCount = 1;
        ui32 ApproxDimension = 1;
        ui32 LeafCapacity = 0;
        ui32 Depth = 0;
        ui64 PermutationOffset = 0;
        bool SignedWeights = false;
        TVector<float> InitialLearn;
        TVector<float> InitialTest;
        TMetalSnapshot BaselineSnapshot;
        TMetalData BaselineData;
    };

    TTrainingDataProviders MetalModelBasedFeatureSubset(
        const TTrainingDataProviders& source, const TVector<ui32>& ignored,
        NPar::ILocalExecutor* executor) {
        TTrainingDataProviders result;
        auto subset = [&](const TTrainingDataProviderPtr& pool) {
            auto objects = pool->ObjectsData->GetFeaturesSubset(ignored, executor);
            TQuantizedObjectsDataProviderPtr quantized = dynamic_cast<TQuantizedObjectsDataProvider*>(objects.Get());
            CB_ENSURE(quantized, "Metal model-based evaluation requires quantized pools");
            auto meta = pool->MetaInfo;
            meta.FeaturesLayout = quantized->GetFeaturesLayout();
            return MakeIntrusive<TTrainingDataProvider>(pool->OriginalFeaturesLayout,
                std::move(meta), pool->ObjectsGrouping, quantized, pool->TargetData);
        };
        result.Learn = subset(source.Learn);
        for (const auto& test : source.Test) result.Test.push_back(subset(test));
        return result;
    }

    ui32 MetalModelBasedBaselineSize(const TMetalModelBasedRun& base,
        const NCatboostOptions::TCatBoostOptions& options,
        const NCatboostOptions::TOutputFilesOptions& output,
        bool hasWeights) {
        const auto& snapshot = base.BaselineSnapshot;
        const ui32 count = snapshot.Depths.size();
        if (!output.UseBestModel) return count;
        auto metrics = CreateMetrics(options.MetricOptions, Nothing(), base.ApproxDimension, hasWeights);
        CB_ENSURE(!metrics.empty(), "Metal model-based evaluation requires an evaluation metric");
        EMetricBestValue bestType;
        float bestValue;
        metrics.front()->GetBestValue(&bestType, &bestValue);
        TErrorTracker tracker(EOverfittingDetectorType::IncToDec, bestType, bestValue, 0, 20, true, true);
        const TString description = metrics.front()->GetDescription();
        for (ui32 tree = 0; tree < snapshot.History.TestMetricsHistory.size(); ++tree) {
            const auto& test = snapshot.History.TestMetricsHistory[tree];
            if (tree + 1 < static_cast<ui32>(Max(0, output.BestModelMinTrees.Get())) || test.empty()) continue;
            const auto value = test.front().find(description);
            if (value != test.front().end()) tracker.AddError(value->second, tree);
        }
        return tracker.GetBestIteration() >= 0 ? Min(count, ui32(tracker.GetBestIteration()) + 1) : count;
    }

    // Replay saved trees against each original quantized history. These are
    // model applications only: no target derivatives or leaf estimates are run.
    void MetalModelBasedAppendPrefix(const TMetalModelBasedRun& base,
        const TTrainingDataProviders& pools, ui32 begin, ui32 end,
        TVector<float>* learn, TVector<float>* test, NPar::ILocalExecutor* executor) {
        const auto& snapshot = base.BaselineSnapshot;
        const auto& data = base.BaselineData;
        const ui32 dimensions = base.ApproxDimension, count = base.PermutationCount;
        CB_ENSURE(begin <= end && end <= snapshot.Depths.size(), "Invalid Metal baseline prefix range");
        CB_ENSURE(learn->size() == ui64(count) * data.Rows * dimensions &&
            test->size() == ui64(pools.Test.front()->GetObjectCount()) * dimensions,
            "Invalid Metal baseline prefix cursor sizes");
        for (ui32 tree = begin; tree < end; ++tree) {
            CheckInterrupted();
            TMetalGreedyTree greedyTree;
            if (snapshot.Greedy) greedyTree = snapshot.GreedyTrees.GetTree(tree, dimensions);
            const ui32 leafCount = snapshot.Greedy ? greedyTree.Info.leaf_count : 1u << snapshot.Depths[tree];
            for (ui32 p = 0; p < count; ++p) {
                const auto& bins = p ? data.AdditionalPermutationBins[p - 1] : data.Bins;
                const float* leaves = snapshot.ModelBasedPermutationCount
                    ? snapshot.ModelBasedPermutationLeaves.data() + (ui64(tree) * count + p) * snapshot.ModelBasedLeafCapacity * dimensions
                    : snapshot.Greedy ? greedyTree.Values.data()
                    : snapshot.Leaves.data() + ui64(tree) * base.LeafCapacity * dimensions;
                for (ui32 row = 0; row < data.Rows; ++row) {
                    ui32 leaf = 0;
                    auto right = [&](ui32 feature, ui32 bin, ui32 type) {
                        CB_ENSURE(feature < data.FeatureCount() && type <= 1 && bin < data.BinsPerFeature,
                            "Saved Metal baseline split is outside the prepared feature grid");
                        const ui8 value = bins[ui64(feature) * data.Rows + row];
                        return type ? value == bin : value > bin;
                    };
                    if (snapshot.Greedy) {
                        ui32 node = 0;
                        for (ui32 step = 0;; ++step) {
                            CB_ENSURE(step < greedyTree.Nodes.size() && node < greedyTree.Nodes.size(),
                                "Saved Metal baseline has an invalid greedy topology");
                            const auto& n = greedyTree.Nodes[node];
                            if (n.leaf != Max<ui32>()) { leaf = n.leaf; break; }
                            node = right(n.feature, n.bin, n.type) ? n.right : n.left;
                        }
                    } else {
                        for (ui32 level = 0; level < snapshot.Depths[tree]; ++level) {
                            const ui64 split = ui64(tree) * base.Depth + level;
                            if (right(snapshot.SplitFeatures[split], snapshot.SplitBins[split], snapshot.SplitTypes[split]))
                                leaf |= 1u << level;
                        }
                    }
                    CB_ENSURE(leaf < leafCount, "Saved Metal baseline leaf index is out of range");
                    for (ui32 dim = 0; dim < dimensions; ++dim) {
                        float& value = (*learn)[(ui64(p) * data.Rows + row) * dimensions + dim];
                        value += leaves[ui64(leaf) * dimensions + dim];
                        CB_ENSURE(std::isfinite(value), "Metal baseline prefix cursor is nonfinite");
                    }
                }
            }
            const auto single = snapshot.Greedy
                ? AppendMetalGreedyTree(data, greedyTree, Min(base.Depth, base.LeafCapacity - 1), nullptr,
                    dimensions, base.SignedWeights)
                : AppendMetalTree(data, snapshot.Depths[tree], base.Depth,
                    MakeConstArrayRef(snapshot.SplitFeatures).Slice(ui64(tree) * base.Depth, base.Depth),
                    MakeConstArrayRef(snapshot.SplitBins).Slice(ui64(tree) * base.Depth, base.Depth),
                    MakeConstArrayRef(snapshot.SplitTypes).Slice(ui64(tree) * base.Depth, base.Depth),
                    MakeConstArrayRef(snapshot.Leaves).Slice(ui64(tree) * base.LeafCapacity * dimensions,
                        ui64(base.LeafCapacity) * dimensions),
                    MakeConstArrayRef(snapshot.Weights).Slice(ui64(tree) * base.LeafCapacity, base.LeafCapacity),
                    nullptr, dimensions, base.SignedWeights);
            const auto delta = ApplyModelMulti(single, *pools.Test.front()->ObjectsData,
                EPredictionType::InternalRawFormulaVal, 0, 0, executor);
            for (ui32 row = 0; row < pools.Test.front()->GetObjectCount(); ++row) {
                for (ui32 dim = 0; dim < dimensions; ++dim) {
                    float& value = (*test)[ui64(row) * dimensions + dim];
                    value = static_cast<float>(double(value) + delta[dim][row]);
                    CB_ENSURE(std::isfinite(value), "Metal baseline evaluation prefix is nonfinite");
                }
            }
        }
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
            TrainModelImpl(internalOptions, catboostOptions, outputOptions, objectiveDescriptor,
                evalMetricDescriptor, std::move(trainingData), std::move(precomputedCtrs), labelConverter,
                trainingCallbacks, customCallbacks, initModel, std::move(initLearnProgress),
                std::move(initModelApplyCompatiblePools), executor, rand, dstModel, evalResultPtrs,
                metricsAndTimeHistory, dstLearnProgress, nullptr);
        }

        void TrainModelImpl(
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
            THolder<TLearnProgress>* dstLearnProgress,
            TMetalModelBasedRun* modelBased) const {
            Y_UNUSED(initLearnProgress);
            Y_UNUSED(rand);
            CB_ENSURE(!precomputedCtrs, "Metal does not yet support precomputed CTRs");
            const bool custom = catboostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::PythonUserDefinedPerObject;
            const TString customSource = objectiveDescriptor ? objectiveDescriptor->MetalSource : TString();
            CB_ENSURE(!objectiveDescriptor || custom,
                      "Metal custom objective descriptors require a scalar custom loss");
            CB_ENSURE(!custom || (objectiveDescriptor && !customSource.empty() && customSource.size() <= 65536 &&
                      customSource.find('\0') == TString::npos),
                      "Metal custom objectives require 1..65536 source bytes without NUL via calc_ders_range_metal");
            CB_ENSURE(!dstLearnProgress, "Metal does not expose CPU learn-progress state");
            NCatboostOptions::TCatBoostOptions options(catboostOptions);
            if (modelBased && !modelBased->LoadBaseline) {
                options.BoostingOptions->BoostFromAverage.Set(false);
            }
            const auto baseline = trainingData.Learn->TargetData->GetBaseline();
            if (initModel || baseline) {
                options.BoostingOptions->BoostFromAverage.SetDefault(false);
                CB_ENSURE(!options.BoostingOptions->BoostFromAverage,
                          "boost_from_average cannot be combined with an initial model or baseline");
            }
            SetMetalDefaultsAndValidate(&options);
            const bool ordered = options.BoostingOptions->BoostingType == EBoostingType::Ordered;
            const bool langevin = options.BoostingOptions->Langevin;
            // GPU defaults are false and zero. Positive temperature alone does
            // not opt into Langevin or inherit the separate CPU-only defaults.
            const bool compoundCtrs = options.CatFeatureParams->MaxTensorComplexity > 1;
            const bool featureParallel = options.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel;
            const auto growPolicy = options.ObliviousTreeOptions->GrowPolicy.Get();
            const bool greedy = growPolicy != EGrowPolicy::SymmetricTree;
            const bool greedySimple = greedy && options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple;
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
            EstimateMetalCtrPriors(*trainingData.Learn, &options);
            TMetalData data = PrepareData(trainingData, options, executor);
            if (modelBased && !modelBased->LoadBaseline) {
                const ui32 count = Max(data.GetPermutationCount(), modelBased->PermutationCount);
                CB_ENSURE(data.GetPermutationCount() == 1 || modelBased->PermutationCount == 1 ||
                    data.GetPermutationCount() == modelBased->PermutationCount,
                    "Metal model-based evaluation permutation histories differ");
                if (data.GetPermutationCount() == 1 && count > 1)
                    data.AdditionalPermutationBins.assign(count - 1, data.Bins);
            }
            TVector<float> featureWeights(data.StaticFeatureCount(), 1.0f);
            if (!options.ObliviousTreeOptions->FeaturePenalties->FeatureWeights.Get().empty()) {
                featureWeights = MakeMetalFeatureWeights(options.ObliviousTreeOptions->FeaturePenalties.Get(),
                    MakeMetalStaticFeatureMetadata(data, *trainingData.Learn, options));
            }
            TVector<ui32> fixedBinarySplits;
            if (greedy && !options.ObliviousTreeOptions->FixedBinarySplits.Get().empty()) {
                TVector<ui32> denseFloatFlatIndices;
                for (ui32 feature : data.FeatureIndices)
                    denseFloatFlatIndices.push_back(data.AllFloatFeatures[feature].Position.FlatIndex);
                fixedBinarySplits = ResolveMetalFixedBinarySplits(
                    options.ObliviousTreeOptions->FixedBinarySplits.Get(),
                    *trainingData.Learn->ObjectsData->GetFeaturesLayout(),
                    *trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo(), denseFloatFlatIndices);
            }
            CB_ENSURE(!initModel || (data.Estimated.AllTextFeatures.empty() && data.Estimated.AllEmbeddingFeatures.empty()),
                "Metal initial-model continuation cannot sum models containing text or embedding features");
            const auto orderedHistories = featureParallel ? MakeMetalFeatureParallelHistoryOrders(*trainingData.Learn, options) :
                TVector<TVector<ui32>>();
            const ui32 permutationCount = featureParallel ? orderedHistories.size() : data.GetPermutationCount();
            const auto objective = options.LossFunctionDescription->GetLossFunction();
            const bool combination = objective == ELossFunction::Combination;
            const bool signedSimpleWeights = !greedy && !featureParallel &&
                options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple &&
                (objective == ELossFunction::QuerySoftMax || combination);
            const bool orderedQuery = ordered && (objective == ELossFunction::QueryRMSE ||
                objective == ELossFunction::QuerySoftMax || objective == ELossFunction::PairLogit || yeti);
            const bool scalarSimple = options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple &&
                !IsMultiClassOnlyMetric(objective) && !IsMetalMultiOutput(objective) && MetalObjective(objective) <= 11;
            // Newly supported Simple and estimated-feature paths count the
            // actual independent/dependent score draws. Preserve the existing
            // chooser for prior Ordered scalar/custom configurations.
            const bool incrementalFeatureParallel = compoundCtrs || orderedQuery || (featureParallel &&
                (!ordered || combination || scalarSimple || data.Estimated.GetFeatureCount() || langevin));
            const bool pairwise = objective == ELossFunction::PairLogit || coupled;
            const bool multioutput = IsMetalMultiOutput(objective);
            const auto target = trainingData.Learn->TargetData->GetTarget();
            CB_ENSURE((target && (target->size() == 1 || (multioutput && !target->empty()))) || (pairwise && !target),
                      "Metal requires scalar targets or matching multioutput target columns");
            const TConstArrayRef<float> targets = target ? (*target)[0] : TConstArrayRef<float>();
            if (yeti) for (float value : targets) CB_ENSURE(value >= 0 && value <= 1,
                "Metal classic YetiRank with PFound requires targets in [0,1]");
            const auto sampleWeights = GetWeights(*trainingData.Learn->TargetData);
            TMaybe<TMetalCombinationData> combinationData;
            if (combination) combinationData = PrepareMetalCombinationData(options.LossFunctionDescription.Get(),
                *trainingData.Learn->TargetData, *trainingData.Learn->ObjectsGrouping);
            const ui32 combinationYetiCount = combination ? combinationData->YetiCount : 0;
            const bool stochasticTarget = yeti || combinationYetiCount;
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
            THolder<TMetalTreeCtrFeatures> treeCtrFeatures;
            TMaybe<TMetalTreeCtrBatch> restoredTreeCtrBatch;
            if (compoundCtrs) {
                const ui64 allRows = ui64(data.Rows) +
                    (trainingData.Test.empty() ? 0 : trainingData.Test.front()->GetObjectCount());
                CB_ENSURE(allRows <= Max<ui32>(), "Metal compound CTR table row count exceeds uint32 capacity");
                treeCtrFeatures = MakeHolder<TMetalTreeCtrFeatures>(*trainingData.Learn, options, executor,
                    data.StaticFeatureCount(), static_cast<ui32>(allRows), data.Categorical.CtrProvider, orderedHistories);
                data.Categorical.CtrProvider = treeCtrFeatures->GetCtrProvider();
            }
            const ui32 maxLeaves = greedy ? MetalGreedyLeafCapacity(greedyPolicy, depth,
                options.ObliviousTreeOptions->MaxLeaves) : 1u << depth;
            // Greedy runtimes cap their leaf banks by the reachable depth even
            // when the ordinary Lossguide snapshot reserves requested MaxLeaves.
            const ui32 modelBasedLeafCapacity = greedy ? Min(maxLeaves,
                greedyPolicy == 2 ? depth + 1 : (1u << Min(depth, 16u))) : maxLeaves;
            const ui32 greedyDepthBound = greedy ? Min(depth, maxLeaves - 1) : 0;
            // Normal training retains actual tree sizes. The checkpoint format
            // still stores padded per-tree buffers, so bound that allocation only
            // when snapshots are requested.
            if (outputOptions.SaveSnapshot() && !(modelBased && modelBased->LoadBaseline)) {
                const ui64 leavesCount = ui64(iterations) * maxLeaves;
                CB_ENSURE((greedy ? leavesCount * (52 + 4 * approxDimension) + ui64(iterations) * 24 :
                          leavesCount * 4 * (approxDimension + 1) + ui64(iterations) * depth * 9) +
                          ui64(data.Rows) * 4 * (2 * approxDimension + optimizerDimension) *
                          permutationCount + (!featureParallel ? ui64(iterations) * modelBasedLeafCapacity * 4 * approxDimension * permutationCount : 0)
                          <= (1ull << 29),
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
            if (modelBased) {
                if (modelBased->LoadBaseline) {
                    modelBased->PermutationCount = permutationCount;
                    modelBased->ApproxDimension = approxDimension;
                    modelBased->LeafCapacity = maxLeaves;
                    modelBased->Depth = depth;
                    modelBased->SignedWeights = greedySimple || signedSimpleWeights;
                    modelBased->InitialLearn = initialPredictions;
                    if (modelBased->InitialLearn.empty())
                        modelBased->InitialLearn.assign(ui64(data.Rows) * approxDimension, bias);
                    const auto& test = *trainingData.Test.front();
                    modelBased->InitialTest.resize(ui64(test.GetObjectCount()) * approxDimension);
                    const auto testBaseline = test.TargetData->GetBaseline();
                    for (ui32 row = 0; row < test.GetObjectCount(); ++row) {
                        for (ui32 dim = 0; dim < approxDimension; ++dim) {
                            const ui32 column = testBaseline && testBaseline->size() != approxDimension
                                ? baselineColumns[dim] : dim;
                            const double value = modelBias[dim] +
                                (testBaseline ? (*testBaseline)[column][row] : 0.0);
                            modelBased->InitialTest[ui64(row) * approxDimension + dim] = static_cast<float>(value);
                        }
                    }
                } else {
                    CB_ENSURE(modelBased->ApproxDimension == approxDimension &&
                        modelBased->InitialLearn.size() == ui64(modelBased->PermutationCount) * data.Rows * approxDimension,
                        "Metal model-based evaluation initial cursor dimensions differ");
                    const ui64 stride = ui64(data.Rows) * approxDimension;
                    initialPredictions.assign(modelBased->InitialLearn.end() - stride, modelBased->InitialLearn.end());
                }
            }
            TMetalSnapshot snapshot;
            snapshot.Greedy = greedy;
            snapshot.YetiRank = yeti && !featureParallel && !langevin;
            snapshot.CombinationYeti = combinationYetiCount && !featureParallel && !langevin;
            snapshot.Langevin = langevin;
            snapshot.TreeCtrs = compoundCtrs;
            TString snapshotPath;
            ui32 restoredIterations = 0;
            if (outputOptions.SaveSnapshot()) {
                NJson::TJsonValue jsonOptions;
                options.Save(&jsonOptions);
                snapshot.Params = ToString(jsonOptions);
                snapshot.Checksum = SnapshotDataChecksum(trainingData, initModel, executor);
                if (data.Estimated.GetFeatureCount()) snapshot.Checksum = UpdateCheckSum(snapshot.Checksum, data.Estimated.Checksum);
                if (custom) snapshot.Checksum = UpdateCheckSum(snapshot.Checksum, TStringBuf(customSource));
                if (featureParallel) {
                    snapshot.Checksum = UpdateCheckSum(snapshot.Checksum, permutationCount);
                    for (const auto& history : orderedHistories) snapshot.Checksum = UpdateCheckSum(snapshot.Checksum, history);
                }
                snapshot.Bias = bias;
                snapshotPath = outputOptions.CreateSnapshotFullPath();
                if (snapshot.Load(snapshotPath, trainingCallbacks)) {
                    if (greedy) snapshot.ValidateGreedy(data.Rows, data.FeatureCount(), greedyPolicy, depth, maxLeaves, modelBased && modelBased->LoadBaseline ? Max<ui32>() : iterations, permutationCount, approxDimension, optimizerDimension, greedySimple);
                    else snapshot.Validate(data.Rows, depth, modelBased && modelBased->LoadBaseline ? Max<ui32>() : iterations, approxDimension,
                                           permutationCount, optimizerDimension, ordered, featureParallel, signedSimpleWeights);
                    if (compoundCtrs) {
                        TStringInput input(snapshot.TreeCtrState);
                        restoredTreeCtrBatch = treeCtrFeatures->Restore(&input);
                        char trailing;
                        CB_ENSURE(input.Read(&trailing, 1) == 0, "Saved Metal tree CTR registry has trailing data");
                    }
                    restoredIterations = snapshot.Depths.size();
                    initialPredictions = snapshot.Predictions;
                    bias = snapshot.Bias;
                    if (!multioutput) std::fill(modelBias.begin(), modelBias.end(), bias);
                }
            }
            if (modelBased && modelBased->LoadBaseline) {
                CB_ENSURE(restoredIterations, "Metal model-based evaluation requires an existing nonempty baseline snapshot");
                snapshot.ModelBasedValidate(permutationCount, modelBasedLeafCapacity, approxDimension);
                CB_ENSURE(permutationCount == 1 || snapshot.ModelBasedPermutationCount,
                    "Metal model-based evaluation requires per-permutation baseline history; create a new baseline snapshot with this build");
                modelBased->BaselineSnapshot = std::move(snapshot);
                modelBased->BaselineData = std::move(data);
                return;
            }
            if (modelBased) {
                snapshot.PermutationPredictions = modelBased->InitialLearn;
                if (modelBased->PermutationCount == 1 && permutationCount > 1) {
                    for (ui32 p = 1; p < permutationCount; ++p)
                        snapshot.PermutationPredictions.insert(snapshot.PermutationPredictions.end(),
                            modelBased->InitialLearn.begin(), modelBased->InitialLearn.end());
                }
            }
            if (outputOptions.SaveSnapshot() && !featureParallel) {
                if (!restoredIterations) {
                    snapshot.ModelBasedPermutationCount = permutationCount;
                    snapshot.ModelBasedLeafCapacity = modelBasedLeafCapacity;
                    snapshot.ModelBasedApproxDimension = approxDimension;
                }
                snapshot.ModelBasedValidate(permutationCount, modelBasedLeafCapacity, approxDimension);
            }
            THolder<TMetalFullMatrixRsm> featureSampler;
            if (options.ObliviousTreeOptions->Rsm.Get() < 1.0) {
                auto metadata = MakeMetalStaticFeatureMetadata(data, *trainingData.Learn, options);
                const auto& bootstrap = options.ObliviousTreeOptions->BootstrapConfig.Get();
                featureSampler = MakeHolder<TMetalFullMatrixRsm>(options.RandomSeed,
                    options.ObliviousTreeOptions->Rsm, objective, bootstrap.GetBootstrapType(),
                    bootstrap.GetTakenFraction(), permutationCount,
                    options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple,
                    data.StaticFeatureCount(), std::move(metadata.RsmFeatures));
                featureSampler->Restore(restoredIterations);
            }
            const auto& regularization = options.ObliviousTreeOptions.Get();
            const bool metaActive = !ordered && !greedy && !vectorBackend && !coupled && !qce && !yetiPair &&
                (regularization.ScoreFunction == EScoreFunction::L2 || regularization.ScoreFunction == EScoreFunction::NewtonL2) &&
                static_cast<float>(regularization.MetaL2Exponent.Get()) != 1.0f && regularization.MetaL2Frequency > 0;
            const bool metaSeeded = metaActive && regularization.MetaL2Frequency <= 1;
            const ui32 metaDataSets = 1 + ui32(permutationCount > 1 && data.HasPermutationDependentFeatures());
            THolder<TMetalMetaL2Context> metaL2;
            THolder<TMetalMetaL2DocRandom> metaDocRandom;
            if (metaSeeded) {
                const auto metadata = MakeMetalStaticFeatureMetadata(data, *trainingData.Learn, options);
                metaL2 = MakeHolder<TMetalMetaL2Context>(regularization.MetaL2Exponent, regularization.MetaL2Frequency,
                    MakeMetalMetaL2StaticDataSets(metadata, permutationCount, data.HasPermutationDependentFeatures()));
                if (!featureParallel && !stochasticTarget && !langevin) {
                    metaDocRandom = MakeHolder<TMetalMetaL2DocRandom>(options.RandomSeed,
                        regularization.BootstrapConfig->GetBootstrapType() != EBootstrapType::No,
                        depth, data.CandidateFeatures.size(), metaDataSets);
                    metaDocRandom->Restore(snapshot.Depths);
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
            CBMOrderedParams orderedParams = {};
            if (ordered) {
                const auto objectiveOptions = MetalObjectiveOptions(options, objectiveId);
                const auto& boosting = options.BoostingOptions.Get();
                orderedParams = {
                    data.Rows, data.FeatureCount(), static_cast<ui32>(data.CandidateFeatures.size()),
                    iterations - restoredIterations, depth, objectiveId,
                    options.ObliviousTreeOptions->ScoreFunction == EScoreFunction::NewtonCosine ? 1u : 0u,
                    objectiveOptions.leaf_estimation_method, params.leaf_estimation_iterations,
                    permutationCount, boosting.MinFoldSize,
                    options.ObliviousTreeOptions->FoldSizeLossNormalization ? 1u : 0u,
                    params.train.learning_rate, params.train.l2_leaf_reg, bias,
                    static_cast<float>(boosting.FoldLenMultiplier), objectiveOptions.objective_param, 0, 0, 0};
            }
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
            THolder<TMetalTreeCtrSession> treeCtrSession;
            THolder<TMetalOrderedRandom> featureParallelRandom;
            THolder<TMetalFeatureParallelYetiRandom> featureParallelYetiRandom;
            TMaybe<TMetalOrderedStochasticShape> orderedStochasticShape;
            if (featureParallel && !ordered && !stochasticTarget && !langevin) {
                featureParallelRandom = MakeHolder<TMetalOrderedRandom>(options.RandomSeed, permutationCount,
                    depth, data.CandidateFeatures.size(), true);
                if (restoredIterations) featureParallelRandom->RestoreState({snapshot.OrderedRandomDrawCount,
                    snapshot.OrderedRandomCompletedIterations, snapshot.OrderedBootstrapInitialized}, restoredIterations);
            }
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
                    CB_ENSURE(snapshot.UsedFeatures.size() == data.StaticFeatureCount() +
                              (treeCtrFeatures ? treeCtrFeatures->GetFeatureCount() : 0),
                              "Metal snapshot feature-use state differs from the prepared features");
                    usedFeatures.assign(snapshot.UsedFeatures.begin(),
                        snapshot.UsedFeatures.begin() + data.StaticFeatureCount());
                }
            }
            if (restoredTreeCtrBatch) {
                ValidateMetalTreeCtrSnapshot(*restoredTreeCtrBatch, ctrUniqueValues, snapshot, ordered, featureWeights);
            }
            THolder<TMetalYetiRandom> yetiRandom;
            if (stochasticTarget && !featureParallel && !langevin) {
                yetiRandom = MakeHolder<TMetalYetiRandom>(options.RandomSeed,
                    options.ObliviousTreeOptions->BootstrapConfig->GetBootstrapType() != EBootstrapType::No,
                    params.leaf_estimation_iterations, depth, data.CandidateFeatures.size(), permutationCount,
                    greedy, greedy ? 2 * maxLeaves : 0, combinationYetiCount ? combinationYetiCount : 1,
                    combinationYetiCount != 0,
                    combinationYetiCount && options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple,
                    metaActive ? metaDataSets : 1);
                if (restoredIterations) yetiRandom->Restore(snapshot.YetiRandom, snapshot.Depths, snapshot.YetiSearchAttempts);
            }
            if (featureParallel && stochasticTarget && !ordered && !langevin) {
                const ui32 components = combinationYetiCount ? combinationYetiCount : 1;
                featureParallelYetiRandom = MakeHolder<TMetalFeatureParallelYetiRandom>(options.RandomSeed,
                    permutationCount, depth, data.CandidateFeatures.size(), params.leaf_estimation_iterations,
                    TVector<ui32>(permutationCount, components), permutationCount * components, combinationYetiCount != 0);
                if (restoredIterations) featureParallelYetiRandom->Restore({snapshot.OrderedRandomDrawCount,
                    snapshot.OrderedRandomCompletedIterations, snapshot.OrderedBootstrapInitialized}, restoredIterations);
            }
            if (ordered && stochasticTarget) {
                orderedStochasticShape = MakeMetalOrderedStochasticShape(orderedParams,
                    *trainingData.Learn->ObjectsGrouping, orderedHistories,
                    options.BoostingOptions->FoldLenMultiplier.Get(), combinationYetiCount ? combinationYetiCount : 1,
                    combinationYetiCount != 0);
                const auto& shape = *orderedStochasticShape;
                if (restoredIterations) CB_ENSURE(snapshot.OrderedDescriptors == shape.Descriptors &&
                    snapshot.OrderedCursors.size() == shape.CursorCount,
                    "Saved Metal Ordered stochastic prefix state differs from the prepared histories");
                if (!langevin) {
                    featureParallelYetiRandom = MakeHolder<TMetalFeatureParallelYetiRandom>(options.RandomSeed,
                        permutationCount, depth, data.CandidateFeatures.size(), params.leaf_estimation_iterations,
                        shape.WeakSeedCounts, shape.LeafTaskCount, combinationYetiCount != 0);
                    if (restoredIterations) featureParallelYetiRandom->Restore({snapshot.OrderedRandomDrawCount,
                        snapshot.OrderedRandomCompletedIterations, snapshot.OrderedBootstrapInitialized}, restoredIterations);
                }
            }
            THolder<TMetalLangevinRandom> langevinRandom;
            if (langevin) {
                langevinRandom = MakeHolder<TMetalLangevinRandom>(options.RandomSeed, featureParallel,
                    permutationCount, static_cast<float>(options.BoostingOptions->DiffusionTemperature.Get()),
                    params.train.learning_rate, executor);
                if (restoredIterations) {
                    const bool expectedCache = (!greedy && !vectorBackend) ||
                        regularization.BootstrapConfig->GetBootstrapType() != EBootstrapType::No;
                    CB_ENSURE(snapshot.LangevinRandom.WeakSeedCacheInitialized == expectedCache,
                        "Saved Metal Langevin seed-cache state differs from the configured training path");
                    // Bound replay by the validated saved feature registry and
                    // configured source walker geometry. Actual draws, including
                    // rejected trials, come from the snapshot's tagged state.
                    const ui64 attempts = Max<ui32>(params.leaf_estimation_iterations, 100);
                    const ui64 components = combinationYetiCount ? combinationYetiCount : ui32(yeti);
                    const ui64 weakCalls = orderedStochasticShape
                        ? *MaxElement(orderedStochasticShape->WeakSeedCounts.begin(), orderedStochasticShape->WeakSeedCounts.end())
                        : components;
                    const ui64 leafCalls = orderedStochasticShape ? orderedStochasticShape->LeafTaskCount
                        : ui64(permutationCount) * components;
                    const ui64 noiseWalkers = greedy || vectorBackend ? permutationCount : 1;
                    const ui64 scoreBatches = greedy || vectorBackend ? 2ull * maxLeaves : depth;
                    const ui64 scorePacks = compoundCtrs ? snapshot.TreeCtrCounts.size() + 2 : metaDataSets;
                    const long double perTree = 1.0L + weakCalls + (attempts + 1.0L) * leafCalls +
                        (2.0L + 2.0L * attempts) * noiseWalkers + static_cast<long double>(scoreBatches) * scorePacks;
                    const long double maximumDraws = 1.0L + TMetalLangevinRandom::WeakSeedCacheDrawCount +
                        restoredIterations * perTree;
                    CB_ENSURE(maximumDraws < static_cast<long double>(Max<ui64>()),
                        "Metal Langevin snapshot random bound overflows");
                    langevinRandom->RestoreState(snapshot.LangevinRandom, restoredIterations,
                        static_cast<ui64>(maximumDraws));
                }
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
                    if (objectiveId == 12 || objectiveId == 13 || yeti)
                        query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping, options.LossFunctionDescription.Get());
                    const auto yetiOptions = yeti ? PrepareMetalYetiRankOptions(options.LossFunctionDescription.Get(), query.Options.group_count)
                                                 : CBMYetiRankOptions{};
                    greedySession = MakeHolder<TMetalGreedySession>(greedyParams, objectiveOptions,
                        data.Bins, targets, sampleWeights, initialPredictions,
                        data.CandidateFeatures, data.CandidateBins, data.CandidateTypes,
                        objectiveId == 12 || objectiveId == 13 ? &query.Options : nullptr,
                        objectiveId == 14 ? pairs.GroupOffsets : query.Offsets,
                        objectiveId == 14 ? &pairs.Options : nullptr, pairs.Winners, pairs.Losers, pairs.Weights,
                        yeti ? &yetiOptions : nullptr);
                } else if (ordered) {
                    TMetalQueryData query;
                    TMetalPairData pairs;
                    if (objectiveId == 14) pairs = PrepareMetalPairData(*trainingData.Learn->TargetData, data.Rows);
                    else if (orderedQuery) query = PrepareMetalQueryData(*trainingData.Learn->ObjectsGrouping, options.LossFunctionDescription.Get());
                    const auto yetiOptions = yeti ? PrepareMetalYetiRankOptions(options.LossFunctionDescription.Get(), query.Options.group_count)
                                                 : CBMYetiRankOptions{};
                    const auto& boosting = options.BoostingOptions.Get();
                    orderedSession = MakeHolder<TMetalOrderedSession>(orderedParams, data.Bins, targets,
                        sampleWeights, initialPredictions, data.CandidateFeatures, data.CandidateBins,
                        options.RandomSeed, *trainingData.Learn->ObjectsData->GetObjectsGrouping(), orderedHistories,
                        data.CandidateTypes, boosting.FoldLenMultiplier.Get(), data.AdditionalPermutationBins,
                        incrementalFeatureParallel, orderedQuery && objectiveId != 14 ? &query : nullptr,
                        objectiveId == 14 ? &pairs : nullptr, yeti ? &yetiOptions : nullptr,
                        combination ? &*combinationData : nullptr, custom ? customSource.c_str() : nullptr);
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
                        options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple ? 3u :
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
                    if (custom) {
                        CB_ENSURE(cbm_session_create_custom(&params, &objectiveOptions, customSource.c_str(),
                            data.Bins.data(), targets.data(), sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(), data.CandidateFeatures.data(),
                            data.CandidateBins.data(), data.CandidateTypes.data(), &session.Handle, error, sizeof(error)) == 0,
                            "Metal custom objective initialization failed: " << error);
                    } else if (combination) {
                        const auto& combined = *combinationData;
                        CB_ENSURE(cbm_session_create_combination(&params, &objectiveOptions, &combined.Options,
                            combined.Components.data(), combined.GroupOffsets.empty() ? nullptr : combined.GroupOffsets.data(),
                            combined.Winners.empty() ? nullptr : combined.Winners.data(),
                            combined.Losers.empty() ? nullptr : combined.Losers.data(),
                            combined.PairWeights.empty() ? nullptr : combined.PairWeights.data(),
                            data.Bins.data(), targets.data(), sampleWeights.empty() ? nullptr : sampleWeights.data(),
                            initialPredictions.empty() ? nullptr : initialPredictions.data(), data.CandidateFeatures.data(),
                            data.CandidateBins.data(), data.CandidateTypes.data(), &session.Handle, error, sizeof(error)) == 0,
                            "Metal Combination initialization failed: " << error);
                    } else if (yeti) {
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
                if (!fixedBinarySplits.empty()) {
                    if (greedySession) greedySession->SetFixedSplits(fixedBinarySplits);
                    else CB_ENSURE(vectorBackend && greedy && cbm_multiclass_session_set_fixed_splits(
                        session.Handle, fixedBinarySplits.size(), fixedBinarySplits.data(), error, sizeof(error)) == 0,
                        "Metal vector fixed split configuration failed: " << error);
                }
                if (greedySession) {
                    greedySession->SetFeatureWeights(featureWeights);
                } else if (greedy && vectorBackend) {
                    CB_ENSURE(cbm_multiclass_session_set_greedy_feature_weights(session.Handle,
                        featureWeights.size(), featureWeights.data(), error, sizeof(error)) == 0,
                        "Metal vector feature weight configuration failed: " << error);
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
                    // Scalar greedy ridge is configured separately from its
                    // source-ignored normalization and meta-L2 options.
                    greedySession->SetAddRidgeToTargetFunction(regularization.AddRidgeToTargetFunctionFlag);
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
                    CB_ENSURE(cbm_ordered_set_add_ridge_to_target_function(orderedSession->GetHandle(),
                        regularization.AddRidgeToTargetFunctionFlag ? 1u : 0u, error, sizeof(error)) == 0,
                        "Metal Ordered ridge configuration failed: " << error);
                    orderedSession->SetBacktracking(params.leaf_estimation_backtracking);
                    orderedSession->SetBootstrap(bootstrapOptions,
                        options.ObliviousTreeOptions->ObservationsToBootstrap == EObservationsToBootstrap::TestOnly);
                    orderedSession->SetScoreNoise(noiseOptions);
                    if (data.Categorical.CtrProvider || !options.ObliviousTreeOptions->FeaturePenalties->FeatureWeights.Get().empty())
                        orderedSession->SetFeaturePenalties(ctrUniqueValues, options.ObliviousTreeOptions->ModelSizeReg, featureWeights);
                    if (restoredIterations && !compoundCtrs) {
                        orderedSession->RestoreState({snapshot.OrderedDescriptors, snapshot.OrderedCursors});
                        if (!stochasticTarget && !langevin) orderedSession->RestoreRandomState({snapshot.OrderedRandomDrawCount,
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
                            ctrUniqueValues.data(), featureWeights.data(), usedFeatures.data(), error, sizeof(error)) == 0,
                            "Metal CTR model-size penalty configuration failed: " << error);
                    }
                    TVector<TConstArrayRef<ui8>> permutationBins{MakeConstArrayRef(data.Bins)};
                    for (const auto& bins : data.AdditionalPermutationBins) permutationBins.push_back(MakeConstArrayRef(bins));
                    if (featureParallel && permutationBins.size() == 1) permutationBins.resize(permutationCount, permutationBins.front());
                    permutations = MakeHolder<TMetalDocParallelPermutations>(session.Handle, data.Rows, data.FeatureCount(),
                        options.RandomSeed, permutationBins, snapshot.PermutationPredictions,
                        snapshot.PermutationMvsLambdas, snapshot.PermutationMvsValid, approxDimension);
                    if (vectorBackend && restoredIterations) {
                        CB_ENSURE(cbm_multiclass_session_restore_optimization_state(session.Handle,
                            data.GetPermutationCount(), snapshot.OptimizationPredictions.data(),
                            error, sizeof(error)) == 0, "Metal multiclass optimizer cursor restoration failed: " << error);
                    }
                }
                if (compoundCtrs) {
                    treeCtrSession = MakeHolder<TMetalTreeCtrSession>(orderedSession ? orderedSession->GetHandle() : session.Handle,
                        ordered, *treeCtrFeatures, data.StaticFeatureCount(), data.BinsPerFeature,
                        data.HasPermutationDependentFeatures());
                    if (restoredTreeCtrBatch) treeCtrSession->Restore(*restoredTreeCtrBatch, snapshot);
                    if (ordered && restoredIterations) {
                        orderedSession->RestoreState({snapshot.OrderedDescriptors, snapshot.OrderedCursors});
                        if (!stochasticTarget && !langevin) orderedSession->RestoreRandomState({snapshot.OrderedRandomDrawCount,
                            snapshot.OrderedRandomCompletedIterations, snapshot.OrderedBootstrapInitialized}, restoredIterations);
                    }
                }
                if (featureParallel && !ordered && !compoundCtrs) {
                    TVector<ui8> activity(data.StaticFeatureCount(), 1);
                    CB_ENSURE(cbm_session_set_feature_activity(session.Handle, activity.size(), activity.data(), error, sizeof(error)) == 0,
                        "Metal FeatureParallel static feature setup failed: " << error);
                }
                if (!ordered && !greedy && !vectorBackend &&
                    (regularization.FoldSizeLossNormalization || regularization.AddRidgeToTargetFunctionFlag || metaActive)) {
                    const bool matrix = coupled || qce || yetiPair;
                    CBMRegularizationOptions config = {};
                    config.normalize_score = !matrix && regularization.FoldSizeLossNormalization;
                    config.normalize_leaf = !matrix && regularization.FoldSizeLossNormalization;
                    config.add_ridge = regularization.AddRidgeToTargetFunctionFlag;
                    config.meta_l2_exponent = metaActive ? static_cast<float>(regularization.MetaL2Exponent.Get()) : 1.0f;
                    config.meta_l2_frequency = metaActive ? regularization.MetaL2Frequency.Get() : 0.0;
                    CB_ENSURE(cbm_session_set_regularization(session.Handle, &config, error, sizeof(error)) == 0,
                        "Metal regularization configuration failed: " << error);
                    if (metaL2) {
                        if (langevinRandom) metaL2->SetSeedProvider([&](ui32, ui32 count) {
                            TVector<ui64> seeds(count);
                            for (auto& seed : seeds) seed = langevinRandom->NextSeed(CBM_LANGEVIN_SEARCH);
                            return seeds;
                        });
                        else if (featureParallelYetiRandom) metaL2->SetSeedProvider([&](ui32 offset, ui32 count) {
                            return featureParallelYetiRandom->PeekScoreSeeds(offset, count);
                        });
                        else if (featureParallelRandom) metaL2->SetSeedProvider([&](ui32 offset, ui32 count) {
                            return featureParallelRandom->PeekScoreSeeds(offset, count);
                        });
                        else if (yetiRandom) metaL2->SetSeedProvider([&](ui32 offset, ui32 count) {
                            return yetiRandom->PeekScoreSeeds(offset, count);
                        });
                        else metaL2->SetSeedProvider([&](ui32 offset, ui32 count) {
                            return metaDocRandom->ScoreSeeds(offset, count);
                        });
                        if (treeCtrFeatures) metaL2->SetDynamicProvider([&]() {
                            TVector<TMetalMetaL2DataSet> datasets;
                            for (const auto& pack : treeCtrFeatures->GetActiveScoringPacks()) {
                                TMetalMetaL2DataSet dataset;
                                dataset.ScoreSeed = pack.BaseTensorHash;
                                dataset.PolicyMask = pack.PolicyMask;
                                for (const auto& feature : pack.Features)
                                    dataset.Features.push_back({feature.AbsoluteFeature, EMetalMetaL2Policy(feature.Policy)});
                                datasets.push_back(std::move(dataset));
                            }
                            return datasets;
                        });
                        CB_ENSURE(cbm_session_set_meta_l2_exponent_callback(session.Handle,
                            TMetalMetaL2Context::Callback, metaL2.Get(), error, sizeof(error)) == 0,
                            "Metal meta-L2 callback configuration failed: " << error);
                    }
                }
                if (langevinRandom) {
                    const float temperature = options.BoostingOptions->DiffusionTemperature;
                    const auto noise = TMetalLangevinRandom::NoiseCallback;
                    const auto seed = TMetalLangevinRandom::SeedCallback;
                    int status = 0;
                    if (orderedSession) status = cbm_ordered_session_set_langevin(orderedSession->GetHandle(),
                        temperature, noise, seed, langevinRandom.Get(), error, sizeof(error));
                    else if (greedySession) status = cbm_greedy_session_set_langevin(greedySession->GetHandle(),
                        temperature, noise, seed, langevinRandom.Get(), error, sizeof(error));
                    else if (vectorBackend) status = cbm_multiclass_session_set_langevin(session.Handle,
                        temperature, noise, seed, langevinRandom.Get(), error, sizeof(error));
                    else status = cbm_session_set_langevin(session.Handle, temperature, featureParallel ? 0u : 1u,
                        noise, seed, langevinRandom.Get(), error, sizeof(error));
                    CB_ENSURE(status == 0, "Metal Langevin configuration failed: " << error);
                }
                if (ordered && stochasticTarget) {
                    const auto shape = orderedSession->GetYetiSeedShape();
                    CB_ENSURE(shape.first == orderedStochasticShape->WeakSeedCounts &&
                        shape.second == orderedStochasticShape->LeafTaskCount,
                        "Metal Ordered runtime stochastic shape differs from its host geometry");
                }
            }
            // The constructors consume only the original static matrices.
            // Published trees resolve appended IDs through the restored/live
            // registry, including a snapshot that needs no further iterations.
            data.TreeCtrs = treeCtrFeatures.Get();
            TFullModel initialBiasModel;
            const TFullModel* progressInitialModel = initModel ? *initModel : nullptr;
            if (multioutput && options.BoostingOptions->BoostFromAverage) {
                initialBiasModel = MetalMultiOutputBiasModel(modelBias);
                progressInitialModel = &initialBiasModel;
            }
            TMetalTrainingProgress progress(options, outputOptions, trainingData, bias, executor,
                progressInitialModel,
                initModel ? &initModelApplyCompatiblePools : nullptr,
                internalOptions.ForceCalcEvalMetricOnEveryIteration, evalMetricDescriptor, approxDimension, baselineColumns,
                modelBased ? MakeConstArrayRef(initialPredictions) : TConstArrayRef<float>(),
                modelBased ? MakeConstArrayRef(modelBased->InitialTest) : TConstArrayRef<float>());
            TObliviousTreeBuilder builder(data.AllFloatFeatures, data.Categorical.AllCatFeatures,
            data.Estimated.AllTextFeatures, data.Estimated.AllEmbeddingFeatures, approxDimension);
            TNonSymmetricTreeModelBuilder greedyBuilder(data.AllFloatFeatures, data.Categorical.AllCatFeatures,
            data.Estimated.AllTextFeatures, data.Estimated.AllEmbeddingFeatures, approxDimension);
            auto* modelBuilder = internalOptions.CalcMetricsOnly ? nullptr : &builder;
            auto* greedyModelBuilder = internalOptions.CalcMetricsOnly ? nullptr : &greedyBuilder;
            CBMStepInfo info = {};
            CB_ENSURE(cbm_device_info(info.stats.device_name, sizeof(info.stats.device_name), error, sizeof(error)) == 0,
                      "Metal device is unavailable: " << error);
            bool continueTraining = true;
            for (ui32 tree = 0; tree < restoredIterations; ++tree) {
                auto singleTree = greedy ? AppendMetalGreedyTree(data, snapshot.GreedyTrees.GetTree(tree, approxDimension),
                    greedyDepthBound, greedyModelBuilder, approxDimension, greedySimple) : AppendMetalTree(data, snapshot.Depths[tree], depth,
                    MakeConstArrayRef(snapshot.SplitFeatures).Slice(ui64(tree) * depth, depth),
                    MakeConstArrayRef(snapshot.SplitBins).Slice(ui64(tree) * depth, depth),
                    MakeConstArrayRef(snapshot.SplitTypes).Slice(ui64(tree) * depth, depth),
                    MakeConstArrayRef(snapshot.Leaves).Slice(ui64(tree) * maxLeaves * approxDimension, ui64(maxLeaves) * approxDimension),
                    MakeConstArrayRef(snapshot.Weights).Slice(ui64(tree) * maxLeaves, maxLeaves), modelBuilder, approxDimension, signedSimpleWeights);
                continueTraining = progress.ReplayIteration(tree, singleTree, snapshot.History);
            }
            if (restoredIterations) {
                progress.RestoreTimeHistory(snapshot.History.TimeHistory);
                progress.RestoreLearnCursor(snapshot.Predictions);
                progress.RestoreBestLearnCursor(snapshot.BestLearnPredictions, snapshot.BestLearnIteration);
            }
            THPTimer snapshotTimer;
            for (ui32 tree = restoredIterations; tree < iterations && continueTraining; ++tree) {
                CheckInterrupted();
                progress.StartIteration();
                ui32 treeDepth = 0;
                TMetalGreedyTree greedyTree;
                const ui64 absoluteIteration = (initModel ? (*initModel)->GetTreeCount() : 0) + tree;
                ui32 yetiSearchAttempts = 0;
                if (langevinRandom && !featureParallel) langevinRandom->BeginIteration();
                if (langevinRandom && combinationYetiCount && !orderedSession) {
                    CB_ENSURE(cbm_session_set_combination_yeti_seed_callback(session.Handle,
                        MetalLangevinTargetSeed<CBM_LANGEVIN_YETI_WEAK>, langevinRandom.Get(), error, sizeof(error)) == 0,
                        "Metal Langevin Combination weak callback setup failed: " << error);
                }
                if (metaL2) metaL2->BeginTree();
                if (metaDocRandom) metaDocRandom->Begin();
                if (featureSampler) {
                    const auto mask = featureSampler->NextMask();
                    CB_ENSURE(cbm_session_set_feature_sampling_mask(session.Handle, mask.size(), mask.data(),
                        error, sizeof(error)) == 0, "Metal feature sampling failed: " << error);
                }
                if (incrementalFeatureParallel) {
                    auto begin = [&](ui32) {
                        if (treeCtrSession) treeCtrSession->BeginTree();
                        if (featureParallelYetiRandom) {
                            const auto seeds = featureParallelYetiRandom->WeakSeeds();
                            if (orderedSession) orderedSession->SetYetiWeakSeeds(seeds);
                            else if (combinationYetiCount) CB_ENSURE(cbm_session_set_combination_yeti_seeds(session.Handle,
                                seeds.size(), seeds.data(), error, sizeof(error)) == 0,
                                "Metal Combination weak seed setup failed: " << error);
                            else CB_ENSURE(cbm_session_set_yeti_oracle_seeds(session.Handle, seeds.size(), seeds.data(), error, sizeof(error)) == 0,
                                "Metal FeatureParallel YetiRank weak seed setup failed: " << error);
                        }
                    };
                    auto selectedSplit = [&](ui32 permutation, const CBMStructureInfo& structure) {
                        if (treeCtrSession) treeCtrSession->Selected(data.GetSplit(structure.feature, structure.bin, structure.type),
                            structure, permutation);
                    };
                    auto scoreDraws = [&]() { return treeCtrSession ? treeCtrSession->ScoreDraws() :
                        1u + (permutationCount > 1 && data.HasPermutationDependentFeatures()); };
                    auto beforeFinish = [&](ui32 searchDrawCount) {
                        if (metaL2) CB_ENSURE(metaL2->GetDrawCount() == searchDrawCount,
                            "Metal meta-L2 feature packs disagree with FeatureParallel scorer draws: " << metaL2->GetError());
                        if (langevinRandom) {
                            const ui32 consumed = metaL2 ? metaL2->GetDrawCount() : 0;
                            CB_ENSURE(consumed <= searchDrawCount, "Metal Langevin meta-L2 scorer count overflows");
                            langevinRandom->AdvanceSearch(searchDrawCount - consumed);
                            if (combinationYetiCount && !orderedSession) CB_ENSURE(
                                cbm_session_set_combination_yeti_seed_callback(session.Handle,
                                    MetalLangevinTargetSeed<CBM_LANGEVIN_YETI_LEAF>, langevinRandom.Get(), error, sizeof(error)) == 0,
                                "Metal Langevin Combination leaf callback setup failed: " << error);
                        }
                        if (featureParallelYetiRandom) {
                            if (combinationYetiCount) {
                                featureParallelYetiRandom->BeginLeafCalls(searchDrawCount);
                                const auto setter = orderedSession ? cbm_ordered_session_set_combination_yeti_seed_callback :
                                    cbm_session_set_combination_yeti_seed_callback;
                                CB_ENSURE(setter(orderedSession ? orderedSession->GetHandle() : session.Handle,
                                    MetalCombinationLeafSeed<TMetalFeatureParallelYetiRandom>, featureParallelYetiRandom.Get(),
                                    error, sizeof(error)) == 0, "Metal Combination leaf seed callback setup failed: " << error);
                            } else {
                                const auto seeds = featureParallelYetiRandom->LeafSeeds(searchDrawCount);
                                if (orderedSession) orderedSession->SetYetiLeafSeeds(seeds);
                                else CB_ENSURE(cbm_session_set_yeti_leaf_seeds(session.Handle, seeds.size(), seeds.data(), error, sizeof(error)) == 0,
                                    "Metal FeatureParallel YetiRank leaf seed setup failed: " << error);
                            }
                        }
                    };
                    if (orderedSession) {
                        if (langevinRandom) orderedSession->StepDynamicSelected(absoluteIteration,
                            langevinRandom->SelectPermutation(absoluteIteration), &info, &treeDepth,
                            splitFeatures, splitBins, splitTypes, leaves, weights, begin, selectedSplit, scoreDraws, beforeFinish);
                        else if (featureParallelYetiRandom) orderedSession->StepDynamicSelected(absoluteIteration,
                            featureParallelYetiRandom->SelectPermutation(), &info, &treeDepth,
                            splitFeatures, splitBins, splitTypes, leaves, weights, begin, selectedSplit, scoreDraws, beforeFinish);
                        else orderedSession->StepDynamic(absoluteIteration, &info, &treeDepth,
                            splitFeatures, splitBins, splitTypes, leaves, weights, begin, selectedSplit, scoreDraws);
                    } else {
                        const ui32 selected = langevinRandom ? langevinRandom->SelectPermutation(absoluteIteration) :
                            featureParallelYetiRandom ? featureParallelYetiRandom->SelectPermutation() :
                            featureParallelRandom->SelectPermutation();
                        begin(selected);
                        CB_ENSURE(cbm_session_select_permutation(session.Handle, selected, error, sizeof(error)) == 0 &&
                            cbm_session_begin_tree(session.Handle, error, sizeof(error)) == 0,
                            "Metal FeatureParallel tree initialization failed: " << error);
                        CBMStructureInfo structure = {};
                        ui32 searchDrawCount = 0;
                        do {
                            if (depth && !data.CandidateFeatures.empty()) searchDrawCount += scoreDraws();
                            CB_ENSURE(cbm_session_grow_tree(session.Handle, &structure, error, sizeof(error)) == 0,
                                "Metal FeatureParallel split search failed: " << error);
                            if (structure.has_split) selectedSplit(selected, structure);
                        } while (!structure.finished);
                        beforeFinish(searchDrawCount);
                        CB_ENSURE(cbm_session_finish_tree(session.Handle, &info, &treeDepth,
                            splitFeatures.data(), splitBins.data(), splitTypes.data(), leaves.data(), weights.data(),
                            error, sizeof(error)) == 0, "Metal FeatureParallel leaf estimation failed: " << error);
                        if (featureParallelRandom) featureParallelRandom->FinishIterationWithDraws(treeDepth, searchDrawCount);
                    }
                    if (featureParallelYetiRandom) {
                        if (combinationYetiCount) {
                            const auto setter = orderedSession ? cbm_ordered_session_set_combination_yeti_seed_callback :
                                cbm_session_set_combination_yeti_seed_callback;
                            CB_ENSURE(setter(orderedSession ? orderedSession->GetHandle() : session.Handle, nullptr, nullptr,
                                error, sizeof(error)) == 0, "Metal Combination leaf seed callback reset failed: " << error);
                        }
                        featureParallelYetiRandom->Complete();
                    }
                } else if (greedy && vectorBackend) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration + (modelBased ? modelBased->PermutationOffset : 0));
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
                        data.FeatureCount(), greedyDepthBound, approxDimension, greedySimple).Depth;
                } else if (greedySession) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration + (modelBased ? modelBased->PermutationOffset : 0));
                    if (yetiRandom) {
                        yetiSearchAttempts = greedySession->PrepareYetiTree(yetiRandom->Begin());
                        greedySession->SetYetiLeafSeeds(yetiRandom->LeafSeeds(yetiSearchAttempts));
                    }
                    greedyTree = greedySession->Step(absoluteIteration);
                    if (yetiRandom) yetiRandom->Complete();
                    info.completed_iterations = greedyTree.Info.completed_iterations;
                    info.finished = greedyTree.Info.finished;
                    info.stats = greedyTree.Info.stats;
                    treeDepth = ValidateMetalGreedyTree(greedyTree.Nodes, greedyTree.Values, greedyTree.Weights,
                        data.FeatureCount(), greedyDepthBound, 1, greedySimple).Depth;
                } else if (orderedSession) {
                    orderedSession->Step(absoluteIteration, &info, &treeDepth,
                        splitFeatures, splitBins, splitTypes, leaves, weights);
                } else if (langevinRandom && !vectorBackend) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration + (modelBased ? modelBased->PermutationOffset : 0));
                    CB_ENSURE(cbm_session_begin_tree(session.Handle, error, sizeof(error)) == 0,
                        "Metal Langevin tree initialization failed: " << error);
                    CBMStructureInfo structure = {};
                    ui32 searchDrawCount = 0;
                    do {
                        CB_ENSURE(cbm_session_grow_tree(session.Handle, &structure, error, sizeof(error)) == 0,
                            "Metal Langevin split search failed: " << error);
                        if (depth && !data.CandidateFeatures.empty()) searchDrawCount += metaDataSets;
                    } while (!structure.finished);
                    const ui32 consumed = metaL2 ? metaL2->GetDrawCount() : 0;
                    CB_ENSURE(consumed <= searchDrawCount, "Metal Langevin meta-L2 scorer count overflows");
                    langevinRandom->AdvanceSearch(searchDrawCount - consumed);
                    if (combinationYetiCount) CB_ENSURE(cbm_session_set_combination_yeti_seed_callback(session.Handle,
                        MetalLangevinTargetSeed<CBM_LANGEVIN_YETI_LEAF>, langevinRandom.Get(), error, sizeof(error)) == 0,
                        "Metal Langevin Combination leaf callback setup failed: " << error);
                    CB_ENSURE(cbm_session_finish_tree(session.Handle, &info, &treeDepth,
                        splitFeatures.data(), splitBins.data(), splitTypes.data(), leaves.data(), weights.data(),
                        error, sizeof(error)) == 0, "Metal Langevin leaf estimation failed: " << error);
                } else if (yetiRandom) {
                    if (permutations) permutations->SelectForIteration(absoluteIteration + (modelBased ? modelBased->PermutationOffset : 0));
                    const auto weakSeeds = yetiRandom->BeginSeeds();
                    const auto weakSetter = combinationYetiCount ? cbm_session_set_combination_yeti_seeds : cbm_session_set_yeti_oracle_seeds;
                    CB_ENSURE(weakSetter(session.Handle, weakSeeds.size(), weakSeeds.data(), error, sizeof(error)) == 0 &&
                        cbm_session_begin_tree(session.Handle, error, sizeof(error)) == 0,
                        "Metal YetiRank weak target failed: " << error);
                    CBMStructureInfo structure = {};
                    ui32 attempts = 0;
                    do {
                        CB_ENSURE(cbm_session_grow_tree(session.Handle, &structure, error, sizeof(error)) == 0,
                            "Metal YetiRank split search failed: " << error);
                        if (depth && !data.CandidateFeatures.empty()) ++attempts;
                    } while (!structure.finished);
                    if (combinationYetiCount) {
                        yetiRandom->BeginLeafCalls(attempts);
                        CB_ENSURE(cbm_session_set_combination_yeti_seed_callback(session.Handle,
                            MetalCombinationLeafSeed<TMetalYetiRandom>, yetiRandom.Get(), error, sizeof(error)) == 0,
                            "Metal Combination leaf seed callback setup failed: " << error);
                    } else {
                        const auto seeds = yetiRandom->LeafSeeds(attempts);
                        CB_ENSURE(cbm_session_set_yeti_leaf_seeds(session.Handle, seeds.size(), seeds.data(), error, sizeof(error)) == 0,
                            "Metal YetiRank leaf seed setup failed: " << error);
                    }
                    CB_ENSURE(cbm_session_finish_tree(session.Handle, &info, &treeDepth, splitFeatures.data(),
                            splitBins.data(), splitTypes.data(), leaves.data(), weights.data(), error, sizeof(error)) == 0,
                        "Metal YetiRank leaf estimation failed: " << error);
                    if (combinationYetiCount) CB_ENSURE(cbm_session_set_combination_yeti_seed_callback(session.Handle,
                        nullptr, nullptr, error, sizeof(error)) == 0, "Metal Combination leaf seed callback reset failed: " << error);
                    yetiRandom->Complete();
                } else {
                    if (permutations) permutations->SelectForIteration(absoluteIteration + (modelBased ? modelBased->PermutationOffset : 0));
                    const auto step = vectorBackend ? cbm_multiclass_session_step : cbm_session_step;
                    CB_ENSURE(step(session.Handle, &info, &treeDepth, splitFeatures.data(),
                        splitBins.data(), splitTypes.data(), leaves.data(), weights.data(), error, sizeof(error)) == 0,
                        "Metal training iteration " << tree << " failed: " << error);
                }
                if (langevinRandom) langevinRandom->FinishIteration();
                if (metaDocRandom) metaDocRandom->Finish(treeDepth);
                if (metaL2 && !featureParallel) CB_ENSURE(metaL2->GetDrawCount() ==
                    (depth && !data.CandidateFeatures.empty() ? Min(treeDepth + 1, depth) * metaDataSets : 0),
                    "Metal meta-L2 DocParallel scorer count differs from the completed tree: " << metaL2->GetError());
                auto singleTreeModel = greedy ? AppendMetalGreedyTree(data, greedyTree, greedyDepthBound, greedyModelBuilder, approxDimension, greedySimple)
                    : AppendMetalTree(data, treeDepth, depth, splitFeatures,
                    splitBins, splitTypes, leaves, weights, modelBuilder, approxDimension, signedSimpleWeights);
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
                    if (snapshot.ModelBasedPermutationCount) {
                        const ui64 size = ui64(permutationCount) * modelBasedLeafCapacity * approxDimension;
                        const ui64 begin = snapshot.ModelBasedPermutationLeaves.size();
                        snapshot.ModelBasedPermutationLeaves.resize(begin + size);
                        const auto copy = vectorBackend ? cbm_multiclass_session_copy_last_permutation_leaves :
                            greedy ? cbm_greedy_session_copy_last_permutation_leaves : cbm_session_copy_last_permutation_leaves;
                        CB_ENSURE(copy(greedySession ? greedySession->GetHandle() : session.Handle,
                            permutationCount, modelBasedLeafCapacity, snapshot.ModelBasedPermutationLeaves.data() + begin,
                            error, sizeof(error)) == 0, "Metal baseline permutation leaf export failed: " << error);
                    }
                    if (greedy && yeti && !langevin) snapshot.YetiSearchAttempts.push_back(yetiSearchAttempts);
                    if (greedy) {
                        snapshot.GreedyTrees.Append(greedyTree, data.FeatureCount(), greedyDepthBound, approxDimension, greedySimple);
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
                        snapshot.BestLearnPredictions = progress.CopyBestLearnCursor();
                        snapshot.BestLearnIteration = progress.GetBestLearnIteration();
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
                            if (!langevin) {
                                const auto randomState = orderedSession->GetRandomState();
                                snapshot.OrderedRandomDrawCount = randomState.DrawCount;
                                snapshot.OrderedRandomCompletedIterations = randomState.CompletedIterations;
                                snapshot.OrderedBootstrapInitialized = randomState.BootstrapInitialized;
                            }
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
                                    data.GetPermutationCount());
                                CB_ENSURE(cbm_multiclass_session_copy_optimization_state(session.Handle,
                                    data.GetPermutationCount(), snapshot.OptimizationPredictions.data(),
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
                        if (treeCtrSession) treeCtrSession->Save(&snapshot);
                        if (featureParallelRandom) {
                            const auto randomState = featureParallelRandom->GetState();
                            snapshot.OrderedRandomDrawCount = randomState.DrawCount;
                            snapshot.OrderedRandomCompletedIterations = randomState.CompletedIterations;
                            snapshot.OrderedBootstrapInitialized = randomState.BootstrapInitialized;
                        }
                        if (featureParallelYetiRandom) {
                            const auto randomState = featureParallelYetiRandom->GetState();
                            snapshot.OrderedRandomDrawCount = randomState.DrawCount;
                            snapshot.OrderedRandomCompletedIterations = randomState.CompletedIterations;
                            snapshot.OrderedBootstrapInitialized = randomState.BootstrapInitialized;
                        }
                        if (yetiRandom) snapshot.YetiRandom = yetiRandom->GetState();
                        if (langevinRandom) snapshot.LangevinRandom = langevinRandom->GetState();
                        snapshot.History = history;
                        snapshot.Save(snapshotPath, info.stats.device_name, trainingCallbacks);
                        snapshotTimer.Reset();
                    }
                }
            }
            const auto& history = progress.GetHistory();
            if (internalOptions.CalcMetricsOnly) {
                progress.Finish(nullptr, evalResultPtrs);
                if (metricsAndTimeHistory) {
                    *metricsAndTimeHistory = history;
                    metricsAndTimeHistory->MetalResumedIterations = restoredIterations;
                    metricsAndTimeHistory->MetalKernelDispatches = info.stats.kernel_dispatches;
                    metricsAndTimeHistory->MetalGpuSeconds = info.stats.gpu_seconds;
                }
                return;
            }
            TFullModel model;
            if (greedy) greedyBuilder.Build(model.ModelTrees.GetMutable());
            else builder.Build(model.ModelTrees.GetMutable());
            model.SetScaleAndBias({1.0, modelBias});
            if (featureSampler) {
                model.ModelInfo["metal_rsm_rng"] = "cuda_single_device_host_shadow_v1";
                model.ModelInfo["metal_rsm_host_draw_count"] = ToString(featureSampler->GetDrawCount());
            }
            if (yeti) model.ModelInfo["metal_yeti_centering"] = "all_rows";
            if (yetiPair) model.ModelInfo["metal_yeti_pair_rng"] = permutationCount > 1
                ? "item_iteration_dataset_domains_v2" : "item_iteration_domains_v1";
            if (data.Categorical.CtrProvider) model.CtrProvider = data.Categorical.CtrProvider->Clone();
            FinalizeMetalCounterTables(data.Categorical, &model);
            model.UpdateDynamicData();
            if (classificationTargetHelper.IsInitialized()) {
                model.ModelInfo["class_params"] = classificationTargetHelper.Serialize();
            }
            if (initModel) {
                model = SumModels({*initModel, &model}, {1.0, 1.0}, {"initModel:", ""});
            }
            progress.Finish(&model, evalResultPtrs);
            if (metricsAndTimeHistory) {
                *metricsAndTimeHistory = history;
                metricsAndTimeHistory->MetalResumedIterations = restoredIterations;
                metricsAndTimeHistory->MetalKernelDispatches = info.stats.kernel_dispatches;
                metricsAndTimeHistory->MetalGpuSeconds = info.stats.gpu_seconds;
            }
            if (model.CtrProvider) {
                model.CtrProvider->DropUnusedTables(model.ModelTrees->GetApplyData()->GetUsedModelCtrBases());
            }
            if (langevinRandom) {
                model.ModelInfo["metal_langevin_host_rng"] = "cuda_shared_mt19937_and_128_element_leaf_blocks_v1";
                model.ModelInfo["metal_langevin_weak_rng"] = ordered || (!featureParallel && !greedy && !vectorBackend)
                    ? "metal_item_iteration_domains_v1" : "no_weak_noise";
                model.ModelInfo["metal_langevin_host_draw_count"] = ToString(langevinRandom->GetState().DrawCount);
            }
            model.ModelInfo["metal_backend"] = "METAL";
            model.ModelInfo["metal_device"] = info.stats.device_name;
            model.ModelInfo["metal_permutations"] = ToString(permutationCount);
            model.ModelInfo["metal_port"] = ordered ? "Native CUDA Ordered/FeatureParallel translation" :
                featureParallel ? "Native CUDA Plain/FeatureParallel translation" : "Native CUDA Plain/DocParallel translation";
            if (treeCtrFeatures) model.ModelInfo["metal_tree_ctr_features"] = ToString(treeCtrFeatures->GetFeatureCount());
            TCoreModelToFullModelConverter converter(options, outputOptions, classificationTargetHelper,
                0, false, EFinalCtrComputationMode::Skip, outputOptions.GetFinalFeatureCalcerComputationMode());
            converter.WithCoreModelFrom(&model).WithObjectsDataFrom(trainingData.Learn->ObjectsData)
                .WithFeatureEstimators(trainingData.FeatureEstimators).WithMetrics(history);
            if (dstModel) {
                converter.Do(true, dstModel, executor, nullptr);
            } else {
                converter.Do(outputOptions.CreateResultModelFullPath(), outputOptions.GetModelFormats(),
                             outputOptions.AddFileFormatExtension(), executor, nullptr);
            }
        }

        void ModelBasedEval(const NCatboostOptions::TCatBoostOptions& catboostOptions,
                            const NCatboostOptions::TOutputFilesOptions& outputOptions,
                            TTrainingDataProviders trainingData,
                            const TLabelConverter& labelConverter,
                            NPar::ILocalExecutor* executor) const override {
            CB_ENSURE(trainingData.Test.size() == 1 && trainingData.Test.front()->GetObjectCount(),
                "Metal model-based evaluation requires exactly one nonempty eval set");
            CB_ENSURE(outputOptions.AllowWriteFiles(), "Metal model-based evaluation requires allow_writing_files=true");
            auto options = catboostOptions;
            SetMetalDefaultsAndValidate(&options);
            const auto objective = options.LossFunctionDescription->GetLossFunction();
            CB_ENSURE(!IsMultiTargetObjective(objective), "Metal model-based evaluation does not support multitarget objectives, like CUDA");
            CB_ENSURE(objective != ELossFunction::PythonUserDefinedPerObject,
                "Metal model-based evaluation does not accept custom objective descriptors");
            CB_ENSURE(options.BoostingOptions->BoostingType == EBoostingType::Plain &&
                options.BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel,
                "Metal model-based evaluation supports Plain DocParallel training only");
            const auto& layout = *trainingData.Learn->ObjectsData->GetFeaturesLayout();
            for (const auto& feature : layout.GetExternalFeaturesMetaInfo()) {
                CB_ENSURE(!feature.IsAvailable || (feature.Type != EFeatureType::Text && feature.Type != EFeatureType::Embedding),
                    "Metal model-based evaluation does not support text or embedding estimators, like CUDA");
            }
            const auto& config = options.ModelBasedEvalOptions.Get();
            CB_ENSURE(config.Offset > 0 && config.ExperimentCount > 0 && config.ExperimentSize > 0 &&
                ui64(config.ExperimentCount.Get()) * config.ExperimentSize.Get() <= ui64(config.Offset.Get()),
                "Metal model-based evaluation needs positive offset, experiment_count and experiment_size with count*size <= offset");
            CB_ENSURE(!config.FeaturesToEvaluate->empty(), "Metal model-based evaluation requires features to evaluate");
            TSet<ui32> allEvaluated;
            for (const auto& group : config.FeaturesToEvaluate.Get()) {
                for (ui32 feature : group) {
                    CB_ENSURE(feature < layout.GetExternalFeatureCount(), "Metal evaluated feature index is out of range");
                    CB_ENSURE(Count(options.DataProcessingOptions->IgnoredFeatures.Get(), feature) == 0,
                        "Metal evaluated feature is explicitly ignored");
                    allEvaluated.insert(feature);
                }
            }
            TSet<ui32> baselineIgnored(options.DataProcessingOptions->IgnoredFeatures->begin(),
                options.DataProcessingOptions->IgnoredFeatures->end());
            if (!config.UseEvaluatedFeaturesInBaselineModel)
                baselineIgnored.insert(allEvaluated.begin(), allEvaluated.end());
            TVector<ui32> baselineIgnoredVector(baselineIgnored.begin(), baselineIgnored.end());
            auto baselinePools = MetalModelBasedFeatureSubset(trainingData, baselineIgnoredVector, executor);
            auto baselineOptions = options;
            baselineOptions.DataProcessingOptions->IgnoredFeatures.Set(baselineIgnoredVector);
            auto baselineOutput = outputOptions;
            baselineOutput.SetSaveSnapshotFlag(true);
            baselineOutput.SetSnapshotFilename(config.BaselineModelSnapshot.Get());
            CB_ENSURE(TFsPath(baselineOutput.CreateSnapshotFullPath()).Exists(),
                "Metal model-based evaluation requires an existing baseline snapshot: " << baselineOutput.CreateSnapshotFullPath());
            TTrainModelInternalOptions internal;
            internal.CalcMetricsOnly = true;
            internal.ForceCalcEvalMetricOnEveryIteration = true;
            TMetalModelBasedRun base;
            base.LoadBaseline = true;
            TrainModelImpl(internal, baselineOptions, baselineOutput, Nothing(), Nothing(), baselinePools,
                Nothing(), labelConverter, nullptr, nullptr, Nothing(), nullptr, {}, executor, Nothing(),
                nullptr, {}, nullptr, nullptr, &base);
            const ui32 baselineSize = MetalModelBasedBaselineSize(base, options, outputOptions,
                trainingData.Learn->MetaInfo.HasWeights);
            CB_ENSURE(baselineSize >= ui32(config.Offset.Get()),
                "Metal model-based evaluation offset must not exceed the retained baseline tree count");
            const ui32 first = baselineSize - config.Offset.Get();
            const ui32 stride = config.Offset.Get() / config.ExperimentCount.Get();
            const auto initialOne = base.InitialLearn;
            for (ui32 p = 1; p < base.PermutationCount; ++p)
                base.InitialLearn.insert(base.InitialLearn.end(), initialOne.begin(), initialOne.end());
            MetalModelBasedAppendPrefix(base, baselinePools, 0, first, &base.InitialLearn, &base.InitialTest, executor);
            for (ui32 set = 0; set < config.FeaturesToEvaluate->size(); ++set) {
                TSet<ui32> ignored = allEvaluated;
                bool active = false;
                for (ui32 feature : config.FeaturesToEvaluate.Get()[set]) {
                    const auto& meta = layout.GetExternalFeatureMetaInfo(feature);
                    // CUDA's evaluated candidates are border-bearing numeric
                    // features; categoricals remain supported background inputs.
                    if (meta.IsAvailable && meta.Type == EFeatureType::Float) {
                        const auto internalFeature = layout.GetInternalFeatureIdx<EFeatureType::Float>(feature);
                        if (!trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo()->GetBorders(internalFeature).empty()) {
                            ignored.erase(feature);
                            active = true;
                            continue;
                        }
                    }
                    CATBOOST_WARNING_LOG << "Ignoring constant or non-numeric evaluated feature " << feature << Endl;
                }
                if (!active) {
                    CATBOOST_WARNING_LOG << "Feature set " << set << " is not evaluated because it consists of ignored, constant or non-numeric features" << Endl;
                    continue;
                }
                ignored.insert(options.DataProcessingOptions->IgnoredFeatures->begin(),
                    options.DataProcessingOptions->IgnoredFeatures->end());
                TVector<ui32> ignoredVector(ignored.begin(), ignored.end());
                auto experimentPools = MetalModelBasedFeatureSubset(trainingData, ignoredVector, executor);
                auto experimentOptions = options;
                experimentOptions.DataProcessingOptions->IgnoredFeatures.Set(ignoredVector);
                experimentOptions.BoostingOptions->IterationCount.Set(config.ExperimentSize.Get());
                auto learn = base.InitialLearn;
                auto test = base.InitialTest;
                for (ui32 fold = 0; fold < ui32(config.ExperimentCount.Get()); ++fold) {
                    CheckInterrupted();
                    auto experimentOutput = outputOptions;
                    experimentOutput.SetSaveSnapshotFlag(false);
                    experimentOutput.SetMetricPeriod(1);
                    experimentOutput.SetTrainDir(JoinFsPaths(outputOptions.GetTrainDir(),
                        NCatboostOptions::GetExperimentName(set, fold)));
                    TMetalModelBasedRun experiment;
                    experiment.PermutationCount = base.PermutationCount;
                    experiment.ApproxDimension = base.ApproxDimension;
                    experiment.PermutationOffset = first + ui64(stride) * fold;
                    experiment.InitialLearn = learn;
                    experiment.InitialTest = test;
                    TEvalResult eval;
                    TrainModelImpl(internal, experimentOptions, experimentOutput, Nothing(), Nothing(), experimentPools,
                        Nothing(), labelConverter, nullptr, nullptr, Nothing(), nullptr, {}, executor, Nothing(),
                        nullptr, {&eval}, nullptr, nullptr, &experiment);
                    if (fold + 1 < ui32(config.ExperimentCount.Get()))
                        MetalModelBasedAppendPrefix(base, baselinePools, first + stride * fold,
                            first + stride * (fold + 1), &learn, &test, executor);
                }
            }
        }
    };

    TTrainerFactory::TRegistrator<TMetalModelTrainer> MetalModelTrainerRegistrator(ETaskType::GPU);
}
}
