#include "loss_description.h"
#include "data_processing_options.h"
#include "json_helper.h"
#include "metric_options.h"

#include <catboost/libs/helpers/json_helpers.h>

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/generic/fwd.h>
#include <util/string/split.h>
#include <util/string/vector.h>
#include <util/string/strip.h>
#include <util/string/subst.h>

ELossFunction ParseLossType(const TStringBuf lossDescription) {
    const TVector<TStringBuf> tokens = StringSplitter(lossDescription).Split(':').Limit(2);
    CB_ENSURE(!tokens.empty(), "custom loss is missing in description: " << lossDescription);
    ELossFunction customLoss;
    CB_ENSURE(TryFromString<ELossFunction>(tokens[0], customLoss), tokens[0] << " loss is not supported");
    return customLoss;
}

TLossParams ParseLossParams(const TStringBuf lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    const TVector<TStringBuf> tokens = StringSplitter(lossDescription).Split(':').Limit(2);
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    TVector<std::pair<TString, TString>> keyValuePairs;
    if (tokens.size() == 2) {
        for (const auto& token : StringSplitter(tokens[1]).Split(';')) {
            const TVector<TString> keyValue = StringSplitter(token.Token()).Split('=').Limit(2);
            CB_ENSURE(keyValue.size() == 2, errorMessage);
            keyValuePairs.emplace_back(keyValue[0], keyValue[1]);
        }
    }

    return TLossParams::FromVector(keyValuePairs);
}

NCatboostOptions::TLossDescription::TLossDescription()
    : LossFunction("type", ELossFunction::RMSE)
    , LossParams("params", TLossParams())
{
}

ELossFunction NCatboostOptions::TLossDescription::GetLossFunction() const {
    return LossFunction.Get();
}

void NCatboostOptions::TLossDescription::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &LossFunction, &LossParams);
}

void NCatboostOptions::TLossDescription::Save(NJson::TJsonValue* options) const {
    SaveFields(options, LossFunction, LossParams);
}

bool NCatboostOptions::TLossDescription::operator==(const TLossDescription& rhs) const {
    return std::tie(LossFunction, LossParams) ==
            std::tie(rhs.LossFunction, rhs.LossParams);
}

const TMap<TString, TString>& NCatboostOptions::TLossDescription::GetLossParamsMap() const {
    return LossParams->GetParamsMap();
};

const TVector<TString>& NCatboostOptions::TLossDescription::GetLossParamKeysOrdered() const {
    return LossParams->GetUserSpecifiedKeyOrder();
};

const TLossParams& NCatboostOptions::TLossDescription::GetLossParams() const {
    return LossParams;
}

// static.
NCatboostOptions::TLossDescription NCatboostOptions::TLossDescription::CloneWithLossFunction(ELossFunction function) {
    TLossDescription desc = *this;
    desc.LossFunction.Set(function);
    return desc;
}

bool NCatboostOptions::TLossDescription::operator!=(const TLossDescription& rhs) const {
    return !(rhs == *this);
}

double NCatboostOptions::GetLogLossBorder(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Logloss);
    return GetParamOrDefault(lossFunctionConfig, "border", GetDefaultTargetBorder());
}

double NCatboostOptions::GetAlpha(const TMap<TString, TString>& lossParams) {
    return GetParamOrDefault(lossParams, "alpha", 0.5);
}

double NCatboostOptions::GetAlpha(const TLossDescription& lossFunctionConfig) {
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    return GetAlpha(lossParams);
}

TVector<double> NCatboostOptions::GetAlphaMultiQuantile(const TMap<TString, TString>& lossParams) {
    const TString median("0.5");
    const TStringBuf alphaParam(lossParams.contains("alpha") ? lossParams.at("alpha") : median);
    TVector<double> alpha;
    for (const auto& value : StringSplitter(alphaParam).Split(',').SkipEmpty()) {
        alpha.emplace_back(FromString<double>(value.Token()));
    }
    return alpha;
}

double NCatboostOptions::GetAlphaQueryCrossEntropy(const TMap<TString, TString>& lossParams) {
    return GetParamOrDefault(lossParams, "alpha", 0.95);
}

double NCatboostOptions::GetAlphaQueryCrossEntropy(const TLossDescription& lossFunctionConfig) {
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    return GetAlphaQueryCrossEntropy(lossParams);
}

void NCatboostOptions::GetApproxScaleQueryCrossEntropy(
    const TLossDescription& lossFunctionConfig,
    TVector<float>* approxScale,
    ui32* approxScaleSize,
    float* defaultScale
) {
    const auto formatErrorMessage = "raw_values_scale should be space separated list of "
        "items group_size,true_count:scale where true_count <= group_size. "
        "If group_size = 0, scale is default scale for non-listed sizes and counts. "
        "If all group_size > 0, default scale is 1.";

    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    const auto& rawValuesScale = GetParamOrDefault(lossParams, "raw_values_scale", TString());
    const TVector<TStringBuf> tokens = StringSplitter(rawValuesScale).Split(' ');

    if (tokens.empty() || (tokens.size() == 1 && tokens[0].empty())) {
        *approxScaleSize = 0;
        *defaultScale = 1;
        approxScale->clear();
        return;
    }

    TVector<ui32> groupSizes;
    TVector<ui32> trueCounts;
    TVector<float> scales;
    for (const auto& token : tokens) {
        const TVector<TString> idxScale = StringSplitter(token).Split(':').Limit(2);
        CB_ENSURE(idxScale.size() == 2, formatErrorMessage);
        const TVector<TString> idx = StringSplitter(idxScale[0]).Split(',').Limit(2);
        CB_ENSURE(idx.size() == 2, formatErrorMessage);
        groupSizes.emplace_back();
        trueCounts.emplace_back();
        scales.emplace_back();
        CB_ENSURE(
            TryFromString<ui32>(idx[0], groupSizes.back())
            && TryFromString<ui32>(idx[1], trueCounts.back())
            && trueCounts.back() <= groupSizes.back(),
            formatErrorMessage);
        CB_ENSURE(TryFromString<float>(idxScale[1], scales.back()), formatErrorMessage);
    }
    *approxScaleSize = SafeIntegerCast<ui32>(*MaxElement(groupSizes.begin(), groupSizes.end()));
    if (*approxScaleSize == 0) {
        *defaultScale = scales[0];
        approxScale->clear();
        return;
    }
    *approxScaleSize += 1;
    *defaultScale = 1;
    const auto defaultScaleIdx = FindIndex(groupSizes, 0);
    if (defaultScaleIdx != NPOS) {
        *defaultScale = scales[defaultScaleIdx];
    }
    approxScale->resize(*approxScaleSize * *approxScaleSize, *defaultScale);
    for (auto idx : xrange(groupSizes.size())) {
        (*approxScale)[groupSizes[idx] * *approxScaleSize + trueCounts[idx]] = scales[idx];
    }
}

int NCatboostOptions::GetYetiRankPermutations(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(
        lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRank ||
        lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRankPairwise);
    return GetParamOrDefault(lossFunctionConfig, "permutations", 10);
}

double NCatboostOptions::GetYetiRankDecay(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(
        lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRank ||
        lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRankPairwise);
    return GetParamOrDefault(lossFunctionConfig, "decay", 0.85);
}

double NCatboostOptions::GetLqParam(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Lq);
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    CB_ENSURE(lossParams.contains("q"), "For " << ELossFunction::Lq << " q parameter is mandatory");
    return FromString<double>(lossParams.at("q"));
}

double NCatboostOptions::GetHuberParam(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Huber);
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    CB_ENSURE(lossParams.contains("delta"), "For " << ELossFunction::Huber << " delta parameter is mandatory");
    return FromString<double>(lossParams.at("delta"));
}

double NCatboostOptions::GetQuerySoftMaxLambdaReg(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::QuerySoftMax);
    return GetParamOrDefault(lossFunctionConfig, "lambda", 0.01);
}

double NCatboostOptions::GetQuerySoftMaxBeta(const TMap<TString, TString>& lossParams) {
    return GetParamOrDefault(lossParams, "beta", 1.0);
}

double NCatboostOptions::GetQuerySoftMaxBeta(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::QuerySoftMax);
    return GetParamOrDefault(lossFunctionConfig, "beta", 1.0);
}

EAucType NCatboostOptions::GetAucType(const TMap<TString, TString>& lossParams) {
    return GetParamOrDefault(lossParams, "type", EAucType::Classic);
}

ui32 NCatboostOptions::GetMaxPairCount(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(IsPairwiseMetric(lossFunctionConfig.GetLossFunction()));
    if (IsPairLogit(lossFunctionConfig.GetLossFunction())) {
        ui32 max_pairs = GetParamOrDefault(lossFunctionConfig, "max_pairs", (ui32)MAX_AUTOGENERATED_PAIRS_COUNT);
        CB_ENSURE(max_pairs > 0, "Max generated pairs count should be positive");
        return max_pairs;
    }
    return (ui32)MAX_AUTOGENERATED_PAIRS_COUNT;
}

double NCatboostOptions::GetStochasticFilterSigma(const TLossDescription& lossDescription) {
    Y_ASSERT(lossDescription.GetLossFunction() == ELossFunction::StochasticFilter);
    return GetParamOrDefault(lossDescription, "sigma", 1.0);
}

int NCatboostOptions::GetStochasticFilterNumEstimations(const TLossDescription& lossDescription) {
    Y_ASSERT(lossDescription.GetLossFunction() == ELossFunction::StochasticFilter);
    return GetParamOrDefault(lossDescription, "num_estimations", 1);
}

double NCatboostOptions::GetTweedieParam(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Tweedie);
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    CB_ENSURE(
        lossParams.contains("variance_power"),
        "For " << ELossFunction::Tweedie << " variance_power parameter is mandatory");
    return FromString<double>(lossParams.at("variance_power"));
}

double NCatboostOptions::GetFocalParamA(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Focal);
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    CB_ENSURE(
        lossParams.contains("focal_alpha"),
        "For " << ELossFunction::Focal << " focal_alpha parameter is mandatory");
    return FromString<double>(lossParams.at("focal_alpha"));
}

double NCatboostOptions::GetFocalParamG(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Focal);
    const auto& lossParams = lossFunctionConfig.GetLossParamsMap();
    CB_ENSURE(
        lossParams.contains("focal_gamma"),
        "For " << ELossFunction::Focal << " focal_gamma parameter is mandatory");
    return FromString<double>(lossParams.at("focal_gamma"));
}

double NCatboostOptions::GetPredictionBorderOrDefault(const TMap<TString, TString>& params, double defaultValue) {
    auto it = params.find(TMetricOptions::PREDICTION_BORDER_PARAM);
    if (it == params.end()) {
        return defaultValue;
    }
    const auto border = FromString<double>(it->second);
    CB_ENSURE(0 <= border && border <= 1.0, "Probability threshold must be in [0, 1] interval.");
    return border;
}

NCatboostOptions::TLossDescription NCatboostOptions::ParseLossDescription(TStringBuf stringLossDescription) {
    TLossDescription description;
    description.LossFunction.Set(ParseLossType(stringLossDescription));
    TLossParams params = ParseLossParams(stringLossDescription);
    description.LossParams.Set(params);
    return description;
}

template <>
void Out<NCatboostOptions::TLossDescription>(IOutputStream& out, const NCatboostOptions::TLossDescription& description) {
    TVector<TString> entries;
    entries.push_back(ToString(description.GetLossFunction()));
    for (const auto& param : description.GetLossParamsMap()) {
        entries.push_back(TString::Join(param.first, "=", ToString(param.second)));
    }
    out << JoinStrings(entries, ",");
}

template <>
inline NCatboostOptions::TLossDescription FromString<NCatboostOptions::TLossDescription>(const TString& stringDescription) {
    NCatboostOptions::TLossDescription description;
    auto descriptionJson = LossDescriptionToJson(stringDescription);
    description.Load(descriptionJson);
    return description;
}

TLossParams::TLossParams(TMap<TString, TString> paramsMap, TVector<TString> userSpecifiedKeyOrder)
    : ParamsMap(std::move(paramsMap))
    , UserSpecifiedKeyOrder(std::move(userSpecifiedKeyOrder))
{}

// static.
TLossParams TLossParams::FromVector(const TVector<std::pair<TString, TString>>& params) {
    TMap<TString, TString> paramsMap;
    TVector<TString> userSpecifiedKeyOrder;
    for (const auto& keyValue : params) {
        const bool inserted = paramsMap.insert({keyValue.first, keyValue.second}).second;
        CB_ENSURE(inserted, "Duplicated loss param found: " << keyValue.first);
        userSpecifiedKeyOrder.push_back(keyValue.first);
    }
    return TLossParams(std::move(paramsMap), std::move(userSpecifiedKeyOrder));
}

// static.
TLossParams TLossParams::FromMap(TMap<TString, TString> paramsMap) {
    TVector<TString> keys;
    keys.reserve(paramsMap.size());
    for (const auto& keyValue: paramsMap) {
        keys.push_back(keyValue.first);
    }
    return TLossParams(std::move(paramsMap), std::move(keys));
}


bool TLossParams::operator==(const TLossParams &that) const {
    return std::tie(ParamsMap, UserSpecifiedKeyOrder) == std::tie(that.ParamsMap, that.UserSpecifiedKeyOrder);
}

bool TLossParams::operator!=(const TLossParams &that) const {
    return !(*this == that);
}


void ValidateHints(const TMap<TString, TString>& hints) {
    TSet<TString> availableHints = {
        "skip_train"
    };

    for (const auto& hint : hints) {
        CB_ENSURE(availableHints.contains(hint.first), TString("No hint called ") + hint.first);
    }

    if (hints.contains("skip_train")) {
        const TString& value = hints.at("skip_train");
        CB_ENSURE(value == "true" || value == "false", "skip_train hint value should be true or false");
    }
}

TMap<TString, TString> ParseHintsDescription(const TStringBuf hintsDescription) {
    const char* errorMessage = "Invalid hints description, it should be in the form "
                               "\"hints=key1~value1|...|keyN~valueN\"";

    const TVector<TStringBuf> tokens = StringSplitter(hintsDescription).Split('|');
    CB_ENSURE(!tokens.empty(), "Hint description should not be empty");

    TMap<TString, TString> hints;
    for (const auto& token : tokens) {
        const TVector<TString> keyValue = StringSplitter(token).Split('~').Limit(2);
        CB_ENSURE(keyValue.size() == 2, errorMessage);
        CB_ENSURE(!hints.contains(keyValue[0]), "Two similar keys in hints description are not allowed");
        hints[keyValue[0]] = keyValue[1];
    }

    ValidateHints(hints);

    return hints;
}

TString MakeHintsDescription(const TMap<TString, TString>& hints) {
    if (hints.empty()) {
        return TString();
    }

    TStringBuilder out;
    auto it = hints.begin();
    out << it->first << '~' << it->second;
    for (++it; it != hints.end(); ++it) {
        out << '|' << it->first << '~' << it->second;
    }

    return out;
}


NJson::TJsonValue LossDescriptionToJson(const TStringBuf lossDescriptionRaw) {
    NJson::TJsonValue descriptionJson;

    ELossFunction lossFunction = ParseLossType(lossDescriptionRaw);
    TLossParams lossParams = ParseLossParams(lossDescriptionRaw);
    NCatboostOptions::TLossDescription lossDescription;
    lossDescription.LossParams.Set(std::move(lossParams));
    lossDescription.LossFunction.Set(lossFunction);
    lossDescription.Save(&descriptionJson);
    return descriptionJson;
}

TString BuildMetricOptionDescription(const NJson::TJsonValue& lossOptions) {
    TString paramType = StripString(ToString(lossOptions["type"]), EqualsStripAdapter('"'));
    paramType += ":";
    TLossParams lossParams;
    TJsonFieldHelper<TLossParams, false>::Read(lossOptions["params"], &lossParams);
    const auto& paramsMap = lossParams.GetParamsMap();
    for (const auto& paramName : lossParams.GetUserSpecifiedKeyOrder()) {
        const TString& paramValue = StripString(paramsMap.at(paramName), EqualsStripAdapter('"'));
        paramType += paramName + "=" + paramValue + ";";
    }

    paramType.pop_back();
    return paramType;
}


static bool IsFromAucFamily(ELossFunction loss) {
    return loss == ELossFunction::AUC
        || loss == ELossFunction::QueryAUC
        || loss == ELossFunction::NormalizedGini;
}

void CheckMetric(const ELossFunction metric, const ELossFunction modelLoss) {
    if (IsUserDefined(metric) || IsUserDefined(modelLoss)) {
        return;
    }

    CB_ENSURE(
        (IsMultiTargetObjective(modelLoss) && IsMultiTargetMetric(metric)) || (!IsMultiTargetObjective(modelLoss) && !IsMultiTargetOnlyMetric(metric)),
        "metric [" + ToString(metric) + "] and loss [" + ToString(modelLoss) + "] are incompatible"
    );

    /* [loss -> metric]
     * ranking             -> ranking compatible
     * binclass only       -> binclass compatible
     * multiclass only     -> multiclass compatible
     * classification only -> classification compatible
     */

    if (IsRankingMetric(modelLoss)) {
        CB_ENSURE(
            // accept classification
            IsBinaryClassCompatibleMetric(metric) && IsBinaryClassCompatibleMetric(modelLoss)
            // accept ranking
            || IsRankingMetric(metric) && (GetRankingType(metric) != ERankingType::CrossEntropy) == (GetRankingType(modelLoss) != ERankingType::CrossEntropy)
            // accept regression
            || IsRegressionMetric(metric) && GetRankingType(modelLoss) == ERankingType::AbsoluteValue
            // accept auc like
            || IsFromAucFamily(metric),
            "metric [" + ToString(metric) + "] is incompatible with loss [" + ToString(modelLoss) + "] (not compatible with ranking)"
        );
    }

    if (IsBinaryClassOnlyMetric(modelLoss)) {
        CB_ENSURE(IsBinaryClassCompatibleMetric(metric),
                  "metric [" + ToString(metric) + "] is incompatible with loss [" + ToString(modelLoss) + "] (no binclass support)");
    }

    if (IsMultiClassOnlyMetric(modelLoss)) {
        CB_ENSURE(IsMultiClassCompatibleMetric(metric),
                  "metric [" + ToString(metric) + "] is incompatible with loss [" + ToString(modelLoss) + "] (no multiclass support)");
    }

    if (IsClassificationOnlyMetric(modelLoss)) {
        CB_ENSURE(IsClassificationMetric(metric),
                  "metric [" + ToString(metric) + "] is incompatible with loss [" + ToString(modelLoss) + "] (no classification support)");
    }

    /* [metric -> loss]
     * binclass only       -> binclass compatible
     * multiclass only     -> multiclass compatible
     * classification only -> classification compatible
     */

    if (IsBinaryClassOnlyMetric(metric)) {
        CB_ENSURE(IsBinaryClassCompatibleMetric(modelLoss),
                  "loss [" + ToString(modelLoss) + "] is incompatible with metric [" + ToString(metric) + "] (no binclass support)");
    }

    if (IsMultiClassOnlyMetric(metric)) {
        CB_ENSURE(IsMultiClassCompatibleMetric(modelLoss),
                  "loss [" + ToString(modelLoss) + "] is incompatible with metric [" + ToString(metric) + "] (no multiclass support)");
    }

    if (IsClassificationOnlyMetric(metric)) {
        CB_ENSURE(IsClassificationMetric(modelLoss),
                  "loss [" + ToString(modelLoss) + "] is incompatible with metric [" + ToString(metric) + "] (no classification support)");
    }
}

ELossFunction GetMetricFromCombination(const TMap<TString, TString>& params) {
    TMaybe<ELossFunction> referenceLoss;
    IterateOverCombination(
        params,
        [&] (const auto& loss, float /*weight*/) {
            if (!referenceLoss) {
                referenceLoss = loss.GetLossFunction();
            } else {
                CheckMetric(*referenceLoss, loss.GetLossFunction());
            }
    });
    CB_ENSURE(referenceLoss, "Combination loss must have one or more non-zero weights");
    return *referenceLoss;
}

void CheckCombinationParameters(const TMap<TString, TString>& params) {
    (void)GetMetricFromCombination(params);
}

TString GetCombinationLossKey(ui32 idx) {
    return "loss" + ToString(idx);
}

TString GetCombinationWeightKey(ui32 idx) {
    return "weight" + ToString(idx);
}
