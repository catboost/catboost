#include "loss_description.h"
#include "data_processing_options.h"
#include "json_helper.h"

#include <util/string/builder.h>
#include <util/string/cast.h>
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

TMap<TString, TString> ParseLossParams(const TStringBuf lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    const TVector<TStringBuf> tokens = StringSplitter(lossDescription).Split(':').Limit(2);
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    TMap<TString, TString> params;
    if (tokens.size() == 2) {
        for (const auto& token : StringSplitter(tokens[1]).Split(';')) {
            const TVector<TString> keyValue = StringSplitter(token.Token()).Split('=').Limit(2);
            CB_ENSURE(keyValue.size() == 2, errorMessage);
            params[keyValue[0]] = keyValue[1];
        }
    }

    return params;
}

NCatboostOptions::TLossDescription::TLossDescription()
    : LossFunction("type", ELossFunction::RMSE)
    , LossParams("params", TMap<TString, TString>())
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

const TMap<TString, TString>& NCatboostOptions::TLossDescription::GetLossParams() const {
    return LossParams.Get();
};

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
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    return GetAlpha(lossParams);
}

double NCatboostOptions::GetAlphaQueryCrossEntropy(const TMap<TString, TString>& lossParams) {
    return GetParamOrDefault(lossParams, "alpha", 0.95);
}

double NCatboostOptions::GetAlphaQueryCrossEntropy(const TLossDescription& lossFunctionConfig) {
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    return GetAlphaQueryCrossEntropy(lossParams);
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
    //TODO(nikitxskv): try to find the best default
    return GetParamOrDefault(lossFunctionConfig, "decay", 0.99);
}

double NCatboostOptions::GetLqParam(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Lq);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    CB_ENSURE(lossParams.contains("q"), "For " << ELossFunction::Lq << " q parameter is mandatory");
    return FromString<double>(lossParams.at("q"));
}

double NCatboostOptions::GetHuberParam(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Huber);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    CB_ENSURE(lossParams.contains("delta"), "For " << ELossFunction::Huber << " delta parameter is mandatory");
    return FromString<double>(lossParams.at("delta"));
}

double NCatboostOptions::GetQuerySoftMaxLambdaReg(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::QuerySoftMax);
    return GetParamOrDefault(lossFunctionConfig, "lambda", 0.01);
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
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    CB_ENSURE(
        lossParams.contains("variance_power"),
        "For " << ELossFunction::Tweedie << " variance_power parameter is mandatory");
    return FromString<double>(lossParams.at("variance_power"));
}

NCatboostOptions::TLossDescription NCatboostOptions::ParseLossDescription(TStringBuf stringLossDescription) {
    TLossDescription description;
    description.LossFunction.Set(ParseLossType(stringLossDescription));
    description.LossParams.Set(ParseLossParams(stringLossDescription));
    return description;
}

template <>
void Out<NCatboostOptions::TLossDescription>(IOutputStream& out, const NCatboostOptions::TLossDescription& description) {
    TVector<TString> entries;
    entries.push_back(ToString(description.GetLossFunction()));
    for (const auto& param : description.GetLossParams()) {
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


NJson::TJsonValue LossDescriptionToJson(const TStringBuf lossDescription) {
    NJson::TJsonValue descriptionJson(NJson::JSON_MAP);

    ELossFunction lossFunction = ParseLossType(lossDescription);
    TMap<TString, TString> lossParams = ParseLossParams(lossDescription);
    descriptionJson["type"] = ToString(lossFunction);
    for (const auto& lossParam : lossParams) {
        descriptionJson["params"][lossParam.first] = lossParam.second;
    }
    return descriptionJson;
}

TString BuildMetricOptionDescription(const NJson::TJsonValue& lossOptions) {
    TString paramType = StripString(ToString(lossOptions["type"]), EqualsStripAdapter('"'));
    paramType += ":";

    for (const auto& elem : lossOptions["params"].GetMap()) {
        const TString& paramName = elem.first;
        const TString& paramValue = StripString(ToString(elem.second), EqualsStripAdapter('"'));
        paramType += paramName + "=" + paramValue + ";";
    }

    paramType.pop_back();
    return paramType;
}


static bool IsFromAucFamily(ELossFunction loss) {
    return loss == ELossFunction::AUC
        || loss == ELossFunction::NormalizedGini;
}

void CheckMetric(const ELossFunction metric, const ELossFunction modelLoss) {
    if (IsUserDefined(metric) || IsUserDefined(modelLoss)) {
        return;
    }

    CB_ENSURE(
        IsMultiRegressionMetric(metric) == IsMultiRegressionMetric(modelLoss),
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
