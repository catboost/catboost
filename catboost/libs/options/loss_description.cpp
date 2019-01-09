#include "loss_description.h"
#include "json_helper.h"

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/iterator.h>
#include <util/string/vector.h>


ELossFunction ParseLossType(const TStringBuf lossDescription) {
    const TVector<TStringBuf> tokens = StringSplitter(lossDescription).SplitLimited(':', 2);
    CB_ENSURE(!tokens.empty(), "custom loss is missing in description: " << lossDescription);
    ELossFunction customLoss;
    CB_ENSURE(TryFromString<ELossFunction>(tokens[0], customLoss), tokens[0] << " loss is not supported");
    return customLoss;
}

TMap<TString, TString> ParseLossParams(const TStringBuf lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    const TVector<TStringBuf> tokens = StringSplitter(lossDescription).SplitLimited(':', 2);
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    TMap<TString, TString> params;
    if (tokens.size() == 2) {
        for (const auto& token : StringSplitter(tokens[1]).Split(';')) {
            const TVector<TString> keyValue = StringSplitter(token.Token()).SplitLimited('=', 2);
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
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    if (lossParams.contains("border")) {
        return FromString<float>(lossParams.at("border"));
    }
    return 0.5;
}

double NCatboostOptions::GetAlpha(const TMap<TString, TString>& lossParams) {
    if (lossParams.contains("alpha")) {
        return FromString<float>(lossParams.at("alpha"));
    }
    return 0.5;
}

double NCatboostOptions::GetAlpha(const TLossDescription& lossFunctionConfig) {
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    return GetAlpha(lossParams);
}

double NCatboostOptions::GetAlphaQueryCrossEntropy(const TMap<TString, TString>& lossParams) {
    if (lossParams.contains("alpha")) {
        return FromString<float>(lossParams.at("alpha"));
    }
    return 0.95;
}

double NCatboostOptions::GetAlphaQueryCrossEntropy(const TLossDescription& lossFunctionConfig) {
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    return GetAlphaQueryCrossEntropy(lossParams);
}

int NCatboostOptions::GetYetiRankPermutations(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRank || lossFunctionConfig.GetLossFunction()  == ELossFunction::YetiRankPairwise);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    if (lossParams.contains("permutations")) {
        return FromString<int>(lossParams.at("permutations"));
    }
    return 10;
}

ESamplingType NCatboostOptions::GetYetiRankSamplingType(const TLossDescription& lossFunctionConfig) {
    CB_ENSURE(lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRankPairwise);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    if (lossParams.contains("sampling_type")) {
        return FromString<ESamplingType>(lossParams.at("sampling_type"));
    }
    return ESamplingType::Groupwise;
}

double NCatboostOptions::GetYetiRankDecay(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRank || lossFunctionConfig.GetLossFunction()  == ELossFunction::YetiRankPairwise);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    if (lossParams.contains("decay")) {
        return FromString<double>(lossParams.at("decay"));
    }
    //TODO(nikitxskv): try to find the best default
    return 0.99;
}

double NCatboostOptions::GetLqParam(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Lq);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    if (lossParams.contains("q")) {
        return FromString<double>(lossParams.at("q"));
    } else {
        CB_ENSURE(false, "For " << ELossFunction::Lq << " q parameter is mandatory");
    }
}

double NCatboostOptions::GetQuerySoftMaxLambdaReg(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::QuerySoftMax);
    const auto& lossParams = lossFunctionConfig.GetLossParams();
    if (lossParams.contains("lambda")) {
        return FromString<double>(lossParams.at("lambda"));
    }
    return 0.01;
}

ui32 NCatboostOptions::GetMaxPairCount(const TLossDescription& lossFunctionConfig) {
    Y_ASSERT(IsPairwiseMetric(lossFunctionConfig.GetLossFunction()));
    if (IsPairLogit(lossFunctionConfig.GetLossFunction())) {
        const auto& lossParams = lossFunctionConfig.GetLossParams();
        if (lossParams.contains("max_pairs")) {
            auto max_pairs = FromString<ui32>(lossParams.at("max_pairs"));
            CB_ENSURE(max_pairs > 0, "Max generated pairs count should be positive");
            return max_pairs;
        }
    }
    return (ui32)MAX_AUTOGENERATED_PAIRS_COUNT;
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
        const TVector<TString> keyValue = StringSplitter(token).SplitLimited('~', 2);
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
