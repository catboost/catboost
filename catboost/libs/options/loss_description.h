#pragma once

#include "enums.h"
#include "option.h"
#include "json_helper.h"
#include <util/string/vector.h>
#include <util/string/iterator.h>
#include <util/string/builder.h>

namespace NCatboostOptions {
    class TLossDescription {
    public:
        explicit TLossDescription()
            : LossFunction("type", ELossFunction::RMSE)
            , LossParams("params", TMap<TString, TString>())
        {
        }

        ELossFunction GetLossFunction() const {
            return LossFunction.Get();
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &LossFunction, &LossParams);
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, LossFunction, LossParams);
        }

        bool operator==(const TLossDescription& rhs) const {
            return std::tie(LossFunction, LossParams) ==
                   std::tie(rhs.LossFunction, rhs.LossParams);
        }

        const TMap<TString, TString>& GetLossParams() const {
            return LossParams.Get();
        };

        bool operator!=(const TLossDescription& rhs) const {
            return !(rhs == *this);
        }

    private:
        TOption<ELossFunction> LossFunction;
        TOption<TMap<TString, TString>> LossParams;
    };

    inline double GetLogLossBorder(const TLossDescription& lossFunctionConfig) {
        Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::Logloss);
        const auto& lossParams = lossFunctionConfig.GetLossParams();
        if (lossParams.has("border")) {
            return FromString<float>(lossParams.at("border"));
        }
        return 0.5;
    }

    inline double GetAlpha(const TLossDescription& lossFunctionConfig) {
        const auto& lossParams = lossFunctionConfig.GetLossParams();
        if (lossParams.has("alpha")) {
            return FromString<float>(lossParams.at("alpha"));
        }
        return 0.5;
    }

    inline int GetYetiRankPermutations(const TLossDescription& lossFunctionConfig) {
        Y_ASSERT(lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRank);
        const auto& lossParams = lossFunctionConfig.GetLossParams();
        if (lossParams.has("permutations")) {
            return FromString<int>(lossParams.at("permutations"));
        }
        return 10;
    }

    inline double GetYetiRankDecay(const TLossDescription& lossFunctionConfig) {
        CB_ENSURE(lossFunctionConfig.GetLossFunction() == ELossFunction::YetiRank);
        auto& lossParams = lossFunctionConfig.GetLossParams();
        if (lossParams.has("decay")) {
            return FromString<double>(lossParams.at("decay"));
        }
        //TODO(nikitxskv): try to find the best default
        return 0.99;
    }
}

template <>
inline TString ToString<NCatboostOptions::TLossDescription>(const NCatboostOptions::TLossDescription& description) {
    TVector<TString> entries;
    entries.push_back(ToString(description.GetLossFunction()));
    for (const auto& param : description.GetLossParams()) {
        entries.push_back(param.first + "=" + ToString(param.second));
    }
    return JoinStrings(entries, ",");
}

inline ELossFunction ParseLossType(const TString& lossDescription) {
    TVector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "custom loss is missing in desctiption: " << lossDescription);
    ELossFunction customLoss;
    CB_ENSURE(TryFromString<ELossFunction>(tokens[0], customLoss), tokens[0] + " loss is not supported");
    return customLoss;
}

inline TMap<TString, TString> ParseLossParams(const TString& lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    TVector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    TMap<TString, TString> params;
    if (tokens.size() == 2) {
        TVector<TString> paramsTokens = StringSplitter(tokens[1]).Split(';').ToList<TString>();

        for (const auto& token : paramsTokens) {
            TVector<TString> keyValue = StringSplitter(token).SplitLimited('=', 2).ToList<TString>();
            CB_ENSURE(keyValue.size() == 2, errorMessage);
            params[keyValue[0]] = keyValue[1];
        }
    }
    return params;
}

inline NJson::TJsonValue LossDescriptionToJson(const TString& lossDescription) {
    NJson::TJsonValue descriptionJson(NJson::JSON_MAP);

    ELossFunction lossFunction = ParseLossType(lossDescription);
    TMap<TString, TString> lossParams = ParseLossParams(lossDescription);
    descriptionJson["type"] = ToString<ELossFunction>(lossFunction);
    for (const auto& lossParam : lossParams) {
        descriptionJson["params"][lossParam.first] = lossParam.second;
    }
    return descriptionJson;
}

template <>
inline NCatboostOptions::TLossDescription FromString<NCatboostOptions::TLossDescription>(const TString& stringDescription) {
    NCatboostOptions::TLossDescription description;
    auto descriptionJson = LossDescriptionToJson(stringDescription);
    description.Load(descriptionJson);
    return description;
}
