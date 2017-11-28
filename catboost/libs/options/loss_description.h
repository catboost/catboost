#pragma once

#include "enums.h"
#include "option.h"
#include "json_helper.h"
#include <util/string/vector.h>
#include <util/string/iterator.h>

namespace NCatboostOptions {
    class TLossDescription {
    public:
        explicit TLossDescription()
            : LossFunction("type", ELossFunction::RMSE)
            , LossParams("params", ymap<TString, TString>())
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

        const ymap<TString, TString>& GetLossParams() const {
            return LossParams.Get();
        };

        bool operator!=(const TLossDescription& rhs) const {
            return !(rhs == *this);
        }

    private:
        TOption<ELossFunction> LossFunction;
        TOption<ymap<TString, TString>> LossParams;
    };

    inline double GetLogLossBorder(const TLossDescription& lossFunctionConfig) {
        CB_ENSURE(lossFunctionConfig.GetLossFunction() == ELossFunction::Logloss);
        auto& lossParams = lossFunctionConfig.GetLossParams();
        if (lossParams.has("border")) {
            return FromString<double>(lossParams.at("border"));
        }
        return 0.5;
    }
}

template <>
inline TString ToString<NCatboostOptions::TLossDescription>(const NCatboostOptions::TLossDescription& description) {
    TVector<TString> entries;
    entries.push_back(ToString(description.GetLossFunction()));
    for (const auto& param : description.GetLossParams()) {
        entries.push_back(param.first + "=" + param.second);
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

inline THashMap<TString, TString> ParseLossParams(const TString& lossDescription) {
    const char* errorMessage = "Invalid metric description, it should be in the form "
                               "\"metric_name:param1=value1;...;paramN=valueN\"";

    TVector<TString> tokens = StringSplitter(lossDescription).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(!tokens.empty(), "Metric description should not be empty");
    CB_ENSURE(tokens.size() <= 2, errorMessage);

    THashMap<TString, TString> params;
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
    THashMap<TString, TString> lossParams = ParseLossParams(lossDescription);
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
