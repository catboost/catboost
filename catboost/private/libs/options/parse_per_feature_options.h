#pragma once

#include "json_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/json/writer/json_value.h>

#include <util/generic/map.h>
#include <util/generic/strbuf.h>
#include <util/string/cast.h>
#include <util/string/split.h>

#include <regex>

namespace NCatboostOptions {
    template<typename TFeatureOptionType>
    static TFeatureOptionType GetOptionValue(const NJson::TJsonValue& option) {
        return FromJson<TFeatureOptionType>(option);
    }

    std::regex GetDenseFormatPattern(const TStringBuf featureOptionRegex); // like: (1,0,0,-1,0)

    std::regex GetSparseFormatPattern(const TStringBuf featureOptionRegex); // like: 0:1,3:-1 or FeatureName1:-1,FeatureName2:-1

    template<typename TFeatureOptionType>
    TMap<TString, TFeatureOptionType> ParsePerFeatureOptionsFromString(const TString& options,
                                                                       const TStringBuf optionName,
                                                                       const TStringBuf featureOptionRegex) {
        TMap<TString, TFeatureOptionType> optionsAsMap;
        std::regex denseFormat = GetDenseFormatPattern(featureOptionRegex);
        std::regex sparseFormat = GetSparseFormatPattern(featureOptionRegex);
        if (std::regex_match(options.data(), denseFormat)) {
            ui32 featureIdx = 0;
            auto optionPieces = StringSplitter(options.substr(1, options.size() - 2))
                .Split(',')
                .SkipEmpty();
            for (const auto& option : optionPieces) {
                const auto value = FromString<TFeatureOptionType>(option.Token());
                optionsAsMap[ToString(featureIdx)] = value;
                featureIdx++;
            }
        } else if (std::regex_match(options.data(), sparseFormat)) {
            for (const auto& oneFeatureMonotonic : StringSplitter(options).Split(',').SkipEmpty()) {
                auto parts = StringSplitter(oneFeatureMonotonic.Token()).Split(':');
                const TString feature(parts.Next()->Token());
                const auto value = FromString<TFeatureOptionType>(parts.Next()->Token());
                optionsAsMap[feature] = value;
            }
        } else {
            CB_ENSURE(false,
                      "Incorrect format of " << optionName << ". Possible formats: \"(1,0,0,-1)\", \"0:1,3:-1\", \"FeatureName1:-1,FeatureName2:-1\".");
        }
        return optionsAsMap;
    }

    template<typename TFeatureOptionType>
    void ConvertFeatureOptionsToCanonicalFormat(const TStringBuf optionName, const TStringBuf optionRegex, NJson::TJsonValue* optionsRef) {
        NJson::TJsonValue canonicalOptions(NJson::EJsonValueType::JSON_MAP);
        switch (optionsRef->GetType()) {
            case NJson::EJsonValueType::JSON_STRING: {
                TMap<TString, TFeatureOptionType> optionsAsMap = ParsePerFeatureOptionsFromString<TFeatureOptionType>(
                    optionsRef->GetString(),
                    optionName,
                    optionRegex
                );
                for (const auto& [key, value] : optionsAsMap) {
                    canonicalOptions.InsertValue(key, value);
                }
            }
                break;
            case NJson::EJsonValueType::JSON_ARRAY: {
                ui32 featureIdx = 0;
                for (const auto& option : optionsRef->GetArray()) {
                    auto value = GetOptionValue<TFeatureOptionType>(option);
                    canonicalOptions.InsertValue(ToString(featureIdx), value);
                    featureIdx++;
                }
            }
                break;
            case NJson::EJsonValueType::JSON_MAP: {
                TMap<TString, int> optionsAsMap;
                for (const auto& [feature, option] : optionsRef->GetMap()) {
                    auto value = GetOptionValue<TFeatureOptionType>(option);
                    canonicalOptions.InsertValue(feature, value);
                }
            }
                break;
            default:
                CB_ENSURE(false, "Incorrect options format");
        }

        *optionsRef = canonicalOptions;
    }
}