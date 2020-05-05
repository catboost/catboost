#include "description_utils.h"

#include <catboost/private/libs/options/enums.h>
#include <util/generic/string.h>
#include <util/string/builder.h>


TString BuildDescriptionFromParams(ELossFunction lossFunction, const TLossParams& params) {
    TStringBuilder buffer;

    // This is kept for backwards compatibility.
    if (lossFunction == ELossFunction::QueryAverage) {
        buffer << "AverageGain";
    } else {
        buffer << ToString(lossFunction);
    }

    if (params.paramsMap.empty()) {
        return buffer;
    }
    buffer << ":";

    TVector<std::pair<TString, TString>> keyAndValues;
    for (const auto& key: params.userSpecifiedKeyOrder) {
        auto it = params.paramsMap.find(key);
        if (it == params.paramsMap.end()) {
            continue;
        }
        keyAndValues.emplace_back(it->first, it->second);
    }
    for (size_t i = 0; i < keyAndValues.size(); ++i) {
        buffer << keyAndValues[i].first << "=" << keyAndValues[i].second
               << (/*If not the last to render, then put a separator.*/i + 1 != keyAndValues.size() ? ";" : "");
    }
    return buffer;
}

