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
    size_t currentParamIdx = 0;
    for (const auto& key: params.userSpecifiedKeyOrder) {
        auto it = params.paramsMap.find(key);
        if (it == params.paramsMap.end()) {
            continue;
        }
        buffer << it->first << "=" << it->second;

        currentParamIdx++;
        // Put key=value pair separator, if the parameter is not last
        if (currentParamIdx != params.userSpecifiedKeyOrder.size()) {
            buffer << ";";
        }
    }
    return buffer;
}

