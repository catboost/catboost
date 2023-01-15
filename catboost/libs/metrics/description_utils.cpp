#include "description_utils.h"

#include <catboost/private/libs/options/enums.h>
#include <util/generic/string.h>
#include <util/string/builder.h>


TString BuildDescriptionFromParamsMap(ELossFunction lossFunction, const TMap<TString, TString>& params) {
    TStringBuilder buffer;

    // This is kept for backwards compatibility.
    if (lossFunction == ELossFunction::QueryAverage) {
        buffer << "AverageGain";
    } else {
        buffer << ToString(lossFunction);
    }

    if (params.empty()) {
        return buffer;
    }
    buffer << ":";
    size_t currentParamIdx = 0;
    for (const auto& keyValue: params) {
        buffer << keyValue.first << "=" << keyValue.second;

        currentParamIdx++;
        // Put key=value pair separator, if the parameter is not last
        if (currentParamIdx != params.size()) {
            buffer << ";";
        }
    }
    return buffer;
}

