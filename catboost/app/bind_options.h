#pragma once

#include <catboost/libs/options/load_options.h>

#include <library/json/json_value.h>

#include <util/generic/string.h>


void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramPath,
                      NCatboostOptions::TPoolLoadParams* params);

void InitOptions(
    const TString& optionsFile,
    NJson::TJsonValue* catBoostJsonOptions,
    NJson::TJsonValue* outputOptionsJson
);

void CopyIgnoredFeaturesToPoolParams(
    const NJson::TJsonValue& catBoostJsonOptions,
    NCatboostOptions::TPoolLoadParams* poolLoadParams
);
