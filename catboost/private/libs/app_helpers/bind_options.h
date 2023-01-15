#pragma once

#include <catboost/private/libs/options/load_options.h>

#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/json/json_value.h>

#include <util/generic/string.h>


void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramPath,
                      NCatboostOptions::TPoolLoadParams* params);
void ParseModelBasedEvalCommandLine(
    int argc,
    const char* argv[],
    NJson::TJsonValue* plainJsonPtr,
    TString* paramPath,
    NCatboostOptions::TPoolLoadParams* params
);

void ParseFeatureEvalCommandLine(
    int argc,
    const char* argv[],
    NJson::TJsonValue* plainJsonPtr,
    NJson::TJsonValue* featureEvalOptions,
    TString* paramPath,
    NCatboostOptions::TPoolLoadParams* params
);

void ParseFeaturesSelectCommandLine(
    int argc,
    const char* argv[],
    NJson::TJsonValue* plainJsonPtr,
    TString* paramPath,
    NCatboostOptions::TPoolLoadParams* params
);

void InitOptions(
    const TString& optionsFile,
    NJson::TJsonValue* catBoostJsonOptions,
    NJson::TJsonValue* outputOptionsJson,
    NJson::TJsonValue* featuresSelectOptions = nullptr
);

void CopyIgnoredFeaturesToPoolParams(
    const NJson::TJsonValue& catBoostJsonOptions,
    NCatboostOptions::TPoolLoadParams* poolLoadParams
);

void BindQuantizerBinarizationParams(
    NLastGetopt::TOpts* parserPtr,
    NJson::TJsonValue* plainJsonPtr
);

void BindQuantizerDataProcessingParams(
    NLastGetopt::TOpts* parserPtr,
    NJson::TJsonValue* plainJsonPtr
);

void BindQuantizerPoolLoadParams(
    NLastGetopt::TOpts* parser,
    NCatboostOptions::TPoolLoadParams* loadParamsPtr
);
