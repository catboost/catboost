#pragma once

#include <catboost/libs/algo/params.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_reader.h>

void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* trainJson,
                      TCmdLineParams* params,
                      TString* paramsPath);

struct TAnalyticalModeCommonParams {
    TString ModelFileName;
    TString OutputPath;
    TString InputPath;
    TString CdFile;
    EPredictionType PredictionType = EPredictionType::RawFormulaVal;
    EFstrType FstrType = EFstrType::FeatureImportance;
    yvector<TString> ClassNames;
    int ThreadCount = 1;

    void BindParserOpts(NLastGetopt::TOpts& parser);
};
