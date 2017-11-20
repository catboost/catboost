#pragma once

#include <catboost/libs/params/params.h>

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
    TVector<EPredictionType> PredictionTypes = {EPredictionType::RawFormulaVal};
    EFstrType FstrType = EFstrType::FeatureImportance;
    TVector<TString> ClassNames;
    int ThreadCount = 1;
    char Delimiter = '\t';
    bool HasHeader = false;
    TString PairsFile = "";

    void BindParserOpts(NLastGetopt::TOpts& parser);
};
