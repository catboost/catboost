#pragma once

#include <catboost/libs/options/enums.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_reader.h>

#include <util/system/info.h>

struct TAnalyticalModeCommonParams {
    TString ModelFileName;
    TString OutputPath;
    TString InputPath;
    TString CdFile;
    TVector<EPredictionType> PredictionTypes = {EPredictionType::RawFormulaVal};
    TVector<TString> OutputColumnsIds = {"DocId", "RawFormulaVal"};
    EFstrType FstrType = EFstrType::FeatureImportance;
    TVector<TString> ClassNames;
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();
    char Delimiter = '\t';
    bool HasHeader = false;
    TString PairsFile = "";

    void BindParserOpts(NLastGetopt::TOpts& parser);
};
