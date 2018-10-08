#pragma once

#include <catboost/libs/data_util/path_with_scheme.h>

#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/options/output_file_options.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_reader.h>

#include <util/system/info.h>

struct TAnalyticalModeCommonParams {
    NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;

    TString ModelFileName;
    EModelType ModelFormat = EModelType::CatboostBinary;
    NCB::TPathWithScheme OutputPath;

    int Verbose;

    NCB::TPathWithScheme InputPath;

    TVector<EPredictionType> PredictionTypes = {EPredictionType::RawFormulaVal};
    TVector<TString> OutputColumnsIds = {"DocId", "RawFormulaVal"};
    EFstrType FstrType = EFstrType::FeatureImportance;
    TVector<TString> ClassNames;
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();

    NCB::TPathWithScheme PairsFilePath;

    void BindParserOpts(NLastGetopt::TOpts& parser);
};

