#pragma once

#include "enums.h"
#include "load_options.h"
#include "output_file_options.h"

#include <catboost/libs/data_util/path_with_scheme.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_reader.h>

#include <util/system/info.h>

namespace NCB {
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

    TString BuildModelFormatHelpMessage();

    void BindDsvPoolFormatParams(
        NLastGetopt::TOpts* parser,
        NCatboostOptions::TDsvPoolFormatParams* dsvPoolFormatParams);

    void BindModelFileParams(NLastGetopt::TOpts* parser, TString* modelFileName, EModelType* modelFormat);
}
