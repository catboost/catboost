#pragma once

#include "enums.h"
#include "load_options.h"
#include "output_file_options.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_reader.h>
#include <library/json/json_value.h>

#include <util/system/info.h>

namespace NCB {
    struct TAnalyticalModeCommonParams {
        NCatboostOptions::TColumnarPoolFormatParams ColumnarPoolFormatParams;

        TString ModelFileName;
        EModelType ModelFormat = EModelType::CatboostBinary;
        NCB::TPathWithScheme OutputPath;

        int Verbose;

        NCB::TPathWithScheme InputPath;

        TVector<EPredictionType> PredictionTypes = {EPredictionType::RawFormulaVal};
        TVector<TString> OutputColumnsIds = {"SampleId", "RawFormulaVal"};
        EFstrType FstrType = EFstrType::FeatureImportance;
        TVector<NJson::TJsonValue> ClassLabels;
        int ThreadCount = NSystemInfo::CachedNumberOfCpus();

        NCB::TPathWithScheme PairsFilePath;
        NCB::TPathWithScheme FeatureNamesPath;

        void BindParserOpts(NLastGetopt::TOpts& parser);
    };

    TString BuildModelFormatHelpMessage();

    void BindColumnarPoolFormatParams(
        NLastGetopt::TOpts* parser,
        NCatboostOptions::TColumnarPoolFormatParams* columnarPoolFormatParams);

    void BindModelFileParams(NLastGetopt::TOpts* parser, TString* modelFileName, EModelType* modelFormat);
}
