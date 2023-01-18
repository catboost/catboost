#pragma once

#include "dataset_reading_params.h"
#include "enums.h"
#include "output_file_options.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <util/generic/maybe.h>
#include <util/system/info.h>

namespace NLastGetopt {
    class TOpts;
}

namespace NCB {
    struct TAnalyticalModeCommonParams {
        NCatboostOptions::TDatasetReadingParams DatasetReadingParams;

        TString ModelFileName;
        EModelType ModelFormat = EModelType::CatboostBinary;
        NCB::TPathWithScheme OutputPath;

        int Verbose;

        TVector<EPredictionType> PredictionTypes = {EPredictionType::RawFormulaVal};
        bool IsUncertaintyPrediction = false;
        TVector<TString> OutputColumnsIds = {"SampleId", "RawFormulaVal"};
        EFstrType FstrType = EFstrType::FeatureImportance;
        int ThreadCount = NSystemInfo::CachedNumberOfCpus();

        ECalcTypeShapValues ShapCalcType = ECalcTypeShapValues::Regular;
        TMaybe<double> BinClassLogitThreshold;

        void BindParserOpts(NLastGetopt::TOpts& parser);
    };

    TString BuildModelFormatHelpMessage();

    void BindModelFileParams(NLastGetopt::TOpts* parser, TString* modelFileName, EModelType* modelFormat);
}
