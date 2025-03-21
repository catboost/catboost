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

        TVector<TString> ModelFileName; // [modelIdx]
        EModelType ModelFormat = EModelType::CatboostBinary;
        NCB::TPathWithScheme OutputPath;

        int Verbose;

        bool IsUncertaintyPrediction = false;
        bool ForceSingleModel = false; // true, if prediction-type is used
        TVector<TVector<TString>> OutputColumnsIds; // [modelIdx]
        EFstrType FstrType = EFstrType::FeatureImportance;
        int ThreadCount = NSystemInfo::CachedNumberOfCpus();
        TString AnnotationsJson;

        ECalcTypeShapValues ShapCalcType = ECalcTypeShapValues::Regular;
        TMaybe<double> BinClassLogitThreshold;

        TString BlendingExpression;

        void BindParserOpts(NLastGetopt::TOpts& parser);
    };

    TString BuildModelFormatHelpMessage();

    void BindModelFileParams(NLastGetopt::TOpts* parser, TVector<TString>* modelFileName, EModelType* modelFormat);
}
