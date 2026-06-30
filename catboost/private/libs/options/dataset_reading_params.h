#pragma once

#include "load_options.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NLastGetopt {
    class TOpts;
}

namespace NCatboostOptions {
    class TDatasetReadingBaseParams {
    public:
        void BindParserOpts(NLastGetopt::TOpts* parser);

    public:
        NCatboostOptions::TColumnarPoolFormatParams ColumnarPoolFormatParams;

        NCB::TPathWithScheme FeatureNamesPath;
        NCB::TPathWithScheme PoolMetaInfoPath;
    };

    class TSingleDatasetReadingParams : public TDatasetReadingBaseParams {
    public:
        void BindParserOpts(NLastGetopt::TOpts* parser);
        void ValidatePoolParams() const;

    public:
        NCB::TPathWithScheme PoolPath;
    };


    class TDatasetReadingParams : public TSingleDatasetReadingParams {
    public:
        void BindParserOpts(NLastGetopt::TOpts* parser);
        void ValidatePoolParams() const;

    public:
        TVector<NJson::TJsonValue> ClassLabels;

        NCB::TPathWithScheme PairsFilePath;
        NCB::TPathWithScheme GraphFilePath;

        bool LoadSampleIds = false;
        bool ForceUnitAutoPairWeights = false;

        TVector<ui32> IgnoredFeatures;
    };

    void BindColumnarPoolFormatParams(
        NLastGetopt::TOpts* parser,
        NCatboostOptions::TColumnarPoolFormatParams* columnarPoolFormatParams);
}
