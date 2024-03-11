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
        void ValidatePoolParams() const;

    public:
        NCatboostOptions::TColumnarPoolFormatParams ColumnarPoolFormatParams;

        NCB::TPathWithScheme PoolPath;

        NCB::TPathWithScheme FeatureNamesPath;
        NCB::TPathWithScheme PoolMetaInfoPath;
    };

    class TDatasetReadingParams : public TDatasetReadingBaseParams {
    public:
        void BindParserOpts(NLastGetopt::TOpts* parser);
        void ValidatePoolParams() const;

    public:
        TVector<NJson::TJsonValue> ClassLabels;

        NCB::TPathWithScheme PairsFilePath;

        bool LoadSampleIds = false;
        bool ForceUnitAutoPairWeights = false;

        TVector<ui32> IgnoredFeatures;
    };

    void BindColumnarPoolFormatParams(
        NLastGetopt::TOpts* parser,
        NCatboostOptions::TColumnarPoolFormatParams* columnarPoolFormatParams);
}
