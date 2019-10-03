#pragma once

#include "enums.h"
#include "cross_validation_params.h"

#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <library/binsaver/bin_saver.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    struct TColumnarPoolFormatParams {
        NCB::TDsvFormatOptions DsvFormat;
        NCB::TPathWithScheme CdFilePath;

        TColumnarPoolFormatParams() = default;

        void Validate() const;

        SAVELOAD(DsvFormat, CdFilePath);
    };

    struct TPoolLoadParams {
        TCvDataPartitionParams CvParams;

        TColumnarPoolFormatParams ColumnarPoolFormatParams;

        NCB::TPathWithScheme LearnSetPath;
        TVector<NCB::TPathWithScheme> TestSetPaths;

        NCB::TPathWithScheme PairsFilePath;
        NCB::TPathWithScheme TestPairsFilePath;

        NCB::TPathWithScheme GroupWeightsFilePath;
        NCB::TPathWithScheme TestGroupWeightsFilePath;

        NCB::TPathWithScheme BaselineFilePath;
        NCB::TPathWithScheme TestBaselineFilePath;
        TVector<TString> ClassNames;

        TVector<ui32> IgnoredFeatures;
        TString BordersFile;

        TPoolLoadParams() = default;

        void Validate() const;
        void Validate(TMaybe<ETaskType> taskType) const;

        SAVELOAD(
            CvParams, ColumnarPoolFormatParams, LearnSetPath, TestSetPaths,
            PairsFilePath, TestPairsFilePath, GroupWeightsFilePath, TestGroupWeightsFilePath,
            BaselineFilePath, TestBaselineFilePath, ClassNames, IgnoredFeatures,
            BordersFile
        );
    };

    void ValidatePoolParams(
        const NCB::TPathWithScheme& poolPath,
        const TColumnarPoolFormatParams& poolFormatParams
    );
}
