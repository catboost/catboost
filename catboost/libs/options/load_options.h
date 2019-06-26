#pragma once

#include "enums.h"
#include "cross_validation_params.h"

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/data_util/path_with_scheme.h>

#include <library/binsaver/bin_saver.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    struct TDsvPoolFormatParams {
        NCB::TDsvFormatOptions Format;
        NCB::TPathWithScheme CdFilePath;

        TDsvPoolFormatParams() = default;

        void Validate() const;

        SAVELOAD(Format, CdFilePath);
    };

    struct TPoolLoadParams {
        TCvDataPartitionParams CvParams;

        TDsvPoolFormatParams DsvPoolFormatParams;

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
            CvParams, DsvPoolFormatParams, LearnSetPath, TestSetPaths,
            PairsFilePath, TestPairsFilePath, GroupWeightsFilePath, TestGroupWeightsFilePath,
            BaselineFilePath, TestBaselineFilePath, ClassNames, IgnoredFeatures,
            BordersFile
        );
    };

    void ValidatePoolParams(
        const NCB::TPathWithScheme& poolPath,
        const TDsvPoolFormatParams& poolFormatParams
    );
}
