#pragma once

#include "enums.h"
#include "option.h"
#include "json_helper.h"
#include "cross_validation_params.h"

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/data_util/exists_checker.h>
#include <catboost/libs/data_util/path_with_scheme.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/system/types.h>


namespace NCatboostOptions {
    struct TDsvPoolFormatParams {
        NCB::TDsvFormatOptions Format;

        NCB::TPathWithScheme CdFilePath;

        TDsvPoolFormatParams() = default;

        void Validate() const {
            if (CdFilePath.Inited()) {
                CB_ENSURE(CheckExists(CdFilePath), "CD-file doesn't exist");
            }
        }
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

        TVector<int> IgnoredFeatures;
        TString BordersFile;

        TPoolLoadParams() = default;

        void Validate(TMaybe<ETaskType> taskType = {}) const {
            DsvPoolFormatParams.Validate();

            CB_ENSURE(LearnSetPath.Inited(), "Error: provide learn dataset");
            CB_ENSURE(CheckExists(LearnSetPath), "Error: features path doesn't exist");

            if (taskType.Defined()) {
                if (taskType.GetRef() == ETaskType::GPU) {
                    CB_ENSURE(TestSetPaths.size() < 2, "Multiple eval sets are not supported on GPU");
                }
                if (taskType.GetRef() == ETaskType::CPU) {
                    CB_ENSURE(BordersFile.empty(), "Borders file is not supported on CPU");
                }
            }
            for (const auto& testFile : TestSetPaths) {
                CB_ENSURE(CheckExists(testFile), "Error: test file '" << testFile << "' doesn't exist");
            }

            if (PairsFilePath.Inited()) {
                CB_ENSURE(CheckExists(PairsFilePath), "Error: pairs file doesn't exist");
            }

            if (TestPairsFilePath.Inited()) {
                CB_ENSURE(CheckExists(TestPairsFilePath), "Error: test pairs file doesn't exist");
            }

            if (GroupWeightsFilePath.Inited()) {
                CB_ENSURE(CheckExists(GroupWeightsFilePath), "Error: group weights file doesn't exist");
            }

            if (TestGroupWeightsFilePath.Inited()) {
                CB_ENSURE(CheckExists(TestGroupWeightsFilePath),
                    "Error: test group weights file doesn't exist");
            }
        }
    };
}
