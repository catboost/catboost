#pragma once

#include "enums.h"
#include "option.h"
#include "json_helper.h"
#include "cross_validation_params.h"

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/system/fs.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    struct TPoolLoadParams {
        TCvDataPartitionParams CvParams;

        TString LearnFile;
        TString CdFile;
        TVector<TString> TestFiles;

        TString PairsFile;
        TString TestPairsFile;

        bool HasHeader = false;
        char Delimiter = '\t';
        TVector<int> IgnoredFeatures;

        TPoolLoadParams() = default;

        void Validate(TMaybe<ETaskType> taskType = {}) const {
            CB_ENSURE(LearnFile.size(), "Error: provide learn dataset");
            CB_ENSURE(NFs::Exists(LearnFile), "Error: features file doesn't exist");

            if (!CdFile.empty()) {
                CB_ENSURE(NFs::Exists(CdFile), "CD-file doesn't exist");
            }
            if (taskType.Defined() && taskType.GetRef() == ETaskType::GPU) {
                CB_ENSURE(TestFiles.size() < 2, "Multiple eval sets are not supported on GPU");
            }
            for (const auto& testFile : TestFiles) {
                CB_ENSURE(NFs::Exists(testFile), "Error: test file '" << testFile << "' doesn't exist");
            }

            if (!PairsFile.empty()) {
                CB_ENSURE(NFs::Exists(PairsFile), "Error: pairs file doesn't exist");
            }

            if (!TestPairsFile.empty()) {
                CB_ENSURE(NFs::Exists(TestPairsFile), "Error: test pairs file doesn't exist");
            }
        }
    };
}
