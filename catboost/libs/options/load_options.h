#pragma once

#include "enums.h"
#include "option.h"
#include "json_helper.h"
#include "cross_validation_params.h"

#include <util/system/types.h>
#include <util/generic/string.h>
#include <util/system/fs.h>

namespace NCatboostOptions {
    struct TPoolLoadParams {
        TCvDataPartitionParams CvParams;

        TString LearnFile;
        TString CdFile;
        TString TestFile;

        TString PairsFile;
        TString TestPairsFile;

        bool HasHeader = false;
        char Delimiter = '\t';
        TVector<int> IgnoredFeatures;

        TPoolLoadParams() = default;

        void Validate() const {
            CB_ENSURE(LearnFile.size(), "Error: provide learn dataset");
            CB_ENSURE(NFs::Exists(LearnFile), "Error: features file doesn't exist");

            if (!CdFile.empty()) {
                CB_ENSURE(NFs::Exists(CdFile), "CD-file doesn't exist");
            }
            if (!TestFile.empty()) {
                CB_ENSURE(NFs::Exists(TestFile), "Error: test file doesn't exist");
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
