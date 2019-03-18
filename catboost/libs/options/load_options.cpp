#include "load_options.h"

#include <catboost/libs/data_util/exists_checker.h>

void NCatboostOptions::TDsvPoolFormatParams::Validate() const {
    if (CdFilePath.Inited()) {
        CB_ENSURE(CheckExists(CdFilePath), "CD-file doesn't exist");
    }
}

void NCatboostOptions::TPoolLoadParams::Validate() const {
    Validate({});
}

void NCatboostOptions::TPoolLoadParams::Validate(TMaybe<ETaskType> taskType) const {
    DsvPoolFormatParams.Validate();

    CB_ENSURE(LearnSetPath.Inited(), "Error: provide learn dataset");
    CB_ENSURE(CheckExists(LearnSetPath), "Error: features path doesn't exist");

    if (taskType.Defined()) {
        if (taskType.GetRef() == ETaskType::GPU) {
            CB_ENSURE(TestSetPaths.size() < 2, "Multiple eval sets are not supported on GPU");
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

    if (BaselineFilePath.Inited()) {
        CB_ENSURE(CheckExists(BaselineFilePath), "Error: baseline file doesn't exist");
    }

    if (TestBaselineFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestBaselineFilePath),
                  "Error: test baseline file doesn't exist");
    }
}
