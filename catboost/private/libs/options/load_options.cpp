#include "load_options.h"

#include <catboost/private/libs/data_util/exists_checker.h>

void NCatboostOptions::TColumnarPoolFormatParams::Validate() const {
    if (CdFilePath.Inited()) {
        CB_ENSURE(CheckExists(CdFilePath), "CD-file doesn't exist");
    }
}

void NCatboostOptions::TPoolLoadParams::Validate() const {
    Validate({});
}

void NCatboostOptions::TPoolLoadParams::Validate(TMaybe<ETaskType> taskType) const {
    ValidateLearn();

    if (taskType.Defined()) {
        if (taskType.GetRef() == ETaskType::GPU) {
            CB_ENSURE(TestSetPaths.size() < 2, "Multiple eval sets are not supported on GPU");
        }
    }
    for (const auto& testSetPath : TestSetPaths) {
        CB_ENSURE(CheckExists(testSetPath), "Error: test file '" << testSetPath << "' doesn't exist");
        ValidatePoolParams(testSetPath, ColumnarPoolFormatParams);
    }

    if (TestPairsFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestPairsFilePath), "Error: test pairs file doesn't exist");
    }

    if (TestGroupWeightsFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestGroupWeightsFilePath),
                "Error: test group weights file doesn't exist");
    }

    if (TestBaselineFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestBaselineFilePath),
                  "Error: test baseline file doesn't exist");
    }
}

void NCatboostOptions::TPoolLoadParams::ValidateLearn() const {
    ColumnarPoolFormatParams.Validate();

    CB_ENSURE(LearnSetPath.Inited(), "Error: provide learn dataset");
    CB_ENSURE(CheckExists(LearnSetPath), "Error: features path doesn't exist");
    ValidatePoolParams(LearnSetPath, ColumnarPoolFormatParams);

    if (PairsFilePath.Inited()) {
        CB_ENSURE(CheckExists(PairsFilePath), "Error: pairs file doesn't exist");
    }

    if (GroupWeightsFilePath.Inited()) {
        CB_ENSURE(CheckExists(GroupWeightsFilePath), "Error: group weights file doesn't exist");
    }

    if (BaselineFilePath.Inited()) {
        CB_ENSURE(CheckExists(BaselineFilePath), "Error: baseline file doesn't exist");
    }
}

void NCatboostOptions::ValidatePoolParams(
    const NCB::TPathWithScheme& poolPath,
    const TColumnarPoolFormatParams& poolFormatParams
) {
    CB_ENSURE(
        poolPath.Scheme == "dsv" || !poolFormatParams.DsvFormat.HasHeader,
        "HasHeader parameter supported for \"dsv\" pools only."
    );
}
