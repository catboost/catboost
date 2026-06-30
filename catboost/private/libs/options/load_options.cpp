#include "load_options.h"

#include <catboost/private/libs/data_util/exists_checker.h>

void NCatboostOptions::TColumnarPoolFormatParams::Validate() const {
    if (CdFilePath.Inited()) {
        CB_ENSURE(CheckExists(CdFilePath), "CD-file '" << CdFilePath << "' doesn't exist");
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
        CB_ENSURE(CheckExists(TestPairsFilePath), "Error: test pairs file '" << TestPairsFilePath << "' doesn't exist");
    }

    if (TestGroupWeightsFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestGroupWeightsFilePath),
                "Error: test group weights file '" << TestGroupWeightsFilePath << "' doesn't exist");
    }

    if (TestTimestampsFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestTimestampsFilePath),
                "Error: test timestamps file '" << TestTimestampsFilePath << "' doesn't exist");
    }

    if (TestBaselineFilePath.Inited()) {
        CB_ENSURE(CheckExists(TestBaselineFilePath),
                  "Error: test baseline file '" << TestBaselineFilePath << "' doesn't exist");
    }

    if (!PrecomputedMetadataFile.empty()) {
        CB_ENSURE(CheckExists(NCB::TPathWithScheme(PrecomputedMetadataFile)),
                  "Error: precomputed metadata file '" << PrecomputedMetadataFile << "' doesn't exist");
    }
}

void NCatboostOptions::TPoolLoadParams::ValidateLearn() const {
    ColumnarPoolFormatParams.Validate();

    CB_ENSURE(LearnSetPath.Inited(), "Error: provide learn dataset");
    CB_ENSURE(CheckExists(LearnSetPath), "Error: features path '" << LearnSetPath << "' doesn't exist");
    ValidatePoolParams(LearnSetPath, ColumnarPoolFormatParams);

    if (PairsFilePath.Inited()) {
        CB_ENSURE(CheckExists(PairsFilePath), "Error: pairs file '" << PairsFilePath << "' doesn't exist");
    }

    if (GraphFilePath.Inited()) {
        CB_ENSURE(CheckExists(GraphFilePath), "Error: graph file '" << GraphFilePath << "' doesn't exist");
    }


    if (GroupWeightsFilePath.Inited()) {
        CB_ENSURE(CheckExists(GroupWeightsFilePath), "Error: group weights file '" << GroupWeightsFilePath << "' doesn't exist");
    }

    if (TimestampsFilePath.Inited()) {
        CB_ENSURE(CheckExists(TimestampsFilePath), "Error: timestamps file '" << TimestampsFilePath << "' doesn't exist");
    }

    if (BaselineFilePath.Inited()) {
        CB_ENSURE(CheckExists(BaselineFilePath), "Error: baseline file '" << BaselineFilePath << "' doesn't exist");
    }
}

bool NCatboostOptions::TPoolLoadParams::HavePairs() const {
    return PairsFilePath.Inited() || TestPairsFilePath.Inited();
}

bool NCatboostOptions::TPoolLoadParams::HaveGraph() const {
    return GraphFilePath.Inited() || TestGraphFilePath.Inited();
}

void NCatboostOptions::ValidatePoolParams(
    const NCB::TPathWithScheme& poolPath,
    const NCB::TDsvFormatOptions& dsvFormat
) {
    CB_ENSURE(
        poolPath.Scheme == "dsv" || !dsvFormat.HasHeader,
        "HasHeader parameter supported for \"dsv\" pools only."
    );
}

void NCatboostOptions::ValidatePoolParams(
    const NCB::TPathWithScheme& poolPath,
    const TColumnarPoolFormatParams& poolFormatParams
) {
    NCatboostOptions::ValidatePoolParams(poolPath, poolFormatParams.DsvFormat);
}
