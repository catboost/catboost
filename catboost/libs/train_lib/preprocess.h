#pragma once

#include <catboost/libs/data/dataset.h>
#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/data_processing_options.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/options/output_file_options.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/fwd.h>

#include <functional>


namespace NJson {
    class TJsonValue;
}


/// Preprocess targets and weights of the `data` as required by loss.
void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                const TLabelConverter& labelConverter,
                TDataset& learnOrTestData);

/// Check consistency of the data with loss and with each other, after Preprocess.
/// Check 1 of 2: consistency of the learnData itself.
void CheckLearnConsistency(
    const NCatboostOptions::TLossDescription& lossDescription,
    bool allowConstLabel,
    const TDataset& learnData
);
/// Check 2 of 2: consistency of the testData sets with the learnData.
void CheckTestConsistency(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TDataset& learnData,
    const TDataset& testData
);

void UpdateUndefinedRandomSeed(
    ETaskType taskType,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    NJson::TJsonValue* updatedJsonParams,
    std::function<void(TIFStream*, TString&)> paramsLoader
);

void UpdateUndefinedClassNames(
    const NCatboostOptions::TDataProcessingOptions& dataProcessingOptions,
    NJson::TJsonValue* updatedJsonParams
);
