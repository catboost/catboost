#pragma once

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/algo/dataset.h>

#include <util/generic/vector.h>

/// Preprocess targets and weights of the `data` as required by loss.
void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                TDataset& learnOrTestData);

/// Check consistency of the data with loss and with each other, after Preprocess.
/// Check 1 of 2: consistency of the learnData itself.
void CheckLearnConsistency(const NCatboostOptions::TLossDescription& lossDescription,
                           const TDataset& learnData);
/// Check 2 of 2: consistency of the testData sets with the learnData.
void CheckTestConsistency(const NCatboostOptions::TLossDescription& lossDescription,
                          const TDataset& learnData,
                          const TDataset& testData);
