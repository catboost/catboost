#pragma once

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/algo/dataset.h>

#include <util/generic/vector.h>

/// Preprocess targets and weights of the `data` as required by loss.
void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                TDataset& learnOrTestData);

TDataset BuildTrainData(const TPool& pool);

/// Check consistency of the data with each other and with loss, after Preprocess.
void CheckConsistency(const NCatboostOptions::TLossDescription& lossDescription,
                      const TDataset& learnData,
                      const TDataset& testData);
