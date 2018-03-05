#pragma once

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/algo/train_data.h>

#include <util/generic/vector.h>

// Preprocessing of weights and targets. Loss specific check that target, weights and queryId are correct.
void PreprocessAndCheck(const NCatboostOptions::TLossDescription& lossDescription,
                        int learnSampleCount,
                        const TVector<ui32>& queryId,
                        const TVector<TPair>& pairs,
                        const TVector<float>& classWeights,
                        TVector<float>* weights,
                        TVector<float>* target);

/// Preprocess targets and weights of the `data` as required by loss.
void Preprocess(const NCatboostOptions::TLossDescription& lossDescription,
                const TVector<float>& classWeights,
                TTrainData& data);

TTrainData BuildTrainData(const TPool& pool);

/// Check consistency of the data with each other and with loss.
void CheckConsistency1(ELossFunction lossFunction,
                       const TTrainData& learnData,
                       const TTrainData& testData);

/// Check consistency of the data with each other and with loss, after Preprocess.
void CheckConsistency2(const NCatboostOptions::TLossDescription& lossDescription,
                       const TTrainData& learnData,
                       const TTrainData& testData);
