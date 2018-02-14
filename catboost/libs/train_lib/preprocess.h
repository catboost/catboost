#pragma once

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/algo/train_data.h>

#include <util/generic/vector.h>

// Preprocessing of weights and targets. Loss specific check that target, weights and queryId are correct.
void PreprocessAndCheck(const NCatboostOptions::TLossDescription& lossDescription,
                        int learnSampleCount,
                        const TVector<ui32>& queryId,
                        const TVector<float>& classWeights,
                        TVector<float>* weights,
                        TVector<float>* target);

TTrainData BuildTrainData(ELossFunction lossFunction, const TPool& train, const TPool& test);
