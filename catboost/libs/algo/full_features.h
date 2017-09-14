#pragma once
#include "params.h"

#include <catboost/libs/data/pool.h>

#include <library/binsaver/bin_saver.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>

using TIndexType = ui32;

struct TAllFeatures {
    yvector<yvector<ui8>> FloatHistograms; // [featureIdx][doc]
    // FloatHistograms[featureIdx] might be empty if feature is const.
    yvector<yvector<int>> CatFeatures; // [featureIdx][doc]
    yvector<yvector<int>> CatFeaturesRemapped; // [featureIdx][doc]
    yvector<yvector<int>> OneHotValues; // [featureIdx][valueIdx]
    yvector<bool> IsOneHot;
};

const int LearnNotSet = -1;

void PrepareAllFeaturesFromPermutedDocs(const yvector<TDocInfo>& docInfos,
                                        const yvector<size_t>& docIndices,
                                        const yhash_set<int>& categFeatures,
                                        const yvector<yvector<float>>& allBorders,
                                        const yvector<bool>& hasNans,
                                        const yvector<int>& ignoredFeatures,
                                        int learnSampleCount,
                                        size_t oneHotMaxSize,
                                        ENanMode NanMode,
                                        NPar::TLocalExecutor& localExecutor,
                                        TAllFeatures* allFeatures);

void PrepareAllFeatures(const yvector<TDocInfo>& docInfos,
                        const yhash_set<int>& categFeatures,
                        const yvector<yvector<float>>& allBorders,
                        const yvector<bool>& hasNans,
                        const yvector<int>& ignoredFeatures,
                        int learnSampleCount,
                        size_t oneHotMaxSize,
                        ENanMode nanMode,
                        NPar::TLocalExecutor& localExecutor,
                        TAllFeatures* allFeatures);
