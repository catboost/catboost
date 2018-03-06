#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/features.h>
#include <library/binsaver/bin_saver.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>


struct TAllFeatures {
    TVector<TVector<ui8>> FloatHistograms; // [featureIdx][doc]
    // FloatHistograms[featureIdx] might be empty if feature is const.
    TVector<TVector<int>> CatFeaturesRemapped; // [featureIdx][doc]
    TVector<TVector<int>> OneHotValues; // [featureIdx][valueIdx]
    TVector<bool> IsOneHot;
};

const int LearnNotSet = -1;

void PrepareAllFeaturesFromPermutedDocs(const TVector<size_t>& docIndices,
                                        const THashSet<int>& categFeatures,
                                        const TVector<TFloatFeature>& floatFeatures,
                                        const TVector<int>& ignoredFeatures,
                                        int learnSampleCount,
                                        size_t oneHotMaxSize,
                                        ENanMode NanMode,
                                        bool allowClearPool,
                                        NPar::TLocalExecutor& localExecutor,
                                        TDocumentStorage* docStorage,
                                        TAllFeatures* allFeatures);

void PrepareAllFeatures(const THashSet<int>& categFeatures,
                        const TVector<TFloatFeature>& floatFeatures,
                        const TVector<int>& ignoredFeatures,
                        int learnSampleCount,
                        size_t oneHotMaxSize,
                        ENanMode nanMode,
                        bool allowClearPool,
                        NPar::TLocalExecutor& localExecutor,
                        TDocumentStorage* docStorage,
                        TAllFeatures* allFeatures);
