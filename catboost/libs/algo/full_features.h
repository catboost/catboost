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
    size_t GetDocCount() const;
    SAVELOAD(FloatHistograms, CatFeaturesRemapped, OneHotValues, IsOneHot);
};

inline int GetDocCount(const TAllFeatures& allFeatures) {
    return static_cast<int>(allFeatures.GetDocCount());
}

/// Binarize data from `learnDocStorage` into `learnFeatures`.
/// One-hot encode categorial features if represented by `oneHotMaxSize` or fewer values.
/// @param categFeatures - Indices of cat-features
/// @param floatFeatures - Borders for binarization
/// @param ignoredFeatures - Make empty binarized slots for these features
/// @param ignoreRedundantCatFeatures - Make empty binarized slots if all cat-values are same
/// @param oneHotMaxSize - Limit on the number of cat-values for one-hot encoding
/// @param nanMode - Select interpretation of NaN values of float features
/// @param clearPool - Discard features from `learnDocStorage` right after binarization
/// @param localExecutor - Thread provider
/// @param selectedDocIndices - Samples in `learnDocStorage` to binarize (empty == all)
/// @param learnDocStorage - Discardable raw features
/// @param learnFeatures - Destination for binarization
void PrepareAllFeaturesLearn(const THashSet<int>& categFeatures,
                             const TVector<TFloatFeature>& floatFeatures,
                             const TVector<int>& ignoredFeatures,
                             bool ignoreRedundantCatFeatures,
                             size_t oneHotMaxSize,
                             ENanMode nanMode,
                             bool clearPool,
                             NPar::TLocalExecutor& localExecutor,
                             const TVector<size_t>& selectedDocIndices,
                             TDocumentStorage* learnDocStorage,
                             TAllFeatures* learnFeatures);

/// Binarize data from `docStorage` into `testFeatures`.
/// Align feature processing to that of `learnFeatures`.
/// @param categFeatures - Indices of cat-features
/// @param floatFeatures - Borders for binarization
/// @param learnFeatures - Binarized learn features for reference
/// @param nanMode - Select interpretation of NaN values of float features
/// @param clearPool - Discard features from `testDocStorage` right after binarization
/// @param localExecutor - Thread provider
/// @param selectedDocIndices - Samples in `testDocStorage` to binarize (empty == all)
/// @param testDocStorage - Discardable raw features
/// @param testFeatures - Destination for binarization
void PrepareAllFeaturesTest(const THashSet<int>& categFeatures,
                            const TVector<TFloatFeature>& floatFeatures,
                            const TAllFeatures& learnFeatures,
                            ENanMode nanMode,
                            bool clearPool,
                            NPar::TLocalExecutor& localExecutor,
                            const TVector<size_t>& selectedDocIndices,
                            TDocumentStorage* testDocStorage,
                            TAllFeatures* testFeatures);
