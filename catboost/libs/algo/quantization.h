#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/data/dataset.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/features.h>

#include <library/binsaver/bin_saver.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


/// Binarize data from `learnDocStorage` into `learnFeatures`.
/// One-hot encode categorial features if represented by `oneHotMaxSize` or fewer values.
/// @param categFeatures - Indices of cat-features
/// @param floatFeatures - Borders for binarization
/// @param ignoredFeatures - Make empty binarized slots for these features
/// @param ignoreRedundantCatFeatures - Make empty binarized slots if all cat-values are same
/// @param oneHotMaxSize - Limit on the number of cat-values for one-hot encoding
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
                             bool clearPool,
                             NPar::TLocalExecutor& localExecutor,
                             const TVector<size_t>& selectedDocIndices,
                             TDocumentStorage* learnDocStorage,
                             TAllFeatures* learnFeatures);

/// Binarize data from `testDocStorage` into `testFeatures`.
/// Align feature processing to that of `learnFeatures`.
/// @param categFeatures - Indices of cat-features
/// @param floatFeatures - Borders for binarization
/// @param learnFeatures - Binarized learn features for reference
/// @param clearPool - Discard features from `testDocStorage` right after binarization
/// @param localExecutor - Thread provider
/// @param selectedDocIndices - Samples in `testDocStorage` to binarize (empty == all)
/// @param testDocStorage - Discardable raw features
/// @param testFeatures - Destination for binarization
void PrepareAllFeaturesTest(const THashSet<int>& categFeatures,
                            const TVector<TFloatFeature>& floatFeatures,
                            const TAllFeatures& learnFeatures,
                            bool allowNansOnlyInTest,
                            bool clearPool,
                            NPar::TLocalExecutor& localExecutor,
                            const TVector<size_t>& selectedDocIndices,
                            TDocumentStorage* testDocStorage,
                            TAllFeatures* testFeatures);

void QuantizeTrainPools(
    const TClearablePoolPtrs& pools,
    const TVector<TFloatFeature>& floatFeatures,
    const TVector<int>& ignoredFeatures,
    size_t oneHotMaxSize,
    NPar::TLocalExecutor& localExecutor,
    TDataset* learnData,
    TVector<TDataset>* testDatasets
);
