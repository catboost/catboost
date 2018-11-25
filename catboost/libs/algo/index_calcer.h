#pragma once

#include "fold.h"
#include "split.h"

#include <catboost/libs/data/dataset.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/data/quantized_features.h>
#include <catboost/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

void SetPermutedIndices(const TSplit& split,
                        const TAllFeatures& features,
                        int curDepth,
                        const TFold& fold,
                        TVector<TIndexType>* indices,
                        NPar::TLocalExecutor* localExecutor);

TVector<bool> GetIsLeafEmpty(int curDepth, const TVector<TIndexType>& indices);

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty);

TVector<TIndexType> BuildIndices(const TFold& fold,
                                 const TSplitTree& tree,
                                 const TDataset& learnData,
                                 const TDatasetPtrs& testDataPtrs,
                                 NPar::TLocalExecutor* localExecutor);

struct TFullModel;

void BinarizeFeatures(const TFullModel& model,
                      const TPool& pool,
                      size_t start,
                      size_t end,
                      TVector<ui8>* result);

TVector<ui8> BinarizeFeatures(const TFullModel& model,
                              const TPool& pool,
                              size_t start,
                              size_t end);

TVector<ui8> BinarizeFeatures(const TFullModel& model, const TPool& pool);

TVector<TIndexType> BuildIndicesForBinTree(const TFullModel& model,
                                           const TVector<ui8>& binarizedFeatures,
                                           size_t treeId);
