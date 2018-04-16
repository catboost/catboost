#pragma once

#include "fold.h"
#include "learn_context.h"
#include "split.h"

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
                                 const TDataset* testData,
                                 NPar::TLocalExecutor* localExecutor);

struct TFullModel;

TVector<ui8> BinarizeFeatures(const TFullModel& model,
                              const TPool& pool,
                              size_t start,
                              size_t end);

TVector<ui8> BinarizeFeatures(const TFullModel& model, const TPool& pool);

TVector<TIndexType> BuildIndicesForBinTree(const TFullModel& model,
                                           const TVector<ui8>& binarizedFeatures,
                                           size_t treeId);
