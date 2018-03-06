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
                        TLearnContext* ctx);

int GetRedundantSplitIdx(int curDepth, const TVector<TIndexType>& indices);

void DeleteSplit(int curDepth, int redundantIdx, TSplitTree* tree, TVector<TIndexType>* indices);

TVector<TIndexType> BuildIndices(const TFold& fold,
                          const TSplitTree& tree,
                          const TTrainData& data,
                          NPar::TLocalExecutor* localExecutor);

int GetDocCount(const TAllFeatures& features);

struct TFullModel;

TVector<ui8> BinarizeFeatures(const TFullModel& model, const TPool& pool);

TVector<TIndexType> BuildIndicesForBinTree(const TFullModel& model,
                                           const TVector<ui8>& binarizedFeatures,
                                           size_t treeId);
