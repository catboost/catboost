#pragma once

#include "element_range.h"
#include "exception.h"

#include <catboost/libs/data_types/pair.h>

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>

// Returns pair <start, end> for each part.
TVector<std::pair<size_t, size_t>> Split(size_t docCount, int partCount);
TVector<std::pair<size_t, size_t>> Split(size_t docCount, const TVector<ui32>& queryId, int partCount);

// Returns vector of document indices for each part.
TVector<TVector<size_t>> StratifiedSplit(const TVector<float>& target, int partCount);

// Split pairs into learn and test pairs
void SplitPairs(
    const TVector<TPair>& pairs,
    int testDocsBegin,
    int testDocsEnd,
    TVector<TPair>* learnPairs,
    TVector<TPair>* testPairs
);
