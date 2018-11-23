#pragma once

#include "element_range.h"
#include "exception.h"

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/pair.h>

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>

// Returns pair <start, end> for each part.
TVector<std::pair<ui32, ui32>> Split(ui32 docCount, ui32 partCount);
TVector<std::pair<ui32, ui32>> Split(ui32 docCount, const TVector<TGroupId>& queryId, ui32 partCount);

// Returns vector of document indices for each part.
TVector<TVector<ui32>> StratifiedSplit(const TVector<float>& target,ui32 partCount);

// Split pairs into learn and test pairs, without changing doc indices
void SplitPairs(
    const TVector<TPair>& pairs,
    ui32 testDocsBegin,
    ui32 testDocsEnd,
    TVector<TPair>* learnPairs,
    TVector<TPair>* testPairs
);

// Split pairs into learn and test pairs, changing doc indices
void SplitPairsAndReindex(
    const TVector<TPair>& pairs,
    ui32 testDocsBegin,
    ui32 testDocsEnd,
    TVector<TPair>* learnPairs,
    TVector<TPair>* testPairs
);
