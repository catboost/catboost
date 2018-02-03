#pragma once

#include "element_range.h"
#include "exception.h"

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>

// Returns pair <start, end> for each part.
TVector<std::pair<size_t, size_t>> Split(size_t docCount, int partCount);
TVector<std::pair<size_t, size_t>> Split(size_t docCount, const TVector<ui32>& queryId, int partCount);

// Returns vector of document indices for each part.
TVector<TVector<size_t>> StratifiedSplit(const TVector<float>& target, int partCount);
