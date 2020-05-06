#pragma once

#include <util/generic/fwd.h>

template <class TKey,
          class TValue,
          class TKeyHash = THash<TKey>,
          size_t MaxLoadFactor = 50, // in percents
          size_t LogInitSize = 8>
class TDenseHash;

template <class TKey,
          class TKeyHash = THash<TKey>,
          size_t MaxLoadFactor = 50,  // in percents
          size_t LogInitSize = 8>
class TDenseHashSet;
