#pragma once

#include "ref_counted.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// TODO(babenko): consider increasing to 128 due to cache line pairing in L2 prefetcher.
constexpr size_t CacheLineSize = 64;

class TChunkedMemoryPool;

DECLARE_REFCOUNTED_STRUCT(IMemoryChunkProvider)
DECLARE_REFCOUNTED_STRUCT(IMemoryUsageTracker)
DECLARE_REFCOUNTED_STRUCT(IReservingMemoryUsageTracker)
DECLARE_REFCOUNTED_STRUCT(TSharedRangeHolder)

using TMemoryTag = ui32;
constexpr TMemoryTag NullMemoryTag = 0;
constexpr TMemoryTag MaxMemoryTag = (1ULL << 22) - 1;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
