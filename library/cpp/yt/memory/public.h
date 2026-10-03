#pragma once

#include "ref_counted.h"

#include <library/cpp/yt/system/cache_line_size.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TChunkedMemoryPool;

DECLARE_REFCOUNTED_STRUCT(IMemoryChunkProvider)
DECLARE_REFCOUNTED_STRUCT(ISimpleMemoryUsageTracker)
DECLARE_REFCOUNTED_STRUCT(TSharedRangeHolder)

using TMemoryTag = ui32;
constexpr TMemoryTag NullMemoryTag = 0;
constexpr TMemoryTag MaxMemoryTag = (1ULL << 22) - 1;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
