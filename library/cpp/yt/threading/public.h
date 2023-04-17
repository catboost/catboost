#pragma once

#include <cstddef>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

// TODO(babenko): consider increasing to 128 due to cache line pairing in L2 prefetcher.
constexpr size_t CacheLineSize = 64;

#define YT_DECLARE_SPIN_LOCK(type, name) \
    type name{__LOCATION__}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

