#pragma once

#ifdef NETLIBA_WITH_NALF
#include <library/cpp/malloc/nalf/alloc_helpers.h>
using TWithCustomAllocator = TWithNalfIncrementalAlloc;
template <typename T>
using TCustomAllocator = TNalfIncrementalAllocator<T>;
#else
#include <memory>
typedef struct {
} TWithCustomAllocator;
template <typename T>
using TCustomAllocator = std::allocator<T>;
#endif
