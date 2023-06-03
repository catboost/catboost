#pragma once

#include "public.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////
// Memory tagging API
//
// Each allocation can be tagged with a number (from 1 to MaxMemoryTag).
// Setting this to NullMemoryTag disables tagging.
// Internally, YTAlloc tracks the number of bytes used by each tag.
//
// Tagged allocations are somewhat slower. Others (large and huge) are not affected
// (but for these performance implications are negligible anyway).
//
// The current memory tag used for allocations is stored in TLS.

// Updates the current tag value in TLS.
void SetCurrentMemoryTag(TMemoryTag tag);

// Returns the current tag value from TLS.
TMemoryTag GetCurrentMemoryTag();

// Returns the memory usage for a given tag.
// The value is somewhat approximate and racy.
size_t GetMemoryUsageForTag(TMemoryTag tag);

// A batched version of GetMemoryUsageForTag.
void GetMemoryUsageForTags(const TMemoryTag* tags, size_t count, size_t* results);

////////////////////////////////////////////////////////////////////////////////

//! An RAII guard for setting the current memory tag in a scope.
class TMemoryTagGuard
{
public:
    TMemoryTagGuard();
    explicit TMemoryTagGuard(TMemoryTag tag);

    TMemoryTagGuard(const TMemoryTagGuard& other) = delete;
    TMemoryTagGuard(TMemoryTagGuard&& other);

    ~TMemoryTagGuard();

private:
    bool Active_;
    TMemoryTag PreviousTag_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define MEMORY_TAG_INL_H_
#include "memory_tag-inl.h"
#undef MEMORY_TAG_INL_H_
