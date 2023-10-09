#include "core-inl.h"

#include <util/system/compiler.h>

#include <library/cpp/malloc/api/malloc.h>

#include <library/cpp/yt/memory/memory_tag.h>

namespace NYT::NYTAlloc {

////////////////////////////////////////////////////////////////////////////////
// YTAlloc public API

#ifdef YT_ALLOC_ENABLED

void* Allocate(size_t size)
{
    return AllocateInline(size);
}

void* AllocateSmall(size_t rank)
{
    return AllocateSmallInline(rank);
}

void* AllocatePageAligned(size_t size)
{
    return AllocatePageAlignedInline(size);
}

void Free(void* ptr)
{
    FreeInline(ptr);
}

void FreeNonNull(void* ptr)
{
    FreeNonNullInline(ptr);
}

size_t GetAllocationSize(const void* ptr)
{
    return GetAllocationSizeInline(ptr);
}

size_t GetAllocationSize(size_t size)
{
    return GetAllocationSizeInline(size);
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYTAlloc

namespace NYT {

using namespace NYTAlloc;

////////////////////////////////////////////////////////////////////////////////
// Memory tags API bridge

TMemoryTag GetCurrentMemoryTag()
{
    return NYTAlloc::TThreadManager::GetCurrentMemoryTag();
}

void SetCurrentMemoryTag(TMemoryTag tag)
{
    TThreadManager::SetCurrentMemoryTag(tag);
}

void GetMemoryUsageForTags(const TMemoryTag* tags, size_t count, size_t* results)
{
    InitializeGlobals();
    StatisticsManager->GetTaggedMemoryUsage(tags, count, results);
}

size_t GetMemoryUsageForTag(TMemoryTag tag)
{
    size_t result;
    GetMemoryUsageForTags(&tag, 1, &result);
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

namespace NYT::NYTAlloc {

////////////////////////////////////////////////////////////////////////////////
// Memory zone API bridge

void SetCurrentMemoryZone(EMemoryZone zone)
{
    TThreadManager::SetCurrentMemoryZone(zone);
}

EMemoryZone GetCurrentMemoryZone()
{
    return TThreadManager::GetCurrentMemoryZone();
}

EMemoryZone GetAllocationMemoryZone(const void* ptr)
{
    auto uintptr = reinterpret_cast<uintptr_t>(ptr);
    if (uintptr >= MinUntaggedSmallPtr && uintptr < MaxUntaggedSmallPtr ||
        uintptr >= MinTaggedSmallPtr && uintptr < MaxTaggedSmallPtr ||
        uintptr >= DumpableLargeZoneStart && uintptr < DumpableLargeZoneEnd)
    {
        return EMemoryZone::Normal;
    } else if (uintptr >= UndumpableLargeZoneStart && uintptr < UndumpableLargeZoneEnd) {
        return EMemoryZone::Undumpable;
    } else {
        return EMemoryZone::Unknown;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Fiber id API

void SetCurrentFiberId(TFiberId id)
{
    TThreadManager::SetCurrentFiberId(id);
}

TFiberId GetCurrentFiberId()
{
    return TThreadManager::GetCurrentFiberId();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYTAlloc

////////////////////////////////////////////////////////////////////////////////
// Libc malloc bridge

#ifdef YT_ALLOC_ENABLED

using namespace NYT::NYTAlloc;

extern "C" void* malloc(size_t size)
{
    return AllocateInline(size);
}

extern "C" void* valloc(size_t size)
{
    return AllocatePageAlignedInline(size);
}

extern "C" void* aligned_alloc(size_t alignment, size_t size)
{
    // Alignment must be a power of two.
    Y_ABORT_UNLESS((alignment & (alignment - 1)) == 0);
    // Alignment must not exceed the page size.
    Y_ABORT_UNLESS(alignment <= PageSize);
    if (alignment <= 16) {
        // Proper alignment here is automatic.
        return Allocate(size);
    } else {
        return AllocatePageAligned(size);
    }
}

extern "C" void* pvalloc(size_t size)
{
    return valloc(AlignUp(size, PageSize));
}

extern "C" int posix_memalign(void** ptrPtr, size_t alignment, size_t size)
{
    *ptrPtr = aligned_alloc(alignment, size);
    return 0;
}

extern "C" void* memalign(size_t alignment, size_t size)
{
    return aligned_alloc(alignment, size);
}

extern "C" void* __libc_memalign(size_t alignment, size_t size)
{
    return aligned_alloc(alignment, size);
}

extern "C" void free(void* ptr)
{
    FreeInline(ptr);
}

extern "C" void* calloc(size_t n, size_t elemSize)
{
    // Overflow check.
    auto size = n * elemSize;
    if (elemSize != 0 && size / elemSize != n) {
        return nullptr;
    }

    void* result = Allocate(size);
    ::memset(result, 0, size);
    return result;
}

extern "C" void cfree(void* ptr)
{
    Free(ptr);
}

extern "C" void* realloc(void* oldPtr, size_t newSize)
{
    if (!oldPtr) {
        return Allocate(newSize);
    }

    if (newSize == 0) {
        Free(oldPtr);
        return nullptr;
    }

    void* newPtr = Allocate(newSize);
    size_t oldSize = GetAllocationSize(oldPtr);
    ::memcpy(newPtr, oldPtr, std::min(oldSize, newSize));
    Free(oldPtr);
    return newPtr;
}

extern "C" size_t malloc_usable_size(void* ptr) noexcept
{
    return GetAllocationSize(ptr);
}

extern "C" size_t nallocx(size_t size, int /* flags */) noexcept
{
    return GetAllocationSize(size);
}

#endif

namespace NMalloc {

////////////////////////////////////////////////////////////////////////////////
// Arcadia malloc API bridge

TMallocInfo MallocInfo()
{
    TMallocInfo info;
    info.Name = "ytalloc";
    return info;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NMalloc
