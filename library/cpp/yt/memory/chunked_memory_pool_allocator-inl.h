#ifndef CHUNKED_MEMORY_POOL_ALLOCATOR_INL_H_
#error "Direct inclusion of this file is not allowed, include chunked_memory_pool_allocator.h"
// For the sake of sane code completion.
#include "chunked_memory_pool_allocator.h"
#endif

#include <util/system/align.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
TChunkedMemoryPoolAllocator<T> TChunkedMemoryPoolAllocator<T>::select_on_container_copy_construction(
    const TChunkedMemoryPoolAllocator& allocator) noexcept
{
    return allocator;
}

template <class T>
TChunkedMemoryPoolAllocator<T>::TChunkedMemoryPoolAllocator(
    TChunkedMemoryPool* pool) noexcept
    : Pool_(pool)
{ }

template <class T>
template <class U>
TChunkedMemoryPoolAllocator<T>::TChunkedMemoryPoolAllocator(
    const TChunkedMemoryPoolAllocator<U>& other) noexcept
    : Pool_(other.Pool_)
{ }

template <class T>
T* TChunkedMemoryPoolAllocator<T>::allocate(std::size_t count)
{
    return reinterpret_cast<T*>(Pool_->AllocateAligned(count * sizeof(T), alignof(T)));
}

template <class T>
void TChunkedMemoryPoolAllocator<T>::deallocate(
    T* /*pointer*/,
    std::size_t /*count*/) const noexcept
{ }

template <class T>
bool TChunkedMemoryPoolAllocator<T>::operator==(
    const TChunkedMemoryPoolAllocator& other) const noexcept
{
    return Pool_ == other.Pool_;
}

template <class T>
bool TChunkedMemoryPoolAllocator<T>::operator!=(
    const TChunkedMemoryPoolAllocator& other) const noexcept
{
    return !(*this == other);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
