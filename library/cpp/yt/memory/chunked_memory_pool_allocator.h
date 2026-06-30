#pragma once

#include "chunked_memory_pool.h"

#include <cstddef>
#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Made to use TChunkedMemoryPool with std containers.
// Does not own pool so lifetime should be no greater than lifetime of the pool.
// Redirects allocations to pool.
// Deallocations are noop.
template <class T>
class TChunkedMemoryPoolAllocator
{
public:
    using value_type = T;

    // All defines for AllocatorAwareContainer are deliberately set to the same as default.
    using is_always_equal = std::false_type;

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;

    static TChunkedMemoryPoolAllocator select_on_container_copy_construction(
        const TChunkedMemoryPoolAllocator& allocator) noexcept;

    explicit TChunkedMemoryPoolAllocator(TChunkedMemoryPool* pool) noexcept;

    template <class U>
    TChunkedMemoryPoolAllocator(const TChunkedMemoryPoolAllocator<U>& other) noexcept;

    [[nodiscard]] T* allocate(std::size_t count);
    void deallocate(T* pointer, std::size_t count) const noexcept;

    // https://en.cppreference.com/w/cpp/named_req/Allocator
    // a1 == a2
    // * true only if the storage allocated by the allocator a1 can be deallocated through a2.
    // * Establishes reflexive, symmetric, and transitive relationship.
    // * Does not throw exceptions.
    bool operator==(const TChunkedMemoryPoolAllocator& other) const noexcept;
    bool operator!=(const TChunkedMemoryPoolAllocator& other) const noexcept;

private:
    template <class U>
    friend class TChunkedMemoryPoolAllocator;

    TChunkedMemoryPool* const Pool_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CHUNKED_MEMORY_POOL_ALLOCATOR_INL_H_
#include "chunked_memory_pool_allocator-inl.h"
#undef CHUNKED_MEMORY_POOL_ALLOCATOR_INL_H_
