#pragma once

#include <memory>

template <class Allocator, class T>
using TReboundAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

class IAllocator {
public:
    struct TBlock {
        void* Data;
        size_t Len;
    };

    virtual ~IAllocator() = default;

    virtual TBlock Allocate(size_t len) = 0;
    virtual void Release(const TBlock& block) = 0;
};

class TDefaultAllocator: public IAllocator {
public:
    TBlock Allocate(size_t len) override;
    void Release(const TBlock& block) override;

    static IAllocator* Instance() noexcept;
};
