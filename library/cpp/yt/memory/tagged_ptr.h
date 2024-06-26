#pragma once

#include <util/system/types.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

using TPackedPtr = uintptr_t;
static_assert(sizeof(TPackedPtr) == 8);

constexpr size_t PackedPtrAddressBits = 48;
constexpr size_t PackedPtrTagBits = 16;
constexpr TPackedPtr PackedPtrAddressMask = (1ULL << PackedPtrAddressBits) - 1;
constexpr TPackedPtr PackedPtrTagMask = ~PackedPtrAddressMask;

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TTaggedPtr
{
    TTaggedPtr() = default;
    TTaggedPtr(T* ptr, ui16 tag);
    explicit TTaggedPtr(T* ptr);

    T* Ptr;
    ui16 Tag;

    TPackedPtr Pack() const;
    static TTaggedPtr<T> Unpack(TPackedPtr packedPtr);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define TAGGED_PTR_INL_H_
#include "tagged_ptr-inl.h"
#undef TAGGED_PTR_INL_H_
