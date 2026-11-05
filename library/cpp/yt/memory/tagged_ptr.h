#pragma once

#include <util/system/types.h>

#include <library/cpp/yt/misc/strong_typedef.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_STRONG_TYPEDEF(TPackedPtr, ui64);

#if defined(_64_)
    constexpr size_t PackedPtrAddressBits = 48;
#elif defined(_32_)
    constexpr size_t PackedPtrAddressBits = 32;
#else
    #error Unsupported platform
#endif
constexpr size_t PackedPtrTagBits = 16;
constexpr TPackedPtr::TUnderlying PackedPtrAddressMask = (1ULL << PackedPtrAddressBits) - 1;
constexpr TPackedPtr::TUnderlying PackedPtrTagMask = ~PackedPtrAddressMask;

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TTaggedPtr
{
    TTaggedPtr() = default;
    TTaggedPtr(T* ptr, ui16 tag);
    explicit TTaggedPtr(T* ptr);

    T* Ptr = nullptr;
    ui16 Tag = 0;

    TPackedPtr Pack() const;
    static TTaggedPtr<T> Unpack(TPackedPtr packedPtr);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define TAGGED_PTR_INL_H_
#include "tagged_ptr-inl.h"
#undef TAGGED_PTR_INL_H_
