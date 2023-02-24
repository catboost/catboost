#ifndef TAGGED_PTR_INL_H_
#error "Direct inclusion of this file is not allowed, include tagged_ptr.h"
// For the sake of sane code completion.
#include "tagged_ptr.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE TTaggedPtr<T>::TTaggedPtr(T* ptr, ui16 tag)
    : Ptr(ptr)
    , Tag(tag)
{ }

template <class T>
Y_FORCE_INLINE TTaggedPtr<T>::TTaggedPtr(T* ptr)
    : Ptr(ptr)
    , Tag(0)
{ }

template <class T>
Y_FORCE_INLINE TPackedPtr TTaggedPtr<T>::Pack() const
{
    YT_ASSERT((reinterpret_cast<TPackedPtr>(Ptr) & PackedPtrTagMask) == 0);
    return (static_cast<TPackedPtr>(Tag) << PackedPtrAddrsssBits) | reinterpret_cast<TPackedPtr>(Ptr);
}

template <class T>
Y_FORCE_INLINE TTaggedPtr<T> TTaggedPtr<T>::Unpack(TPackedPtr packedPtr)
{
    return {
        reinterpret_cast<T*>(packedPtr & PackedPtrAddressMask),
        static_cast<ui16>(packedPtr >> PackedPtrAddrsssBits),
    };
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
