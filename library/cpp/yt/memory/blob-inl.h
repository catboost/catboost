#ifndef BLOB_INL_H_
#error "Direct inclusion of this file is not allowed, include blob.h"
// For the sake of sane code completion.
#include "blob.h"
#endif

#include <library/cpp/yt/misc/port.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE const char* TBlob::Begin() const
{
    return Begin_;
}

Y_FORCE_INLINE char* TBlob::Begin()
{
    return Begin_;
}

Y_FORCE_INLINE const char* TBlob::End() const
{
    return Begin_ + Size_;
}

Y_FORCE_INLINE char* TBlob::End()
{
    return Begin_ + Size_;
}

Y_FORCE_INLINE size_t TBlob::size() const
{
    return Size_;
}

Y_FORCE_INLINE size_t TBlob::Size() const
{
    return Size_;
}

Y_FORCE_INLINE size_t TBlob::Capacity() const
{
    return Capacity_;
}

Y_FORCE_INLINE TStringBuf TBlob::ToStringBuf() const
{
    return TStringBuf(Begin_, Size_);
}

Y_FORCE_INLINE TRef TBlob::ToRef() const
{
    return TRef(Begin_, Size_);
}

Y_FORCE_INLINE char TBlob::operator [] (size_t index) const
{
    return Begin_[index];
}

Y_FORCE_INLINE char& TBlob::operator [] (size_t index)
{
    return Begin_[index];
}

Y_FORCE_INLINE void TBlob::Clear()
{
    Size_ = 0;
}

Y_FORCE_INLINE bool TBlob::IsEmpty() const
{
    return Size_ == 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
