#pragma once

#include "ref.h"
#include "ref_counted.h"

#include <library/cpp/yt/misc/port.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Default memory tag for TBlob.
struct TDefaultBlobTag
{ };

//! A home-grown optimized replacement for |std::vector<char>| suitable for carrying
//! large chunks of data.
/*!
 *  Compared to |std::vector<char>|, this class supports uninitialized allocations
 *  when explicitly requested to.
 */
class TBlob
{
public:
    //! Constructs a blob with a given size.
    explicit TBlob(
        TRefCountedTypeCookie tagCookie = GetRefCountedTypeCookie<TDefaultBlobTag>(),
        size_t size = 0,
        bool initiailizeStorage = true,
        bool pageAligned = false);

    //! Copies a chunk of memory into a new instance.
    TBlob(
        TRefCountedTypeCookie tagCookie,
        TRef data,
        bool pageAligned = false);

    //! Copies the data.
    TBlob(const TBlob& other);

    //! Moves the data (takes the ownership).
    TBlob(TBlob&& other) noexcept;

    //! Reclaims the memory.
    ~TBlob();

    //! Ensures that capacity is at least #capacity.
    void Reserve(size_t newCapacity);

    //! Changes the size to #newSize.
    /*!
     *  If #size exceeds the current capacity,
     *  we make sure the new capacity grows exponentially.
     *  Hence calling #Resize N times to increase the size by N only
     *  takes amortized O(1) time per call.
     */
    void Resize(size_t newSize, bool initializeStorage = true);

    //! Returns the start pointer.
    Y_FORCE_INLINE const char* Begin() const
    {
        return Begin_;
    }

    //! Returns the start pointer.
    Y_FORCE_INLINE char* Begin()
    {
        return Begin_;
    }

    //! Returns the end pointer.
    Y_FORCE_INLINE const char* End() const
    {
        return Begin_ + Size_;
    }

    //! Returns the end pointer.
    Y_FORCE_INLINE char* End()
    {
        return Begin_ + Size_;
    }

    //! Returns the size.
    Y_FORCE_INLINE size_t size() const
    {
        return Size_;
    }

    //! Returns the size.
    Y_FORCE_INLINE size_t Size() const
    {
        return Size_;
    }

    //! Returns the capacity.
    Y_FORCE_INLINE size_t Capacity() const
    {
        return Capacity_;
    }

    //! Returns the TStringBuf instance for the occupied part of the blob.
    Y_FORCE_INLINE TStringBuf ToStringBuf() const
    {
        return TStringBuf(Begin_, Size_);
    }

    //! Returns the TRef instance for the occupied part of the blob.
    Y_FORCE_INLINE TRef ToRef() const
    {
        return TRef(Begin_, Size_);
    }

    //! Provides by-value access to the underlying storage.
    Y_FORCE_INLINE char operator [] (size_t index) const
    {
        return Begin_[index];
    }

    //! Provides by-ref access to the underlying storage.
    Y_FORCE_INLINE char& operator [] (size_t index)
    {
        return Begin_[index];
    }

    //! Clears the instance but does not reclaim the memory.
    Y_FORCE_INLINE void Clear()
    {
        Size_ = 0;
    }

    //! Returns |true| if size is zero.
    Y_FORCE_INLINE bool IsEmpty() const
    {
        return Size_ == 0;
    }

    //! Overwrites the current instance.
    TBlob& operator = (const TBlob& rhs);

    //! Takes the ownership.
    TBlob& operator = (TBlob&& rhs) noexcept;

    //! Appends a chunk of memory to the end.
    void Append(const void* data, size_t size);

    //! Appends a chunk of memory to the end.
    void Append(TRef ref);

    //! Appends a single char to the end.
    void Append(char ch);

    //! Swaps the current and other instances
    void Swap(TBlob& other);

    friend void swap(TBlob& left, TBlob& right);

private:
    char* Begin_ = nullptr;
    size_t Size_ = 0;
    size_t Capacity_ = 0;
    bool PageAligned_ = false;

#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    TRefCountedTypeCookie TagCookie_ = NullRefCountedTypeCookie;
#endif

    char* DoAllocate(size_t newCapacity);
    void Allocate(size_t newCapacity);
    void Reallocate(size_t newCapacity);
    void Free();

    void Reset();

    void SetTagCookie(TRefCountedTypeCookie tagCookie);
    void SetTagCookie(const TBlob& other);
};

void swap(TBlob& left, TBlob& right);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

