#pragma once

#include "ref.h"
#include "ref_counted.h"

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
        bool initializeStorage = true,
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
    const char* Begin() const;

    //! Returns the start pointer.
    char* Begin();

    //! Returns the end pointer.
    const char* End() const;

    //! Returns the end pointer.
    char* End();

    //! Returns the size.
    size_t size() const;

    //! Returns the size.
    size_t Size() const;

    //! Returns the capacity.
    size_t Capacity() const;

    //! Returns the TStringBuf instance for the occupied part of the blob.
    TStringBuf ToStringBuf() const;

    //! Returns the TRef instance for the occupied part of the blob.
    TRef ToRef() const;

    //! Provides by-value access to the underlying storage.
    char operator [] (size_t index) const;

    //! Provides by-ref access to the underlying storage.
    char& operator [] (size_t index);

    //! Clears the instance but does not reclaim the memory.
    void Clear();

    //! Returns |true| if size is zero.
    bool IsEmpty() const;

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

#define BLOB_INL_H_
#include "blob-inl.h"
#undef BLOB_INL_H_
