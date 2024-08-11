#pragma once

#include "new.h"
#include "range.h"
#include "shared_range.h"

#include <library/cpp/yt/string/format.h>

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Forward declaration.
class TBlob;

//! A non-owning reference to a range of memory.
class TRef
    : public TRange<char>
{
public:
    //! Creates a null TRef.
    TRef() = default;

    //! Creates a TRef for a given block of memory.
    TRef(const void* data, size_t size);

    //! Creates a TRef for a given range of memory.
    TRef(const void* begin, const void* end);

    //! Creates an empty TRef.
    static TRef MakeEmpty();

    //! Creates a non-owning TRef for a given blob.
    static TRef FromBlob(const TBlob& blob);

    //! Creates a non-owning TRef for a given string.
    static TRef FromString(const TString& str);

    //! Creates a non-owning TRef for a given std::string.
    static TRef FromString(const std::string& str);

    //! Creates a non-owning TRef for a given stringbuf.
    static TRef FromStringBuf(TStringBuf strBuf);

    //! Creates a non-owning TRef for a given pod structure.
    template <class T>
    static TRef FromPod(const T& data);

    //! Converts to TStringBuf.
    TStringBuf ToStringBuf() const;

    //! Creates a TRef for a part of existing range.
    TRef Slice(size_t startOffset, size_t endOffset) const;

    //! Compares the content for bitwise equality.
    static bool AreBitwiseEqual(TRef lhs, TRef rhs);
};

////////////////////////////////////////////////////////////////////////////////

//! A non-owning reference to a mutable range of memory.
//! Use with caution :)
class TMutableRef
    : public TMutableRange<char>
{
public:
    //! Creates a null TMutableRef.
    //! Note empty TMutableRef is not the same as null TMutableRef.
    //! `operator bool` can be used to check if ref is nonnull.
    TMutableRef() = default;

    //! Creates a TMutableRef for a given block of memory.
    TMutableRef(void* data, size_t size);

    //! Creates a TMutableRef for a given range of memory.
    TMutableRef(void* begin, void* end);

    //! Creates an empty TMutableRef.
    static TMutableRef MakeEmpty();

    //! Converts a TMutableRef to TRef.
    operator TRef() const;

    //! Creates a non-owning TMutableRef for a given blob.
    static TMutableRef FromBlob(TBlob& blob);

    //! Creates a non-owning TMutableRef for a given pod structure.
    template <class T>
    static TMutableRef FromPod(T& data);

    //! Creates a non-owning TMutableRef for a given TString.
    //! Ensures that the string is not shared.
    static TMutableRef FromString(TString& str);

    //! Creates a non-owning TMutableRef for a given std::string.
    static TMutableRef FromString(std::string& str);

    //! Creates a TMutableRef for a part of existing range.
    TMutableRef Slice(size_t startOffset, size_t endOffset) const;
};

////////////////////////////////////////////////////////////////////////////////

//! Default tag type for memory blocks allocated via TSharedRef.
/*!
 *  Each newly allocated TSharedRef blob is associated with a tag type
 *  that appears in ref-counted statistics.
 */
struct TDefaultSharedBlobTag { };

//! A reference to a range of memory with shared ownership.
class TSharedRef
    : public TSharedRange<char>
{
public:
    //! Creates a null TSharedRef.
    TSharedRef() = default;

    //! Creates a TSharedRef with a given holder.
    TSharedRef(TRef ref, TSharedRangeHolderPtr holder);

    //! Creates a TSharedRef from a pointer and length.
    TSharedRef(const void* data, size_t length, TSharedRangeHolderPtr holder);

    //! Creates a TSharedRef from a range.
    TSharedRef(const void* begin, const void* end, TSharedRangeHolderPtr holder);

    //! Creates an empty TSharedRef.
    static TSharedRef MakeEmpty();

    //! Converts a TSharedRef to TRef.
    operator TRef() const;


    //! Creates a TSharedRef from a string.
    //! Since strings are ref-counted, no data is copied.
    //! The memory is marked with a given tag.
    template <class TTag>
    static TSharedRef FromString(TString str);

    //! Creates a TSharedRef from a string.
    //! Since strings are ref-counted, no data is copied.
    //! The memory is marked with TDefaultSharedBlobTag.
    static TSharedRef FromString(TString str);

    //! Creates a TSharedRef reference from a string.
    //! Since strings are ref-counted, no data is copied.
    //! The memory is marked with a given tag.
    static TSharedRef FromString(TString str, TRefCountedTypeCookie tagCookie);

    //! Creates a TSharedRef for a given blob taking ownership of its content.
    static TSharedRef FromBlob(TBlob&& blob);

    //! Converts to TStringBuf.
    TStringBuf ToStringBuf() const;

    //! Creates a copy of a given TRef.
    //! The memory is marked with a given tag.
    static TSharedRef MakeCopy(TRef ref, TRefCountedTypeCookie tagCookie);

    //! Creates a copy of a given TRef.
    //! The memory is marked with a given tag.
    template <class TTag>
    static TSharedRef MakeCopy(TRef ref);

    //! Creates a TSharedRef for a part of existing range.
    TSharedRef Slice(size_t startOffset, size_t endOffset) const;

    //! Creates a TSharedRef for a part of existing range.
    TSharedRef Slice(const void* begin, const void* end) const;

    //! Creates a vector of slices with specified size.
    std::vector<TSharedRef> Split(size_t partSize) const;

private:
    friend class TSharedRefArrayImpl;
};

////////////////////////////////////////////////////////////////////////////////

//! Various options for allocating TSharedMutableRef.
struct TSharedMutableRefAllocateOptions
{
    bool InitializeStorage = true;
    bool ExtendToUsableSize = false;
};

//! A reference to a mutable range of memory with shared ownership.
//! Use with caution :)
class TSharedMutableRef
    : public TSharedMutableRange<char>
{
public:
    //! Creates a null TSharedMutableRef.
    TSharedMutableRef() = default;

    //! Creates a TSharedMutableRef with a given holder.
    TSharedMutableRef(const TMutableRef& ref, TSharedRangeHolderPtr holder);

    //! Creates a TSharedMutableRef from a pointer and length.
    TSharedMutableRef(void* data, size_t length, TSharedRangeHolderPtr holder);

    //! Creates a TSharedMutableRef from a range.
    TSharedMutableRef(void* begin, void* end, TSharedRangeHolderPtr holder);

    //! Creates an empty TSharedMutableRef.
    static TSharedMutableRef MakeEmpty();

    //! Converts a TSharedMutableRef to TMutableRef.
    operator TMutableRef() const;

    //! Converts a TSharedMutableRef to TSharedRef.
    operator TSharedRef() const;

    //! Converts a TSharedMutableRef to TRef.
    operator TRef() const;


    //! Allocates a new shared block of memory.
    //! The memory is marked with a given tag.
    template <class TTag>
    static TSharedMutableRef Allocate(size_t size, TSharedMutableRefAllocateOptions options = {});

    //! Allocates a new shared block of memory.
    //! The memory is marked with TDefaultSharedBlobTag.
    static TSharedMutableRef Allocate(size_t size, TSharedMutableRefAllocateOptions options = {});

    //! Allocates a new shared block of memory.
    //! The memory is marked with a given tag.
    static TSharedMutableRef Allocate(size_t size, TSharedMutableRefAllocateOptions options, TRefCountedTypeCookie tagCookie);

    //! Allocates a new page aligned shared block of memory.
    //! #size must be divisible by page size.
    //! The memory is marked with a given tag.
    template <class TTag>
    static TSharedMutableRef AllocatePageAligned(size_t size, TSharedMutableRefAllocateOptions options = {});

    //! Allocates a new page aligned shared block of memory.
    //! #size must be divisible by page size.
    //! The memory is marked with TDefaultSharedBlobTag.
    static TSharedMutableRef AllocatePageAligned(size_t size, TSharedMutableRefAllocateOptions options = {});

    //! Allocates a new page aligned shared block of memory.
    //! #size must be divisible by page size.
    //! The memory is marked with a given tag.
    static TSharedMutableRef AllocatePageAligned(size_t size, TSharedMutableRefAllocateOptions options, TRefCountedTypeCookie tagCookie);

    //! Creates a TSharedMutableRef for the whole blob taking ownership of its content.
    static TSharedMutableRef FromBlob(TBlob&& blob);

    //! Creates a copy of a given TRef.
    //! The memory is marked with a given tag.
    static TSharedMutableRef MakeCopy(TRef ref, TRefCountedTypeCookie tagCookie);

    //! Creates a copy of a given TRef.
    //! The memory is marked with a given tag.
    template <class TTag>
    static TSharedMutableRef MakeCopy(TRef ref);

    //! Creates a reference for a part of existing range.
    TSharedMutableRef Slice(size_t startOffset, size_t endOffset) const;

    //! Creates a reference for a part of existing range.
    TSharedMutableRef Slice(void* begin, void* end) const;
};

////////////////////////////////////////////////////////////////////////////////

DECLARE_REFCOUNTED_CLASS(TSharedRefArrayImpl)

//! A smart-pointer to a ref-counted immutable sequence of TSharedRef-s.
class TSharedRefArray
{
public:
    TSharedRefArray() = default;
    TSharedRefArray(const TSharedRefArray& other);
    TSharedRefArray(TSharedRefArray&& other) noexcept;

    explicit TSharedRefArray(const TSharedRef& part);
    explicit TSharedRefArray(TSharedRef&& part);

    struct TCopyParts
    { };
    struct TMoveParts
    { };

    template <class TParts>
    TSharedRefArray(const TParts& parts, TCopyParts);
    template <class TParts>
    TSharedRefArray(TParts&& parts, TMoveParts);

    TSharedRefArray& operator = (const TSharedRefArray& other);
    TSharedRefArray& operator = (TSharedRefArray&& other);

    explicit operator bool() const;

    void Reset();

    size_t Size() const;
    size_t size() const;
    i64 ByteSize() const;
    bool Empty() const;
    const TSharedRef& operator [] (size_t index) const;

    const TSharedRef* Begin() const;
    const TSharedRef* End() const;

    std::vector<TSharedRef> ToVector() const;
    TString ToString() const;

    //! Creates a copy of a given TSharedRefArray.
    //! The memory is marked with a given tag.
    static TSharedRefArray MakeCopy(const TSharedRefArray& array, TRefCountedTypeCookie tagCookie);

private:
    friend class TSharedRefArrayBuilder;

    TSharedRefArrayImplPtr Impl_;

    explicit TSharedRefArray(TSharedRefArrayImplPtr impl);

    template <class... As>
    static TSharedRefArrayImplPtr NewImpl(
        size_t size,
        size_t poolCapacity,
        TRefCountedTypeCookie cookie,
        As&&... args);
};

// STL interop.
const TSharedRef* begin(const TSharedRefArray& array);
const TSharedRef* end(const TSharedRefArray& array);

////////////////////////////////////////////////////////////////////////////////

struct TDefaultSharedRefArrayBuilderTag { };

//! A helper for creating TSharedRefArray.
class TSharedRefArrayBuilder
{
public:
    //! Creates a builder instance.
    /*
     *  The user must provide the total (resulting) part count in #size.
     *
     *  Additionally, the user may request a certain memory pool of size #poolCapacity
     *  to be created. Parts occupiying space in the above pool are created with #AllocateAndAdd
     *  calls.
     *
     *  The pool (if any) and the array are created within a single memory allocation tagged with
     *  #tagCookie.
     *
     *  If less than #size parts are added, the trailing ones are null.
     */
    explicit TSharedRefArrayBuilder(
        size_t size,
        size_t poolCapacity = 0,
        TRefCountedTypeCookie tagCookie = GetRefCountedTypeCookie<TDefaultSharedRefArrayBuilderTag>());

    //! Adds an existing TSharedRef part to the constructed array.
    void Add(TSharedRef part);

    //! Allocates #size memory from the pool and adds a part to the constuctured array.
    /*!
     *  The resulting TMutableRef enables the user to fill the just-created part appropriately.
     *  The total sum of #size during all #AllocateAndAll calls must now exceed #allocationCapacity
     *  passed to the ctor.
     *
     *  The memory is being claimed from the pool contiguously; the user must
     *  take care of the alignment issues on its own.
     */
    TMutableRef AllocateAndAdd(size_t size);

    //! Finalizes the construction; returns the constructed TSharedRefArray.
    TSharedRefArray Finish();

private:
    const size_t AllocationCapacity_;
    TSharedRefArrayImplPtr Impl_;
    char* CurrentAllocationPtr_;
    size_t CurrentPartIndex_ = 0;
};


////////////////////////////////////////////////////////////////////////////////

void FormatValue(TStringBuilderBase* builder, const TRef& ref, TStringBuf spec);
void FormatValue(TStringBuilderBase* builder, const TMutableRef& ref, TStringBuf spec);
void FormatValue(TStringBuilderBase* builder, const TSharedRef& ref, TStringBuf spec);
void FormatValue(TStringBuilderBase* builder, const TSharedMutableRef& ref, TStringBuf);

size_t GetPageSize();
size_t RoundUpToPage(size_t bytes);

size_t GetByteSize(TRef ref);
size_t GetByteSize(const TSharedRefArray& array);
template <class T>
size_t GetByteSize(TRange<T> parts);
template <class T>
size_t GetByteSize(const std::vector<T>& parts);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define REF_INL_H_
#include "ref-inl.h"
#undef REF_INL_H_

//! Serialize TSharedRef like vector<char>. Useful for ::Save, ::Load serialization/deserialization. See util/ysaveload.h.
template <>
class TSerializer<NYT::TSharedRef>: public TVectorSerializer<NYT::TSharedRange<char>> {};
