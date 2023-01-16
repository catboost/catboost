#include "ref.h"
#include "blob.h"

#include <library/cpp/ytalloc/api/ytalloc.h>

#include <util/system/info.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

// N.B. We would prefer these arrays to be zero sized
// but zero sized arrays are not supported in MSVC.
const char EmptyRefData[1] = {0};
char MutableEmptyRefData[1] = {0};

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

class TBlobHolder
    : public TRefCounted
{
public:
    explicit TBlobHolder(TBlob&& blob)
        : Blob_(std::move(blob))
    { }

private:
    const TBlob Blob_;
};

////////////////////////////////////////////////////////////////////////////////

class TStringHolder
    : public TRefCounted
{
public:
    TStringHolder(TString&& string, TRefCountedTypeCookie cookie)
        : String_(std::move(string))
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        , Cookie_(cookie)
#endif
    {
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        TRefCountedTrackerFacade::AllocateTagInstance(Cookie_);
        TRefCountedTrackerFacade::AllocateSpace(Cookie_, String_.length());
#endif
    }
    ~TStringHolder()
    {
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        TRefCountedTrackerFacade::FreeTagInstance(Cookie_);
        TRefCountedTrackerFacade::FreeSpace(Cookie_, String_.length());
#endif
    }

    const TString& String() const
    {
        return String_;
    }

private:
    const TString String_;
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    const TRefCountedTypeCookie Cookie_;
#endif
};

////////////////////////////////////////////////////////////////////////////////

template <class TDerived>
class TAllocationHolderBase
    : public TRefCounted
{
public:
    TAllocationHolderBase(size_t size, TRefCountedTypeCookie cookie)
        : Size_(size)
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        , Cookie_(cookie)
#endif
    { }

    ~TAllocationHolderBase()
    {
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        TRefCountedTrackerFacade::FreeTagInstance(Cookie_);
        TRefCountedTrackerFacade::FreeSpace(Cookie_, Size_);
#endif
    }

    TMutableRef GetRef()
    {
        return TMutableRef(static_cast<TDerived*>(this)->GetBegin(), Size_);
    }

protected:
    const size_t Size_;
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    const TRefCountedTypeCookie Cookie_;
#endif

    void Initialize(bool initializeStorage)
    {
        if (initializeStorage) {
            ::memset(static_cast<TDerived*>(this)->GetBegin(), 0, Size_);
        }
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        TRefCountedTrackerFacade::AllocateTagInstance(Cookie_);
        TRefCountedTrackerFacade::AllocateSpace(Cookie_, Size_);
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////

class TDefaultAllocationHolder
    : public TAllocationHolderBase<TDefaultAllocationHolder>
    , public TWithExtraSpace<TDefaultAllocationHolder>
{
public:
    TDefaultAllocationHolder(size_t size, bool initializeStorage, TRefCountedTypeCookie cookie)
        : TAllocationHolderBase(size, cookie)
    {
        Initialize(initializeStorage);
    }

    char* GetBegin()
    {
        return static_cast<char*>(GetExtraSpacePtr());
    }
};

////////////////////////////////////////////////////////////////////////////////

class TPageAlignedAllocationHolder
    : public TAllocationHolderBase<TPageAlignedAllocationHolder>
{
public:
    TPageAlignedAllocationHolder(size_t size, bool initializeStorage, TRefCountedTypeCookie cookie)
        : TAllocationHolderBase(size, cookie)
        , Begin_(static_cast<char*>(NYTAlloc::AllocatePageAligned(size)))
    {
        Initialize(initializeStorage);
    }

    ~TPageAlignedAllocationHolder()
    {
        NYTAlloc::Free(Begin_);
    }

    char* GetBegin()
    {
        return Begin_;
    }

private:
    char* const Begin_;
};

////////////////////////////////////////////////////////////////////////////////

TRef TRef::FromBlob(const TBlob& blob)
{
    return TRef(blob.Begin(), blob.Size());
}

bool TRef::AreBitwiseEqual(TRef lhs, TRef rhs)
{
    if (lhs.Size() != rhs.Size()) {
        return false;
    }
    if (lhs.Size() == 0) {
        return true;
    }
    return ::memcmp(lhs.Begin(), rhs.Begin(), lhs.Size()) == 0;
}

////////////////////////////////////////////////////////////////////////////////

TMutableRef TMutableRef::FromBlob(TBlob& blob)
{
    return TMutableRef(blob.Begin(), blob.Size());
}

////////////////////////////////////////////////////////////////////////////////

TSharedRef TSharedRef::FromString(TString str, TRefCountedTypeCookie tagCookie)
{
    auto holder = New<TStringHolder>(std::move(str), tagCookie);
    auto ref = TRef::FromString(holder->String());
    return TSharedRef(ref, std::move(holder));
}

TSharedRef TSharedRef::FromBlob(TBlob&& blob)
{
    auto ref = TRef::FromBlob(blob);
    auto holder = New<TBlobHolder>(std::move(blob));
    return TSharedRef(ref, std::move(holder));
}

TSharedRef TSharedRef::MakeCopy(TRef ref, TRefCountedTypeCookie tagCookie)
{
    if (!ref) {
        return {};
    }
    if (ref.Empty()) {
        return TSharedRef::MakeEmpty();
    }
    auto result = TSharedMutableRef::Allocate(ref.Size(), false, tagCookie);
    ::memcpy(result.Begin(), ref.Begin(), ref.Size());
    return result;
}

std::vector<TSharedRef> TSharedRef::Split(size_t partSize) const
{
    YT_VERIFY(partSize > 0);
    std::vector<TSharedRef> result;
    result.reserve(Size() / partSize + 1);
    auto sliceBegin = Begin();
    while (sliceBegin < End()) {
        auto sliceEnd = sliceBegin + partSize;
        if (sliceEnd < sliceBegin || sliceEnd > End()) {
            sliceEnd = End();
        }
        result.push_back(Slice(sliceBegin, sliceEnd));
        sliceBegin = sliceEnd;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////

TSharedMutableRef TSharedMutableRef::Allocate(size_t size, bool initializeStorage, TRefCountedTypeCookie tagCookie)
{
    auto holder = NewWithExtraSpace<TDefaultAllocationHolder>(size, size, initializeStorage, tagCookie);
    auto ref = holder->GetRef();
    return TSharedMutableRef(ref, std::move(holder));
}

TSharedMutableRef TSharedMutableRef::AllocatePageAligned(size_t size, bool initializeStorage, TRefCountedTypeCookie tagCookie)
{
    auto holder = New<TPageAlignedAllocationHolder>(size, initializeStorage, tagCookie);
    auto ref = holder->GetRef();
    return TSharedMutableRef(ref, std::move(holder));
}

TSharedMutableRef TSharedMutableRef::FromBlob(TBlob&& blob)
{
    auto ref = TMutableRef::FromBlob(blob);
    auto holder = New<TBlobHolder>(std::move(blob));
    return TSharedMutableRef(ref, std::move(holder));
}

TSharedMutableRef TSharedMutableRef::MakeCopy(TRef ref, TRefCountedTypeCookie tagCookie)
{
    if (!ref) {
        return {};
    }
    if (ref.Empty()) {
        return TSharedMutableRef::MakeEmpty();
    }
    auto result = Allocate(ref.Size(), false, tagCookie);
    ::memcpy(result.Begin(), ref.Begin(), ref.Size());
    return result;
}

////////////////////////////////////////////////////////////////////////////////

TString ToString(TRef ref)
{
    return TString(ref.Begin(), ref.End());
}

TString ToString(const TMutableRef& ref)
{
    return ToString(TRef(ref));
}

TString ToString(const TSharedRef& ref)
{
    return ToString(TRef(ref));
}

TString ToString(const TSharedMutableRef& ref)
{
    return ToString(TRef(ref));
}

size_t GetPageSize()
{
    static const size_t PageSize = NSystemInfo::GetPageSize();
    return PageSize;
}

size_t RoundUpToPage(size_t bytes)
{
    static const size_t PageSize = NSystemInfo::GetPageSize();
    YT_ASSERT((PageSize & (PageSize - 1)) == 0);
    return (bytes + PageSize - 1) & (~(PageSize - 1));
}

size_t GetByteSize(const TSharedRefArray& array)
{
    size_t size = 0;
    if (array) {
        for (const auto& part : array) {
            size += part.Size();
        }
    }
    return size;
}

////////////////////////////////////////////////////////////////////////////////

i64 TSharedRefArray::ByteSize() const
{
    i64 result = 0;
    if (*this) {
        for (const auto& part : *this) {
            result += part.Size();
        }
    }
    return result;
}

std::vector<TSharedRef> TSharedRefArray::ToVector() const
{
    if (!Impl_) {
        return {};
    }

    return std::vector<TSharedRef>(Begin(), End());
}

////////////////////////////////////////////////////////////////////////////////

TSharedRefArrayBuilder::TSharedRefArrayBuilder(
    size_t size,
    size_t poolCapacity,
    TRefCountedTypeCookie tagCookie)
    : AllocationCapacity_(poolCapacity)
    , Impl_(TSharedRefArray::NewImpl(
        size,
        poolCapacity,
        tagCookie,
        size))
    , CurrentAllocationPtr_(Impl_->GetBeginAllocationPtr())
{ }

void TSharedRefArrayBuilder::Add(TSharedRef part)
{
    YT_ASSERT(CurrentPartIndex_ < Impl_->Size());
    Impl_->MutableBegin()[CurrentPartIndex_++] = std::move(part);
}

TMutableRef TSharedRefArrayBuilder::AllocateAndAdd(size_t size)
{
    YT_ASSERT(CurrentPartIndex_ < Impl_->Size());
    YT_ASSERT(CurrentAllocationPtr_ + size <= Impl_->GetBeginAllocationPtr() + AllocationCapacity_);
    TMutableRef ref(CurrentAllocationPtr_, size);
    CurrentAllocationPtr_ += size;
    TRefCountedPtr holder(Impl_.Get(), false);
    TSharedRef sharedRef(ref, std::move(holder));
    Add(std::move(sharedRef));
    return ref;
}

TSharedRefArray TSharedRefArrayBuilder::Finish()
{
    return TSharedRefArray(std::move(Impl_));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
