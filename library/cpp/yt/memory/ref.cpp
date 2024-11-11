#include "ref.h"

#include "blob.h"

#include <library/cpp/yt/malloc/malloc.h>

#include <library/cpp/yt/misc/port.h>

#include <library/cpp/yt/string/format.h>

#include <util/system/info.h>
#include <util/system/align.h>

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
    : public TSharedRangeHolder
{
public:
    explicit TBlobHolder(TBlob&& blob)
        : Blob_(std::move(blob))
    { }

    // TSharedRangeHolder overrides.
    std::optional<size_t> GetTotalByteSize() const override
    {
        return Blob_.Capacity();
    }

private:
    const TBlob Blob_;
};

////////////////////////////////////////////////////////////////////////////////

template <class TString>
class TStringHolder
    : public TSharedRangeHolder
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
#else
        Y_UNUSED(cookie);
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

    // TSharedRangeHolder overrides.
    std::optional<size_t> GetTotalByteSize() const override
    {
        return String_.capacity();
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
    : public TSharedRangeHolder
{
public:
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
    size_t Size_;
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    TRefCountedTypeCookie Cookie_;
#endif

    void Initialize(
        size_t size,
        TSharedMutableRefAllocateOptions options,
        TRefCountedTypeCookie cookie)
    {
        Size_ = size;
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
        Cookie_ = cookie;
#else
        Y_UNUSED(cookie);
#endif
        if (options.InitializeStorage) {
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
    TDefaultAllocationHolder(
        size_t size,
        TSharedMutableRefAllocateOptions options,
        TRefCountedTypeCookie cookie)
    {
        if (options.ExtendToUsableSize) {
            if (auto usableSize = GetUsableSpaceSize(); usableSize != 0) {
                size = usableSize;
            }
        }
        Initialize(size, options, cookie);
    }

    char* GetBegin()
    {
        return static_cast<char*>(GetExtraSpacePtr());
    }

    // TSharedRangeHolder overrides.
    std::optional<size_t> GetTotalByteSize() const override
    {
        return Size_;
    }
};

////////////////////////////////////////////////////////////////////////////////

class TPageAlignedAllocationHolder
    : public TAllocationHolderBase<TPageAlignedAllocationHolder>
{
public:
    TPageAlignedAllocationHolder(
        size_t size,
        TSharedMutableRefAllocateOptions options,
        TRefCountedTypeCookie cookie)
        : Begin_(static_cast<char*>(::aligned_malloc(size, GetPageSize())))
    {
        Initialize(size, options, cookie);
    }

    ~TPageAlignedAllocationHolder()
    {
        ::free(Begin_);
    }

    char* GetBegin()
    {
        return Begin_;
    }

    // TSharedRangeHolder overrides.
    std::optional<size_t> GetTotalByteSize() const override
    {
        return AlignUp(Size_, GetPageSize());
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
    return FromStringImpl(std::move(str), tagCookie);
}

TSharedRef TSharedRef::FromString(std::string str, TRefCountedTypeCookie tagCookie)
{
    return FromStringImpl(std::move(str), tagCookie);
}

template <class TString>
TSharedRef TSharedRef::FromStringImpl(TString str, TRefCountedTypeCookie tagCookie)
{
    auto holder = New<TStringHolder<TString>>(std::move(str), tagCookie);
    auto ref = TRef::FromString(holder->String());
    return TSharedRef(ref, std::move(holder));
}

TSharedRef TSharedRef::FromString(const char* str)
{
    return FromString(std::string(str));
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
    auto result = TSharedMutableRef::Allocate(ref.Size(), {.InitializeStorage = false}, tagCookie);
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

TSharedMutableRef TSharedMutableRef::Allocate(size_t size, TSharedMutableRefAllocateOptions options, TRefCountedTypeCookie tagCookie)
{
    auto holder = NewWithExtraSpace<TDefaultAllocationHolder>(size, size, options, tagCookie);
    auto ref = holder->GetRef();
    return TSharedMutableRef(ref, std::move(holder));
}

TSharedMutableRef TSharedMutableRef::AllocatePageAligned(size_t size, TSharedMutableRefAllocateOptions options, TRefCountedTypeCookie tagCookie)
{
    auto holder = New<TPageAlignedAllocationHolder>(size, options, tagCookie);
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
    auto result = Allocate(ref.Size(), {.InitializeStorage = false}, tagCookie);
    ::memcpy(result.Begin(), ref.Begin(), ref.Size());
    return result;
}

////////////////////////////////////////////////////////////////////////////////

void FormatValue(TStringBuilderBase* builder, const TRef& ref, TStringBuf spec)
{
    FormatValue(builder, TStringBuf{ref.Begin(), ref.End()}, spec);
}

void FormatValue(TStringBuilderBase* builder, const TMutableRef& ref, TStringBuf spec)
{
    FormatValue(builder, TRef(ref), spec);
}

void FormatValue(TStringBuilderBase* builder, const TSharedRef& ref, TStringBuf spec)
{
    FormatValue(builder, TRef(ref), spec);
}

void FormatValue(TStringBuilderBase* builder, const TSharedMutableRef& ref, TStringBuf spec)
{
    FormatValue(builder, TRef(ref), spec);
}

size_t GetPageSize()
{
    static const size_t PageSize = NSystemInfo::GetPageSize();
    return PageSize;
}

size_t RoundUpToPage(size_t bytes)
{
    return AlignUp<size_t>(bytes, GetPageSize());
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

TString TSharedRefArray::ToString() const
{
    if (!Impl_) {
        return {};
    }

    TString result;
    size_t size = 0;
    for (const auto& part : *this) {
        size += part.size();
    }
    result.ReserveAndResize(size);
    char* ptr = result.begin();
    for (const auto& part : *this) {
        size += part.size();
        ::memcpy(ptr, part.begin(), part.size());
        ptr += part.size();
    }
    return result;
}

TSharedRefArray TSharedRefArray::MakeCopy(
    const TSharedRefArray& array,
    TRefCountedTypeCookie tagCookie)
{
    TSharedRefArrayBuilder builder(
        array.Size(),
        array.ByteSize(),
        tagCookie);
    for (const auto& part : array) {
        auto partCopy = builder.AllocateAndAdd(part.Size());
        ::memcpy(partCopy.Begin(), part.Begin(), part.Size());
    }
    return builder.Finish();
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
    TSharedRangeHolderPtr holder(Impl_.Get(), false);
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
