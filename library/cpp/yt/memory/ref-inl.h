#ifndef REF_INL_H_
#error "Direct inclusion of this file is not allowed, include ref.h"
// For the sake of sane code completion.
#include "ref.h"
#endif

#include <library/cpp/yt/misc/concepts.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

extern const char EmptyRefData[];
extern char MutableEmptyRefData[];

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE TRef::TRef(const void* data, size_t size)
    : TRange<char>(static_cast<const char*>(data), size)
{ }

Y_FORCE_INLINE TRef::TRef(const void* begin, const void* end)
    : TRange<char>(static_cast<const char*>(begin), static_cast<const char*>(end))
{ }

Y_FORCE_INLINE TRef TRef::MakeEmpty()
{
    return TRef(NDetail::EmptyRefData, NDetail::EmptyRefData);
}

Y_FORCE_INLINE TRef TRef::FromString(const TString& str)
{
    return FromStringBuf(str);
}

Y_FORCE_INLINE TRef TRef::FromString(const std::string& str)
{
    return TRef(str.data(), str.size());
}

Y_FORCE_INLINE TRef TRef::FromStringBuf(TStringBuf strBuf)
{
    return TRef(strBuf.data(), strBuf.length());
}

template <class T>
Y_FORCE_INLINE TRef TRef::FromPod(const T& data)
{
    static_assert(TTypeTraits<T>::IsPod || (std::is_standard_layout_v<T> && std::is_trivial_v<T>), "T must be a pod-type.");
    return TRef(&data, sizeof(data));
}

Y_FORCE_INLINE TStringBuf TRef::ToStringBuf() const
{
    return TStringBuf(Begin(), Size());
}

Y_FORCE_INLINE TRef TRef::Slice(size_t startOffset, size_t endOffset) const
{
    YT_ASSERT(endOffset >= startOffset && endOffset <= Size());
    return TRef(Begin() + startOffset, endOffset - startOffset);
}

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE TMutableRef::TMutableRef(void* data, size_t size)
    : TMutableRange<char>(static_cast<char*>(data), size)
{ }

Y_FORCE_INLINE TMutableRef::TMutableRef(void* begin, void* end)
    : TMutableRange<char>(static_cast<char*>(begin), static_cast<char*>(end))
{ }

Y_FORCE_INLINE TMutableRef TMutableRef::MakeEmpty()
{
    return TMutableRef(NDetail::MutableEmptyRefData, NDetail::MutableEmptyRefData);
}

Y_FORCE_INLINE TMutableRef::operator TRef() const
{
    return TRef(Begin(), Size());
}

template <class T>
Y_FORCE_INLINE TMutableRef TMutableRef::FromPod(T& data)
{
    static_assert(TTypeTraits<T>::IsPod || (std::is_standard_layout_v<T> && std::is_trivial_v<T>), "T must be a pod-type.");
    return TMutableRef(&data, sizeof(data));
}

Y_FORCE_INLINE TMutableRef TMutableRef::FromString(TString& str)
{
    // NB: begin() invokes CloneIfShared().
    return TMutableRef(str.begin(), str.length());
}

Y_FORCE_INLINE TMutableRef TMutableRef::FromString(std::string& str)
{
    return TMutableRef(str.data(), str.length());
}

Y_FORCE_INLINE TMutableRef TMutableRef::Slice(size_t startOffset, size_t endOffset) const
{
    YT_ASSERT(endOffset >= startOffset && endOffset <= Size());
    return TMutableRef(Begin() + startOffset, endOffset - startOffset);
}

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE TSharedRef::TSharedRef(TRef ref, TSharedRangeHolderPtr holder)
    : TSharedRange<char>(ref, std::move(holder))
{ }

Y_FORCE_INLINE TSharedRef::TSharedRef(const void* data, size_t length, TSharedRangeHolderPtr holder)
    : TSharedRange<char>(static_cast<const char*>(data), length, std::move(holder))
{ }

Y_FORCE_INLINE TSharedRef::TSharedRef(const void* begin, const void* end, TSharedRangeHolderPtr holder)
    : TSharedRange<char>(static_cast<const char*>(begin), static_cast<const char*>(end), std::move(holder))
{ }

Y_FORCE_INLINE TSharedRef TSharedRef::MakeEmpty()
{
    return TSharedRef(TRef::MakeEmpty(), nullptr);
}

Y_FORCE_INLINE TSharedRef::operator TRef() const
{
    return TRef(Begin(), Size());
}

template <class TTag>
Y_FORCE_INLINE TSharedRef TSharedRef::FromString(TString str)
{
    static_assert(IsEmptyClass<TTag>());
    return FromString(std::move(str), GetRefCountedTypeCookie<TTag>());
}

Y_FORCE_INLINE TSharedRef TSharedRef::FromString(TString str)
{
    return FromString<TDefaultSharedBlobTag>(std::move(str));
}

template <class TTag>
Y_FORCE_INLINE TSharedRef TSharedRef::FromString(std::string str)
{
    static_assert(IsEmptyClass<TTag>());
    return FromString(std::move(str), GetRefCountedTypeCookie<TTag>());
}

Y_FORCE_INLINE TSharedRef TSharedRef::FromString(std::string str)
{
    return FromString<TDefaultSharedBlobTag>(std::move(str));
}

Y_FORCE_INLINE TStringBuf TSharedRef::ToStringBuf() const
{
    return TStringBuf(Begin(), Size());
}

template <class TTag>
Y_FORCE_INLINE TSharedRef TSharedRef::MakeCopy(TRef ref)
{
    static_assert(IsEmptyClass<TTag>());
    return MakeCopy(ref, GetRefCountedTypeCookie<TTag>());
}

Y_FORCE_INLINE TSharedRef TSharedRef::Slice(size_t startOffset, size_t endOffset) const
{
    YT_ASSERT(endOffset >= startOffset && endOffset <= Size());
    return TSharedRef(Begin() + startOffset, endOffset - startOffset, Holder_);
}

Y_FORCE_INLINE  TSharedRef TSharedRef::Slice(const void* begin, const void* end) const
{
    YT_ASSERT(begin >= Begin());
    YT_ASSERT(end <= End());
    return TSharedRef(begin, end, Holder_);
}

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE TSharedMutableRef::TSharedMutableRef(const TMutableRef& ref, TSharedRangeHolderPtr holder)
    : TSharedMutableRange<char>(ref, std::move(holder))
{ }

Y_FORCE_INLINE TSharedMutableRef::TSharedMutableRef(void* data, size_t length, TSharedRangeHolderPtr holder)
    : TSharedMutableRange<char>(static_cast<char*>(data), length, std::move(holder))
{ }

Y_FORCE_INLINE TSharedMutableRef::TSharedMutableRef(void* begin, void* end, TSharedRangeHolderPtr holder)
    : TSharedMutableRange<char>(static_cast<char*>(begin), static_cast<char*>(end), std::move(holder))
{ }

Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::MakeEmpty()
{
    return TSharedMutableRef(TMutableRef::MakeEmpty(), nullptr);
}

Y_FORCE_INLINE TSharedMutableRef::operator TMutableRef() const
{
    return TMutableRef(Begin(), Size());
}

Y_FORCE_INLINE TSharedMutableRef::operator TSharedRef() const
{
    return TSharedRef(Begin(), Size(), Holder_);
}

Y_FORCE_INLINE TSharedMutableRef::operator TRef() const
{
    return TRef(Begin(), Size());
}

Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::Allocate(size_t size, TSharedMutableRefAllocateOptions options)
{
    return Allocate<TDefaultSharedBlobTag>(size, options);
}

Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::AllocatePageAligned(size_t size, TSharedMutableRefAllocateOptions options)
{
    return AllocatePageAligned<TDefaultSharedBlobTag>(size, options);
}

template <class TTag>
Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::MakeCopy(TRef ref)
{
    static_assert(IsEmptyClass<TTag>());
    return MakeCopy(ref, GetRefCountedTypeCookie<TTag>());
}

Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::Slice(size_t startOffset, size_t endOffset) const
{
    YT_ASSERT(endOffset >= startOffset && endOffset <= Size());
    return TSharedMutableRef(Begin() + startOffset, endOffset - startOffset, Holder_);
}

Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::Slice(void* begin, void* end) const
{
    YT_ASSERT(begin >= Begin());
    YT_ASSERT(end <= End());
    return TSharedMutableRef(begin, end, Holder_);
}

template <class TTag>
Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::Allocate(size_t size, TSharedMutableRefAllocateOptions options)
{
    static_assert(IsEmptyClass<TTag>());
    return Allocate(size, options, GetRefCountedTypeCookie<TTag>());
}

template <class TTag>
Y_FORCE_INLINE TSharedMutableRef TSharedMutableRef::AllocatePageAligned(size_t size, TSharedMutableRefAllocateOptions options)
{
    static_assert(IsEmptyClass<TTag>());
    return AllocatePageAligned(size, options, GetRefCountedTypeCookie<TTag>());
}

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE size_t GetByteSize(TRef ref)
{
    return ref ? ref.Size() : 0;
}

template <class T>
size_t GetByteSize(TRange<T> parts)
{
    size_t size = 0;
    for (const auto& part : parts) {
        size += part.Size();
    }
    return size;
}

template <class T>
size_t GetByteSize(const std::vector<T>& parts)
{
    return GetByteSize(TRange(parts));
}

////////////////////////////////////////////////////////////////////////////////

class TSharedRefArrayImpl
    : public TSharedRangeHolder
    , public TWithExtraSpace<TSharedRefArrayImpl>
{
public:
    TSharedRefArrayImpl(
        size_t extraSpaceSize,
        TRefCountedTypeCookie tagCookie,
        size_t size)
        : Size_(size)
        , ExtraSpaceSize_(extraSpaceSize)
        , TagCookie_(tagCookie)
    {
        for (size_t index = 0; index < Size_; ++index) {
            new (MutableBegin() + index) TSharedRef();
        }
        RegisterWithRefCountedTracker();
    }

    TSharedRefArrayImpl(
        size_t extraSpaceSize,
        TRefCountedTypeCookie tagCookie,
        const TSharedRef& part)
        : Size_(1)
        , ExtraSpaceSize_(extraSpaceSize)
        , TagCookie_(tagCookie)
    {
        new (MutableBegin()) TSharedRef(part);
        RegisterWithRefCountedTracker();
    }

    TSharedRefArrayImpl(
        size_t extraSpaceSize,
        TRefCountedTypeCookie tagCookie,
        TSharedRef&& part)
        : Size_(1)
        , ExtraSpaceSize_(extraSpaceSize)
        , TagCookie_(tagCookie)
    {
        new (MutableBegin()) TSharedRef(std::move(part));
        RegisterWithRefCountedTracker();
    }

    template <class TParts>
    TSharedRefArrayImpl(
        size_t extraSpaceSize,
        TRefCountedTypeCookie tagCookie,
        const TParts& parts,
        TSharedRefArray::TCopyParts)
        : Size_(parts.size())
        , ExtraSpaceSize_(extraSpaceSize)
        , TagCookie_(tagCookie)
    {
        for (size_t index = 0; index < Size_; ++index) {
            new (MutableBegin() + index) TSharedRef(parts[index]);
        }
        RegisterWithRefCountedTracker();
    }

    template <class TParts>
    TSharedRefArrayImpl(
        size_t extraSpaceSize,
        TRefCountedTypeCookie tagCookie,
        TParts&& parts,
        TSharedRefArray::TMoveParts)
        : Size_(parts.size())
        , ExtraSpaceSize_(extraSpaceSize)
        , TagCookie_(tagCookie)
    {
        for (size_t index = 0; index < Size_; ++index) {
            new (MutableBegin() + index) TSharedRef(std::move(parts[index]));
        }
        RegisterWithRefCountedTracker();
    }

    ~TSharedRefArrayImpl()
    {
        for (size_t index = 0; index < Size_; ++index) {
            auto& part = MutableBegin()[index];
            if (part.GetHolder() == this) {
                part.Holder_.Release();
            }
            part.TSharedRef::~TSharedRef();
        }
        UnregisterFromRefCountedTracker();
    }


    size_t Size() const
    {
        return Size_;
    }

    bool Empty() const
    {
        return Size_ == 0;
    }

    const TSharedRef& operator [] (size_t index) const
    {
        YT_ASSERT(index < Size());
        return Begin()[index];
    }


    const TSharedRef* Begin() const
    {
        return static_cast<const TSharedRef*>(GetExtraSpacePtr());
    }

    const TSharedRef* End() const
    {
        return Begin() + Size_;
    }


    // TSharedRangeHolder overrides.
    std::optional<size_t> GetTotalByteSize() const override
    {
        size_t result = 0;
        for (size_t index = 0; index < Size(); ++index) {
            const auto& part = (*this)[index];
            if (!part) {
                continue;
            }
            auto partSize = part.GetHolder()->GetTotalByteSize();
            if (!partSize) {
                return std::nullopt;
            }
            result += *partSize;
        }
        return result;
    }

private:
    friend class TSharedRefArrayBuilder;

    const size_t Size_;
    const size_t ExtraSpaceSize_;
    const TRefCountedTypeCookie TagCookie_;


    void RegisterWithRefCountedTracker()
    {
        TRefCountedTrackerFacade::AllocateTagInstance(TagCookie_);
        TRefCountedTrackerFacade::AllocateSpace(TagCookie_, ExtraSpaceSize_);
    }

    void UnregisterFromRefCountedTracker()
    {
        TRefCountedTrackerFacade::FreeTagInstance(TagCookie_);
        TRefCountedTrackerFacade::FreeSpace(TagCookie_, ExtraSpaceSize_);
    }


    TSharedRef* MutableBegin()
    {
        return static_cast<TSharedRef*>(GetExtraSpacePtr());
    }

    TSharedRef* MutableEnd()
    {
        return MutableBegin() + Size_;
    }

    char* GetBeginAllocationPtr()
    {
        return static_cast<char*>(static_cast<void*>(MutableEnd()));
    }
};

DEFINE_REFCOUNTED_TYPE(TSharedRefArrayImpl)

////////////////////////////////////////////////////////////////////////////////

struct TSharedRefArrayTag { };

Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(TIntrusivePtr<TSharedRefArrayImpl> impl)
    : Impl_(std::move(impl))
{ }

Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(const TSharedRefArray& other)
    : Impl_(other.Impl_)
{ }

Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(TSharedRefArray&& other) noexcept
    : Impl_(std::move(other.Impl_))
{ }

Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(const TSharedRef& part)
    : Impl_(NewImpl(1, 0, GetRefCountedTypeCookie<TSharedRefArrayTag>(), part))
{ }

Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(TSharedRef&& part)
    : Impl_(NewImpl(1, 0, GetRefCountedTypeCookie<TSharedRefArrayTag>(), std::move(part)))
{ }

template <class TParts>
Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(const TParts& parts, TSharedRefArray::TCopyParts)
    : Impl_(NewImpl(parts.size(), 0, GetRefCountedTypeCookie<TSharedRefArrayTag>(), parts, TSharedRefArray::TCopyParts{}))
{ }

template <class TParts>
Y_FORCE_INLINE TSharedRefArray::TSharedRefArray(TParts&& parts, TSharedRefArray::TMoveParts)
    : Impl_(NewImpl(parts.size(), 0, GetRefCountedTypeCookie<TSharedRefArrayTag>(), std::move(parts), TSharedRefArray::TMoveParts{}))
{ }

Y_FORCE_INLINE TSharedRefArray& TSharedRefArray::operator=(const TSharedRefArray& other)
{
    Impl_ = other.Impl_;
    return *this;
}

Y_FORCE_INLINE TSharedRefArray& TSharedRefArray::operator=(TSharedRefArray&& other)
{
    Impl_ = std::move(other.Impl_);
    return *this;
}

Y_FORCE_INLINE void TSharedRefArray::Reset()
{
    Impl_.Reset();
}

Y_FORCE_INLINE TSharedRefArray::operator bool() const
{
    return Impl_.operator bool();
}

Y_FORCE_INLINE size_t TSharedRefArray::Size() const
{
    return Impl_ ? Impl_->Size() : 0;
}

Y_FORCE_INLINE size_t TSharedRefArray::size() const
{
    return Impl_ ? Impl_->Size() : 0;
}

Y_FORCE_INLINE bool TSharedRefArray::Empty() const
{
    return Impl_ ? Impl_->Empty() : true;
}

Y_FORCE_INLINE const TSharedRef& TSharedRefArray::operator[](size_t index) const
{
    YT_ASSERT(Impl_);
    return (*Impl_)[index];
}

Y_FORCE_INLINE const TSharedRef* TSharedRefArray::Begin() const
{
    return Impl_ ? Impl_->Begin() : nullptr;
}

Y_FORCE_INLINE const TSharedRef* TSharedRefArray::End() const
{
    return Impl_ ? Impl_->End() : nullptr;
}

template <class... As>
TSharedRefArrayImplPtr TSharedRefArray::NewImpl(
    size_t size,
    size_t poolCapacity,
    TRefCountedTypeCookie tagCookie,
    As&&... args)
{
    auto extraSpaceSize = sizeof(TSharedRef) * size + poolCapacity;
    return NewWithExtraSpace<TSharedRefArrayImpl>(
        extraSpaceSize,
        extraSpaceSize,
        tagCookie,
        std::forward<As>(args)...);
}

Y_FORCE_INLINE const TSharedRef* begin(const TSharedRefArray& array)
{
    return array.Begin();
}

Y_FORCE_INLINE const TSharedRef* end(const TSharedRefArray& array)
{
    return array.End();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
