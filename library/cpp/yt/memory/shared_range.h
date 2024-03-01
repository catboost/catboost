#pragma once

#include "public.h"
#include "intrusive_ptr.h"
#include "range.h"
#include "ref_counted.h"

#include <library/cpp/yt/assert/assert.h>

#include <util/ysaveload.h>

#include <optional>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t N>
class TCompactVector;

////////////////////////////////////////////////////////////////////////////////

struct TSharedRangeHolderCloneOptions
{
    bool KeepMemoryReferenceTracking = true;
};

struct TSharedRangeHolder
    : public TRefCounted
{
    //! Clones the holder possibly adjusting its flavor based on #options.
    /*!
     *  The default implementation just returns this.
     */
    virtual TSharedRangeHolderPtr Clone(const TSharedRangeHolderCloneOptions& options);

    //! Returns the (estimated) total number of bytes being held or |null| if unable to estimate.
    /*!
     *  The returned value is static and never changes.
     *  The default implementation returns |null|.
     */
    virtual std::optional<size_t> GetTotalByteSize() const;
};

DEFINE_REFCOUNTED_TYPE(TSharedRangeHolder)

////////////////////////////////////////////////////////////////////////////////

//! TRange with ownership semantics.
template <class T>
class TSharedRange
    : public TRange<T>
{
public:
    //! Constructs a null TSharedRange.
    TSharedRange()
    { }

    //! Constructs an empty TSharedRange from a nullptr expression.
    TSharedRange(std::nullptr_t)
        : TRange<T>(nullptr, 0UL)
        , Holder_(nullptr)
    { }

    //! Constructs a TSharedRange from TRange.
    TSharedRange(TRange<T> range, TSharedRangeHolderPtr holder)
        : TRange<T>(range)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedRange from a pointer and length.
    TSharedRange(const T* data, size_t length, TSharedRangeHolderPtr holder)
        : TRange<T>(data, length)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedRange from a range.
    TSharedRange(const T* begin, const T* end, TSharedRangeHolderPtr holder)
        : TRange<T>(begin, end)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedRange from a TCompactVector.
    template <size_t N>
    TSharedRange(const TCompactVector<T, N>& elements, TSharedRangeHolderPtr holder)
        : TRange<T>(elements)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedRange from an std::vector.
    TSharedRange(const std::vector<T>& elements, TSharedRangeHolderPtr holder)
        : TRange<T>(elements)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedRange from a C array.
    template <size_t N>
    TSharedRange(const T (& elements)[N], TSharedRangeHolderPtr holder)
        : TRange<T>(elements)
        , Holder_(std::move(holder))
    { }

    TSharedRange(const TSharedRange& other) = default;

    TSharedRange(TSharedRange&& other) noexcept
        : TSharedRange()
    {
        other.Swap(*this);
    }

    TSharedRange& operator=(TSharedRange other) noexcept
    {
        other.Swap(*this);
        return *this;
    }

    void Swap(TSharedRange& other) noexcept
    {
        DoSwap(TRange<T>::Data_, other.Data_);
        DoSwap(TRange<T>::Length_, other.Length_);
        Holder_.Swap(other.Holder_);
    }

    void Reset()
    {
        TRange<T>::Data_ = nullptr;
        TRange<T>::Length_ = 0;
        Holder_.Reset();
    }

    TSharedRange<T> Slice(size_t startOffset, size_t endOffset) const
    {
        YT_ASSERT(startOffset <= this->Size());
        YT_ASSERT(endOffset >= startOffset && endOffset <= this->Size());
        return TSharedRange<T>(this->Begin() + startOffset, endOffset - startOffset, Holder_);
    }

    TSharedRange<T> Slice(const T* begin, const T* end) const
    {
        YT_ASSERT(begin >= this->Begin());
        YT_ASSERT(end <= this->End());
        return TSharedRange<T>(begin, end, Holder_);
    }

    const TSharedRangeHolderPtr& GetHolder() const
    {
        return Holder_;
    }

    TSharedRangeHolderPtr&& ReleaseHolder()
    {
        return std::move(Holder_);
    }

protected:
    TSharedRangeHolderPtr Holder_;

};

////////////////////////////////////////////////////////////////////////////////

//! Constructs a combined holder from a vector of typed holders.
TSharedRangeHolderPtr MakeCompositeSharedRangeHolder(std::vector<TSharedRangeHolderPtr> holders);

//! Constructs a combined holder instance by taking ownership of a given list of holders.
template <class... THolders>
TSharedRangeHolderPtr MakeSharedRangeHolder(THolders&&... holders)
{
    struct THolder
        : public TSharedRangeHolder
    {
        std::tuple<typename std::decay<THolders>::type...> Holders;
    };

    auto holder = New<THolder>();
    holder->Holders = std::tuple<THolders...>(std::forward<THolders>(holders)...);
    return holder;
}

template <class T, class TContainer, class... THolders>
TSharedRange<T> DoMakeSharedRange(TContainer&& elements, THolders&&... holders)
{
    struct THolder
        : public TSharedRangeHolder
    {
        typename std::decay<TContainer>::type Elements;
        std::tuple<typename std::decay<THolders>::type...> Holders;
    };

    auto holder = New<THolder>();
    holder->Holders = std::tuple<THolders...>(std::forward<THolders>(holders)...);
    holder->Elements = std::forward<TContainer>(elements);

    auto range = MakeRange<T>(holder->Elements);

    return TSharedRange<T>(range, std::move(holder));
}

//! Constructs a TSharedRange by taking ownership of an std::vector.
template <class T, class... THolders>
TSharedRange<T> MakeSharedRange(std::vector<T>&& elements, THolders&&... holders)
{
    return DoMakeSharedRange<T>(std::move(elements), std::forward<THolders>(holders)...);
}

//! Constructs a TSharedRange by taking ownership of an TCompactVector.
template <class T, size_t N, class... THolders>
TSharedRange<T> MakeSharedRange(TCompactVector<T, N>&& elements, THolders&&... holders)
{
    return DoMakeSharedRange<T>(std::move(elements), std::forward<THolders>(holders)...);
}

//! Constructs a TSharedRange by copying an std::vector.
template <class T, class... THolders>
TSharedRange<T> MakeSharedRange(const std::vector<T>& elements, THolders&&... holders)
{
    return DoMakeSharedRange<T>(elements, std::forward<THolders>(holders)...);
}

template <class T, class... THolders>
TSharedRange<T> MakeSharedRange(TRange<T> range, THolders&&... holders)
{
    return TSharedRange<T>(range, MakeSharedRangeHolder(std::forward<THolders>(holders)...));
}

template <class T, class THolder>
TSharedRange<T> MakeSharedRange(TRange<T> range, TSharedRangeHolderPtr holder)
{
    return TSharedRange<T>(range, std::move(holder));
}

template <class U, class T>
TSharedRange<U> ReinterpretCastRange(const TSharedRange<T>& range)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must have equal sizes.");
    return TSharedRange<U>(reinterpret_cast<const U*>(range.Begin()), range.Size(), range.GetHolder());
}

////////////////////////////////////////////////////////////////////////////////

//! TMutableRange with ownership semantics.
//! Use with caution :)
template <class T>
class TSharedMutableRange
    : public TMutableRange<T>
{
public:
    //! Constructs a null TSharedMutableRange.
    TSharedMutableRange()
    { }

    //! Constructs a TSharedMutableRange from TMutableRange.
    TSharedMutableRange(TMutableRange<T> range, TSharedRangeHolderPtr holder)
        : TMutableRange<T>(range)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedMutableRange from a pointer and length.
    TSharedMutableRange(T* data, size_t length, TSharedRangeHolderPtr holder)
        : TMutableRange<T>(data, length)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedMutableRange from a range.
    TSharedMutableRange(T* begin, T* end, TSharedRangeHolderPtr holder)
        : TMutableRange<T>(begin, end)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedMutableRange from a TCompactVector.
    template <size_t N>
    TSharedMutableRange(TCompactVector<T, N>& elements, TSharedRangeHolderPtr holder)
        : TMutableRange<T>(elements)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedMutableRange from an std::vector.
    TSharedMutableRange(std::vector<T>& elements, TSharedRangeHolderPtr holder)
        : TMutableRange<T>(elements)
        , Holder_(std::move(holder))
    { }

    //! Constructs a TSharedMutableRange from a C array.
    template <size_t N>
    TSharedMutableRange(T (& elements)[N], TSharedRangeHolderPtr holder)
        : TMutableRange<T>(elements)
        , Holder_(std::move(holder))
    { }

    TSharedMutableRange(const TSharedMutableRange& other) = default;

    TSharedMutableRange(TSharedMutableRange&& other) noexcept
        : TSharedMutableRange()
    {
        other.Swap(*this);
    }

    TSharedMutableRange& operator=(TSharedMutableRange other) noexcept
    {
        other.Swap(*this);
        return *this;
    }

    void Swap(TSharedMutableRange& other) noexcept
    {
        DoSwap(TRange<T>::Data_, other.Data_);
        DoSwap(TRange<T>::Length_, other.Length_);
        Holder_.Swap(other.Holder_);
    }

    void Reset()
    {
        TRange<T>::Data_ = nullptr;
        TRange<T>::Length_ = 0;
        Holder_.Reset();
    }

    TSharedMutableRange<T> Slice(size_t startOffset, size_t endOffset) const
    {
        YT_ASSERT(startOffset <= this->Size());
        YT_ASSERT(endOffset >= startOffset && endOffset <= this->Size());
        return TSharedMutableRange<T>(this->Begin() + startOffset, endOffset - startOffset, Holder_);
    }

    TSharedMutableRange<T> Slice(T* begin, T* end) const
    {
        YT_ASSERT(begin >= this->Begin());
        YT_ASSERT(end <= this->End());
        return TSharedMutableRange<T>(begin, end, Holder_);
    }

    TSharedRangeHolderPtr GetHolder() const
    {
        return Holder_;
    }

    TSharedRangeHolderPtr&& ReleaseHolder()
    {
        return std::move(Holder_);
    }

protected:
    TSharedRangeHolderPtr Holder_;
};

template <class T, class TContainer, class... THolders>
TSharedMutableRange<T> DoMakeSharedMutableRange(TContainer&& elements, THolders&&... holders)
{
    struct THolder
        : public TSharedRangeHolder
    {
        typename std::decay<TContainer>::type Elements;
        std::tuple<typename std::decay<THolders>::type...> Holders;
    };

    auto holder = New<THolder>();
    holder->Holders = std::tuple<THolders...>(std::forward<THolders>(holders)...);
    holder->Elements = std::forward<TContainer>(elements);

    auto range = TMutableRange<T>(holder->Elements);

    return TSharedMutableRange<T>(range, holder);
}

//! Constructs a TSharedMutableRange by taking ownership of an std::vector.
template <class T, class... THolders>
TSharedMutableRange<T> MakeSharedMutableRange(std::vector<T>&& elements, THolders&&... holders)
{
    return DoMakeSharedMutableRange<T>(std::move(elements), std::forward<THolders>(holders)...);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
