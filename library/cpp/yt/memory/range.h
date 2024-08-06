#pragma once

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/hash.h>

#include <array>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>
#include <type_traits>

// For size_t.
#include <stddef.h>

namespace google::protobuf {

////////////////////////////////////////////////////////////////////////////////
// Forward declarations

template <class T>
class RepeatedField;

template <class T>
class RepeatedPtrField;

////////////////////////////////////////////////////////////////////////////////

} // namespace google::protobuf

namespace NYT {

////////////////////////////////////////////////////////////////////////////////
// Forward declarations

template <class T, size_t N>
class TCompactVector;

////////////////////////////////////////////////////////////////////////////////

//! TRange (inspired by TArrayRef from LLVM)
/*!
 * Represents a constant reference to an array (zero or more elements
 * consecutively in memory), i. e. a start pointer and a length. It allows
 * various APIs to take consecutive elements easily and conveniently.
 *
 * This class does not own the underlying data, it is expected to be used in
 * situations where the data resides in some other buffer, whose lifetime
 * extends past that of the TRange. For this reason, it is not in general
 * safe to store an TRange.
 *
 * This is intended to be trivially copyable, so it should be passed by
 * value.
 */
template <class T>
class TRange
{
public:
    using iterator = const T*;
    using const_iterator = const T*;
    using size_type = size_t;

    static constexpr bool CAnyRepeatedField = requires {
        { google::protobuf::RepeatedField<T>{} }
            ->
        std::same_as<google::protobuf::RepeatedField<T>>;
    };

    //! Constructs a null TRange.
    TRange()
        : Data_(nullptr)
        , Length_(0)
    { }

    //! Constructs a TRange from a pointer and length.
    TRange(const T* data, size_t length)
        : Data_(data)
        , Length_(length)
    { }

    //! Constructs a TRange from a range.
    TRange(const T* begin, const T* end)
        : Data_(begin)
        , Length_(end - begin)
    { }

    //! Constructs a TRange from a TCompactVector.
    template <size_t N>
    TRange(const TCompactVector<T, N>& elements)
        : Data_(elements.data())
        , Length_(elements.size())
    { }

    //! Constructs a TRange from an std::vector.
    template <class A>
    TRange(const std::vector<T, A>& elements)
        : Data_(elements.data())
        , Length_(elements.size())
    { }

    //! Constructs a TRange from an std::span.
    TRange(const std::span<T>& elements)
        : Data_(elements.data())
        , Length_(elements.size())
    { }

    //! Constructs a TRange from a C array.
    template <size_t N>
    TRange(const T (&elements)[N])
        : Data_(elements)
        , Length_(N)
    { }

    //! Constructs a TRange from std::initializer_list.
    TRange(std::initializer_list<T> elements)
        : Data_(elements.begin())
        , Length_(elements.size())
    { }

    //! Constructs a TRange from std::array.
    template <size_t N>
    TRange(const std::array<T, N>& elements)
        : Data_(elements.data())
        , Length_(N)
    { }

    //! Constructs a TRange from std::optional.
    //! Range will contain 0-1 elements.
    explicit TRange(const std::optional<T>& element)
        : Data_(element ? &*element : nullptr)
        , Length_(element ? 1 : 0)
    { }

    //! Constructs a TRange from RepeatedField.
    template<std::same_as<T> U = T>
    requires CAnyRepeatedField
    TRange(const google::protobuf::RepeatedField<U>& elements)
        : Data_(elements.data())
        , Length_(elements.size())
    { }

    //! Constructs a TRange from RepeatedPtrField.
    template<class TPointed>
    TRange(const google::protobuf::RepeatedPtrField<TPointed>& elements)
        : Data_(elements.data())
        , Length_(elements.size())
    { }

    const_iterator Begin() const
    {
        return Data_;
    }

    // STL interop, for gcc.
    const_iterator begin() const
    {
        return Begin();
    }

    const_iterator End() const
    {
        return Data_ + Length_;
    }

    // STL interop, for gcc.
    const_iterator end() const
    {
        return End();
    }

    bool Empty() const
    {
        return Length_ == 0;
    }

    bool empty() const
    {
        return Empty();
    }

    explicit operator bool() const
    {
        return Data_ != nullptr;
    }

    size_t Size() const
    {
        return Length_;
    }

    size_t size() const
    {
        return Size();
    }

    const T* Data() const
    {
        return Data_;
    }

    const T* data() const
    {
        return Data();
    }

    const T& operator[](size_t index) const
    {
        YT_ASSERT(index < Size());
        return Data_[index];
    }


    const T& Front() const
    {
        YT_ASSERT(Length_ > 0);
        return Data_[0];
    }

    const T& Back() const
    {
        YT_ASSERT(Length_ > 0);
        return Data_[Length_ - 1];
    }


    TRange<T> Slice(size_t startOffset, size_t endOffset) const
    {
        YT_ASSERT(startOffset <= endOffset && endOffset <= Size());
        return TRange<T>(Begin() + startOffset, endOffset - startOffset);
    }

    std::vector<T> ToVector() const
    {
        return std::vector<T>(Data_, Data_ + Length_);
    }

protected:
    //! The start of the array, in an external buffer.
    const T* Data_;

    //! The number of elements.
    size_t Length_;

};

// STL interop.
template <class T>
typename TRange<T>::const_iterator begin(TRange<T> ref)
{
    return ref.Begin();
}

template <class T>
typename TRange<T>::const_iterator end(TRange<T> ref)
{
    return ref.End();
}

// Deduction guide for the constructor from RepeatedPtrField.
template<class TPointed>
TRange(const google::protobuf::RepeatedPtrField<TPointed>& elements) -> TRange<const TPointed*>;

////////////////////////////////////////////////////////////////////////////////

template <class U, class T>
TRange<U> ReinterpretCastRange(TRange<T> range)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must have equal sizes.");
    return TRange<U>(reinterpret_cast<const U*>(range.Begin()), range.Size());
}

////////////////////////////////////////////////////////////////////////////////

// TMutableRange (inspired by TMutableArrayRef from LLVM)
/*
 * Represents a mutable reference to an array (zero or more elements
 * consecutively in memory), i. e. a start pointer and a length.
 * It allows various APIs to take and modify consecutive elements easily and
 * conveniently.
 *
 * This class does not own the underlying data, it is expected to be used in
 * situations where the data resides in some other buffer, whose lifetime
 * extends past that of the TMutableRange. For this reason, it is not in
 * general safe to store a TMutableRange.
 *
 * This is intended to be trivially copyable, so it should be passed by value.
 */
template <class T>
class TMutableRange
    : public TRange<T>
{
public:
    using iterator = T*;

    //! Constructs a null TMutableRange.
    TMutableRange()
    { }

    //! Constructs a TMutableRange from a pointer and length.
    TMutableRange(T* data, size_t length)
        : TRange<T>(data, length)
    { }

    //! Constructs a TMutableRange from a range.
    TMutableRange(T* begin, T* end)
        : TRange<T>(begin, end)
    { }

    //! Constructs a TMutableRange from a TCompactVector.
    template <size_t N>
    TMutableRange(TCompactVector<T, N>& elements)
        : TRange<T>(elements)
    { }

    //! Constructs a TMutableRange from an std::vector.
    TMutableRange(std::vector<T>& elements)
        : TRange<T>(elements)
    { }

    //! Constructs a TMutableRange from an std::span.
    TMutableRange(std::span<T>& elements)
        : TRange<T>(elements)
    { }

    //! Constructs a TMutableRange from std::array.
    template <size_t N>
    TMutableRange(std::array<T, N>& elements)
        : TRange<T>(elements.data(), N)
    { }

    //! Construct a TMutableRange from an std::optional
    //! Range will contain 0-1 elements.
    explicit TMutableRange(std::optional<T>& optional)
        : TRange<T>(optional)
    { }

    //! Constructs a TMutableRange from a C array.
    template <size_t N>
    TMutableRange(T (& elements)[N])
        : TRange<T>(elements)
    { }

    //! Constructs a TMutableRange from RepeatedField.
    template<std::same_as<T> U = T>
    requires TRange<T>::CAnyRepeatedField
    TMutableRange(google::protobuf::RepeatedField<U>& elements)
        : TRange<T>(elements)
    { }

    //! Constructs a TMutableRange from RepeatedPtrField.
    template<class TPointed>
    TMutableRange(google::protobuf::RepeatedPtrField<TPointed>& elements)
        : TRange<T>(elements)
    { }

    using TRange<T>::Begin;
    using TRange<T>::End;
    using TRange<T>::Front;
    using TRange<T>::Back;
    using TRange<T>::operator[];

    iterator Begin() const
    {
        return const_cast<T*>(this->Data_);
    }

    // STL interop, for gcc.
    iterator begin() const
    {
        return Begin();
    }

    iterator End() const
    {
        return this->Begin() + this->Size();
    }

    // STL interop, for gcc.
    iterator end() const
    {
        return End();
    }

    T& operator[](size_t index)
    {
        YT_ASSERT(index <= this->Size());
        return Begin()[index];
    }

    T& Front()
    {
        YT_ASSERT(this->Length_ > 0);
        return Begin()[0];
    }

    T& Back()
    {
        YT_ASSERT(this->Length_ > 0);
        return Begin()[this->Length_ - 1];
    }

    TMutableRange<T> Slice(size_t startOffset, size_t endOffset) const
    {
        YT_ASSERT(startOffset <= endOffset && endOffset <= this->Size());
        return TMutableRange<T>(Begin() + startOffset, endOffset - startOffset);
    }

    TMutableRange<T> Slice(T* begin, T* end) const
    {
        YT_ASSERT(begin >= Begin());
        YT_ASSERT(end <= End());
        return TMutableRange<T>(begin, end);
    }
};

// STL interop.
template <class T>
typename TMutableRange<T>::iterator begin(TMutableRange<T> ref)
{
    return ref.Begin();
}

template <class T>
typename TMutableRange<T>::iterator end(TMutableRange<T> ref)
{
    return ref.End();
}

// Deduction guide for the constructor from RepeatedPtrField.
template<class TPointed>
TMutableRange(google::protobuf::RepeatedPtrField<TPointed>& elements) -> TMutableRange<TPointed*>;

////////////////////////////////////////////////////////////////////////////////

template <class U, class T>
TMutableRange<U> ReinterpretCastMutableRange(TMutableRange<T> range)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must have equal sizes.");
    return TMutableRange<U>(reinterpret_cast<U*>(range.Begin()), range.Size());
}

////////////////////////////////////////////////////////////////////////////////

// Mark TMutableRange and TMutableRange as PODs.
namespace NMpl {

template <class T>
struct TIsPod;

template <class T>
struct TIsPod<TRange<T>>
{
    static const bool Value = true;
};

template <class T>
struct TIsPod<TMutableRange<T>>
{
    static const bool Value = true;
};

} // namespace NMpl

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

template <class T>
struct hash<NYT::TRange<T>>
{
    size_t operator()(const NYT::TRange<T>& range) const
    {
        size_t result = 0;
        for (const auto& element : range) {
            NYT::HashCombine(result, element);
        }
        return result;
    }
};

template <class T>
struct hash<NYT::TMutableRange<T>>
{
    size_t operator()(const NYT::TMutableRange<T>& range) const
    {
        size_t result = 0;
        for (const auto& element : range) {
            NYT::HashCombine(result, element);
        }
        return result;
    }
};


