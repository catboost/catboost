#pragma once

#include <type_traits>

#include <util/system/types.h>
#include <util/generic/typetraits.h>
#include <util/generic/fwd.h>

class IOutputStream;
namespace NPrivate {
    void PrintFlags(IOutputStream& stream, ui64 value, size_t size);
} // namespace NPrivate

/**
 * `TFlags` wrapper provides a type-safe mechanism for storing OR combinations
 * of enumeration values.
 *
 * This class is intended to be used mainly via helper macros. For example:
 * @code
 * class TAligner {
 * public:
 *     enum EOrientation {
 *         Vertical = 1,
 *         Horizontal = 2
 *     };
 *     Y_DECLARE_FLAGS(EOrientations, EOrientation);
 *
 *     // ...
 * };
 *
 * Y_DECLARE_OPERATORS_FOR_FLAGS(TAligner::EOrientations);
 * @endcode
 */
template <class Enum>
class TFlags {
    static_assert(std::is_enum<Enum>::value, "Expecting an enumeration here.");

public:
    using TEnum = Enum;
    using TInt = std::underlying_type_t<Enum>;

    constexpr TFlags(std::nullptr_t = 0)
        : Value_(0)
    {
    }

    constexpr TFlags(Enum value)
        : Value_(static_cast<TInt>(value))
    {
    }

    /* Generated copy/move ctor/assignment are OK. */

    constexpr operator TInt() const {
        return Value_;
    }

    constexpr TInt ToBaseType() const {
        return Value_;
    }

    constexpr static TFlags FromBaseType(TInt value) {
        return TFlags(TFlag(value));
    }

    constexpr friend TFlags operator|(TFlags l, TFlags r) {
        return TFlags(TFlag(l.Value_ | r.Value_));
    }

    constexpr friend TFlags operator|(TEnum l, TFlags r) {
        return TFlags(TFlag(static_cast<TInt>(l) | r.Value_));
    }

    constexpr friend TFlags operator|(TFlags l, TEnum r) {
        return TFlags(TFlag(l.Value_ | static_cast<TInt>(r)));
    }

    constexpr friend TFlags operator^(TFlags l, TFlags r) {
        return TFlags(TFlag(l.Value_ ^ r.Value_));
    }

    constexpr friend TFlags
    operator^(TEnum l, TFlags r) {
        return TFlags(TFlag(static_cast<TInt>(l) ^ r.Value_));
    }

    constexpr friend TFlags
    operator^(TFlags l, TEnum r) {
        return TFlags(TFlag(l.Value_ ^ static_cast<TInt>(r)));
    }

    constexpr friend TFlags
    operator&(TFlags l, TFlags r) {
        return TFlags(TFlag(l.Value_ & r.Value_));
    }

    constexpr friend TFlags operator&(TEnum l, TFlags r) {
        return TFlags(TFlag(static_cast<TInt>(l) & r.Value_));
    }

    constexpr friend TFlags operator&(TFlags l, TEnum r) {
        return TFlags(TFlag(l.Value_ & static_cast<TInt>(r)));
    }

    constexpr friend bool operator==(TFlags l, TFlags r) {
        return l.Value_ == r.Value_;
    }

    constexpr friend bool operator==(TEnum l, TFlags r) {
        return static_cast<TInt>(l) == r.Value_;
    }

    constexpr friend bool operator==(TFlags l, TEnum r) {
        return l.Value_ == static_cast<TInt>(r);
    }

    constexpr friend bool operator!=(TFlags l, TFlags r) {
        return l.Value_ != r.Value_;
    }

    constexpr friend bool operator!=(TEnum l, TFlags r) {
        return static_cast<TInt>(l) != r.Value_;
    }

    constexpr friend bool operator!=(TFlags l, TEnum r) {
        return l.Value_ != static_cast<TInt>(r);
    }

    TFlags& operator&=(TFlags flags) {
        *this = *this & flags;
        return *this;
    }

    TFlags& operator&=(Enum value) {
        *this = *this & value;
        return *this;
    }

    TFlags& operator|=(TFlags flags) {
        *this = *this | flags;
        return *this;
    }

    TFlags& operator|=(Enum value) {
        *this = *this | value;
        return *this;
    }

    TFlags& operator^=(TFlags flags) {
        *this = *this ^ flags;
        return *this;
    }

    TFlags& operator^=(Enum flags) {
        *this = *this ^ flags;
        return *this;
    }

    constexpr TFlags operator~() const {
        return TFlags(TFlag(~Value_));
    }

    constexpr bool operator!() const {
        return !Value_;
    }

    constexpr explicit operator bool() const {
        return Value_;
    }

    constexpr bool HasFlag(Enum value) const {
        return (Value_ & static_cast<TInt>(value)) == static_cast<TInt>(value);
    }

    constexpr bool HasFlags(TFlags flags) const {
        return (Value_ & flags.Value_) == flags.Value_;
    }

    constexpr bool HasAnyOfFlags(TFlags flags) const {
        return (Value_ & flags.Value_) != 0;
    }

    TFlags RemoveFlag(Enum value) {
        Value_ &= ~static_cast<TInt>(value);
        return *this;
    }

    TFlags RemoveFlags(TFlags flags) {
        Value_ &= ~flags.Value_;
        return *this;
    }

    friend IOutputStream& operator<<(IOutputStream& stream Y_LIFETIME_BOUND, const TFlags& flags) {
        ::NPrivate::PrintFlags(stream, static_cast<ui64>(flags.Value_), sizeof(TInt));
        return stream;
    }

private:
    struct TFlag {
        constexpr TFlag() {
        }
        constexpr explicit TFlag(TInt value)
            : Value(value)
        {
        }

        TInt Value = 0;
    };

    constexpr explicit TFlags(TFlag value)
        : Value_(value.Value)
    {
    }

private:
    TInt Value_;
};

template <class T>
struct TPodTraits<::TFlags<T>> {
    enum {
        IsPod = TTypeTraits<T>::IsPod
    };
};

template <class Enum>
struct THash<::TFlags<Enum>> {
    size_t operator()(const TFlags<Enum>& flags) const noexcept {
        return THash<typename ::TFlags<Enum>::TInt>()(flags);
    }
};

/**
 * This macro defines a flags type for the provided enum.
 *
 * @param FLAGS                         Name of the flags type to declare.
 * @param ENUM                          Name of the base enum type to use.
 */
#define Y_DECLARE_FLAGS(FLAGS, ENUM) \
    using FLAGS = ::TFlags<ENUM>

/**
 * This macro declares global operator functions for enum base of `FLAGS` type.
 * This way operations on individual enum values will provide a type-safe
 * `TFlags` object.
 *
 * @param FLAGS                         Flags type to declare operator for.
 */
#define Y_DECLARE_OPERATORS_FOR_FLAGS(FLAGS)                           \
    Y_DECLARE_UNUSED                                                   \
    constexpr inline FLAGS operator|(FLAGS::TEnum l, FLAGS::TEnum r) { \
        return FLAGS(l) | r;                                           \
    }                                                                  \
    Y_DECLARE_UNUSED                                                   \
    constexpr inline FLAGS operator~(FLAGS::TEnum value) {             \
        return ~FLAGS(value);                                          \
    }                                                                  \
    Y_SEMICOLON_GUARD
