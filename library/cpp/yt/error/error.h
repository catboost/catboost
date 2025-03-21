#pragma once

#include <library/cpp/yt/error/error_code.h>

#include <library/cpp/yt/threading/public.h>

#include <library/cpp/yt/error/mergeable_dictionary.h>

#include <library/cpp/yt/yson_string/convert.h>
#include <library/cpp/yt/yson_string/string.h>

#include <library/cpp/yt/misc/property.h>

#include <util/system/compiler.h>
#include <util/system/getpid.h>

#include <util/generic/size_literals.h>

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! An opaque wrapper around |int| value capable of conversions from |int|s and
//! arbitrary enum types.
class TErrorCode
{
public:
    using TUnderlying = int;

    constexpr TErrorCode();
    explicit constexpr TErrorCode(int value);
    template <class E>
        requires std::is_enum_v<E>
    constexpr TErrorCode(E value);

    constexpr operator int() const;

    template <class E>
        requires std::is_enum_v<E>
    constexpr operator E() const;

    template <class E>
        requires std::is_enum_v<E>
    constexpr bool operator == (E rhs) const;

    constexpr bool operator == (TErrorCode rhs) const;
private:
    int Value_;
};

////////////////////////////////////////////////////////////////////////////////

void FormatValue(TStringBuilderBase* builder, TErrorCode code, TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

// Forward declaration.
class TErrorException;

template <class TValue>
concept CErrorNestable = requires (TError& error, TValue&& operand)
{
    { error <<= std::forward<TValue>(operand) } -> std::same_as<TError&>;
};

template <>
class [[nodiscard]] TErrorOr<void>
{
public:
    TErrorOr();
    ~TErrorOr();

    TErrorOr(const TError& other);
    TErrorOr(TError&& other) noexcept;

    TErrorOr(const TErrorException& errorEx) noexcept;

    TErrorOr(const std::exception& ex);

    struct TDisableFormat
    { };
    static constexpr TDisableFormat DisableFormat = {};

    TErrorOr(std::string message, TDisableFormat);
    TErrorOr(TErrorCode code, std::string message, TDisableFormat);

    template <class... TArgs>
    explicit TErrorOr(
        TFormatString<TArgs...> format,
        TArgs&&... arg);

    template <class... TArgs>
    TErrorOr(
        TErrorCode code,
        TFormatString<TArgs...> format,
        TArgs&&... args);

    TError& operator = (const TError& other);
    TError& operator = (TError&& other) noexcept;

    static TError FromSystem();
    static TError FromSystem(int error);
    static TError FromSystem(const TSystemError& error);

    TErrorCode GetCode() const;
    TError& SetCode(TErrorCode code);

    TErrorCode GetNonTrivialCode() const;
    THashSet<TErrorCode> GetDistinctNonTrivialErrorCodes() const;

    const std::string& GetMessage() const;
    TError& SetMessage(std::string message);

    bool HasOriginAttributes() const;
    TProcessId GetPid() const;
    TStringBuf GetThreadName() const;
    NThreading::TThreadId GetTid() const;

    bool HasDatetime() const;
    TInstant GetDatetime() const;

    bool HasAttributes() const noexcept;

    const TErrorAttributes& Attributes() const;
    TErrorAttributes* MutableAttributes();

    const std::vector<TError>& InnerErrors() const;
    std::vector<TError>* MutableInnerErrors();

    // Used for deserialization only.
    TOriginAttributes* MutableOriginAttributes() const noexcept;
    void UpdateOriginAttributes();

    TError Truncate(
        int maxInnerErrorCount = 2,
        i64 stringLimit = 16_KB,
        const THashSet<TStringBuf>& attributeWhitelist = {}) const &;
    TError Truncate(
        int maxInnerErrorCount = 2,
        i64 stringLimit = 16_KB,
        const THashSet<TStringBuf>& attributeWhitelist = {}) &&;

    bool IsOK() const;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    void ThrowOnError(U&& u) const &;
    template <class... TArgs>
    void ThrowOnError(TFormatString<TArgs...> format, TArgs&&... args) const &;
    template <class... TArgs>
    void ThrowOnError(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) const &;
    inline void ThrowOnError() const &;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    void ThrowOnError(U&& u) &&;
    template <class... TArgs>
    void ThrowOnError(TFormatString<TArgs...> format, TArgs&&... args) &&;
    template <class... TArgs>
    void ThrowOnError(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) &&;
    inline void ThrowOnError() &&;

    template <CInvocable<bool(const TError&)> TFilter>
    std::optional<TError> FindMatching(const TFilter& filter) const;
    template <CInvocable<bool(TErrorCode)> TFilter>
    std::optional<TError> FindMatching(const TFilter& filter) const;
    std::optional<TError> FindMatching(TErrorCode code) const;
    std::optional<TError> FindMatching(const THashSet<TErrorCode>& codes) const;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    TError Wrap(U&& u) const &;
    template <class... TArgs>
    TError Wrap(TFormatString<TArgs...> format, TArgs&&... args) const &;
    template <class... TArgs>
    TError Wrap(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) const &;
    TError Wrap() const &;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    TError Wrap(U&& u) &&;
    template <class... TArgs>
    TError Wrap(TFormatString<TArgs...> format, TArgs&&... args) &&;
    template <class... TArgs>
    TError Wrap(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) &&;
    TError Wrap() &&;

    //! Perform recursive aggregation of error codes and messages over the error tree.
    //! Result of this aggregation is suitable for error clustering in groups of
    //! "similar" errors. Refer to yt/yt/library/error_skeleton/skeleton_ut.cpp for examples.
    //!
    //! This method builds skeleton from scratch by doing complete error tree traversal,
    //! so calling it in computationally hot code is discouraged.
    //!
    //! In order to prevent core -> re2 dependency, implementation belongs to a separate library
    //! yt/yt/library/error_skeleton. Calling this method without PEERDIR'ing implementation
    //! results in an exception.
    std::string GetSkeleton() const;

    TError& operator <<= (const TErrorAttribute& attribute) &;
    TError& operator <<= (const std::vector<TErrorAttribute>& attributes) &;
    TError& operator <<= (const TError& innerError) &;
    TError& operator <<= (TError&& innerError) &;
    TError& operator <<= (const std::vector<TError>& innerErrors) &;
    TError& operator <<= (std::vector<TError>&& innerErrors) &;
    TError& operator <<= (TAnyMergeableDictionaryRef attributes) &;

    template <CErrorNestable TValue>
    TError&& operator << (TValue&& operand) &&;

    template <CErrorNestable TValue>
    TError operator << (TValue&& operand) const &;

    template <CErrorNestable TValue>
    TError&& operator << (const std::optional<TValue>& rhs) &&;

    template <CErrorNestable TValue>
    TError operator << (const std::optional<TValue>& rhs) const &;

    // The |enricher| is called during TError construction and before TErrorOr<> construction. Meant
    // to enrich the error, e.g. by setting generic attributes. The |RegisterEnricher| method is not
    // threadsafe and is meant to be called from single-threaded bootstrapping code. Multiple
    // enrichers are supported and will be called in order of registration.
    using TEnricher = std::function<void(TError&)>;
    static void RegisterEnricher(TEnricher enricher);

private:
    class TImpl;
    std::unique_ptr<TImpl> Impl_;

    explicit TErrorOr(std::unique_ptr<TImpl> impl);

    void MakeMutable();
    void Enrich();

    friend class TErrorAttributes;

    static TEnricher Enricher_;
};

////////////////////////////////////////////////////////////////////////////////

bool operator == (const TError& lhs, const TError& rhs);

////////////////////////////////////////////////////////////////////////////////

void FormatValue(TStringBuilderBase* builder, const TError& error, TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

using TErrorVisitor = std::function<void(const TError&, int depth)>;

//! Traverses the error tree in DFS order.
void TraverseError(
    const TError& error,
    const TErrorVisitor& visitor,
    int depth = 0);

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TErrorTraits
{
    using TWrapped = TErrorOr<T>;
    using TUnwrapped = T;
};

template <class T>
struct TErrorTraits<TErrorOr<T>>
{
    using TUnderlying = T;
    using TWrapped = TErrorOr<T>;
    using TUnwrapped = T;
};

////////////////////////////////////////////////////////////////////////////////

class TErrorException
    : public std::exception
{
public:
    DEFINE_BYREF_RW_PROPERTY(TError, Error);

public:
    TErrorException() = default;
    TErrorException(const TErrorException& other) = default;
    TErrorException(TErrorException&& other) = default;

    const char* what() const noexcept override;

private:
    mutable std::string CachedWhat_;
};

// Make these templates to avoid type erasure during throw.
template <class TException>
    requires std::derived_from<std::remove_cvref_t<TException>, TErrorException>
TException&& operator <<= (TException&& ex, const TError& error);
template <class TException>
    requires std::derived_from<std::remove_cvref_t<TException>, TErrorException>
TException&& operator <<= (TException&& ex, TError&& error);

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

struct TErrorAdaptor
{
    template <class TArg>
        requires std::constructible_from<TError, TArg>
    TError operator << (TArg&& rhs) const;

    template <class TArg>
        requires
            std::constructible_from<TError, TArg> &&
            std::derived_from<std::remove_cvref_t<TArg>, TError>
    TArg&& operator << (TArg&& rhs) const;
};

// Make these to correctly forward TError to Wrap call.
template <class TErrorLike, class U>
    requires
        std::derived_from<std::remove_cvref_t<TErrorLike>, TError> &&
        (!CStringLiteral<std::remove_cvref_t<U>>)
void ThrowErrorExceptionIfFailed(TErrorLike&& error, U&& u);

template <class TErrorLike, class... TArgs>
    requires std::derived_from<std::remove_cvref_t<TErrorLike>, TError>
void ThrowErrorExceptionIfFailed(TErrorLike&& error, TFormatString<TArgs...> format, TArgs&&... args);

template <class TErrorLike, class... TArgs>
    requires std::derived_from<std::remove_cvref_t<TErrorLike>, TError>
void ThrowErrorExceptionIfFailed(TErrorLike&& error, TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args);

template <class TErrorLike>
    requires std::derived_from<std::remove_cvref_t<TErrorLike>, TError>
void ThrowErrorExceptionIfFailed(TErrorLike&& error);

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

#define THROW_ERROR \
    throw ::NYT::TErrorException() <<= ::NYT::NDetail::TErrorAdaptor() <<

#define THROW_ERROR_EXCEPTION(head, ...) \
    THROW_ERROR ::NYT::TError(head __VA_OPT__(,) __VA_ARGS__)

// NB: When given an error and a string as arguments, this macro automatically wraps
// new error around the initial one.
#define THROW_ERROR_EXCEPTION_IF_FAILED(error, ...) \
    ::NYT::NDetail::ThrowErrorExceptionIfFailed((error) __VA_OPT__(,) __VA_ARGS__); \

#define THROW_ERROR_EXCEPTION_UNLESS(condition, head, ...) \
    if ((condition)) {\
    } else { \
        THROW_ERROR ::NYT::TError(head __VA_OPT__(,) __VA_ARGS__); \
    }

#define THROW_ERROR_EXCEPTION_IF(condition, head, ...) \
    THROW_ERROR_EXCEPTION_UNLESS(!(condition), head, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////

template <class T>
class [[nodiscard]] TErrorOr
    : public TError
{
public:
    TErrorOr();

    TErrorOr(const T& value);
    TErrorOr(T&& value) noexcept;

    TErrorOr(const TErrorOr<T>& other);
    TErrorOr(TErrorOr<T>&& other) noexcept;

    TErrorOr(const TError& other);
    TErrorOr(TError&& other) noexcept;

    TErrorOr(const TErrorException& errorEx) noexcept;

    TErrorOr(const std::exception& ex);

    template <class U>
    TErrorOr(const TErrorOr<U>& other);
    template <class U>
    TErrorOr(TErrorOr<U>&& other) noexcept;

    TErrorOr<T>& operator = (const TErrorOr<T>& other)
        requires std::is_copy_assignable_v<T>;
    TErrorOr<T>& operator = (TErrorOr<T>&& other) noexcept
        requires std::is_nothrow_move_assignable_v<T>;

    const T& Value() const & Y_LIFETIME_BOUND;
    T& Value() & Y_LIFETIME_BOUND;
    T&& Value() && Y_LIFETIME_BOUND;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    const T& ValueOrThrow(U&& u) const & Y_LIFETIME_BOUND;
    template <class... TArgs>
    const T& ValueOrThrow(TFormatString<TArgs...> format, TArgs&&... args) const & Y_LIFETIME_BOUND;
    template <class... TArgs>
    const T& ValueOrThrow(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) const & Y_LIFETIME_BOUND;
    const T& ValueOrThrow() const & Y_LIFETIME_BOUND;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    T& ValueOrThrow(U&& u) & Y_LIFETIME_BOUND;
    template <class... TArgs>
    T& ValueOrThrow(TFormatString<TArgs...> format, TArgs&&... args) & Y_LIFETIME_BOUND;
    template <class... TArgs>
    T& ValueOrThrow(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) & Y_LIFETIME_BOUND;
    T& ValueOrThrow() & Y_LIFETIME_BOUND;

    template <class U>
        requires (!CStringLiteral<std::remove_cvref_t<U>>)
    T&& ValueOrThrow(U&& u) && Y_LIFETIME_BOUND;
    template <class... TArgs>
    T&& ValueOrThrow(TFormatString<TArgs...> format, TArgs&&... args) && Y_LIFETIME_BOUND;
    template <class... TArgs>
    T&& ValueOrThrow(TErrorCode code, TFormatString<TArgs...> format, TArgs&&... args) && Y_LIFETIME_BOUND;
    T&& ValueOrThrow() && Y_LIFETIME_BOUND;

    const T& ValueOrDefault(const T& defaultValue Y_LIFETIME_BOUND) const & Y_LIFETIME_BOUND;
    T& ValueOrDefault(T& defaultValue Y_LIFETIME_BOUND) & Y_LIFETIME_BOUND;
    constexpr T ValueOrDefault(T&& defaultValue) const &;
    constexpr T ValueOrDefault(T&& defaultValue) &&;

private:
    std::optional<T> Value_;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
void FormatValue(TStringBuilderBase* builder, const TErrorOr<T>& error, TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

template <class F, class... As>
auto RunNoExcept(F&& functor, As&&... args) noexcept -> decltype(functor(std::forward<As>(args)...))
{
    return functor(std::forward<As>(args)...);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define STRIPPED_ERROR_INL_H_
#include "error-inl.h"
#undef STRIPPED_ERROR_INL_H_
