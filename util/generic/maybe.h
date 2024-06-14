#pragma once

#include <utility>

#include "maybe_traits.h"
#include "yexception.h"

#include <util/system/align.h>
#include <util/system/compiler.h>
#include <util/stream/output.h>
#include <util/ysaveload.h>

namespace NMaybe {
    struct TPolicyUndefinedExcept {
        [[noreturn]] static void OnEmpty(const std::type_info& valueTypeInfo);
    };

    struct TPolicyUndefinedFail {
        [[noreturn]] static void OnEmpty(const std::type_info& valueTypeInfo);
    };
}

struct TNothing {
    explicit constexpr TNothing(int) noexcept {
    }
};

constexpr TNothing NothingObject{0};

constexpr TNothing Nothing() noexcept {
    return NothingObject;
}

constexpr bool operator==(TNothing, TNothing) noexcept {
    return true;
}

template <class T, class Policy /*= ::NMaybe::TPolicyUndefinedExcept*/>
class TMaybe: private TMaybeBase<T> {
public:
    using TInPlace = NMaybe::TInPlace;

private:
    static_assert(!std::is_same<std::remove_cv_t<T>, TNothing>::value,
                  "Instantiation of TMaybe with a TNothing type is ill-formed");
    static_assert(!std::is_same<std::remove_cv_t<T>, TInPlace>::value,
                  "Instantiation of TMaybe with a TInPlace type is ill-formed");
    static_assert(!std::is_reference<T>::value,
                  "Instantiation of TMaybe with reference type is ill-formed");
    static_assert(!std::is_array<T>::value,
                  "Instantiation of TMaybe with array type is ill-formed");
    static_assert(std::is_destructible<T>::value,
                  "Instantiation of TMaybe with non-destructible type is ill-formed");

    template <class U>
    struct TConstructibleFromMaybeSomehow {
    public:
        static constexpr bool value = std::is_constructible<T, TMaybe<U, Policy>&>::value ||
                                      std::is_constructible<T, const TMaybe<U, Policy>&>::value ||
                                      std::is_constructible<T, TMaybe<U, Policy>&&>::value ||
                                      std::is_constructible<T, const TMaybe<U, Policy>&&>::value ||
                                      std::is_convertible<TMaybe<U, Policy>&, T>::value ||
                                      std::is_convertible<const TMaybe<U, Policy>&, T>::value ||
                                      std::is_convertible<TMaybe<U, Policy>&&, T>::value ||
                                      std::is_convertible<const TMaybe<U, Policy>&&, T>::value;
    };

    template <class U>
    struct TAssignableFromMaybeSomehow {
    public:
        static constexpr bool value = TConstructibleFromMaybeSomehow<U>::value ||
                                      std::is_assignable<T&, TMaybe<U, Policy>&>::value ||
                                      std::is_assignable<T&, const TMaybe<U, Policy>&>::value ||
                                      std::is_assignable<T&, TMaybe<U, Policy>&&>::value ||
                                      std::is_assignable<T&, const TMaybe<U, Policy>&&>::value;
    };

    template <class U>
    struct TImplicitCopyCtor {
    public:
        static constexpr bool value = std::is_constructible<T, const U&>::value &&
                                      std::is_convertible<const U&, T>::value &&
                                      !TConstructibleFromMaybeSomehow<U>::value;
    };

    template <class U>
    struct TExplicitCopyCtor {
    public:
        static constexpr bool value = std::is_constructible<T, const U&>::value &&
                                      !std::is_convertible<const U&, T>::value &&
                                      !TConstructibleFromMaybeSomehow<U>::value;
    };

    template <class U>
    struct TImplicitMoveCtor {
    public:
        static constexpr bool value = std::is_constructible<T, U&&>::value &&
                                      std::is_convertible<U&&, T>::value &&
                                      !TConstructibleFromMaybeSomehow<U>::value;
    };

    template <class U>
    struct TExplicitMoveCtor {
    public:
        static constexpr bool value = std::is_constructible<T, U&&>::value &&
                                      !std::is_convertible<U&&, T>::value &&
                                      !TConstructibleFromMaybeSomehow<U>::value;
    };

    template <class U>
    struct TCopyAssignable {
    public:
        static constexpr bool value = std::is_constructible<T, const U&>::value &&
                                      std::is_assignable<T&, const U&>::value &&
                                      !TAssignableFromMaybeSomehow<U>::value;
    };

    template <class U>
    struct TMoveAssignable {
    public:
        static constexpr bool value = std::is_constructible<T, U&&>::value &&
                                      std::is_assignable<T&, U&&>::value &&
                                      !TAssignableFromMaybeSomehow<U>::value;
    };

    template <class U>
    struct TImplicitAnyCtor {
    public:
        using UDec = std::decay_t<U>;

        static constexpr bool value = std::is_constructible<T, U>::value &&
                                      std::is_convertible<U, T>::value &&
                                      !std::is_same<UDec, TInPlace>::value &&
                                      !std::is_same<UDec, TMaybe>::value;
    };

    template <class U>
    struct TExplicitAnyCtor {
    public:
        using UDec = std::decay_t<U>;
        static constexpr bool value = std::is_constructible<T, U>::value &&
                                      !std::is_convertible<U, T>::value &&
                                      !std::is_same<UDec, TInPlace>::value &&
                                      !std::is_same<UDec, TMaybe>::value;
    };

    template <class U>
    struct TAssignableFromAny {
    public:
        using UDec = std::decay_t<U>;
        static constexpr bool value = !std::is_same<UDec, TMaybe>::value &&
                                      std::is_constructible<T, U>::value &&
                                      std::is_assignable<T&, U>::value &&
                                      (!std::is_scalar<T>::value || !std::is_same<UDec, T>::value);
    };

    template <typename>
    struct TIsMaybe {
        static constexpr bool value = false;
    };

    template <typename U, typename P>
    struct TIsMaybe<TMaybe<U, P>> {
        static constexpr bool value = true;
    };

    template <typename F, typename... Args>
    static auto Call(F&& f, Args&&... args) -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));

    template <typename F, typename... Args>
    using TCallResult = std::remove_reference_t<std::remove_cv_t<decltype(Call(std::declval<F>(), std::declval<Args>()...))>>;

    template <typename U, typename F>
    static constexpr F&& CheckReturnsMaybe(F&& f) {
        using ReturnType = TCallResult<F, U>;
        static_assert(TIsMaybe<ReturnType>::value, "Function must return TMaybe");
        return f;
    }

    using TBase = TMaybeBase<T>;

public:
    using value_type = T;
    using TValueType = value_type;

    TMaybe() noexcept = default;

    constexpr TMaybe(const TMaybe&) = default;
    constexpr TMaybe(TMaybe&&) = default;

    template <class... Args>
    constexpr explicit TMaybe(TInPlace, Args&&... args)
        : TBase(TInPlace{}, std::forward<Args>(args)...)
    {
    }

    template <class U, class... TArgs>
    constexpr explicit TMaybe(TInPlace, std::initializer_list<U> il, TArgs&&... args)
        : TBase(TInPlace{}, il, std::forward<TArgs>(args)...)
    {
    }

    constexpr TMaybe(TNothing) noexcept {
    }

    template <class U, class = std::enable_if_t<TImplicitCopyCtor<U>::value>>
    TMaybe(const TMaybe<U, Policy>& right) {
        if (right.Defined()) {
            new (Data()) T(right.GetRef());
            this->Defined_ = true;
        }
    }

    template <class U, std::enable_if_t<TExplicitCopyCtor<U>::value, bool> = false>
    explicit TMaybe(const TMaybe<U, Policy>& right) {
        if (right.Defined()) {
            new (Data()) T(right.GetRef());
            this->Defined_ = true;
        }
    }

    template <class U, class = std::enable_if_t<TImplicitMoveCtor<U>::value>>
    TMaybe(TMaybe<U, Policy>&& right) noexcept(std::is_nothrow_constructible<T, U&&>::value) {
        if (right.Defined()) {
            new (Data()) T(std::move(right.GetRef()));
            this->Defined_ = true;
        }
    }

    template <class U, std::enable_if_t<TExplicitMoveCtor<U>::value, bool> = false>
    explicit TMaybe(TMaybe<U, Policy>&& right) noexcept(std::is_nothrow_constructible<T, U&&>::value) {
        if (right.Defined()) {
            new (Data()) T(std::move(right.GetRef()));
            this->Defined_ = true;
        }
    }

    template <class U = T, class = std::enable_if_t<TImplicitAnyCtor<U>::value>>
    constexpr TMaybe(U&& right)
        : TBase(TInPlace{}, std::forward<U>(right))
    {
    }

    template <class U = T, std::enable_if_t<TExplicitAnyCtor<U>::value, bool> = false>
    constexpr explicit TMaybe(U&& right)
        : TBase(TInPlace{}, std::forward<U>(right))
    {
    }

    ~TMaybe() = default;

    constexpr TMaybe& operator=(const TMaybe&) = default;
    constexpr TMaybe& operator=(TMaybe&&) = default;

    TMaybe& operator=(TNothing) noexcept {
        Clear();
        return *this;
    }

    template <class U = T>
    std::enable_if_t<TAssignableFromAny<U>::value, TMaybe&> operator=(U&& right) {
        if (Defined()) {
            *Data() = std::forward<U>(right);
        } else {
            Init(std::forward<U>(right));
        }
        return *this;
    }

    template <class U>
    std::enable_if_t<TCopyAssignable<U>::value,
                     TMaybe&>
    operator=(const TMaybe<U, Policy>& right) {
        if (right.Defined()) {
            if (Defined()) {
                *Data() = right.GetRef();
            } else {
                Init(right.GetRef());
            }
        } else {
            Clear();
        }

        return *this;
    }

    template <class U>
    std::enable_if_t<TMoveAssignable<U>::value,
                     TMaybe&>
    operator=(TMaybe<U, Policy>&& right) noexcept(
        std::is_nothrow_assignable<T&, U&&>::value&& std::is_nothrow_constructible<T, U&&>::value)
    {
        if (right.Defined()) {
            if (Defined()) {
                *Data() = std::move(right.GetRef());
            } else {
                Init(std::move(right.GetRef()));
            }
        } else {
            Clear();
        }

        return *this;
    }

    template <typename... Args>
    T& ConstructInPlace(Args&&... args) {
        Clear();
        Init(std::forward<Args>(args)...);
        return *Data();
    }

    Y_REINITIALIZES_OBJECT void Clear() noexcept {
        if (Defined()) {
            this->Defined_ = false;
            Data()->~T();
        }
    }

    constexpr bool Defined() const noexcept {
        return this->Defined_;
    }

    Y_PURE_FUNCTION constexpr bool Empty() const noexcept {
        return !Defined();
    }

    void CheckDefined() const {
        if (Y_UNLIKELY(!Defined())) {
            Policy::OnEmpty(typeid(TValueType));
        }
    }

    const T* Get() const noexcept Y_LIFETIME_BOUND {
        return Defined() ? Data() : nullptr;
    }

    T* Get() noexcept Y_LIFETIME_BOUND {
        return Defined() ? Data() : nullptr;
    }

    constexpr const T& GetRef() const& Y_LIFETIME_BOUND {
        CheckDefined();

        return *Data();
    }

    constexpr T& GetRef() & Y_LIFETIME_BOUND {
        CheckDefined();

        return *Data();
    }

    constexpr const T&& GetRef() const&& Y_LIFETIME_BOUND {
        CheckDefined();

        return std::move(*Data());
    }

    constexpr T&& GetRef() && Y_LIFETIME_BOUND {
        CheckDefined();

        return std::move(*Data());
    }

    constexpr const T& operator*() const& Y_LIFETIME_BOUND {
        return GetRef();
    }

    constexpr T& operator*() & Y_LIFETIME_BOUND {
        return GetRef();
    }

    constexpr const T&& operator*() const&& Y_LIFETIME_BOUND {
        return std::move(GetRef());
    }

    constexpr T&& operator*() && Y_LIFETIME_BOUND {
        return std::move(GetRef());
    }

    constexpr const T* operator->() const Y_LIFETIME_BOUND {
        return &GetRef();
    }

    constexpr T* operator->() Y_LIFETIME_BOUND {
        return &GetRef();
    }

    constexpr const T& GetOrElse(const T& elseValue Y_LIFETIME_BOUND) const Y_LIFETIME_BOUND {
        return Defined() ? *Data() : elseValue;
    }

    constexpr T& GetOrElse(T& elseValue Y_LIFETIME_BOUND) Y_LIFETIME_BOUND {
        return Defined() ? *Data() : elseValue;
    }

    constexpr T&& GetOrElse(T&& elseValue Y_LIFETIME_BOUND) && Y_LIFETIME_BOUND {
        return Defined() ? std::move(*Data()) : std::move(elseValue);
    }

    constexpr const TMaybe& OrElse(const TMaybe& elseValue Y_LIFETIME_BOUND) const noexcept Y_LIFETIME_BOUND {
        return Defined() ? *this : elseValue;
    }

    constexpr TMaybe& OrElse(TMaybe& elseValue Y_LIFETIME_BOUND) Y_LIFETIME_BOUND {
        return Defined() ? *this : elseValue;
    }

    constexpr TMaybe&& OrElse(TMaybe&& elseValue Y_LIFETIME_BOUND) && Y_LIFETIME_BOUND {
        return Defined() ? std::move(*this) : std::move(elseValue);
    }

    template <typename F>
    constexpr auto AndThen(F&& func) & {
        using ReturnType = TCallResult<F, T&>;

        if (Defined()) {
            return std::forward<F>(CheckReturnsMaybe<T&>(func))(*Data());
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto AndThen(F&& func) const& {
        using ReturnType = TCallResult<F, const T&>;

        if (Defined()) {
            return std::forward<F>(CheckReturnsMaybe<const T&>(func))(*Data());
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto AndThen(F&& func) && {
        using ReturnType = TCallResult<F, T&&>;

        if (Defined()) {
            return std::forward<F>(CheckReturnsMaybe<T&&>(func))(std::move(*Data()));
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto AndThen(F&& func) const&& {
        using ReturnType = TCallResult<F, const T&&>;

        if (Defined()) {
            return std::forward<F>(CheckReturnsMaybe<const T&&>(func))(std::move(*Data()));
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto Transform(F&& func) & {
        using ReturnType = TMaybe<TCallResult<F, T&>>;

        if (Defined()) {
            return ReturnType(std::forward<F>(func)(*Data()));
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto Transform(F&& func) const& {
        using ReturnType = TMaybe<TCallResult<F, const T&>>;

        if (Defined()) {
            return ReturnType(std::forward<F>(func)(*Data()));
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto Transform(F&& func) && {
        using ReturnType = TMaybe<TCallResult<F, T&&>>;

        if (Defined()) {
            return ReturnType(std::forward<F>(func)(std::move(*Data())));
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr auto Transform(F&& func) const&& {
        using ReturnType = TMaybe<TCallResult<F, const T&&>>;

        if (Defined()) {
            return ReturnType(std::forward<F>(func)(std::move(*Data())));
        }

        return ReturnType{};
    }

    template <typename F>
    constexpr TMaybe Or(F&& func) const& {
        using ResultType = TCallResult<F>;
        static_assert(std::is_same<ResultType, TMaybe>::value, "Function must return TMaybe with the same type");

        if (Defined()) {
            return *this;
        }

        return std::forward<F>(func)();
    }

    template <typename F>
    constexpr TMaybe Or(F&& func) && {
        using ResultType = TCallResult<F>;
        static_assert(std::is_same<ResultType, TMaybe>::value, "Function must return TMaybe with the same type");

        if (Defined()) {
            return std::move(*this);
        }

        return std::forward<F>(func)();
    }

    template <typename U>
    TMaybe<U, Policy> Cast() const {
        return Defined() ? TMaybe<U, Policy>(*Data()) : TMaybe<U, Policy>();
    }

    constexpr explicit operator bool() const noexcept {
        return Defined();
    }

    void Save(IOutputStream* out) const {
        const bool defined = Defined();

        ::Save<bool>(out, defined);

        if (defined) {
            ::Save(out, *Data());
        }
    }

    void Load(IInputStream* in) {
        bool defined;

        ::Load(in, defined);

        if (defined) {
            if (!Defined()) {
                ConstructInPlace();
            }

            ::Load(in, *Data());
        } else {
            Clear();
        }
    }

    void Swap(TMaybe& other) {
        if (this->Defined_ == other.Defined_) {
            if (this->Defined_) {
                ::DoSwap(this->Data_, other.Data_);
            }
        } else {
            if (this->Defined_) {
                other.Init(std::move(this->Data_));
                this->Clear();
            } else {
                this->Init(std::move(other.Data_));
                other.Clear();
            }
        }
    }

    void swap(TMaybe& other) {
        Swap(other);
    }

private:
    constexpr const T* Data() const noexcept Y_LIFETIME_BOUND {
        return std::addressof(this->Data_);
    }

    constexpr T* Data() noexcept Y_LIFETIME_BOUND {
        return std::addressof(this->Data_);
    }

    template <typename... Args>
    void Init(Args&&... args) {
        new (Data()) T(std::forward<Args>(args)...);
        this->Defined_ = true;
    }
};

template <class T>
using TMaybeFail = TMaybe<T, NMaybe::TPolicyUndefinedFail>;

template <class T, class TPolicy = ::NMaybe::TPolicyUndefinedExcept>
constexpr TMaybe<std::decay_t<T>, TPolicy> MakeMaybe(T&& value) {
    return TMaybe<std::decay_t<T>, TPolicy>(std::forward<T>(value));
}

template <class T, class... TArgs>
constexpr TMaybe<T> MakeMaybe(TArgs&&... args) {
    return TMaybe<T>(typename TMaybe<T>::TInPlace{}, std::forward<TArgs>(args)...);
}

template <class T, class U, class... TArgs>
constexpr TMaybe<T> MakeMaybe(std::initializer_list<U> il, TArgs&&... args) {
    return TMaybe<T>(typename TMaybe<T>::TInPlace{}, il, std::forward<TArgs>(args)...);
}

template <class T, class TPolicy>
void Swap(TMaybe<T, TPolicy>& lhs, TMaybe<T, TPolicy>& rhs) {
    lhs.Swap(rhs);
}

template <class T, class TPolicy>
void swap(TMaybe<T, TPolicy>& lhs, TMaybe<T, TPolicy>& rhs) {
    lhs.Swap(rhs);
}

template <typename T, class TPolicy>
struct THash<TMaybe<T, TPolicy>> {
    constexpr size_t operator()(const TMaybe<T, TPolicy>& data) const {
        return (data.Defined()) ? THash<T>()(data.GetRef()) : 42;
    }
};

// Comparisons between TMaybe
template <class T, class TPolicy>
constexpr bool operator==(const ::TMaybe<T, TPolicy>& left, const ::TMaybe<T, TPolicy>& right) {
    return (static_cast<bool>(left) != static_cast<bool>(right))
               ? false
               : (
                     !static_cast<bool>(left)
                         ? true
                         : *left == *right);
}

template <class T, class TPolicy>
constexpr bool operator!=(const TMaybe<T, TPolicy>& left, const TMaybe<T, TPolicy>& right) {
    return !(left == right);
}

template <class T, class TPolicy>
constexpr bool operator<(const TMaybe<T, TPolicy>& left, const TMaybe<T, TPolicy>& right) {
    return (!static_cast<bool>(right))
               ? false
               : (
                     !static_cast<bool>(left)
                         ? true
                         : (*left < *right));
}

template <class T, class TPolicy>
constexpr bool operator>(const TMaybe<T, TPolicy>& left, const TMaybe<T, TPolicy>& right) {
    return right < left;
}

template <class T, class TPolicy>
constexpr bool operator<=(const TMaybe<T, TPolicy>& left, const TMaybe<T, TPolicy>& right) {
    return !(right < left);
}

template <class T, class TPolicy>
constexpr bool operator>=(const TMaybe<T, TPolicy>& left, const TMaybe<T, TPolicy>& right) {
    return !(left < right);
}

// Comparisons with TNothing
template <class T, class TPolicy>
constexpr bool operator==(const TMaybe<T, TPolicy>& left, TNothing) noexcept {
    return !static_cast<bool>(left);
}

template <class T, class TPolicy>
constexpr bool operator==(TNothing, const TMaybe<T, TPolicy>& right) noexcept {
    return !static_cast<bool>(right);
}

template <class T, class TPolicy>
constexpr bool operator!=(const TMaybe<T, TPolicy>& left, TNothing) noexcept {
    return static_cast<bool>(left);
}

template <class T, class TPolicy>
constexpr bool operator!=(TNothing, const TMaybe<T, TPolicy>& right) noexcept {
    return static_cast<bool>(right);
}

template <class T, class TPolicy>
constexpr bool operator<(const TMaybe<T, TPolicy>&, TNothing) noexcept {
    return false;
}

template <class T, class TPolicy>
constexpr bool operator<(TNothing, const TMaybe<T, TPolicy>& right) noexcept {
    return static_cast<bool>(right);
}

template <class T, class TPolicy>
constexpr bool operator<=(const TMaybe<T, TPolicy>& left, TNothing) noexcept {
    return !static_cast<bool>(left);
}

template <class T, class TPolicy>
constexpr bool operator<=(TNothing, const TMaybe<T, TPolicy>&) noexcept {
    return true;
}

template <class T, class TPolicy>
constexpr bool operator>(const TMaybe<T, TPolicy>& left, TNothing) noexcept {
    return static_cast<bool>(left);
}

template <class T, class TPolicy>
constexpr bool operator>(TNothing, const TMaybe<T, TPolicy>&) noexcept {
    return false;
}

template <class T, class TPolicy>
constexpr bool operator>=(const TMaybe<T, TPolicy>&, TNothing) noexcept {
    return true;
}

template <class T, class TPolicy>
constexpr bool operator>=(TNothing, const TMaybe<T, TPolicy>& right) noexcept {
    return !static_cast<bool>(right);
}

// Comparisons with T

template <class T, class TPolicy>
constexpr bool operator==(const TMaybe<T, TPolicy>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T, class TPolicy>
constexpr bool operator==(const T& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T, class TPolicy>
constexpr bool operator!=(const TMaybe<T, TPolicy>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T, class TPolicy>
constexpr bool operator!=(const T& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T, class TPolicy>
constexpr bool operator<(const TMaybe<T, TPolicy>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? std::less<T>{}(*maybe, value) : true;
}

template <class T, class TPolicy>
constexpr bool operator<(const T& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? std::less<T>{}(value, *maybe) : false;
}

template <class T, class TPolicy>
constexpr bool operator<=(const TMaybe<T, TPolicy>& maybe, const T& value) {
    return !(maybe > value);
}

template <class T, class TPolicy>
constexpr bool operator<=(const T& value, const TMaybe<T, TPolicy>& maybe) {
    return !(value > maybe);
}

template <class T, class TPolicy>
constexpr bool operator>(const TMaybe<T, TPolicy>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? value < maybe : false;
}

template <class T, class TPolicy>
constexpr bool operator>(const T& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? maybe < value : true;
}

template <class T, class TPolicy>
constexpr bool operator>=(const TMaybe<T, TPolicy>& maybe, const T& value) {
    return !(maybe < value);
}

template <class T, class TPolicy>
constexpr bool operator>=(const T& value, const TMaybe<T, TPolicy>& maybe) {
    return !(value < maybe);
}

// Comparison with values convertible to T

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator==(const ::TMaybe<T, TPolicy>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator==(const U& value, const ::TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator!=(const TMaybe<T, TPolicy>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator!=(const U& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<(const TMaybe<T, TPolicy>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? std::less<T>{}(*maybe, value) : true;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<(const U& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? std::less<T>{}(value, *maybe) : false;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<=(const TMaybe<T, TPolicy>& maybe, const U& value) {
    return !(maybe > value);
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<=(const U& value, const TMaybe<T, TPolicy>& maybe) {
    return !(value > maybe);
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>(const TMaybe<T, TPolicy>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? value < maybe : false;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>(const U& value, const TMaybe<T, TPolicy>& maybe) {
    return static_cast<bool>(maybe) ? maybe < value : true;
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>=(const TMaybe<T, TPolicy>& maybe, const U& value) {
    return !(maybe < value);
}

template <class T, class TPolicy, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>=(const U& value, const TMaybe<T, TPolicy>& maybe) {
    return !(value < maybe);
}

class IOutputStream;

template <class T, class TPolicy>
inline IOutputStream& operator<<(IOutputStream& out, const TMaybe<T, TPolicy>& maybe) {
    if (maybe.Defined()) {
        out << *maybe;
    } else {
        out << TStringBuf("(empty maybe)");
    }
    return out;
}
