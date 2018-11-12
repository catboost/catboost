#pragma once

#include <utility>

#include "maybe_traits.h"
#include "yexception.h"

#include <util/system/align.h>
#include <util/stream/output.h>
#include <util/ysaveload.h>

namespace NMaybe {
    struct TPolicyUndefinedExcept {
        static void OnEmpty() {
            ythrow yexception() << AsStringBuf("TMaybe is empty");
        }
    };

    struct TPolicyUndefinedFail {
        static void OnEmpty() {
            Y_FAIL("TMaybe is empty");
        }
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
class TMaybe : private TMaybeBase<T> {
public:
    using TInPlace = NMaybe::TInPlace;

private:
    static_assert(!std::is_same<std::remove_cv_t<T>, TNothing>::value,
                  "Instantiation of TMaybe with a TNothing type is ill-formed");
    static_assert(!std::is_same<std::remove_cv_t<T>, TInPlace>::value,
                  "Instantiation of TMaybe with a TInPlace type is ill-formed");
    static_assert(!std::is_reference<T>::value,
                  "Instantiation of TMaybe with reference type is ill-formed");
    static_assert(std::is_destructible<T>::value,
                  "Instantiation of TMaybe with non-destructible type is ill-formed");

    template <class U>
    using TConstructibleFromMaybeSomehow = TDisjunction<std::is_constructible<T, TMaybe<U, Policy>&>,
                                                        std::is_constructible<T, const TMaybe<U, Policy>&>,
                                                        std::is_constructible<T, TMaybe<U, Policy>&&>,
                                                        std::is_constructible<T, const TMaybe<U, Policy>&&>,
                                                        std::is_convertible<TMaybe<U, Policy>&, T>,
                                                        std::is_convertible<const TMaybe<U, Policy>&, T>,
                                                        std::is_convertible<TMaybe<U, Policy>&&, T>,
                                                        std::is_convertible<const TMaybe<U, Policy>&&, T>>;

    template <class U>
    using TAssignableFromMaybeSomehow = TDisjunction<TConstructibleFromMaybeSomehow<U>,
                                                     std::is_assignable<T&, TMaybe<U, Policy>&>,
                                                     std::is_assignable<T&, const TMaybe<U, Policy>&>,
                                                     std::is_assignable<T&, TMaybe<U, Policy>&&>,
                                                     std::is_assignable<T&, const TMaybe<U, Policy>&&>>;

    using TBase = TMaybeBase<T>;

public:
    using value_type = T;
    using TValueType = value_type;

    TMaybe() noexcept = default;

    constexpr TMaybe(const TMaybe&) = default;
    constexpr TMaybe(TMaybe&&) = default;

    template <class... Args>
    constexpr explicit TMaybe(TInPlace, Args&&... args)
        : TBase(TInPlace{}, std::forward<Args>(args)...) {}

    template <class U, class... TArgs>
    constexpr explicit TMaybe(TInPlace, std::initializer_list<U> il, TArgs&&... args)
        : TBase(TInPlace{}, il, std::forward<TArgs>(args)...) {}

    constexpr TMaybe(TNothing) noexcept {}

    constexpr TMaybe(const T& right)
        : TBase(TInPlace{}, right) {}

    constexpr TMaybe(T&& right)
        : TBase(TInPlace{}, std::move(right)) {}

    template <class U, class = std::enable_if_t<
        std::is_constructible<T, const U&>::value
        && std::is_convertible<const U&, T>::value
        && !TConstructibleFromMaybeSomehow<U>::value>>
    TMaybe(const TMaybe<U, Policy>& right) {
        if (right.Defined()) {
            new (Data()) T(right.GetRef());
            this->Defined_ = true;
        }
    }

    template <class U,
              std::enable_if_t<
                  std::is_constructible<T, const U&>::value &&
                      !std::is_convertible<const U&, T>::value &&
                      !TConstructibleFromMaybeSomehow<U>::value,
                  bool> = false>
    explicit TMaybe(const TMaybe<U, Policy>& right) {
        if (right.Defined()) {
            new (Data()) T(right.GetRef());
            this->Defined_ = true;
        }
    }

    template <class U, class = std::enable_if_t<
        std::is_constructible<T, const U&>::value
        && std::is_convertible<const U&, T>::value
        && !TConstructibleFromMaybeSomehow<U>::value>>
    TMaybe(TMaybe<U, Policy>&& right) noexcept(std::is_nothrow_constructible<T, U&&>::value) {
        if (right.Defined()) {
            new (Data()) T(std::move(right.GetRef()));
            this->Defined_ = true;
        }
    }

    template <class U,
              std::enable_if_t<
                  std::is_constructible<T, U&&>::value &&
                      !std::is_convertible<U&&, T>::value &&
                      !TConstructibleFromMaybeSomehow<U>::value,
                  bool> = false>
    explicit TMaybe(TMaybe<U, Policy>&& right) noexcept(
        std::is_nothrow_constructible<T, U&&>::value)
    {
        if (right.Defined()) {
            new (Data()) T(std::move(right.GetRef()));
            this->Defined_ = true;
        }
    }

    ~TMaybe() = default;

    TMaybe& operator=(const TMaybe&) = default;
    TMaybe& operator=(TMaybe&&) = default;

    TMaybe& operator=(TNothing) noexcept {
        Clear();
        return *this;
    }

    TMaybe& operator=(const T& right) {
        if (Defined()) {
            *Data() = right;
        } else {
            Init(right);
        }

        return *this;
    }

    TMaybe& operator=(T&& right) {
        if (Defined()) {
            *Data() = std::move(right);
        } else {
            Init(std::move(right));
        }

        return *this;
    }

    template <class U, class = std::enable_if_t<
        std::is_constructible<T, const U&>::value
        && std::is_assignable<T&, const U&>::value
        && !TAssignableFromMaybeSomehow<U>::value>>
    TMaybe& operator=(const TMaybe<U, Policy>& right) {
        if (right.Defined()) {
            if (Defined()) {
                GetRef() = right.GetRef();
            } else {
                Init(right.GetRef());
            }
        } else {
            Clear();
        }

        return *this;
    }

    template <class U, class = std::enable_if_t<
        std::is_constructible<T, U&&>::value
        && std::is_assignable<T&, U&&>::value
        && !TAssignableFromMaybeSomehow<U>::value>>
    TMaybe& operator=(TMaybe<U, Policy>&& right) noexcept(
        std::is_nothrow_assignable<T&, U&&>::value
        && std::is_nothrow_constructible<T, U&&>::value)
    {
        if (right.Defined()) {
            if (Defined()) {
                GetRef() = std::move(right.GetRef());
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

    void Clear() noexcept {
        if (Defined()) {
            this->Defined_ = false;
            Data()->~T();
        }
    }

    constexpr bool Defined() const noexcept {
        return this->Defined_;
    }

    Y_PURE_FUNCTION
    constexpr bool Empty() const noexcept {
        return !Defined();
    }

    void CheckDefined() const {
        if (!Defined()) {
            Policy::OnEmpty();
        }
    }

    const T* Get() const noexcept {
        return Defined() ? Data() : nullptr;
    }

    T* Get() noexcept {
        return Defined() ? Data() : nullptr;
    }

    constexpr const T& GetRef() const {
        CheckDefined();

        return *Data();
    }

    constexpr T& GetRef() {
        CheckDefined();

        return *Data();
    }

    constexpr const T& operator*() const {
        return GetRef();
    }

    constexpr T& operator*() {
        return GetRef();
    }

    constexpr const T* operator->() const {
        return &GetRef();
    }

    constexpr T* operator->() {
        return &GetRef();
    }

    constexpr const T& GetOrElse(const T& elseValue) const {
        return Defined() ? *Data() : elseValue;
    }

    constexpr T& GetOrElse(T& elseValue) {
        return Defined() ? *Data() : elseValue;
    }

    constexpr const TMaybe& OrElse(const TMaybe& elseValue) const noexcept {
        return Defined() ? *this : elseValue;
    }

    constexpr TMaybe& OrElse(TMaybe& elseValue) {
        return Defined() ? *this : elseValue;
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
            ::Save(out, GetRef());
        }
    }

    void Load(IInputStream* in) {
        bool defined;

        ::Load(in, defined);

        if (defined) {
            if (!Defined()) {
                ConstructInPlace();
            }

            ::Load(in, GetRef());
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
    constexpr const T* Data() const noexcept {
        return std::addressof(this->Data_);
    }

    constexpr T* Data() noexcept {
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
        out << AsStringBuf("(empty maybe)");
    }
    return out;
}
