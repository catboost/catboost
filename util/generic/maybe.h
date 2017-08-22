#pragma once

#include <utility>
#include "yexception.h"

#include <util/system/align.h>
#include <util/stream/output.h>
#include <util/ysaveload.h>

namespace NMaybe {
    struct TPolicyUndefinedExcept {
        static void OnEmpty() {
            ythrow yexception() << STRINGBUF("TMaybe is empty");
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
class TMaybe {
public:
    struct TInPlace {
    };

    static_assert(!std::is_same<std::remove_cv_t<T>, TNothing>::value, "Instantiation of TMaybe with a TNothing type is ill-formed");
    static_assert(!std::is_same<std::remove_cv_t<T>, TInPlace>::value, "Instantiation of TMaybe with a TInPlace type is ill-formed");

    using TValueType = T;

    constexpr TMaybe() noexcept {
    }

    template <class... TArgs>
    constexpr explicit TMaybe(TInPlace, TArgs&&... args)
        : Defined_(true)
        , Data_(std::forward<TArgs>(args)...)
    {
    }

    template <class U, class... TArgs>
    constexpr explicit TMaybe(TInPlace, std::initializer_list<U> il, TArgs&&... args)
        : Defined_(true)
        , Data_(il, std::forward<TArgs>(args)...)
    {
    }

    constexpr TMaybe(TNothing) noexcept {
    }

    constexpr TMaybe(const T& right)
        : Defined_(true)
        , Data_(right)
    {
    }

    constexpr TMaybe(T&& right)
        : Defined_(true)
        , Data_(std::move(right))
    {
    }

    inline TMaybe(const TMaybe& right)
        : Defined_(right.Defined_)
    {
        if (Defined_) {
            new (Data()) T(right.Data_);
        }
    }

    inline TMaybe(TMaybe&& right) noexcept(std::is_nothrow_move_constructible<T>::value)
        : Defined_(right.Defined_)
    {
        if (Defined_) {
            new (Data()) T(std::move(right.Data_));
        }
    }

    inline ~TMaybe() {
        Clear();
    }

    template <typename... Args>
    inline T& ConstructInPlace(Args&&... args) {
        Clear();
        Init(std::forward<Args>(args)...);
        return *Data();
    }

    inline TMaybe& operator=(TNothing) noexcept {
        Clear();
        return *this;
    }

    inline TMaybe& operator=(const T& right) {
        if (Defined()) {
            *Data() = right;
        } else {
            Init(right);
        }

        return *this;
    }

    inline TMaybe& operator=(T&& right) {
        if (Defined()) {
            *Data() = std::move(right);
        } else {
            Init(std::move(right));
        }

        return *this;
    }

    inline TMaybe& operator=(const TMaybe& right) {
        if (right.Defined()) {
            operator=(*right.Data());
        } else {
            Clear();
        }

        return *this;
    }

    inline TMaybe& operator=(TMaybe&& right) noexcept(std::is_nothrow_move_assignable<T>::value&&
                                                          std::is_nothrow_move_constructible<T>::value) {
        if (right.Defined()) {
            operator=(std::move(*right.Data()));
        } else {
            Clear();
        }

        return *this;
    }

    inline void Clear() {
        if (Defined()) {
            Defined_ = false;
            Data()->~T();
        }
    }

    inline bool Defined() const noexcept {
        return Defined_;
    }

    inline bool Empty() const noexcept {
        return !Defined();
    }

    inline void CheckDefined() const {
        if (!Defined()) {
            Policy::OnEmpty();
        }
    }

    inline const T* Get() const noexcept {
        return Defined() ? Data() : nullptr;
    }

    inline T* Get() noexcept {
        return Defined() ? Data() : nullptr;
    }

    inline const T& GetRef() const {
        CheckDefined();

        return *Data();
    }

    inline T& GetRef() {
        CheckDefined();

        return *Data();
    }

    inline const T& operator*() const {
        return GetRef();
    }

    inline T& operator*() {
        return GetRef();
    }

    inline const T* operator->() const {
        return &GetRef();
    }

    inline T* operator->() {
        return &GetRef();
    }

    inline const T& GetOrElse(const T& elseValue) const {
        return Defined() ? *Data() : elseValue;
    }

    inline T& GetOrElse(T& elseValue) {
        return Defined() ? *Data() : elseValue;
    }

    inline const TMaybe& OrElse(const TMaybe& elseValue) const noexcept {
        return Defined() ? *this : elseValue;
    }

    inline TMaybe& OrElse(TMaybe& elseValue) {
        return Defined() ? *this : elseValue;
    }

    template <typename U>
    TMaybe<U> Cast() const {
        return Defined() ? TMaybe<U>(*Data()) : TMaybe<U>();
    }

    constexpr explicit operator bool() const noexcept {
        return Defined_;
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

    inline void Swap(TMaybe& other) {
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

    inline void swap(TMaybe& other) {
        Swap(other);
    }

private:
    bool Defined_{false};
    union {
        char NullState_{'\0'};
        T Data_;
    };

    inline const T* Data() const noexcept {
        return std::addressof(Data_);
    }

    inline T* Data() noexcept {
        return std::addressof(Data_);
    }

    template <typename... Args>
    inline void Init(Args&&... args) {
        new (Data()) T(std::forward<Args>(args)...);
        Defined_ = true;
    }
};

template <class T>
using TMaybeFail = TMaybe<T, NMaybe::TPolicyUndefinedFail>;

template <class T>
constexpr TMaybe<std::decay_t<T>> MakeMaybe(T&& value) {
    return TMaybe<std::decay_t<T>>(std::forward<T>(value));
}

template <class T, class... TArgs>
constexpr TMaybe<T> MakeMaybe(TArgs&&... args) {
    return TMaybe<T>(typename TMaybe<T>::TInPlace{}, std::forward<TArgs>(args)...);
}

template <class T, class U, class... TArgs>
constexpr TMaybe<T> MakeMaybe(std::initializer_list<U> il, TArgs&&... args) {
    return TMaybe<T>(typename TMaybe<T>::TInPlace{}, il, std::forward<TArgs>(args)...);
}

template <class T>
void Swap(TMaybe<T>& lhs, TMaybe<T>& rhs) {
    lhs.Swap(rhs);
}

template <class T>
void swap(TMaybe<T>& lhs, TMaybe<T>& rhs) {
    lhs.Swap(rhs);
}

// Comparisons between TMaybe
template <class T>
constexpr bool operator==(const ::TMaybe<T>& left, const ::TMaybe<T>& right) {
    return (static_cast<bool>(left) != static_cast<bool>(right))
               ? false
               : (
                     !static_cast<bool>(left)
                         ? true
                         : *left == *right);
}

template <class T>
constexpr bool operator!=(const TMaybe<T>& left, const TMaybe<T>& right) {
    return !(left == right);
}

template <class T>
constexpr bool operator<(const TMaybe<T>& left, const TMaybe<T>& right) {
    return (!static_cast<bool>(right))
               ? false
               : (
                     !static_cast<bool>(left)
                         ? true
                         : (*left < *right));
}

template <class T>
constexpr bool operator>(const TMaybe<T>& left, const TMaybe<T>& right) {
    return right < left;
}

template <class T>
constexpr bool operator<=(const TMaybe<T>& left, const TMaybe<T>& right) {
    return !(right < left);
}

template <class T>
constexpr bool operator>=(const TMaybe<T>& left, const TMaybe<T>& right) {
    return !(left < right);
}

// Comparisons with TNothing
template <class T>
constexpr bool operator==(const TMaybe<T>& left, TNothing) noexcept {
    return !static_cast<bool>(left);
}

template <class T>
constexpr bool operator==(TNothing, const TMaybe<T>& right) noexcept {
    return !static_cast<bool>(right);
}

template <class T>
constexpr bool operator!=(const TMaybe<T>& left, TNothing) noexcept {
    return static_cast<bool>(left);
}

template <class T>
constexpr bool operator!=(TNothing, const TMaybe<T>& right) noexcept {
    return static_cast<bool>(right);
}

template <class T>
constexpr bool operator<(const TMaybe<T>&, TNothing) noexcept {
    return false;
}

template <class T>
constexpr bool operator<(TNothing, const TMaybe<T>& right) noexcept {
    return static_cast<bool>(right);
}

template <class T>
constexpr bool operator<=(const TMaybe<T>& left, TNothing) noexcept {
    return !static_cast<bool>(left);
}

template <class T>
constexpr bool operator<=(TNothing, const TMaybe<T>&) noexcept {
    return true;
}

template <class T>
constexpr bool operator>(const TMaybe<T>& left, TNothing) noexcept {
    return static_cast<bool>(left);
}

template <class T>
constexpr bool operator>(TNothing, const TMaybe<T>&) noexcept {
    return false;
}

template <class T>
constexpr bool operator>=(const TMaybe<T>&, TNothing) noexcept {
    return true;
}

template <class T>
constexpr bool operator>=(TNothing, const TMaybe<T>& right) noexcept {
    return !static_cast<bool>(right);
}

// Comparisons with T

template <class T>
constexpr bool operator==(const TMaybe<T>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T>
constexpr bool operator==(const T& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T>
constexpr bool operator!=(const TMaybe<T>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T>
constexpr bool operator!=(const T& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T>
constexpr bool operator<(const TMaybe<T>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? std::less<T>{}(*maybe, value) : true;
}

template <class T>
constexpr bool operator<(const T& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? std::less<T>{}(value, *maybe) : false;
}

template <class T>
constexpr bool operator<=(const TMaybe<T>& maybe, const T& value) {
    return !(maybe > value);
}

template <class T>
constexpr bool operator<=(const T& value, const TMaybe<T>& maybe) {
    return !(value > maybe);
}

template <class T>
constexpr bool operator>(const TMaybe<T>& maybe, const T& value) {
    return static_cast<bool>(maybe) ? value < maybe : false;
}

template <class T>
constexpr bool operator>(const T& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? maybe < value : true;
}

template <class T>
constexpr bool operator>=(const TMaybe<T>& maybe, const T& value) {
    return !(maybe < value);
}

template <class T>
constexpr bool operator>=(const T& value, const TMaybe<T>& maybe) {
    return !(value < maybe);
}

// Comparison with values convertible to T

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator==(const ::TMaybe<T>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator==(const U& value, const ::TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? *maybe == value : false;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator!=(const TMaybe<T>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator!=(const U& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? !(*maybe == value) : true;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<(const TMaybe<T>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? std::less<T>{}(*maybe, value) : true;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<(const U& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? std::less<T>{}(value, *maybe) : false;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<=(const TMaybe<T>& maybe, const U& value) {
    return !(maybe > value);
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator<=(const U& value, const TMaybe<T>& maybe) {
    return !(value > maybe);
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>(const TMaybe<T>& maybe, const U& value) {
    return static_cast<bool>(maybe) ? value < maybe : false;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>(const U& value, const TMaybe<T>& maybe) {
    return static_cast<bool>(maybe) ? maybe < value : true;
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>=(const TMaybe<T>& maybe, const U& value) {
    return !(maybe < value);
}

template <class T, class U, std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
constexpr bool operator>=(const U& value, const TMaybe<T>& maybe) {
    return !(value < maybe);
}

class IOutputStream;

template <class T, class TPolicy>
static inline IOutputStream& operator<<(IOutputStream& out, const TMaybe<T, TPolicy>& maybe) {
    if (maybe.Defined()) {
        out << *maybe;
    } else {
        out << STRINGBUF("(empty maybe)");
    }
    return out;
}
