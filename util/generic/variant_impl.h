#pragma once

#ifndef VARIANT_IMPL_H_
#error "Direct inclusion of this file is not allowed, include variant.h"
#endif

#include <util/generic/hash.h>

namespace NVariant {
    template <class T>
    class TVisitorEquals {
    public:
        TVisitorEquals(const T& left)
            : Left_(left)
        {
        }

        template <class T2>
        bool operator()(const T2& right) const {
            return Left_ == right;
        }

    private:
        const T& Left_;
    };

    struct TVisitorHash {
        template <class T>
        size_t operator()(const T& value) const {
            return THash<T>()(value);
        };
    };

    struct TVisitorDestroy {
        template <class T>
        void operator()(T& value) const {
            Y_UNUSED(value);
            value.~T();
        };
    };

    template <class T>
    class TVisitorCopyConstruct {
    public:
        TVisitorCopyConstruct(T* var)
            : Var_(var)
        {
        }

        template <class T2>
        void operator()(const T2& value) {
            new (Var_) T(value);
        };

    private:
        T* const Var_;
    };

    template <class T>
    class TVisitorMoveConstruct {
    public:
        TVisitorMoveConstruct(T* var)
            : Var_(var)
        {
        }

        template <class T2>
        void operator()(T2& value) {
            new (Var_) T(std::move(value));
        };

    private:
        T* const Var_;
    };

} // namespace NVariant

template <class... Ts>
struct THash<::TVariant<Ts...>> {
public:
    inline size_t operator()(const ::TVariant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.Tag());
        const size_t valueHash = v.Visit(NVariant::TVisitorHash());
        return CombineHashes(tagHash, valueHash);
    }
};

template <class... Ts>
template <class T>
void ::TVariant<Ts...>::AssignValue(const T& value) {
    static_assert(
        ::NVariant::TTagTraits<T, Ts...>::Tag != -1,
        "Type not in TVariant.");

    Tag_ = ::NVariant::TTagTraits<T, Ts...>::Tag;
    new (&Storage_) T(value);
}

template <class... Ts>
template <class T>
void ::TVariant<Ts...>::AssignValue(T&& value) {
    static_assert(
        ::NVariant::TTagTraits<T, Ts...>::Tag != -1,
        "Type not in TVariant.");

    Tag_ = ::NVariant::TTagTraits<T, Ts...>::Tag;
    new (&Storage_) T(std::move(value));
}

template <class... Ts>
template <class T, class... TArgs>
void ::TVariant<Ts...>::EmplaceValue(TArgs&&... args) {
    static_assert(
        ::NVariant::TTagTraits<T, Ts...>::Tag != -1,
        "Type not in TVariant.");

    Tag_ = ::NVariant::TTagTraits<T, Ts...>::Tag;
    new (&Storage_) T(std::forward<TArgs>(args)...);
}

template <class... Ts>
void ::TVariant<Ts...>::AssignVariant(const TVariant& other) {
    other.Visit(::NVariant::TVisitorCopyConstruct<TVariant>(this));
}

template <class... Ts>
void ::TVariant<Ts...>::AssignVariant(TVariant&& other) {
    other.Visit(::NVariant::TVisitorMoveConstruct<TVariant>(this));
}

template <class... Ts>
template <class T>
T& ::TVariant<Ts...>::UncheckedAs() {
    return *reinterpret_cast<T*>(&Storage_);
}

template <class... Ts>
template <class T>
const T& ::TVariant<Ts...>::UncheckedAs() const {
    return *reinterpret_cast<const T*>(&Storage_);
}

template <class... Ts>
template <class T>
::TVariant<Ts...>::TVariant(const T& value) {
    AssignValue(value);
}

template <class... Ts>
template <class T, class>
::TVariant<Ts...>::TVariant(T&& value) {
    AssignValue(std::move(value));
}

template <class... Ts>
template <class T, class... TArgs, class>
::TVariant<Ts...>::TVariant(TVariantTypeTag<T>, TArgs&&... args) {
    EmplaceValue<T>(std::forward<TArgs>(args)...);
}

template <class... Ts>
::TVariant<Ts...>::TVariant(const TVariant& other) {
    AssignVariant(other);
}

template <class... Ts>
::TVariant<Ts...>::TVariant(TVariant&& other) {
    AssignVariant(std::move(other));
}

template <class... Ts>
template <class T>
::TVariant<Ts...>& ::TVariant<Ts...>::operator=(const T& value) {
    if (&value != &UncheckedAs<T>()) {
        Destroy();
        AssignValue(value);
    }
    return *this;
}

template <class... Ts>
template <class T, class>
::TVariant<Ts...>& ::TVariant<Ts...>::operator=(T&& value) {
    if (&value != &UncheckedAs<T>()) {
        Destroy();
        AssignValue(std::move(value));
    }
    return *this;
}

template <class... Ts>
::TVariant<Ts...>& ::TVariant<Ts...>::operator=(const TVariant& other) {
    if (&other != this) {
        Destroy();
        AssignVariant(other);
    }
    return *this;
}

template <class... Ts>
::TVariant<Ts...>& ::TVariant<Ts...>::operator=(TVariant&& other) {
    if (&other != this) {
        Destroy();
        AssignVariant(std::move(other));
    }
    return *this;
}

template <class... Ts>
template <class T>
bool ::TVariant<Ts...>::operator==(const T& other) const {
    static_assert(
        ::NVariant::TTagTraits<T, Ts...>::Tag != -1,
        "Type not in TVariant.");

    if (Tag_ != ::NVariant::TTagTraits<T, Ts...>::Tag) {
        return false;
    }
    return UncheckedAs<T>() == other;
}

template <class... Ts>
bool ::TVariant<Ts...>::operator==(const TVariant& other) const {
    if (Tag_ != other.Tag_) {
        return false;
    }
    return other.Visit(NVariant::TVisitorEquals<TVariant>(*this));
}

template <class... Ts>
template <class T>
T& ::TVariant<Ts...>::As() {
    typedef ::NVariant::TTagTraits<T, Ts...> TTagTraits;
    static_assert(TTagTraits::Tag != -1, "Type not in TVariant.");
    Y_ASSERT(Tag_ == TTagTraits::Tag);
    return UncheckedAs<T>();
}

template <class... Ts>
template <class T>
const T& ::TVariant<Ts...>::As() const {
    typedef ::NVariant::TTagTraits<T, Ts...> TTagTraits;
    static_assert(TTagTraits::Tag != -1, "Type not in TVariant.");
    Y_ASSERT(Tag_ == TTagTraits::Tag);
    return UncheckedAs<T>();
}

template <class... Ts>
template <class T>
T* ::TVariant<Ts...>::TryAs() {
    typedef ::NVariant::TTagTraits<T, Ts...> TTagTraits;
    static_assert(TTagTraits::Tag != -1, "Type not in TVariant.");
    return Tag_ == TTagTraits::Tag ? &UncheckedAs<T>() : nullptr;
}

template <class... Ts>
template <class T>
const T* ::TVariant<Ts...>::TryAs() const {
    typedef ::NVariant::TTagTraits<T, Ts...> TTagTraits;
    static_assert(TTagTraits::Tag != -1, "Type not in TVariant.");
    return Tag_ == TTagTraits::Tag ? &UncheckedAs<T>() : nullptr;
}

template <class... Ts>
template <class T>
bool ::TVariant<Ts...>::Is() const {
    typedef ::NVariant::TTagTraits<T, Ts...> TTagTraits;
    static_assert(TTagTraits::Tag != -1, "Type not in TVariant.");
    return Tag_ == TTagTraits::Tag;
}

template <class... Ts>
::TVariant<Ts...>::~TVariant() {
    Destroy();
}

template <class... Ts>
void ::TVariant<Ts...>::Destroy() {
    Visit(::NVariant::TVisitorDestroy());
}

template <class... Ts>
int ::TVariant<Ts...>::Tag() const {
    return Tag_;
}

template <class... Ts>
template <class Visitor>
typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType TVariant<Ts...>::Visit(Visitor&& visitor) {
    return ::NVariant::TVisitTraits<Ts...>::template Visit<typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType>(Tag_, &Storage_, std::forward<Visitor>(visitor));
}

template <class... Ts>
template <class Visitor>
typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType TVariant<Ts...>::Visit(Visitor&& visitor) const {
    return ::NVariant::TVisitTraits<Ts...>::template Visit<typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType>(Tag_, &Storage_, std::forward<Visitor>(visitor));
}
