#pragma once

#include "variant_traits.h"
#include "variant_visitors.h"

#include <util/digest/numeric.h>
#include <util/generic/typetraits.h>
#include <util/system/yassert.h>

#include <type_traits>


template <class T>
struct TVariantTypeTag {};
class TWrongVariantError : public yexception {};
class TEmptyVariantError : public yexception {};

//! |boost::variant|-like discriminated union with C++ 11 features.
template <class... Ts>
class TVariant {
    template <class T>
    using TTag = ::NVariant::TTagTraits<std::decay_t<T>, Ts...>;

    template <class Visitor>
    using TVisRes = typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType;
    using TVisTraits = ::NVariant::TVisitTraits<Ts...>;

    static_assert(::NVariant::TTypeTraits<Ts...>::NoRefs,
                  "TVariant type arguments cannot be references.");

    static_assert(::NVariant::TTypeTraits<Ts...>::NoDuplicates,
                  "TVariant type arguments cannot contain duplicate types.");

public:
    TVariant() {}

    TVariant(const TVariant& other) {
        CopyVariant(other);
    }

    TVariant(TVariant&& other) {
        MoveVariant(std::move(other));
    }

    template <class T, class = std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value>>
    TVariant(T&& value) {
        AssignValue(std::forward<T>(value));
    }

    //! Constructs an instance in-place.
    template <class T, class... TArgs>
    TVariant(TVariantTypeTag<T>, TArgs&&... args) {
        EmplaceValue<T>(std::forward<TArgs>(args)...);
    }

    ~TVariant() {
        Destroy();
    }

    TVariant& operator=(const TVariant& other) {
        if (&other != this) {
            Destroy();
            CopyVariant(other);
        }
        return *this;
    }

    TVariant& operator=(TVariant&& other) {
        if (&other != this) {
            Destroy();
            MoveVariant(std::move(other));
        }
        return *this;
    }

    template <class T, class = std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value>>
    TVariant& operator=(T&& value) {
        if (&value != ReinterpretAs<T>()) {
            Destroy();
            AssignValue(std::forward<T>(value));
        }
        return *this;
    }

    explicit Y_WARN_UNUSED_RESULT operator bool() const {
        return !Empty();
    }

    bool Y_WARN_UNUSED_RESULT Empty() const {
        return Tag_ == ::NVariant::T_EMPTY;
    }

    void Destroy() {
        if (!Empty()) {
            UncheckedVisit(::NVariant::TVisitorDestroy());
        }
        Tag_ = ::NVariant::T_EMPTY;
    }

    template <class T, class... TArgs>
    void Emplace(TArgs&&... args) {
        Destroy();
        EmplaceValue<T>(std::forward<TArgs>(args)...);
    };

    bool operator==(const TVariant& other) const {
        return Tag_ == other.Tag_ && (!other || other.UncheckedVisit(NVariant::TVisitorEquals<TVariant>(*this)));
    }

    template <class T, class = std::enable_if_t<!std::is_same<std::decay_t<T>, TVariant>::value>>
    bool operator==(const T& value) const {
        return Is<T>() && *ReinterpretAs<T>() == value;
    }

    //! Casts the instance to a given type.
    //! Tag validation is only performed in debug builds.
    template <class T>
    T& As() {
        // TODO(velavokr): enabling this check breaks lots of tests.
        // Y_ENSURE_EX(Is<T>(), TWrongVariantError());
        Y_ASSERT(Is<T>());
        return *ReinterpretAs<T>();
    }

    //! Similar to its non-const version.
    template <class T>
    const T& As() const {
        // TODO(velavokr): enabling this check breaks lots of tests.
        // Y_ENSURE_EX(Is<T>(), TWrongVariantError());
        Y_ASSERT(Is<T>());
        return *ReinterpretAs<T>();
    }

    //! Casts the instance to the given type.
    //! Converts the instance if it does not have the right type.
    template <class T, class... TArgs>
    T& AsOrMake(TArgs&& ... args) {
        if (!Is<T>()) {
            Emplace<T>(std::forward<TArgs>(args)...);
        }
        return As<T>();
    }

    //! Casts the instance to the given type.
    //! Returns a constructed default if the instance does not have the right type.
    template <class T, class... TArgs>
    const T AsOrMake(TArgs&& ... args) const {
        if (!Is<T>()) {
            return T(std::forward<TArgs>(args)...);
        }
        return As<T>();
    }

    //! Checks if the instance holds a given of a given type.
    //! Returns the pointer to the value on success or |nullptr| on failure.
    template <class T>
    T* TryAs() {
        static_assert(TTag<T>::Tag != NVariant::T_INVALID, "Type not in TVariant.");
        return Is<T>() ? ReinterpretAs<T>() : nullptr;
    }

    //! Similar to its non-const version.
    template <class T>
    const T* TryAs() const {
        static_assert(TTag<T>::Tag != ::NVariant::T_INVALID, "Type not in TVariant.");
        return Is<T>() ? ReinterpretAs<T>() : nullptr;
    }

    //! Returns |true| iff the instance holds a value of a given type.
    template <class T>
    bool Y_WARN_UNUSED_RESULT Is() const {
        static_assert(TTag<T>::Tag != ::NVariant::T_INVALID, "Type not in TVariant.");
        return Tag_ == TTag<T>::Tag;
    }

    //! Pass mutable internal value to visitor
    template <class Visitor>
    TVisRes<Visitor> Visit(Visitor&& visitor) {
        Y_ENSURE_EX(!Empty(), TEmptyVariantError());
        return UncheckedVisit(std::forward<Visitor>(visitor));
    }

    //! Pass const internal value to visitor
    template <class Visitor>
    TVisRes<Visitor> Visit(Visitor&& visitor) const {
        Y_ENSURE_EX(!Empty(), TEmptyVariantError());
        return UncheckedVisit(std::forward<Visitor>(visitor));
    }

    //! Returns the discriminating tag of the instance.
    int Tag() const {
        return Tag_;
    }

    //! Returns the discriminating tag the given type.
    template <class T>
    static constexpr int TagOf() {
        // NB: Must keep this inline due to constexpr.
        return ::NVariant::TTagTraits<T, Ts...>::Tag;
    }

    template <class T>
    static constexpr auto TypeOf() {
        return TVariantTypeTag<T>();
    }

public:
    //! Pass mutable internal value to visitor
    //! Tag validation is only performed in debug builds.
    template <class Visitor>
    TVisRes<Visitor> UncheckedVisit(Visitor&& visitor) {
        Y_ASSERT(!Empty());
        return TVisTraits::template Visit<TVisRes<Visitor>>(Tag_, &Storage_, std::forward<Visitor>(visitor));
    }

    //! Pass const internal value to visitor
    //! Tag validation is only performed in debug builds.
    template <class Visitor>
    TVisRes<Visitor> UncheckedVisit(Visitor&& visitor) const {
        Y_ASSERT(!Empty());
        return TVisTraits::template Visit<TVisRes<Visitor>>(Tag_, &Storage_, std::forward<Visitor>(visitor));
    }

private:
    template <class T>
    void AssignValue(T&& value) {
        using namespace NVariant;
        static_assert(TTag<T>::Tag != ::NVariant::T_INVALID, "Type not in TVariant.");
        Tag_ = TTag<T>::Tag;
        new (&Storage_) std::decay_t<T>(std::forward<T>(value));
    }

    template <class T, class... TArgs>
    void EmplaceValue(TArgs&&... args) {
        using namespace NVariant;
        static_assert(TTag<T>::Tag != ::NVariant::T_INVALID, "Type not in TVariant.");
        Tag_ = TTag<T>::Tag;
        new (&Storage_) std::decay_t<T>(std::forward<TArgs>(args)...);
    };

    void CopyVariant(const TVariant& other) {
        if (other) {
            other.UncheckedVisit(::NVariant::TVisitorCopyConstruct<TVariant>(this));
        }
    }

    void MoveVariant(TVariant&& other) {
        if (other) {
            other.UncheckedVisit(::NVariant::TVisitorMoveConstruct<TVariant>(this));
        }
    }

    template <class T>
    auto* ReinterpretAs() {
        return reinterpret_cast<std::decay_t<T>*>(&Storage_);
    }

    template <class T>
    const auto* ReinterpretAs() const {
        return reinterpret_cast<const std::decay_t<T>*>(&Storage_);
    }

private:
    int Tag_ = ::NVariant::T_EMPTY;
    std::aligned_union_t<0, Ts...> Storage_;
};


template <class... Ts>
struct THash<::TVariant<Ts...>> {
public:
    inline size_t operator()(const ::TVariant<Ts...>& v) const {
        const size_t tagHash = IntHash(v.Tag());
        const size_t valueHash = v ? v.UncheckedVisit(NVariant::TVisitorHash()) : 0;
        return CombineHashes(tagHash, valueHash);
    }
};
