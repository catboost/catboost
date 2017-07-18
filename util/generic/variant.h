#pragma once

#include <util/generic/typetraits.h>
#include <util/system/yassert.h>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////

namespace NVariant {
    template <class X, class... Ts>
    struct TTagTraits;

    template <class... Ts>
    struct TTypeTraits;

    template <class Visitor, class... Ts>
    struct TVisitorResult;

} // namespace NVariant

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TVariantTypeTag {};

//! |boost::variant|-like discriminated union with C++ 11 features.
template <class... Ts>
class TVariant {
public:
    static_assert(
        ::NVariant::TTypeTraits<Ts...>::NoRefs,
        "TVariant type arguments cannot be references.");

    static_assert(
        ::NVariant::TTypeTraits<Ts...>::NoDuplicates,
        "TVariant type arguments cannot contain duplicate types.");

    //! Variants cannot be default-constructed.
    TVariant() = delete;

    //! Constructs an instance by copying another instance.
    TVariant(const TVariant& other);

    //! Constructs an instance by moving another instance.
    TVariant(TVariant&& other);

    //! Constructs an instance by copying a given value.
    template <class T>
    TVariant(const T& value);

    //! Constructs an instance by moving a given value.
    template <
        class T,
        class = std::enable_if_t<
            !std::is_reference<T>::value &&
            !std::is_same<std::decay_t<T>, TVariant<Ts...>>::value>>
    TVariant(T&& value);

    //! Constructs an instance in-place.
    template <
        class T,
        class... TArgs,
        class = std::enable_if_t<
            !std::is_reference<T>::value &&
            !std::is_same<std::decay_t<T>, TVariant<Ts...>>::value>>
    TVariant(TVariantTypeTag<T>, TArgs&&... args);

    //! Destroys the instance.
    ~TVariant();

    //! Assigns a given value.
    template <class T>
    TVariant& operator=(const T& value);

    //! Moves a given value.
    template <
        class T,
        class = std::enable_if_t<
            !std::is_reference<T>::value &&
            !std::is_same<std::decay_t<T>, TVariant<Ts...>>::value>>
    TVariant& operator=(T&& value);

    //! Assigns a given instance.
    TVariant& operator=(const TVariant& other);

    //! Moves a given instance.
    TVariant& operator=(TVariant&& other);

    //! Compare with other value.
    template <class T>
    bool operator==(const T& value) const;

    //! Compare with other instance.
    bool operator==(const TVariant& other) const;

    //! Returns the discriminating tag of the instance.
    int Tag() const;

    //! Returns the discriminating tag the given type.
    template <class T>
    static constexpr int TagOf() {
        // NB: Must keep this inline due to constexpr.
        return ::NVariant::TTagTraits<T, Ts...>::Tag;
    }

    //! Casts the instance to a given type.
    //! Tag validation is only performed in debug builds.
    template <class T>
    T& As();

    //! Similar to its non-const version.
    template <class T>
    const T& As() const;

    //! Checks if the instance holds a given of a given type.
    //! Returns the pointer to the value on success or |nullptr| on failure.
    template <class T>
    T* TryAs();

    //! Similar to its non-const version.
    template <class T>
    const T* TryAs() const;

    //! Returns |true| iff the instance holds a value of a given type.
    template <class T>
    bool Is() const;

    //! Pass mutable internal value to visitor
    template <class Visitor>
    typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType Visit(Visitor&& visitor);

    //! Pass const internal value to visitor
    template <class Visitor>
    typename ::NVariant::TVisitorResult<Visitor, Ts...>::TType Visit(Visitor&& visitor) const;

private:
    int Tag_;
    std::aligned_union_t<0, Ts...> Storage_;

    template <class T>
    void AssignValue(const T& value);

    template <class T>
    void AssignValue(T&& value);

    template <class T, class... TArgs>
    void EmplaceValue(TArgs&&... args);

    void AssignVariant(const TVariant& other);

    void AssignVariant(TVariant&& other);

    void Destroy();

    template <class T>
    T& UncheckedAs();

    template <class T>
    const T& UncheckedAs() const;
};

#define VARIANT_TRAITS_H_
#include "variant_traits.h"
#undef VARIANT_TRAITS_H_

#define VARIANT_IMPL_H_
#include "variant_impl.h"
#undef VARIANT_IMPL_H_
