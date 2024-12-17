#pragma once

#include "type_erasure_detail.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

template <class T, class TCpo>
struct TTypeErasableTraits;

template <class T, auto Cpo, class TRet, bool NoExcept, class TCvThis, class... TArgs>
struct TTypeErasableTraits<T, TOverloadedCpo<Cpo, TRet(TCvThis, TArgs...) noexcept(NoExcept)>>
{
    using TReplaced = TFromThis<T, TCvThis>;

    static constexpr bool Value = NoExcept
        ? requires (TReplaced t, TArgs... args) {
            { Cpo(std::forward<TReplaced>(t), std::forward<TArgs>(args)...) } noexcept -> std::same_as<TRet>;
        }
        : requires (TReplaced t, TArgs... args) {
            { Cpo(std::forward<TReplaced>(t), std::forward<TArgs>(args)...) } -> std::same_as<TRet>;
        };
};

template <class T, class... TCpos>
concept CAnyRefErasable = (TTypeErasableTraits<std::remove_cvref_t<T>, TCpos>::Value && ...);

template <class T, bool EnableCopy, class... TCpos>
concept CAnyObjectErasable =
    CAnyRefErasable<T, TCpos...> &&
    std::movable<std::remove_cvref_t<T>> &&
    (!EnableCopy || std::copyable<std::remove_cvref_t<T>>);

////////////////////////////////////////////////////////////////////////////////

struct TBadAnyCast
    : public std::exception
{ };

////////////////////////////////////////////////////////////////////////////////

// AnyRef.

template <bool UseStaticVTable, class... TCpos>
class TAnyRef
    : private TAnyFragment<TAnyRef<UseStaticVTable, TCpos...>, TCpos>...
{
public:
    using TStorage = TNonOwningStorage;

    TAnyRef() = default;

    template <class TConcrete>
        requires
            CAnyRefErasable<std::remove_cvref_t<TConcrete>, TCpos...> &&
            (!std::same_as<std::remove_cvref_t<TConcrete>, TAnyRef>)
    TAnyRef(TConcrete& concRef) noexcept
    {
        using TDecayed = std::remove_cvref_t<TConcrete>;

        Storage_.Set(const_cast<void*>(reinterpret_cast<const void*>(std::addressof(concRef))));
        Holder_ = THolder::template Create<TDecayed>();
    }

    // Copy/Move are trivial copy. Dtor is also trivial.

    const auto& GetStorage() const &
    {
        return Storage_;
    }

    auto& GetStorage() &
    {
        return Storage_;
    }

    auto&& GetStorage() &&
    {
        return std::move(Storage_);
    }

    const auto& GetVTable() const
    {
        return Holder_.GetVTable();
    }

    template <class T>
        requires CAnyRefErasable<T, TCpos...>
    T& AnyCast() &
    {
        const auto& vtable = GetVTable();
        if (!vtable.IsValid() || !vtable.template IsCurrentlyStored<T>()) {
            throw TBadAnyCast{};
        }

        return Storage_.template As<T>();
    }

    template <class T>
        requires CAnyRefErasable<T, TCpos...>
    const T& AnyCast() const &
    {
        const auto& vtable = Holder_.GetVTable();
        if (!vtable.IsValid() || !vtable.template IsCurrentlyStored<T>()) {
            throw TBadAnyCast{};
        }

        return Storage_.template As<T>();
    }

private:
    using TVTable = TVTable<TStorage, TCpos...>;
    using THolder = TVTableHolder<TStorage, TVTable, UseStaticVTable>;

    TStorage Storage_ = {};
    THolder Holder_ = {};
};

////////////////////////////////////////////////////////////////////////////////

template <class... TCpos>
struct TAnyRefPublicHelper
{
    using T = TAnyRef<(sizeof...(TCpos) >= 3), TCpos...>;
};

template <>
struct TAnyRefPublicHelper<>
{
    using T = TAnyRef<false, TTagInvokeTag<NoopCpo>>;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
struct TIsAnyRef
    : public std::false_type
{ };

template <bool UseStaticVTable, class... TCpos>
struct TIsAnyRef<TAnyRef<UseStaticVTable, TCpos...>>
    : public std::true_type
{ };

////////////////////////////////////////////////////////////////////////////////

template <bool EnableCopy, CStorage TStorage, class... TCpos>
struct TRebindVTable
{
    using TVTable = TVTable<
        TStorage,
        TTagInvokeTag<Deleter>,
        TTagInvokeTag<Mover<TStorage>>,
        TTagInvokeTag<Copier<TStorage>>,
        TCpos...>;
};

template <CStorage TStorage, class... TCpos>
struct TRebindVTable<false, TStorage, TCpos...>
{
    using TVTable = TVTable<
        TStorage,
        TTagInvokeTag<Deleter>,
        TTagInvokeTag<Mover<TStorage>>,
        TCpos...>;
};

////////////////////////////////////////////////////////////////////////////////

template <bool UseStaticVTable, bool EnableCopy, size_t SmallObjectSize, size_t SmallObjectAlign, class... TCpos>
class TAnyObject
    : private TAnyFragment<
        TAnyObject<
            UseStaticVTable,
            EnableCopy,
            SmallObjectSize,
            SmallObjectAlign,
            TCpos...>,
        TCpos>...
{
public:
    using TStorage = TOwningStorage<SmallObjectSize, SmallObjectAlign>;

    TAnyObject() = default;

    template <class TConcrete>
        requires
            (!std::same_as<std::remove_cvref_t<TConcrete>, TAnyObject>) &&
            std::constructible_from<std::remove_cvref_t<TConcrete>, TConcrete> &&
            CAnyObjectErasable<std::remove_cvref_t<TConcrete>, EnableCopy, TCpos...> &&
            (!TIsAnyRef<std::remove_cvref_t<TConcrete>>::value)
    TAnyObject(TConcrete&& concrete)
    {
        Set<TConcrete>(std::forward<TConcrete>(concrete));
    }

    template <class TConcrete, class... TArgs>
        requires
            (!std::same_as<std::remove_cvref_t<TConcrete>, TAnyObject>) &&
            std::constructible_from<std::remove_cvref_t<TConcrete>, TArgs...> &&
            (!std::is_const_v<TConcrete>) &&
            CAnyObjectErasable<std::remove_cvref_t<TConcrete>, EnableCopy, TCpos...> &&
            (!TIsAnyRef<std::remove_cvref_t<TConcrete>>::value)
    TAnyObject(std::in_place_type_t<TConcrete>, TArgs&&... args)
    {
        Set<TConcrete>(std::forward<TArgs>(args)...);
    }

    TAnyObject(TAnyObject&& other)
        : Holder_(other.Holder_)
    {
        if (IsValid()) {
            auto* mover = GetVTable().template GetFunctor<Mover<TStorage>>();
            mover(std::move(other).GetStorage(), GetStorage());

            other.Reset();
        }
    }

    TAnyObject& operator=(TAnyObject&& other)
    {
        if (this == &other) {
            return *this;
        }

        Reset();

        Holder_ = other.Holder_;

        if (IsValid()) {
            auto* mover = GetVTable().template GetFunctor<Mover<TStorage>>();
            mover(std::move(other).GetStorage(), GetStorage());

            other.Reset();
        }

        return *this;
    }

    TAnyObject(const TAnyObject& other) requires EnableCopy
        : Holder_(other.Holder_)
    {
        if (IsValid()) {
            auto* copier = GetVTable().template GetFunctor<Copier<TStorage>>();
            copier(other.GetStorage(), GetStorage());
        }
    }

    TAnyObject& operator=(const TAnyObject& other) requires EnableCopy
    {
        if (this == &other) {
            return *this;
        }

        Reset();

        Holder_ = other.Holder_;

        if (IsValid()) {
            auto* copier = GetVTable().template GetFunctor<Copier<TStorage>>();
            copier(other.GetStorage(), GetStorage());
        }

        return *this;
    }

    constexpr explicit operator bool() const
    {
        return IsValid();
    }

    ~TAnyObject()
    {
        Reset();
    }

    Y_FORCE_INLINE const auto& GetStorage() const &
    {
        return Storage_;
    }

    Y_FORCE_INLINE auto& GetStorage() &
    {
        return Storage_;
    }

    Y_FORCE_INLINE auto&& GetStorage() &&
    {
        return std::move(Storage_);
    }

    Y_FORCE_INLINE const auto& GetVTable() const
    {
        return Holder_.GetVTable();
    }

    template <class T>
    Y_FORCE_INLINE T& AnyCast() &
    {
        using TDecayed = std::remove_cvref_t<T>;
        using TWrapped = TOwningWrapper<TDecayed, TStorage, EnableCopy, TCpos...>;

        const auto& vtable = GetVTable();
        if (!vtable.IsValid() || !vtable.template IsCurrentlyStored<TWrapped>()) {
            throw TBadAnyCast{};
        }

        return Storage_.template As<TWrapped>().Unwrap();
    }

    template <class T>
    Y_FORCE_INLINE const T& AnyCast() const &
    {
        using TDecayed = std::remove_cvref_t<T>;
        using TWrapped = TOwningWrapper<TDecayed, TStorage, EnableCopy, TCpos...>;

        const auto& vtable = GetVTable();
        if (!vtable.IsValid() || !vtable.template IsCurrentlyStored<TWrapped>()) {
            throw TBadAnyCast{};
        }

        return Storage_.template As<TWrapped>().Unwrap();
    }

    Y_FORCE_INLINE bool IsValid() const
    {
        return Holder_.IsValid();
    }

private:
    using TVTable = typename TRebindVTable<EnableCopy, TStorage, TCpos...>::TVTable;
    using THolder = TVTableHolder<TStorage, TVTable, UseStaticVTable>;
    using TTraits = std::allocator_traits<std::allocator<std::byte>>;

    TStorage Storage_ = {};
    THolder Holder_ = {};

    static inline std::allocator<std::byte> Allocator = {};

    Y_FORCE_INLINE void Reset() noexcept
    {
        const auto& vtable = Holder_.GetVTable();
        if (Holder_.IsValid()) {
            auto* deleter = vtable.template GetFunctor<Deleter>();
            deleter(Storage_);

            Holder_.Reset();
        }
    }

    template <class TConcrete, class... TArgs>
    Y_FORCE_INLINE void Set(TArgs&&... args)
    {
        using TDecayed = std::remove_cvref_t<TConcrete>;
        using TWrapped = TOwningWrapper<TDecayed, TStorage, EnableCopy, TCpos...>;

        Reset();

        Holder_ = THolder::template Create<TWrapped>();

        if constexpr (TStorage::template IsStatic<TWrapped>) {
            Storage_.Set();
        } else {
            Storage_.Set(TTraits::allocate(Allocator, sizeof(TWrapped)));
        }

        TTraits::template construct<TWrapped>(
            Allocator,
            &Storage_.template As<TWrapped>(),
            std::forward<TArgs>(args)...);
    }
};

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

// A simple placeholder for overloaded cpo usage.
using NDetail::TErasedThis;

////////////////////////////////////////////////////////////////////////////////

// A wrapper class which carries all the necessary information for type erasure.
template <auto Cpo, class TSignature>
using TOverload = NDetail::TOverloadedCpo<Cpo, TSignature>;

////////////////////////////////////////////////////////////////////////////////

// A non-owning reference.
template <class... TCpos>
using TAnyRef = typename NDetail::TAnyRefPublicHelper<TCpos...>::T;

////////////////////////////////////////////////////////////////////////////////

template <class... TCpos>
using TAnyObject = NDetail::TAnyObject<
    (sizeof...(TCpos) >= 3),
    /*EnableCopy*/ true,
    /*SmallObjectSize*/ sizeof(void*) * 2,
    /*SmallObjectAlign*/ alignof(void*) * 2,
    TCpos...>;

template <class... TCpos>
using TAnyUnique = NDetail::TAnyObject<
    (sizeof...(TCpos) >= 3),
    /*EnableCopy*/ false,
    /*SmallObjectSize*/ sizeof(void*) * 2,
    /*SmallObjectAlign*/ alignof(void*) * 2,
    TCpos...>;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
