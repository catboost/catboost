#pragma once

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/tag_invoke_cpo.h>

#include <util/system/compiler.h>

#include <concepts>
#include <memory>
#include <numeric>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): clang16 has bug which causes it
// ignore noexcept specifier of reference to static member methods.
#if (!__clang__ || __clang_major__ < 18)
    #define YT_TYPE_ERASURE_NOEXCEPT(NoExcept)
#else
    #define YT_TYPE_ERASURE_NOEXCEPT(NoExcept) noexcept(NoExcept)
#endif

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): Next section of code is about most general type erasure storage.
// There are two kinds: One which simply holds a pointer to data stored elsewhere
// and the one which holds the data on its own. The latter can store it inline
// on the byte array or on heap depending on the object size (small object optimization).
// Classes below feature bare-bone implementations of said storages which do not
// control when and which object are being placed inside the storage for they do not
// have access to the virtual table (ctors/dtors) of the type. Said operations are handled
// by type erasure containers themselves.
////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): This class is used to check
// whether some class has a templated method which
// accepts any type parameter.
struct TConceptWitness
{ };

////////////////////////////////////////////////////////////////////////////////

// Semantic requirement:
// 1) Pointer provided by |Ptr| call
// must be pointing to the beginning of some object.
// 2) IsStatic must be a constexpr variable.
// NB(arkady-e1ppa): value of bool template in Ptr
// is whether currently stored object is static or not.
template <class T>
concept CPointerProvider = requires (T* t, const T* ct) {
    { t->template Ptr<true>() } -> std::same_as<void*>;
    { ct->template Ptr<true>() } -> std::same_as<void*>;
    { t->template Ptr<false>() } -> std::same_as<void*>;
    { ct->template Ptr<false>() } -> std::same_as<void*>;
    { T::template IsStatic<TConceptWitness> } -> std::same_as<const bool&>;
};

////////////////////////////////////////////////////////////////////////////////

// CRTP base to implement cast methods for each
// storage.
template <class TDerived>
struct TStorageCasterBase
{
    template <bool IsStatic>
    Y_FORCE_INLINE void* GetPtr() const noexcept
    {
        return static_cast<const TDerived*>(this)->template Ptr<IsStatic>();
    }

    template <class TDecayedConcrete>
    Y_FORCE_INLINE TDecayedConcrete& As() & noexcept
    {
        return this->template ToRef<TDecayedConcrete, TDecayedConcrete*>();
    }

    template <class TDecayedConcrete>
    Y_FORCE_INLINE const TDecayedConcrete& As() const & noexcept
    {
        return this->template ToRef<TDecayedConcrete, const TDecayedConcrete*>();
    }

    template <class TDecayedConcrete>
    Y_FORCE_INLINE TDecayedConcrete&& As() && noexcept
    {
        return std::move(this->template ToRef<TDecayedConcrete, TDecayedConcrete*>());
    }

    template <class TDecayedConcrete, class TPtr>
    Y_FORCE_INLINE decltype(auto) ToRef() const noexcept
    {
        static_assert(std::derived_from<TDerived, TStorageCasterBase>, "Must inherit from TStorageCasterBase");
        static_assert(CPointerProvider<TDerived>, "Class must define Ptr method. See CPointerProvider");
        static_assert(
            std::same_as<
                TDecayedConcrete,
                std::remove_cvref_t<TDecayedConcrete>>,
            "Submitted type must not contain const, reference or volatile qualifiers");

        if constexpr (TDerived::template IsStatic<TDecayedConcrete>) {
            return *std::launder(reinterpret_cast<TPtr>(this->template GetPtr</*IsStatic*/ true>()));
        } else {
            return *static_cast<TPtr>(this->template GetPtr</*IsStatic*/ false>());
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CStorage = requires (const T& cref, T& ref, void* ptr) {
    { cref.template As<TConceptWitness>() } -> std::same_as<const TConceptWitness&>;
    { ref.template As<TConceptWitness>() } -> std::same_as<TConceptWitness&>;
    { std::move(ref).template As<TConceptWitness>() } -> std::same_as<TConceptWitness&&>;
    { ref.Set(ptr) } -> std::same_as<void>;

    // Semantic: Set() should reset the instance making it empty.
    { ref.Set() } -> std::same_as<void>;
} && std::default_initializable<T>;

////////////////////////////////////////////////////////////////////////////////

class TNonOwningStorage
    : public TStorageCasterBase<TNonOwningStorage>
{
public:
    template <bool /*IsStatic*/>
    Y_FORCE_INLINE void* Ptr() const noexcept
    {
        return const_cast<void*>(Data_);
    }

    template <class T>
    static constexpr bool IsStatic = false;

    Y_FORCE_INLINE void Set(void* ptr) noexcept
    {
        Data_ = ptr;
    }

    Y_FORCE_INLINE void Set() noexcept
    {
        Data_ = nullptr;
    }

private:
    void* Data_ = nullptr;
};

static_assert(CPointerProvider<TNonOwningStorage>);
static_assert(CStorage<TNonOwningStorage>);

////////////////////////////////////////////////////////////////////////////////

template <size_t InlineSize, size_t InlineAlign>
class TOwningStorage
    : public TStorageCasterBase<TOwningStorage<InlineSize, InlineAlign>>
{
private:
    static constexpr size_t RealSize = sizeof(void*) > InlineSize
        ? sizeof(void*)
        : InlineSize;

    static constexpr size_t RealAlign = alignof(void*) > InlineAlign
        ? alignof(void*)
        : InlineAlign;

public:
    template <class T>
    static constexpr bool IsStatic =
        (sizeof(T) <= RealSize) &&
        (alignof(T) <= RealAlign);

    Y_FORCE_INLINE void Set(void* ptr) noexcept
    {
        std::construct_at<void*>(reinterpret_cast<void**>(Data_), ptr);
    }

    Y_FORCE_INLINE void Set() noexcept
    {
        std::construct_at(Data_);
    }

    template <bool IsStatic>
    Y_FORCE_INLINE void* Ptr() const noexcept
    {
        auto* mutableThis = const_cast<TOwningStorage*>(this);

        if constexpr (IsStatic) {
            return &mutableThis->Data_;
        } else {
            return *std::launder(reinterpret_cast<void**>(&mutableThis->Data_));
        }
    }

private:
    alignas(RealAlign) std::byte Data_[RealSize] = {};
};

static_assert(CPointerProvider<TOwningStorage<1, 1>>);
static_assert(CStorage<TOwningStorage<1, 1>>);
static_assert(CPointerProvider<TOwningStorage<64, 64>>);
static_assert(CStorage<TOwningStorage<64, 64>>);

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): This part covers details of overloaded CPOs parcing and placeholder types.
// Relevant to public interface parts are exposed in type_erasure.h
////////////////////////////////////////////////////////////////////////////////

// Type erasable "function" = (function name <=> cpo) + (fixed signature).
template <auto Cpo, class TSignature>
struct TOverloadedCpo;

template <auto Cpo, class TRet, bool NoExcept, class... TArgs>
struct TOverloadedCpo<Cpo, TRet(TArgs...) noexcept(NoExcept)>
{ };

////////////////////////////////////////////////////////////////////////////////

// Each implementation of erasable method has
// its own "this" in signature. Since in the erased methods
// we do not have any specific type, we use placeholder.
struct TErasedThis
{ };

////////////////////////////////////////////////////////////////////////////////

template <class T, class U>
struct TFromThisImpl
{
    using TT = U;
};

template <class T>
struct TFromThisImpl<T, const TErasedThis&>
{
    using TT = const T&;
};

template <class T>
struct TFromThisImpl<T, TErasedThis&>
{
    using TT = T&;
};

template <class T>
struct TFromThisImpl<T, TErasedThis&&>
{
    using TT = T&&;
};

template <class T>
struct TFromThisImpl<T, TErasedThis>
{
    using TT = T;
};

template <class T, class U>
using TFromThis = typename TFromThisImpl<T, U>::TT;

////////////////////////////////////////////////////////////////////////////////

template <CStorage TStorage, class TCpo>
class TVTableEntry;

template <CStorage TStorage, auto Cpo, class TRet, bool NoExcept, class TCvThis, class... TArgs>
class TVTableEntry<TStorage, TOverloadedCpo<Cpo, TRet(TCvThis, TArgs...) noexcept(NoExcept)>>
{
private:
    using TReplaced = TFromThis<TStorage, TCvThis>;
    using TFunction = TRet(*)(TReplaced, TArgs...) YT_TYPE_ERASURE_NOEXCEPT(NoExcept);

public:
    TVTableEntry() = default;

    Y_FORCE_INLINE TFunction Get() const noexcept
    {
        return Function_;
    }

    template <class TConcrete>
    Y_FORCE_INLINE static TVTableEntry Create() noexcept
    {
        TVTableEntry entry = {};

        entry.Function_ = &TVTableEntry::StaticInvoke<TConcrete>;

        return entry;
    }

    Y_FORCE_INLINE void Reset() noexcept
    {
        Function_ = nullptr;
    }

    Y_FORCE_INLINE bool IsValid() const noexcept
    {
        return Function_ != nullptr;
    }

    // NB(arkady-e1ppa): This method may or may not work correctly
    // for dynamically-linked libraries.
    template <class T>
    Y_FORCE_INLINE bool IsCurrentlyStored() const noexcept
    {
        return Function_ == &TVTableEntry::StaticInvoke<T>;
    }

private:
    TFunction Function_ = nullptr;

    template <class TConcrete>
    static TRet StaticInvoke(TReplaced storage, TArgs... args) YT_TYPE_ERASURE_NOEXCEPT(NoExcept)
    {
        return Cpo(std::forward<TReplaced>(storage).template As<TConcrete>(), std::forward<TArgs>(args)...);
    }
};

////////////////////////////////////////////////////////////////////////////////

// Add vtable holder for static/local versions.
template <CStorage TStorage, class... TCpos>
class TVTable;

// NB(arkady-e1ppa): We do not support empty vtables since there is no way
// to check validity nor type of the object for such vtables.
template <CStorage TStorage, class TCpo, class... TCpos>
class TVTable<TStorage, TCpo, TCpos...>
    : private TVTableEntry<TStorage, TCpo>
    , private TVTableEntry<TStorage, TCpos>...
{
public:
    TVTable() = default;

    template <class TConcrete>
    Y_FORCE_INLINE static TVTable Create() noexcept
    {
        return TVTable{TCtorTag<TConcrete>{}};
    }

    template <auto C>
    Y_FORCE_INLINE auto GetFunctor() const noexcept
    {
        return TVTableEntry<TStorage, TTagInvokeTag<C>>::Get();
    }

    Y_FORCE_INLINE void Reset() noexcept
    {
        TVTableEntry<TStorage, TCpo>::Reset();
    }

    Y_FORCE_INLINE bool IsValid() const noexcept
    {
        return TVTableEntry<TStorage, TCpo>::IsValid();
    }

    template <class T>
    Y_FORCE_INLINE bool IsCurrentlyStored() const noexcept
    {
        return TVTableEntry<TStorage, TCpo>::template IsCurrentlyStored<T>();
    }

private:
    template <class T>
    struct TCtorTag
    { };

    template <class T>
    explicit TVTable(TCtorTag<T>) noexcept
        : TVTableEntry<TStorage, TCpo>{TVTableEntry<TStorage, TCpo>::template Create<T>()}
        , TVTableEntry<TStorage, TCpos>{TVTableEntry<TStorage, TCpos>::template Create<T>()}...
    { }
};

////////////////////////////////////////////////////////////////////////////////

template <class T, class TStorage>
struct TIsVTable
    : public std::false_type
{ };

template <CStorage TStorage, class... TCpos>
struct TIsVTable<TVTable<TStorage, TCpos...>, TStorage>
    : public std::true_type
{ };

template <class T, class TStorage>
concept CVTableFor =
    CStorage<TStorage> &&
    TIsVTable<std::remove_cvref_t<T>, TStorage>::value;

////////////////////////////////////////////////////////////////////////////////

template <class TStorage, CVTableFor<TStorage> TTable, bool IsStatic>
class TVTableHolder
{
public:
    TVTableHolder() = default;

    bool IsValid() const
    {
        return Table_ != nullptr && Table_->IsValid();
    }

    template <class TConcrete>
    static TVTableHolder Create()
    {
        static TTable staticTable = TTable::template Create<TConcrete>();

        TVTableHolder holder = {};
        holder.Table_ = std::addressof(staticTable);
        return holder;
    }

    const TTable& GetVTable() const
    {
        if (!IsValid()) {
            return EmptyTable;
        }

        return *Table_;
    }

    void Reset()
    {
        Table_ = nullptr;
    }

private:
    TTable* Table_ = nullptr;

    static inline TTable EmptyTable = {};
};

template <class TStorage, CVTableFor<TStorage> TTable>
class TVTableHolder<TStorage, TTable, false>
{
public:
    TVTableHolder() = default;

    bool IsValid() const
    {
        return Table_.IsValid();
    }

    template <class TConcrete>
    static TVTableHolder Create()
    {
        TVTableHolder holder = {};
        holder.Table_ = TTable::template Create<TConcrete>();
        return holder;
    }

    const TTable& GetVTable() const
    {
        return Table_;
    }

    void Reset()
    {
        Table_.Reset();
    }

private:
    TTable Table_ = {};
};

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): Below are Cpos which replicate function of default language features
// such as copy/move ctors, dtor. More can be added later if needed.
// But they are likely to be an overkill.
// General formula is that we have a cpo for default feature and then
// a wrapper of an object which serves the purpose of an adaptor between
// native c++ expression and CPO expression of the same behavior.
////////////////////////////////////////////////////////////////////////////////

struct TDeleter
    : public TTagInvokeCpoBase<TDeleter>
{ };

inline constexpr TOverloadedCpo<TDeleter{}, void(TErasedThis&) noexcept> Deleter = {};

////////////////////////////////////////////////////////////////////////////////

// NB(arkady-e1ppa): We do not use custom allocators with wacky policies
// and therefore don't adopt reallocating move ctor.
template <CStorage TStorage>
struct TMover
    : public TTagInvokeCpoBase<TMover<TStorage>>
{ };

template <CStorage TStorage>
inline constexpr TOverloadedCpo<TMover<TStorage>{}, void(TErasedThis&&, TStorage&)> Mover = {};

////////////////////////////////////////////////////////////////////////////////

template <CStorage TStorage>
struct TCopier
    : public TTagInvokeCpoBase<TCopier<TStorage>>
{ };

template <CStorage TStorage>
inline constexpr TOverloadedCpo<TCopier<TStorage>{}, void(const TErasedThis&, TStorage&)> Copier = {};

////////////////////////////////////////////////////////////////////////////////

struct TNoopCpo
{
    template <class... TArgs>
    constexpr void operator() (TArgs&&... /*args*/) const
    {
        YT_ABORT();
    }
};

inline constexpr TOverloadedCpo<TNoopCpo{}, void(const TErasedThis&)> NoopCpo = {};

////////////////////////////////////////////////////////////////////////////////

template <class W, class T>
concept CWrapperOf = requires (W& ref, const W& cref) {
    { ref.Unwrap() } -> std::same_as<T&>;
    { std::move(ref).Unwrap() } -> std::same_as<T&&>;
    { cref.Unwrap() } -> std::same_as<const T&>;
};

////////////////////////////////////////////////////////////////////////////////

// Wraps value, adding TagInvoke overload for dtor and move/copy ctors.
template <class TDerived, class TConcrete, CStorage TStorage, bool EnableCopy>
class TOwningWrapperBase
{
public:
    // NB(arkady-e1ppa): Wrapper is stored not only the object itself.
    static constexpr bool IsStatic = TStorage::template IsStatic<TDerived>;

    using TTraits = std::allocator_traits<std::allocator<std::byte>>;
    static inline std::allocator<std::byte> Allocator = {};

    template <class... TArgs>
        requires std::constructible_from<TConcrete, TArgs...>
    explicit TOwningWrapperBase(TArgs&&... args)
        noexcept(std::is_nothrow_constructible_v<TConcrete, TArgs...>)
        : Concrete_(std::forward<TArgs>(args)...)
    { }

    Y_FORCE_INLINE void Delete() & noexcept
    {
        TTraits::template destroy<TDerived>(Allocator, AsFinal());

        if constexpr (!IsStatic) {
            TTraits::deallocate(Allocator, reinterpret_cast<std::byte*>(AsFinal()), sizeof(TDerived));
        }
    }

    Y_FORCE_INLINE void Move(TStorage& to) &&
        requires std::movable<TConcrete>
    {
        if constexpr (IsStatic) {
            to.Set();

            TTraits::template construct<TDerived>(
                Allocator,
                &to.template As<TDerived>(),
                std::move(*AsFinal()));
        } else {
            to.Set(static_cast<void*>(AsFinal()));
        }
    }

    Y_FORCE_INLINE void Copy(TStorage& to) const &
        requires (EnableCopy && std::copyable<TConcrete>)
    {
        if constexpr (IsStatic) {
            to.Set();
        } else {
            to.Set(TTraits::allocate(Allocator, sizeof(TDerived)));
        }

        TTraits::template construct<TDerived>(
            Allocator,
            &to.template As<TDerived>(),
            *AsFinal());
    }

    friend Y_FORCE_INLINE void TagInvoke(TDeleter, TDerived& this_) noexcept
    {
        static_assert(std::derived_from<TDerived, TOwningWrapperBase>, "Must derived from TOwningWrapperBase");
        this_.Delete();
    }

    friend Y_FORCE_INLINE void TagInvoke(TMover<TStorage>, TDerived&& this_, TStorage& to)
        noexcept(std::is_nothrow_move_constructible_v<TConcrete>)
    {
        std::move(this_).Move(to);
    }

    friend Y_FORCE_INLINE void TagInvoke(TCopier<TStorage>, const TDerived& this_, TStorage& to)
        noexcept(std::is_nothrow_copy_constructible_v<TConcrete>)
    {
        this_.Copy(to);
    }

    // CWrapperOf<TConcrete>
    Y_FORCE_INLINE TConcrete& Unwrap() &
    {
        return Concrete_;
    }

    Y_FORCE_INLINE const TConcrete& Unwrap() const &
    {
        return Concrete_;
    }

    Y_FORCE_INLINE TConcrete&& Unwrap() &&
    {
        return std::move(Concrete_);
    }

private:
    TConcrete Concrete_;

    Y_FORCE_INLINE TDerived* AsFinal() const
    {
        return static_cast<TDerived*>(const_cast<TOwningWrapperBase*>(this));
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class TDerived, class TCpo>
struct TUnwrappingTagInvokeBase;

// NB(arkady-e1ppa): Technically, we could accept TErasedThis as any argument, but
// there is barely any profit from doing so yet code becomes much more cumbersome
// both here and around TAny(Object|Ref).
template <class TDerived, auto Cpo, class TRet, bool NoExcept, class TCvThis, class... TArgs>
struct TUnwrappingTagInvokeBase<TDerived, TOverloadedCpo<Cpo, TRet(TCvThis, TArgs...) noexcept(NoExcept)>>
{
    using TReplaced = TFromThis<TDerived, TCvThis>;

    friend Y_FORCE_INLINE TRet TagInvoke(TTagInvokeTag<Cpo>, TReplaced wrapper, TArgs... args) YT_TYPE_ERASURE_NOEXCEPT(NoExcept)
    {
        return Cpo(std::forward<TReplaced>(wrapper).Unwrap(), std::forward<TArgs>(args)...);
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class TConcrete, CStorage TStorage, bool EnableCopy, class... TCpos>
class TOwningWrapper final
    : public TOwningWrapperBase<
        TOwningWrapper<
            TConcrete,
            TStorage,
            EnableCopy,
            TCpos...>,
        TConcrete,
        TStorage,
        EnableCopy>
    , public TUnwrappingTagInvokeBase<
        TOwningWrapper<
            TConcrete,
            TStorage,
            EnableCopy,
            TCpos...>,
        TCpos>...
{
    using TBase = TOwningWrapperBase<
        TOwningWrapper<
            TConcrete,
            TStorage,
            EnableCopy,
            TCpos...>,
        TConcrete,
        TStorage,
        EnableCopy>;

    using TBase::TBase;
};

////////////////////////////////////////////////////////////////////////////////

// Every "Any-like" object must be able to provide access to storage
// and a vtable which matches the storage provided.
template <class T>
concept CSomeAnyObject = requires (T& ref, const T& cref) {
    typename T::TStorage;
    { ref.GetStorage() } -> std::same_as<typename T::TStorage&>;
    { std::move(ref).GetStorage() } -> std::same_as<typename T::TStorage&&>;
    { cref.GetStorage() } -> std::same_as<const typename T::TStorage&>;

    { cref.GetVTable() } -> CVTableFor<typename T::TStorage>;
};

////////////////////////////////////////////////////////////////////////////////

template <class TDerived, class TCpo>
struct TAnyFragment;

template <class TDerived, auto Cpo, class TRet, bool NoExcept, class TCvThis, class... TArgs>
struct TAnyFragment<TDerived, TOverloadedCpo<Cpo, TRet(TCvThis, TArgs...) noexcept(NoExcept)>>
{
    using TReplaced = TFromThis<TDerived, TCvThis>;
    using TVTableTag = TOverloadedCpo<Cpo, TRet(TCvThis, TArgs...) noexcept(NoExcept)>;

    friend Y_FORCE_INLINE TRet TagInvoke(TTagInvokeTag<Cpo>, TReplaced any, TArgs... args) YT_TYPE_ERASURE_NOEXCEPT(NoExcept)
    {
        static_assert(CSomeAnyObject<TDerived>);

        auto&& vtable = any.GetVTable();

        YT_VERIFY(vtable.IsValid());

        auto* functor = vtable.template GetFunctor<TVTableTag{}>();

        return functor(std::forward<TReplaced>(any).GetStorage(), std::forward<TArgs>(args)...);
    }
};

#undef YT_TYPE_ERASURE_NOEXCEPT

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
