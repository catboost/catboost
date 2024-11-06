#ifndef NEW_INL_H_
#error "Direct inclusion of this file is not allowed, include new.h"
// For the sake of sane code completion.
#include "new.h"
#endif

#include "ref_tracked.h"

#include <library/cpp/yt/misc/port.h>

#include <library/cpp/yt/malloc//malloc.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TRefCountedCookieHolder
{
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    TRefCountedTypeCookie Cookie = NullRefCountedTypeCookie;

    void InitializeTracking(TRefCountedTypeCookie cookie)
    {
        YT_ASSERT(Cookie == NullRefCountedTypeCookie);
        Cookie = cookie;
        TRefCountedTrackerFacade::AllocateInstance(Cookie);
    }

    ~TRefCountedCookieHolder()
    {
        if (Cookie != NullRefCountedTypeCookie) {
            TRefCountedTrackerFacade::FreeInstance(Cookie);
        }
    }
#endif
};

template <class T>
struct TRefCountedWrapper final
    : public T
    , public TRefTracked<T>
{
    template <class... TArgs>
    explicit TRefCountedWrapper(TArgs&&... args)
        : T(std::forward<TArgs>(args)...)
    { }

    ~TRefCountedWrapper() = default;

    void DestroyRefCounted() override
    {
        T::DestroyRefCountedImpl(this);
    }
};

template <class T, class TDeleter>
class TRefCountedWrapperWithDeleter final
    : public T
    , public TRefTracked<T>
{
public:
    template <class... TArgs>
    explicit TRefCountedWrapperWithDeleter(TDeleter deleter, TArgs&&... args)
        : T(std::forward<TArgs>(args)...)
        , Deleter_(std::move(deleter))
    { }

    ~TRefCountedWrapperWithDeleter() = default;

    void DestroyRefCounted() override
    {
        Deleter_(this);
    }

private:
    const TDeleter Deleter_;
};

template <class T>
struct TRefCountedWrapperWithCookie final
    : public T
    , public TRefCountedCookieHolder
{
    template <class... TArgs>
    explicit TRefCountedWrapperWithCookie(TArgs&&... args)
        : T(std::forward<TArgs>(args)...)
    { }

    ~TRefCountedWrapperWithCookie() = default;

    void DestroyRefCounted() override
    {
        T::DestroyRefCountedImpl(this);
    }
};

namespace NDetail {

template <class... Args>
Y_FORCE_INLINE void CustomInitialize(Args... args)
{
    Y_UNUSED(args...);
}

template <class T>
Y_FORCE_INLINE auto CustomInitialize(T* ptr) -> decltype(&T::InitializeRefCounted, void())
{
    ptr->InitializeRefCounted();
}

template <class T, class... As>
Y_FORCE_INLINE T* NewEpilogue(void* ptr, As&& ... args)
{
    try {
        auto* instance = static_cast<T*>(ptr);
        new (instance) T(std::forward<As>(args)...);
        CustomInitialize(instance);
        return instance;
    } catch (...) {
        // Do not forget to free the memory.
        TFreeMemory<T>::Do(ptr);
        throw;
    }
}

template <class T, bool = std::derived_from<T, TRefCountedBase>>
struct TConstructHelper
{
    static constexpr size_t RefCounterSpace = (sizeof(TRefCounter) + alignof(T) - 1) & ~(alignof(T) - 1);
    static constexpr size_t RefCounterOffset = RefCounterSpace - sizeof(TRefCounter);
    static constexpr size_t Size = RefCounterSpace + sizeof(T);
    static constexpr size_t Alignment = alignof(T);

    template <class... As>
    Y_FORCE_INLINE static T* Construct(void* ptr, As&&... args)
    {
        auto* refCounter = reinterpret_cast<TRefCounter*>(static_cast<char*>(ptr) + RefCounterOffset);
        new (refCounter) TRefCounter();
        auto* object = reinterpret_cast<T*>(refCounter + 1);
        if constexpr (std::is_constructible_v<T, As...>) {
            new(object) T(std::forward<As>(args)...);
        } else {
            new(object) T{std::forward<As>(args)...};
        }
        CustomInitialize(object);
        return object;
    }
};

template <class T>
struct TConstructHelper<T, true>
{
    static constexpr size_t Size = sizeof(TRefCountedWrapper<T>);
    static constexpr size_t Alignment = alignof(TRefCountedWrapper<T>);

    template <class... As>
    Y_FORCE_INLINE static TRefCountedWrapper<T>* Construct(void* ptr, As&&... args)
    {
        using TDerived = TRefCountedWrapper<T>;
        auto* object = new(static_cast<TDerived*>(ptr)) TDerived(std::forward<As>(args)...);
        CustomInitialize(object);
        return object;
    }
};

template <class T, class... As>
Y_FORCE_INLINE TIntrusivePtr<T> SafeConstruct(void* ptr, As&&... args)
{
    try {
        auto* instance = TConstructHelper<T>::Construct(ptr, std::forward<As>(args)...);
        return TIntrusivePtr<T>(instance, /*addReference*/ false);
    } catch (...) {
        // Do not forget to free the memory.
        TFreeMemory<T>::Do(ptr);
        throw;
    }
}

void AbortOnOom();

template <size_t Size, size_t Alignment>
Y_FORCE_INLINE void* AllocateConstSizeAlignedOrCrash()
{
    void* ptr;
#ifdef _win_
    ptr = ::aligned_malloc(Size, Alignment);
#else
    if (Alignment <= alignof(std::max_align_t)) {
        ptr = ::malloc(Size);
    } else {
        ptr = ::aligned_malloc(Size, Alignment);
    }
#endif
    if (Y_UNLIKELY(!ptr)) {
        AbortOnOom();
    }
    return ptr;
}

template <size_t Alignment>
Y_FORCE_INLINE void* AllocateOrCrash(size_t size)
{
    void* ptr;
#ifdef _win_
    ptr = ::aligned_malloc(size, Alignment);
#else
    if (Alignment <= alignof(std::max_align_t)) {
        ptr = ::malloc(size);
    } else {
        ptr = ::aligned_malloc(size, Alignment);
    }
#endif
    if (Y_UNLIKELY(!ptr)) {
        AbortOnOom();
    }
    return ptr;
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class T, class... As, class>
Y_FORCE_INLINE TIntrusivePtr<T> New(
    As&&... args)
{
    void* ptr = NYT::NDetail::AllocateConstSizeAlignedOrCrash<
        NYT::NDetail::TConstructHelper<T>::Size,
        NYT::NDetail::TConstructHelper<T>::Alignment>();
    return NYT::NDetail::SafeConstruct<T>(ptr, std::forward<As>(args)...);
}

template <class T, class... As, class>
Y_FORCE_INLINE TIntrusivePtr<T> TryNew(
    typename T::TAllocator* allocator,
    As&&... args)
{
    auto* ptr = allocator->Allocate(NYT::NDetail::TConstructHelper<T>::Size);
    if (Y_UNLIKELY(!ptr)) {
        return nullptr;
    }
    return NYT::NDetail::SafeConstruct<T>(ptr, std::forward<As>(args)...);
}

template <class T, class... As, class>
Y_FORCE_INLINE TIntrusivePtr<T> New(
    typename T::TAllocator* allocator,
    As&&... args)
{
    auto obj = TryNew<T>(allocator, std::forward<As>(args)...);
    if (Y_UNLIKELY(!obj)) {
        NYT::NDetail::AbortOnOom();
    }
    return obj;
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class... As, class>
Y_FORCE_INLINE TIntrusivePtr<T> NewWithExtraSpace(
    size_t extraSpaceSize,
    As&&... args)
{
    auto totalSize = NYT::NDetail::TConstructHelper<T>::Size + extraSpaceSize;
    void* ptr = NYT::NDetail::AllocateOrCrash<NYT::NDetail::TConstructHelper<T>::Alignment>(totalSize);
    return NYT::NDetail::SafeConstruct<T>(ptr, std::forward<As>(args)...);
}

template <class T, class... As, class>
Y_FORCE_INLINE TIntrusivePtr<T> TryNewWithExtraSpace(
    typename T::TAllocator* allocator,
    size_t extraSpaceSize,
    As&&... args)
{
    auto totalSize = NYT::NDetail::TConstructHelper<T>::Size + extraSpaceSize;
    auto* ptr = allocator->Allocate(totalSize);
    if (Y_UNLIKELY(!ptr)) {
        return nullptr;
    }
    return NYT::NDetail::SafeConstruct<T>(ptr, std::forward<As>(args)...);
}

template <class T, class... As, class>
Y_FORCE_INLINE TIntrusivePtr<T> NewWithExtraSpace(
    typename T::TAllocator* allocator,
    size_t extraSpaceSize,
    As&&... args)
{
    auto obj = TryNewWithExtraSpace<T>(allocator, extraSpaceSize, std::forward<As>(args)...);
    if (Y_UNLIKELY(!obj)) {
        NYT::NDetail::AbortOnOom();
    }
    return obj;
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class TDeleter, class... As>
Y_FORCE_INLINE TIntrusivePtr<T> NewWithDeleter(TDeleter deleter, As&&... args)
{
    using TWrapper = TRefCountedWrapperWithDeleter<T, TDeleter>;
    void* ptr = NYT::NDetail::AllocateConstSizeAlignedOrCrash<sizeof(TWrapper), alignof(TWrapper)>();
    auto* instance = NYT::NDetail::NewEpilogue<TWrapper>(
        ptr,
        std::move(deleter),
        std::forward<As>(args)...);
    return TIntrusivePtr<T>(instance, /*addReference*/ false);
}

////////////////////////////////////////////////////////////////////////////////

template <class T, class TTag, int Counter, class... As>
Y_FORCE_INLINE TIntrusivePtr<T> NewWithLocation(
    const TSourceLocation& location,
    As&&... args)
{
    using TWrapper = TRefCountedWrapperWithCookie<T>;
    void* ptr = NYT::NDetail::AllocateConstSizeAlignedOrCrash<sizeof(TWrapper), alignof(TWrapper)>();
    auto* instance = NYT::NDetail::NewEpilogue<TWrapper>(ptr, std::forward<As>(args)...);
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    instance->InitializeTracking(GetRefCountedTypeCookieWithLocation<T, TTag, Counter>(location));
#else
    Y_UNUSED(location);
#endif
    return TIntrusivePtr<T>(instance, /*addReference*/ false);
}

////////////////////////////////////////////////////////////////////////////////

template <class T>
const void* TWithExtraSpace<T>::GetExtraSpacePtr() const
{
    return static_cast<const T*>(this) + 1;
}

template <class T>
void* TWithExtraSpace<T>::GetExtraSpacePtr()
{
    return static_cast<T*>(this) + 1;
}

template <class T>
size_t TWithExtraSpace<T>::GetUsableSpaceSize() const
{
#ifdef _win_
    return 0;
#else
    return malloc_usable_size(const_cast<T*>(static_cast<const T*>(this))) - sizeof(T);
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
