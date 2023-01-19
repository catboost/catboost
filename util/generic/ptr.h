#pragma once

#include "fwd.h"
#include "utility.h"
#include "intrlist.h"
#include "refcount.h"
#include "typetraits.h"
#include "singleton.h"

#include <type_traits>
#include <utility>

#include <util/system/compiler.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>

template <class T, class U>
using TGuardConversion = typename std::enable_if_t<std::is_convertible<U*, T*>::value>;

template <class T>
inline void AssertTypeComplete() {
    // If compiler triggers this error from destructor of your class with
    // smart pointer, then may be you should move the destructor definition
    // to the .cpp file, where type T have full definition.
    //
    // 'delete' called on pointer to incomplete type is
    // undefined behavior (missing destructor call/corrupted memory manager).
    // 'sizeof' is used to trigger compile-time error.
    static_assert(sizeof(T) != 0, "Type must be complete");
}

template <class T>
inline void CheckedDelete(T* t) {
    AssertTypeComplete<T>();

    delete t;
}

template <class T>
inline void CheckedArrayDelete(T* t) {
    AssertTypeComplete<T>();

    delete[] t;
}

class TNoAction {
public:
    template <class T>
    static inline void Destroy(T*) noexcept {
    }
};

class TDelete {
public:
    template <class T>
    static inline void Destroy(T* t) noexcept {
        CheckedDelete(t);
    }

    /*
     * special handling for nullptr - call nothing
     */
    static inline void Destroy(std::nullptr_t) noexcept {
    }

    /*
     * special handling for void* - call ::operator delete()
     */
    static void Destroy(void* t) noexcept;
};

class TDeleteArray {
public:
    template <class T>
    static inline void Destroy(T* t) noexcept {
        CheckedArrayDelete(t);
    }
};

class TDestructor {
public:
    template <class T>
    static inline void Destroy(T* t) noexcept {
        (void)t;
        t->~T();
    }
};

class TFree {
public:
    template <class T>
    static inline void Destroy(T* t) noexcept {
        DoDestroy((void*)t);
    }

private:
    /*
     * we do not want dependancy on cstdlib here...
     */
    static void DoDestroy(void* t) noexcept;
};

template <class Base, class T>
class TPointerCommon {
public:
    using TValueType = T;

    inline T* operator->() const noexcept {
        T* ptr = AsT();
        Y_ASSERT(ptr);
        return ptr;
    }

#ifndef __cpp_impl_three_way_comparison
    template <class C>
    inline bool operator==(const C& p) const noexcept {
        return (p == AsT());
    }

    template <class C>
    inline bool operator!=(const C& p) const noexcept {
        return (p != AsT());
    }
#endif

    inline explicit operator bool() const noexcept {
        return nullptr != AsT();
    }

protected:
    inline T* AsT() const noexcept {
        return (static_cast<const Base*>(this))->Get();
    }

    static inline T* DoRelease(T*& t) noexcept {
        T* ret = t;
        t = nullptr;
        return ret;
    }
};

template <class Base, class T>
class TPointerBase: public TPointerCommon<Base, T> {
public:
    inline T& operator*() const noexcept {
        Y_ASSERT(this->AsT());

        return *(this->AsT());
    }

    inline T& operator[](size_t n) const noexcept {
        Y_ASSERT(this->AsT());

        return (this->AsT())[n];
    }
};

/*
 * void*-like pointers does not have operator*
 */
template <class Base>
class TPointerBase<Base, void>: public TPointerCommon<Base, void> {
};

template <class T, class D>
class TAutoPtr: public TPointerBase<TAutoPtr<T, D>, T> {
public:
    inline TAutoPtr(T* t = nullptr) noexcept
        : T_(t)
    {
    }

    inline TAutoPtr(const TAutoPtr& t) noexcept
        : T_(t.Release())
    {
    }

    inline ~TAutoPtr() {
        DoDestroy();
    }

    inline TAutoPtr& operator=(const TAutoPtr& t) noexcept {
        if (this != &t) {
            Reset(t.Release());
        }

        return *this;
    }

    inline T* Release() const noexcept Y_WARN_UNUSED_RESULT {
        return this->DoRelease(T_);
    }

    Y_REINITIALIZES_OBJECT inline void Reset(T* t) noexcept {
        if (T_ != t) {
            DoDestroy();
            T_ = t;
        }
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        Destroy();
    }

    inline void Destroy() noexcept {
        Reset(nullptr);
    }

    inline void Swap(TAutoPtr& r) noexcept {
        DoSwap(T_, r.T_);
    }

    inline T* Get() const noexcept {
        return T_;
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    inline void DoDestroy() noexcept {
        if (T_) {
            D::Destroy(T_);
        }
    }

private:
    mutable T* T_;
};

template <class T, class D>
class THolder: public TPointerBase<THolder<T, D>, T> {
public:
    constexpr THolder() noexcept
        : T_(nullptr)
    {
    }

    constexpr THolder(std::nullptr_t) noexcept
        : T_(nullptr)
    {
    }

    explicit THolder(T* t) noexcept
        : T_(t)
    {
    }

    inline THolder(TAutoPtr<T, D> t) noexcept
        : T_(t.Release())
    {
    }

    template <class U, class = TGuardConversion<T, U>>
    inline THolder(TAutoPtr<U, D> t) noexcept
        : T_(t.Release())
    {
    }

    inline THolder(THolder&& that) noexcept
        : T_(that.Release())
    {
    }

    template <class U, class = TGuardConversion<T, U>>
    inline THolder(THolder<U, D>&& that) noexcept
        : T_(that.Release())
    {
    }

    THolder(const THolder&) = delete;
    THolder& operator=(const THolder&) = delete;

    inline ~THolder() {
        DoDestroy();
    }

    inline void Destroy() noexcept {
        Reset(nullptr);
    }

    inline T* Release() noexcept Y_WARN_UNUSED_RESULT {
        return this->DoRelease(T_);
    }

    Y_REINITIALIZES_OBJECT inline void Reset(T* t) noexcept {
        if (T_ != t) {
            DoDestroy();
            T_ = t;
        }
    }

    Y_REINITIALIZES_OBJECT inline void Reset(TAutoPtr<T, D> t) noexcept {
        Reset(t.Release());
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        Destroy();
    }

    inline void Swap(THolder& r) noexcept {
        DoSwap(T_, r.T_);
    }

    inline T* Get() const noexcept {
        return T_;
    }

    inline operator TAutoPtr<T, D>() noexcept {
        return Release();
    }

    THolder& operator=(std::nullptr_t) noexcept {
        this->Reset(nullptr);
        return *this;
    }

    THolder& operator=(THolder&& that) noexcept {
        this->Reset(that.Release());
        return *this;
    }

    template <class U>
    THolder& operator=(THolder<U, D>&& that) noexcept {
        this->Reset(that.Release());
        return *this;
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    inline void DoDestroy() noexcept {
        if (T_) {
            D::Destroy(T_);
        }
    }

private:
    T* T_;
};

template <typename T, typename... Args>
[[nodiscard]] THolder<T> MakeHolder(Args&&... args) {
    return THolder<T>(new T(std::forward<Args>(args)...));
}

/*
 * usage:
 * class T: public TRefCounted<T>
 * and we get methods Ref() && UnRef() with
 * proper destruction of last UnRef()
 */
template <class T, class C, class D>
class TRefCounted {
public:
    inline TRefCounted(long initval = 0) noexcept
        : Counter_(initval)
    {
    }

    inline ~TRefCounted() = default;

    inline void Ref(intptr_t d) noexcept {
        auto resultCount = Counter_.Add(d);
        Y_ASSERT(resultCount >= d);
        (void)resultCount;
    }

    inline void Ref() noexcept {
        auto resultCount = Counter_.Inc();
        Y_ASSERT(resultCount != 0);
        (void)resultCount;
    }

    inline void UnRef(intptr_t d) noexcept {
        auto resultCount = Counter_.Sub(d);
        Y_ASSERT(resultCount >= 0);
        if (resultCount == 0) {
            D::Destroy(static_cast<T*>(this));
        }
    }

    inline void UnRef() noexcept {
        UnRef(1);
    }

    inline intptr_t RefCount() const noexcept {
        return Counter_.Val();
    }

    inline void DecRef() noexcept {
        auto resultCount = Counter_.Dec();
        Y_ASSERT(resultCount >= 0);
        (void)resultCount;
    }

    TRefCounted(const TRefCounted&)
        : Counter_(0)
    {
    }

    void operator=(const TRefCounted&) {
    }

private:
    C Counter_;
};

/**
 * Atomically reference-counted base with a virtual destructor.
 *
 * @note Plays well with inheritance, should be used for refcounted base classes.
 */
struct TThrRefBase: public TRefCounted<TThrRefBase, TAtomicCounter> {
    virtual ~TThrRefBase();
};

/**
 * Atomically reference-counted base.
 *
 * Deletes refcounted object as type T.
 *
 * @warning Additional care should be taken with regard to inheritance.  If used
 * as a base class, @p T should either declare a virtual destructor, or be
 * derived from @p TThrRefBase instead. Otherwise, only destructor of class @p T
 * would be called, potentially slicing the object and creating memory leaks.
 *
 * @note To avoid accidental inheritance when it is not originally intended,
 * class @p T should be marked as final.
 */
template <class T, class D = TDelete>
using TAtomicRefCount = TRefCounted<T, TAtomicCounter, D>;

/**
 * Non-atomically reference-counted base.
 *
 * @warning Not thread-safe. Use with great care. If in doubt, use @p ThrRefBase
 * or @p TAtomicRefCount instead.
 */
template <class T, class D = TDelete>
using TSimpleRefCount = TRefCounted<T, TSimpleCounter, D>;

template <class T>
class TDefaultIntrusivePtrOps {
public:
    static inline void Ref(T* t) noexcept {
        Y_ASSERT(t);

        t->Ref();
    }

    static inline void UnRef(T* t) noexcept {
        Y_ASSERT(t);

        t->UnRef();
    }

    static inline void DecRef(T* t) noexcept {
        Y_ASSERT(t);

        t->DecRef();
    }

    static inline long RefCount(const T* t) noexcept {
        Y_ASSERT(t);

        return t->RefCount();
    }
};

template <class T, class Ops>
class TIntrusivePtr: public TPointerBase<TIntrusivePtr<T, Ops>, T> {
    template <class U, class O>
    friend class TIntrusivePtr;

    template <class U, class O>
    friend class TIntrusiveConstPtr;

public:
    struct TNoIncrement {
    };

    inline TIntrusivePtr(T* t = nullptr) noexcept
        : T_(t)
    {
        Ops();
        Ref();
    }

    inline TIntrusivePtr(T* t, TNoIncrement) noexcept
        : T_(t)
    {
        Ops();
    }

    inline ~TIntrusivePtr() {
        UnRef();
    }

    inline TIntrusivePtr(const TIntrusivePtr& p) noexcept
        : T_(p.T_)
    {
        Ref();
    }

    // NOTE:
    // without std::enable_if_t compiler sometimes tries to use this constructor inappropriately
    // e.g.
    //     struct A {};
    //     struct B {};
    //     void Func(TIntrusivePtr<A>);
    //     void Func(TIntrusivePtr<B>);
    //     ...
    //     Func(TIntrusivePtr<A>(new A)); // <--- compiler can't decide which version of Func to use
    template <class U, class = TGuardConversion<T, U>>
    inline TIntrusivePtr(const TIntrusivePtr<U>& p) noexcept
        : T_(p.Get())
    {
        Ref();
    }

    template <class U, class = TGuardConversion<T, U>>
    inline TIntrusivePtr(TIntrusivePtr<U>&& p) noexcept
        : T_(p.T_)
    {
        p.T_ = nullptr;
    }

    inline TIntrusivePtr(TIntrusivePtr&& p) noexcept
        : T_(nullptr)
    {
        Swap(p);
    }

    inline TIntrusivePtr& operator=(TIntrusivePtr p) noexcept {
        p.Swap(*this);

        return *this;
    }

    // Effectively replace both:
    // Reset(const TIntrusivePtr&)
    // Reset(TIntrusivePtr&&)
    Y_REINITIALIZES_OBJECT inline void Reset(TIntrusivePtr t) noexcept {
        Swap(t);
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        Drop();
    }

    inline T* Get() const noexcept {
        return T_;
    }

    inline void Swap(TIntrusivePtr& r) noexcept {
        DoSwap(T_, r.T_);
    }

    inline void Drop() noexcept {
        TIntrusivePtr(nullptr).Swap(*this);
    }

    inline T* Release() const noexcept Y_WARN_UNUSED_RESULT {
        T* res = T_;
        if (T_) {
            Ops::DecRef(T_);
            T_ = nullptr;
        }
        return res;
    }

    inline long RefCount() const noexcept {
        return T_ ? Ops::RefCount(T_) : 0;
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    inline void Ref() noexcept {
        if (T_) {
            Ops::Ref(T_);
        }
    }

    inline void UnRef() noexcept {
        if (T_) {
            Ops::UnRef(T_);
        }
    }

private:
    mutable T* T_;
};

template <class T, class Ops>
struct THash<TIntrusivePtr<T, Ops>>: THash<const T*> {
    using THash<const T*>::operator();
    inline size_t operator()(const TIntrusivePtr<T, Ops>& ptr) const {
        return THash<const T*>::operator()(ptr.Get());
    }
};

// Behaves like TIntrusivePtr but returns const T* to prevent user from accidentally modifying the referenced object.
template <class T, class Ops>
class TIntrusiveConstPtr: public TPointerBase<TIntrusiveConstPtr<T, Ops>, const T> {
public:
    inline TIntrusiveConstPtr(T* t = nullptr) noexcept // we need a non-const pointer to Ref(), UnRef() and eventually delete it.
        : T_(t)
    {
        Ops();
        Ref();
    }

    inline ~TIntrusiveConstPtr() {
        UnRef();
    }

    inline TIntrusiveConstPtr(const TIntrusiveConstPtr& p) noexcept
        : T_(p.T_)
    {
        Ref();
    }

    inline TIntrusiveConstPtr(TIntrusiveConstPtr&& p) noexcept
        : T_(nullptr)
    {
        Swap(p);
    }

    inline TIntrusiveConstPtr(TIntrusivePtr<T> p) noexcept
        : T_(p.T_)
    {
        p.T_ = nullptr;
    }

    template <class U, class = TGuardConversion<T, U>>
    inline TIntrusiveConstPtr(const TIntrusiveConstPtr<U>& p) noexcept
        : T_(p.T_)
    {
        Ref();
    }

    template <class U, class = TGuardConversion<T, U>>
    inline TIntrusiveConstPtr(TIntrusiveConstPtr<U>&& p) noexcept
        : T_(p.T_)
    {
        p.T_ = nullptr;
    }

    inline TIntrusiveConstPtr& operator=(TIntrusiveConstPtr p) noexcept {
        p.Swap(*this);

        return *this;
    }

    // Effectively replace both:
    // Reset(const TIntrusiveConstPtr&)
    // Reset(TIntrusiveConstPtr&&)
    Y_REINITIALIZES_OBJECT inline void Reset(TIntrusiveConstPtr t) noexcept {
        Swap(t);
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        Drop();
    }

    inline const T* Get() const noexcept {
        return T_;
    }

    inline void Swap(TIntrusiveConstPtr& r) noexcept {
        DoSwap(T_, r.T_);
    }

    inline void Drop() noexcept {
        TIntrusiveConstPtr(nullptr).Swap(*this);
    }

    inline long RefCount() const noexcept {
        return T_ ? Ops::RefCount(T_) : 0;
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    inline void Ref() noexcept {
        if (T_ != nullptr) {
            Ops::Ref(T_);
        }
    }

    inline void UnRef() noexcept {
        if (T_ != nullptr) {
            Ops::UnRef(T_);
        }
    }

private:
    T* T_;

    template <class U, class O>
    friend class TIntrusiveConstPtr;
};

template <class T, class Ops>
struct THash<TIntrusiveConstPtr<T, Ops>>: THash<const T*> {
    using THash<const T*>::operator();
    inline size_t operator()(const TIntrusiveConstPtr<T, Ops>& ptr) const {
        return THash<const T*>::operator()(ptr.Get());
    }
};

template <class T, class Ops>
class TSimpleIntrusiveOps {
    using TFunc = void (*)(T*)
#if __cplusplus >= 201703
        noexcept
#endif
        ;

    static void DoRef(T* t) noexcept {
        Ops::Ref(t);
    }

    static void DoUnRef(T* t) noexcept {
        Ops::UnRef(t);
    }

public:
    inline TSimpleIntrusiveOps() noexcept {
        InitStaticOps();
    }

    inline ~TSimpleIntrusiveOps() = default;

    static inline void Ref(T* t) noexcept {
        Ref_(t);
    }

    static inline void UnRef(T* t) noexcept {
        UnRef_(t);
    }

private:
    static inline void InitStaticOps() noexcept {
        struct TInit {
            inline TInit() noexcept {
                Ref_ = DoRef;
                UnRef_ = DoUnRef;
            }
        };

        Singleton<TInit>();
    }

private:
    static TFunc Ref_;
    static TFunc UnRef_;
};

template <class T, class Ops>
typename TSimpleIntrusiveOps<T, Ops>::TFunc TSimpleIntrusiveOps<T, Ops>::Ref_ = nullptr;

template <class T, class Ops>
typename TSimpleIntrusiveOps<T, Ops>::TFunc TSimpleIntrusiveOps<T, Ops>::UnRef_ = nullptr;

template <typename T, class Ops = TDefaultIntrusivePtrOps<T>, typename... Args>
[[nodiscard]] TIntrusivePtr<T, Ops> MakeIntrusive(Args&&... args) {
    return new T{std::forward<Args>(args)...};
}

template <typename T, class Ops = TDefaultIntrusivePtrOps<T>, typename... Args>
[[nodiscard]] TIntrusiveConstPtr<T, Ops> MakeIntrusiveConst(Args&&... args) {
    return new T{std::forward<Args>(args)...};
}

template <class T, class C, class D>
class TSharedPtr: public TPointerBase<TSharedPtr<T, C, D>, T> {
    template <class TT, class CC, class DD>
    friend class TSharedPtr;

public:
    inline TSharedPtr() noexcept
        : T_(nullptr)
        , C_(nullptr)
    {
    }

    inline TSharedPtr(T* t) {
        THolder<T, D> h(t);

        Init(h);
    }

    inline TSharedPtr(TAutoPtr<T, D> t) {
        Init(t);
    }

    inline TSharedPtr(T* t, C* c) noexcept
        : T_(t)
        , C_(c)
    {
    }

    template <class TT, class = TGuardConversion<T, TT>>
    inline TSharedPtr(THolder<TT>&& t) {
        Init(t);
    }

    inline ~TSharedPtr() {
        UnRef();
    }

    inline TSharedPtr(const TSharedPtr& t) noexcept
        : T_(t.T_)
        , C_(t.C_)
    {
        Ref();
    }

    inline TSharedPtr(TSharedPtr&& t) noexcept
        : T_(nullptr)
        , C_(nullptr)
    {
        Swap(t);
    }

    template <class TT, class = TGuardConversion<T, TT>>
    inline TSharedPtr(const TSharedPtr<TT, C, D>& t) noexcept
        : T_(t.T_)
        , C_(t.C_)
    {
        Ref();
    }

    template <class TT, class = TGuardConversion<T, TT>>
    inline TSharedPtr(TSharedPtr<TT, C, D>&& t) noexcept
        : T_(t.T_)
        , C_(t.C_)
    {
        t.T_ = nullptr;
        t.C_ = nullptr;
    }

    inline TSharedPtr& operator=(TSharedPtr t) noexcept {
        t.Swap(*this);

        return *this;
    }

    // Effectively replace both:
    // Reset(const TSharedPtr& t)
    // Reset(TSharedPtr&& t)
    Y_REINITIALIZES_OBJECT inline void Reset(TSharedPtr t) noexcept {
        Swap(t);
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        Drop();
    }

    inline void Drop() noexcept {
        TSharedPtr().Swap(*this);
    }

    inline T* Get() const noexcept {
        return T_;
    }

    inline C* ReferenceCounter() const noexcept {
        return C_;
    }

    inline void Swap(TSharedPtr& r) noexcept {
        DoSwap(T_, r.T_);
        DoSwap(C_, r.C_);
    }

    inline long RefCount() const noexcept {
        return C_ ? C_->Val() : 0;
    }

    template <class TT>
    [[nodiscard]] inline TSharedPtr<TT, C, D> As() & noexcept {
        static_assert(std::has_virtual_destructor<TT>(), "Type should have a virtual dtor");
        static_assert(std::is_base_of<T, TT>(), "When downcasting from T to TT, T should be a parent of TT");
        if (const auto ttPtr = dynamic_cast<TT*>(T_)) {
            TSharedPtr<TT, C, D> ttSharedPtr(ttPtr, C_);
            ttSharedPtr.Ref();
            return ttSharedPtr;
        } else {
            return TSharedPtr<TT, C, D>{};
        }
    }

    template <class TT>
    [[nodiscard]] inline TSharedPtr<TT, C, D> As() && noexcept {
        static_assert(std::has_virtual_destructor<TT>(), "Type should have a virtual dtor");
        static_assert(std::is_base_of<T, TT>(), "When downcasting from T to TT, T should be a parent of TT");
        if (const auto ttPtr = dynamic_cast<TT*>(T_)) {
            TSharedPtr<TT, C, D> ttSharedPtr(ttPtr, C_);
            T_ = nullptr;
            C_ = nullptr;
            return ttSharedPtr;
        } else {
            return TSharedPtr<TT, C, D>{};
        }
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    template <class X>
    inline void Init(X& t) {
        C_ = !!t ? new C(1) : nullptr;
        T_ = t.Release();
    }

    inline void Ref() noexcept {
        if (C_) {
            C_->Inc();
        }
    }

    inline void UnRef() noexcept {
        if (C_ && !C_->Dec()) {
            DoDestroy();
        }
    }

    inline void DoDestroy() noexcept {
        if (T_) {
            D::Destroy(T_);
        }

        delete C_;
    }

private:
    T* T_;
    C* C_;
};

template <class T, class C, class D>
struct THash<TSharedPtr<T, C, D>>: THash<const T*> {
    using THash<const T*>::operator();
    inline size_t operator()(const TSharedPtr<T, C, D>& ptr) const {
        return THash<const T*>::operator()(ptr.Get());
    }
};

template <class T, class D = TDelete>
using TAtomicSharedPtr = TSharedPtr<T, TAtomicCounter, D>;

// use with great care. if in doubt, use TAtomicSharedPtr instead
template <class T, class D = TDelete>
using TSimpleSharedPtr = TSharedPtr<T, TSimpleCounter, D>;

template <typename T, typename C, typename... Args>
[[nodiscard]] TSharedPtr<T, C> MakeShared(Args&&... args) {
    return new T{std::forward<Args>(args)...};
}

template <typename T, typename... Args>
[[nodiscard]] inline TAtomicSharedPtr<T> MakeAtomicShared(Args&&... args) {
    return MakeShared<T, TAtomicCounter>(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
[[nodiscard]] inline TSimpleSharedPtr<T> MakeSimpleShared(Args&&... args) {
    return MakeShared<T, TSimpleCounter>(std::forward<Args>(args)...);
}

class TCopyClone {
public:
    template <class T>
    static inline T* Copy(T* t) {
        if (t)
            return t->Clone();
        return nullptr;
    }
};

class TCopyNew {
public:
    template <class T>
    static inline T* Copy(T* t) {
        if (t)
            return new T(*t);
        return nullptr;
    }
};

template <class T, class C, class D>
class TCopyPtr: public TPointerBase<TCopyPtr<T, C, D>, T> {
public:
    inline TCopyPtr(T* t = nullptr) noexcept
        : T_(t)
    {
    }

    inline TCopyPtr(const TCopyPtr& t)
        : T_(C::Copy(t.Get()))
    {
    }

    inline TCopyPtr(TCopyPtr&& t) noexcept
        : T_(nullptr)
    {
        Swap(t);
    }

    inline ~TCopyPtr() {
        DoDestroy();
    }

    inline TCopyPtr& operator=(TCopyPtr t) noexcept {
        t.Swap(*this);

        return *this;
    }

    inline T* Release() noexcept Y_WARN_UNUSED_RESULT {
        return DoRelease(T_);
    }

    Y_REINITIALIZES_OBJECT inline void Reset(T* t) noexcept {
        if (T_ != t) {
            DoDestroy();
            T_ = t;
        }
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        Destroy();
    }

    inline void Destroy() noexcept {
        Reset(nullptr);
    }

    inline void Swap(TCopyPtr& r) noexcept {
        DoSwap(T_, r.T_);
    }

    inline T* Get() const noexcept {
        return T_;
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    inline void DoDestroy() noexcept {
        if (T_)
            D::Destroy(T_);
    }

private:
    T* T_;
};

// Copy-on-write pointer
template <class TPtr, class TCopy>
class TCowPtr: public TPointerBase<TCowPtr<TPtr, TCopy>, const typename TPtr::TValueType> {
    using T = typename TPtr::TValueType;

public:
    inline TCowPtr() = default;

    inline TCowPtr(const TPtr& p)
        : T_(p)
    {
    }

    inline TCowPtr(T* p)
        : T_(p)
    {
    }

    inline const T* Get() const noexcept {
        return Const();
    }

    inline const T* Const() const noexcept {
        return T_.Get();
    }

    inline T* Mutable() {
        Unshare();

        return T_.Get();
    }

    inline bool Shared() const noexcept {
        return T_.RefCount() > 1;
    }

    inline void Swap(TCowPtr& r) noexcept {
        T_.Swap(r.T_);
    }

    Y_REINITIALIZES_OBJECT inline void Reset(TCowPtr p) {
        p.Swap(*this);
    }

    Y_REINITIALIZES_OBJECT inline void Reset() {
        T_.Reset();
    }

#ifdef __cpp_impl_three_way_comparison
    template <class Other>
    inline bool operator==(const Other& p) const noexcept {
        return (p == Get());
    }
#endif
private:
    inline void Unshare() {
        if (Shared()) {
            Reset(TCopy::Copy(T_.Get()));
        }
    }

private:
    TPtr T_;
};

// saves .Get() on argument passing. Intended usage: Func(TPtrArg<X> p); ... TIntrusivePtr<X> p2;  Func(p2);
template <class T>
class TPtrArg {
    T* Ptr;

public:
    TPtrArg(T* p)
        : Ptr(p)
    {
    }
    TPtrArg(const TIntrusivePtr<T>& p)
        : Ptr(p.Get())
    {
    }
    operator T*() const {
        return Ptr;
    }
    T* operator->() const {
        return Ptr;
    }
    T* Get() const {
        return Ptr;
    }
};
