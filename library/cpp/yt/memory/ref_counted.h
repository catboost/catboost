#pragma once

#include <library/cpp/yt/misc/port.h>

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/ytalloc/api/ytalloc.h>

#include <atomic>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A technical base class for ref-counted objects and promise states.
class TRefCountedBase
{
public:
    TRefCountedBase() = default;

    // Make destructor protected
    virtual ~TRefCountedBase() noexcept = default;

    virtual void DestroyRefCounted() = 0;

private:
    TRefCountedBase(const TRefCountedBase&) = delete;
    TRefCountedBase(TRefCountedBase&&) = delete;

    TRefCountedBase& operator=(const TRefCountedBase&) = delete;
    TRefCountedBase& operator=(TRefCountedBase&&) = delete;
};

////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct TFreeMemory
{
    static void Do(void* ptr)
    {
        NYTAlloc::FreeNonNull(ptr);
    }
};

template <class T>
struct TFreeMemory<T, std::void_t<typename T::TAllocator>>
{
    static void Do(void* ptr)
    {
        using TAllocator = typename T::TAllocator;
        TAllocator::Free(ptr);
    }
};

////////////////////////////////////////////////////////////////////////////////

class TRefCounter
{
public:
    //! Returns current number of strong references to the object.
    /*!
     * Note that you should never ever use this method in production code.
     * This method is mainly for debugging purposes.
     */
    int GetRefCount() const noexcept;

    //! Increments the strong reference counter.
    void Ref() const noexcept;

    //! Increments the strong reference counter if it is not null.
    bool TryRef() const noexcept;

    //! Decrements the strong reference counter.
    bool Unref() const;

    //! Returns current number of weak references to the object.
    int GetWeakRefCount() const noexcept;

    //! Increments the weak reference counter.
    void WeakRef() const noexcept;

    //! Decrements the weak reference counter.
    bool WeakUnref() const;

private:
    mutable std::atomic<int> StrongCount_ = 1;
    mutable std::atomic<int> WeakCount_ = 1;
};

////////////////////////////////////////////////////////////////////////////////

template <class T>
const TRefCounter* GetRefCounter(const T* obj);

template <class T>
void DestroyRefCounted(const T* obj);

template <class T>
void DeallocateRefCounted(const T* obj);

////////////////////////////////////////////////////////////////////////////////

// API

template <class T>
void Ref(T* obj);

template <class T>
void Unref(T* obj);

////////////////////////////////////////////////////////////////////////////////

struct TRefCounted
    : public TRefCountedBase
    , public TRefCounter
{
    void Unref() const;

    void WeakUnref() const;

    template <class T>
    static void DestroyRefCountedImpl(T* ptr);
};

////////////////////////////////////////////////////////////////////////////////

// Forward declaration.
template <class T>
class TIntrusivePtr;

using TRefCountedPtr = TIntrusivePtr<TRefCounted>;

// A bunch of helpful macros that enable working with intrusive pointers to incomplete types.
/*
 *  Typically when you have a forward-declared type |T| and an instance
 *  of |TIntrusivePtr<T>| you need the complete definition of |T| to work with
 *  the pointer even if you're not actually using the members of |T|.
 *  E.g. the dtor of |TIntrusivePtr<T>|, should you ever need it, must be able
 *  to unref an instance of |T| and eventually destroy it.
 *  This may force #inclusion of way more headers than really seems necessary.
 *
 *  |DECLARE_REFCOUNTED_STRUCT|, |DECLARE_REFCOUNTED_CLASS|, and |DEFINE_REFCOUNTED_TYPE|
 *  alleviate this issue by forcing TIntrusivePtr to work with the free-standing overloads
 *  of |Ref| and |Unref| instead of their template version.
 *  These overloads are declared together with the forward declaration of |T| and
 *  are subsequently defined afterwards.
 */

#define DECLARE_REFCOUNTED_TYPE(type) \
    typedef ::NYT::TIntrusivePtr<type> type ## Ptr; \
    \
    [[maybe_unused]] const ::NYT::TRefCounter* GetRefCounter(const type* obj) ATTRIBUTE_USED; \
    [[maybe_unused]] void DestroyRefCounted(const type* obj) ATTRIBUTE_USED; \
    [[maybe_unused]] void DeallocateRefCounted(const type* obj) ATTRIBUTE_USED;

//! Forward-declares a class type, defines an intrusive pointer for it, and finally
//! declares Ref/Unref overloads. Use this macro in |public.h|-like files.
#define DECLARE_REFCOUNTED_CLASS(type) \
    class type; \
    DECLARE_REFCOUNTED_TYPE(type)

//! Forward-declares a struct type, defines an intrusive pointer for it, and finally
//! declares Ref/Unref overloads. Use this macro in |public.h|-like files.
#define DECLARE_REFCOUNTED_STRUCT(type) \
    struct type; \
    DECLARE_REFCOUNTED_TYPE(type)

//! Provides implementations for Ref/Unref overloads. Use this macro right
//! after the type's full definition.
#define DEFINE_REFCOUNTED_TYPE(type) \
    [[maybe_unused]] Y_FORCE_INLINE const ::NYT::TRefCounter* GetRefCounter(const type* obj) \
    { \
        return ::NYT::TRefCountedHelper<type>::GetRefCounter(obj); \
    } \
    [[maybe_unused]] Y_FORCE_INLINE void DestroyRefCounted(const type* obj) \
    { \
        ::NYT::TRefCountedHelper<type>::Destroy(obj); \
    } \
    [[maybe_unused]] Y_FORCE_INLINE void DeallocateRefCounted(const type* obj) \
    { \
        ::NYT::TRefCountedHelper<type>::Deallocate(obj); \
    }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define REF_COUNTED_INL_H_
#include "ref_counted-inl.h"
#undef REF_COUNTED_INL_H_
