#pragma once

#include "intrusive_ptr.h"

#include <library/cpp/yt/misc/source_location.h>

#include <util/system/defaults.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

/*!
 * \defgroup yt_new New<T> safe smart pointer constructors
 * \ingroup yt_new
 *
 * This is collection of safe smart pointer constructors.
 *
 * \page yt_new_rationale Rationale
 * New<T> function family was designed to prevent the following problem.
 * Consider the following piece of code.
 *
 * \code
 *     class TFoo
 *         : public virtual TRefCounted
 *     {
 *     public:
 *         TFoo();
 *     };
 *
 *     using TFooPtr = TIntrusivePtr<TFoo>;
 *
 *     void RegisterObject(TFooPtr foo)
 *     {
 *         ...
 *     }
 *
 *     TFoo::TFoo()
 *     {
 *         // ... do something before
 *         RegisterObject(this);
 *         // ... do something after
 *     }
 * \endcode
 *
 * What will happen on <tt>new TFoo()</tt> construction? After memory allocation
 * the reference counter for newly created instance would be initialized to zero.
 * Afterwards, the control goes to TFoo constructor. To invoke
 * <tt>RegisterObject</tt> a new temporary smart pointer to the current instance
 * have to be created effectively incrementing the reference counter (now one).
 * After <tt>RegisterObject</tt> returns the control to the constructor
 * the temporary pointer is destroyed effectively decrementing the reference
 * counter to zero hence triggering object destruction during its initialization.
 *
 * To avoid this undefined behavior <tt>New<T></tt> was introduced.
 * <tt>New<T></tt> holds a fake
 * reference to the object during its construction effectively preventing
 * premature destruction.
 *
 * \note An initialization like <tt>TIntrusivePtr&lt;T&gt; p = new T()</tt>
 * would result in a dangling reference due to internals of #New<T> and
 * #TRefCountedBase.
 */

////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct THasAllocator
{
    using TFalse = void;
};

template <class T>
struct THasAllocator<T, std::void_t<typename T::TAllocator>>
{
    using TTrue = void;
};

////////////////////////////////////////////////////////////////////////////////

//! Allocates a new instance of |T| using the standard allocator.
//! Aborts the process on out-of-memory condition.
template <class T, class... As, class = typename THasAllocator<T>::TFalse>
TIntrusivePtr<T> New(As&&... args);

//! Allocates a new instance of |T| using a custom #allocator.
//! Returns null on allocation failure.
template <class T, class... As, class = typename THasAllocator<T>::TTrue>
TIntrusivePtr<T> TryNew(typename T::TAllocator* allocator, As&&... args);

//! Same as #TryNewWit but aborts on allocation failure.
template <class T, class... As, class = typename THasAllocator<T>::TTrue>
TIntrusivePtr<T> New(typename T::TAllocator* allocator, As&&... args);

//! Allocates an instance of |T|
//! Aborts the process on out-of-memory condition.
template <class T, class... As, class = typename THasAllocator<T>::TFalse>
TIntrusivePtr<T> NewWithExtraSpace(size_t extraSpaceSize, As&&... args);

//! Allocates a new instance of |T| with additional storage of #extraSpaceSize bytes
//! using a custom #allocator.
//! Returns null on allocation failure.
template <class T, class... As, class = typename THasAllocator<T>::TTrue>
TIntrusivePtr<T> TryNewWithExtraSpace(typename T::TAllocator* allocator, size_t extraSpaceSize, As&&... args);

//! Same as #TryNewWithExtraSpace but aborts on allocation failure.
template <class T, class... As, class = typename THasAllocator<T>::TTrue>
TIntrusivePtr<T> NewWithExtraSpace(typename T::TAllocator* allocator, size_t extraSpaceSize, As&&... args);

//! Allocates a new instance of |T| with a custom #deleter.
//! Aborts the process on out-of-memory condition.
template <class T, class TDeleter, class... As>
TIntrusivePtr<T> NewWithDeleter(TDeleter deleter, As&&... args);

//! Allocates a new instance of |T|.
//! The allocation is additionally marked with #location.
//! Aborts the process on out-of-memory condition.
template <class T, class TTag, int Counter, class... As>
TIntrusivePtr<T> NewWithLocation(const TSourceLocation& location, As&&... args);

//! Enables calling #New and co for types with private ctors.
#define DECLARE_NEW_FRIEND() \
    template <class DECLARE_NEW_FRIEND_T> \
    friend struct NYT::TRefCountedWrapper;

////////////////////////////////////////////////////////////////////////////////

//! CRTP mixin enabling access to instance's extra space.
template <class T>
class TWithExtraSpace
{
protected:
    const void* GetExtraSpacePtr() const;
    void* GetExtraSpacePtr();
    size_t GetUsableSpaceSize() const;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define NEW_INL_H_
#include "new-inl.h"
#undef NEW_INL_H_
