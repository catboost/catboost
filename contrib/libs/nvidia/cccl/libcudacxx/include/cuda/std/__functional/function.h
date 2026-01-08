// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_FUNCTION_H
#define _LIBCUDACXX___FUNCTIONAL_FUNCTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/unary_function.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/allocator_arg_t.h>
#include <cuda/std/__memory/allocator_destructor.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__memory/builtin_new_allocator.h>
#include <cuda/std/__memory/compressed_pair.h>
#include <cuda/std/__memory/uses_allocator.h>
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/piecewise_construct.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/tuple>

#ifndef __cuda_std__

#  if _CCCL_HAS_EXCEPTIONS()
#    include <function>
#  endif // !_CCCL_HAS_EXCEPTIONS()

#  include <cuda/std/__cccl/prologue.h>

[[noreturn]] _CCCL_API inline void __throw_bad_function_call()
{
#  if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw ::std::bad_function_call();), (_CUDA_VSTD_NOVERSION::terminate();))
#  else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  _CUDA_VSTD_NOVERSION::terminate();
#  endif // !_CCCL_HAS_EXCEPTIONS()
}

template <class _Fp>
class _CCCL_TYPE_VISIBILITY_DEFAULT function; // undefined

namespace __function
{

template <class _Rp>
struct __maybe_derive_from_unary_function
{};

template <class _Rp, class _A1>
struct __maybe_derive_from_unary_function<_Rp(_A1)> : public __unary_function<_A1, _Rp>
{};

template <class _Rp>
struct __maybe_derive_from_binary_function
{};

template <class _Rp, class _A1, class _A2>
struct __maybe_derive_from_binary_function<_Rp(_A1, _A2)> : public __binary_function<_A1, _A2, _Rp>
{};

template <class _Fp>
_CCCL_API inline bool __not_null(_Fp const&)
{
  return true;
}

template <class _Fp>
_CCCL_API inline bool __not_null(_Fp* __ptr)
{
  return __ptr;
}

template <class _Ret, class _Class>
_CCCL_API inline bool __not_null(_Ret _Class::* __ptr)
{
  return __ptr;
}

template <class _Fp>
_CCCL_API inline bool __not_null(function<_Fp> const& __f)
{
  return !!__f;
}

#  ifdef _LIBCUDACXX_HAS_EXTENSION_BLOCKS
template <class _Rp, class... _Args>
_CCCL_API inline bool __not_null(_Rp (^__p)(_Args...))
{
  return __p;
}
#  endif

} // namespace __function

namespace __function
{

// __alloc_func holds a functor and an allocator.

template <class _Fp, class _Ap, class _FB>
class __alloc_func;
template <class _Fp, class _FB>
class __default_alloc_func;

template <class _Fp, class _Ap, class _Rp, class... _ArgTypes>
class __alloc_func<_Fp, _Ap, _Rp(_ArgTypes...)>
{
  __compressed_pair<_Fp, _Ap> __f_;

public:
  using _Target _CCCL_NODEBUG_ALIAS = _Fp;
  using _Alloc _CCCL_NODEBUG_ALIAS  = _Ap;

  _CCCL_API inline const _Target& __target() const
  {
    return __f_.first();
  }

  // WIN32 APIs may define __allocator, so use __get_allocator instead.
  _CCCL_API inline const _Alloc& __get_allocator() const
  {
    return __f_.second();
  }

  _CCCL_API inline explicit __alloc_func(_Target&& __f)
      : __f_(piecewise_construct, _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::move(__f)), _CUDA_VSTD::forward_as_tuple())
  {}

  _CCCL_API inline explicit __alloc_func(const _Target& __f, const _Alloc& __a)
      : __f_(piecewise_construct, _CUDA_VSTD::forward_as_tuple(__f), _CUDA_VSTD::forward_as_tuple(__a))
  {}

  _CCCL_API inline explicit __alloc_func(const _Target& __f, _Alloc&& __a)
      : __f_(piecewise_construct, _CUDA_VSTD::forward_as_tuple(__f), _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::move(__a)))
  {}

  _CCCL_API inline explicit __alloc_func(_Target&& __f, _Alloc&& __a)
      : __f_(piecewise_construct,
             _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::move(__f)),
             _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::move(__a)))
  {}

  _CCCL_API inline _Rp operator()(_ArgTypes&&... __arg)
  {
    using _Invoker = __invoke_void_return_wrapper<_Rp>;
    return _Invoker::__call(__f_.first(), _CUDA_VSTD::forward<_ArgTypes>(__arg)...);
  }

  _CCCL_API inline __alloc_func* __clone() const
  {
    using __alloc_traits = allocator_traits<_Alloc>;
    using _AA            = typename __rebind_alloc_helper<__alloc_traits, __alloc_func>::type;
    _AA __a(__f_.second());
    using _Dp = __allocator_destructor<_AA>;
    unique_ptr<__alloc_func, _Dp> __hold(__a.allocate(1), _Dp(__a, 1));
    ::new ((void*) __hold.get()) __alloc_func(__f_.first(), _Alloc(__a));
    return __hold.release();
  }

  _CCCL_API inline void destroy() noexcept
  {
    __f_.~__compressed_pair<_Target, _Alloc>();
  }

  static void __destroy_and_delete(__alloc_func* __f)
  {
    using __alloc_traits = allocator_traits<_Alloc>;
    using _FunAlloc      = typename __rebind_alloc_helper<__alloc_traits, __alloc_func>::type;
    _FunAlloc __a(__f->__get_allocator());
    __f->destroy();
    __a.deallocate(__f, 1);
  }
};

template <class _Fp, class _Rp, class... _ArgTypes>
class __default_alloc_func<_Fp, _Rp(_ArgTypes...)>
{
  _Fp __f_;

public:
  using _Target _CCCL_NODEBUG_ALIAS = _Fp;

  _CCCL_API inline const _Target& __target() const
  {
    return __f_;
  }

  _CCCL_API inline explicit __default_alloc_func(_Target&& __f)
      : __f_(_CUDA_VSTD::move(__f))
  {}

  _CCCL_API inline explicit __default_alloc_func(const _Target& __f)
      : __f_(__f)
  {}

  _CCCL_API inline _Rp operator()(_ArgTypes&&... __arg)
  {
    using _Invoker = __invoke_void_return_wrapper<_Rp>;
    return _Invoker::__call(__f_, _CUDA_VSTD::forward<_ArgTypes>(__arg)...);
  }

  _CCCL_API inline __default_alloc_func* __clone() const
  {
    __builtin_new_allocator::__holder_t __hold = __builtin_new_allocator::__allocate_type<__default_alloc_func>(1);
    __default_alloc_func* __res                = ::new ((void*) __hold.get()) __default_alloc_func(__f_);
    (void) __hold.release();
    return __res;
  }

  _CCCL_API inline void destroy() noexcept
  {
    __f_.~_Target();
  }

  static void __destroy_and_delete(__default_alloc_func* __f)
  {
    __f->destroy();
    __builtin_new_allocator::__deallocate_type<__default_alloc_func>(__f, 1);
  }
};

// __base provides an abstract interface for copyable functors.

template <class _Fp>
class _CCCL_TYPE_VISIBILITY_DEFAULT __base;

template <class _Rp, class... _ArgTypes>
class __base<_Rp(_ArgTypes...)>
{
  __base(const __base&);
  __base& operator=(const __base&);

public:
  _CCCL_API inline __base() {}
  _CCCL_API inline virtual ~__base() {}
  virtual __base* __clone() const            = 0;
  virtual void __clone(__base*) const        = 0;
  virtual void destroy() noexcept            = 0;
  virtual void destroy_deallocate() noexcept = 0;
  virtual _Rp operator()(_ArgTypes&&...)     = 0;
#  ifndef _CCCL_NO_RTTI
  virtual const void* target(const type_info&) const noexcept = 0;
  virtual const type_info& target_type() const noexcept       = 0;
#  endif // _CCCL_NO_RTTI
};

// __func implements __base for a given functor type.

template <class _FD, class _Alloc, class _FB>
class __func;

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
class __func<_Fp, _Alloc, _Rp(_ArgTypes...)> : public __base<_Rp(_ArgTypes...)>
{
  __alloc_func<_Fp, _Alloc, _Rp(_ArgTypes...)> __f_;

public:
  _CCCL_API inline explicit __func(_Fp&& __f)
      : __f_(_CUDA_VSTD::move(__f))
  {}

  _CCCL_API inline explicit __func(const _Fp& __f, const _Alloc& __a)
      : __f_(__f, __a)
  {}

  _CCCL_API inline explicit __func(const _Fp& __f, _Alloc&& __a)
      : __f_(__f, _CUDA_VSTD::move(__a))
  {}

  _CCCL_API inline explicit __func(_Fp&& __f, _Alloc&& __a)
      : __f_(_CUDA_VSTD::move(__f), _CUDA_VSTD::move(__a))
  {}

  virtual __base<_Rp(_ArgTypes...)>* __clone() const;
  virtual void __clone(__base<_Rp(_ArgTypes...)>*) const;
  virtual void destroy() noexcept;
  virtual void destroy_deallocate() noexcept;
  virtual _Rp operator()(_ArgTypes&&... __arg);
#  ifndef _CCCL_NO_RTTI
  virtual const void* target(const type_info&) const noexcept;
  virtual const type_info& target_type() const noexcept;
#  endif // _CCCL_NO_RTTI
};

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
__base<_Rp(_ArgTypes...)>* __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::__clone() const
{
  using __alloc_traits = allocator_traits<_Alloc>;
  using _Ap            = typename __rebind_alloc_helper<__alloc_traits, __func>::type;
  _Ap __a(__f_.__get_allocator());
  using _Dp = __allocator_destructor<_Ap>;
  unique_ptr<__func, _Dp> __hold(__a.allocate(1), _Dp(__a, 1));
  ::new ((void*) __hold.get()) __func(__f_.__target(), _Alloc(__a));
  return __hold.release();
}

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
void __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::__clone(__base<_Rp(_ArgTypes...)>* __p) const
{
  ::new ((void*) __p) __func(__f_.__target(), __f_.__get_allocator());
}

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
void __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::destroy() noexcept
{
  __f_.destroy();
}

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
void __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::destroy_deallocate() noexcept
{
  using __alloc_traits = allocator_traits<_Alloc>;
  using _Ap            = typename __rebind_alloc_helper<__alloc_traits, __func>::type;
  _Ap __a(__f_.__get_allocator());
  __f_.destroy();
  __a.deallocate(this, 1);
}

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
_Rp __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::operator()(_ArgTypes&&... __arg)
{
  return __f_(_CUDA_VSTD::forward<_ArgTypes>(__arg)...);
}

#  ifndef _CCCL_NO_RTTI

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
const void* __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::target(const type_info& __ti) const noexcept
{
  if (__ti == typeid(_Fp))
  {
    return _CUDA_VSTD::addressof(__f_.__target());
  }
  return nullptr;
}

template <class _Fp, class _Alloc, class _Rp, class... _ArgTypes>
const type_info& __func<_Fp, _Alloc, _Rp(_ArgTypes...)>::target_type() const noexcept
{
  return typeid(_Fp);
}

#  endif // _CCCL_NO_RTTI

// __value_func creates a value-type from a __func.

template <class _Fp>
class __value_func;

template <class _Rp, class... _ArgTypes>
class __value_func<_Rp(_ArgTypes...)>
{
  typename aligned_storage<3 * sizeof(void*)>::type __buf_;

  using __func = __base<_Rp(_ArgTypes...)>;
  __func* __f_;

  _CCCL_NO_CFI static __func* __as_base(void* __p)
  {
    return reinterpret_cast<__func*>(__p);
  }

public:
  _CCCL_API inline __value_func() noexcept
      : __f_(nullptr)
  {}

  template <class _Fp, class _Alloc>
  _CCCL_API inline __value_func(_Fp&& __f, const _Alloc& __a)
      : __f_(nullptr)
  {
    using __alloc_traits = allocator_traits<_Alloc>;
    using _Fun           = __function::__func<_Fp, _Alloc, _Rp(_ArgTypes...)>;
    using _FunAlloc      = typename __rebind_alloc_helper<__alloc_traits, _Fun>::type;

    if (__function::__not_null(__f))
    {
      _FunAlloc __af(__a);
      if (sizeof(_Fun) <= sizeof(__buf_) && is_nothrow_copy_constructible<_Fp>::value
          && is_nothrow_copy_constructible<_FunAlloc>::value)
      {
        __f_ = ::new ((void*) &__buf_) _Fun(_CUDA_VSTD::move(__f), _Alloc(__af));
      }
      else
      {
        using _Dp = __allocator_destructor<_FunAlloc>;
        unique_ptr<__func, _Dp> __hold(__af.allocate(1), _Dp(__af, 1));
        ::new ((void*) __hold.get()) _Fun(_CUDA_VSTD::move(__f), _Alloc(__a));
        __f_ = __hold.release();
      }
    }
  }

  template <class _Fp, class = enable_if_t<!is_same<decay_t<_Fp>, __value_func>::value>>
  _CCCL_API inline explicit __value_func(_Fp&& __f)
      : __value_func(_CUDA_VSTD::forward<_Fp>(__f), allocator<_Fp>())
  {}

  _CCCL_API inline __value_func(const __value_func& __f)
  {
    if (__f.__f_ == nullptr)
    {
      __f_ = nullptr;
    }
    else if ((void*) __f.__f_ == &__f.__buf_)
    {
      __f_ = __as_base(&__buf_);
      __f.__f_->__clone(__f_);
    }
    else
    {
      __f_ = __f.__f_->__clone();
    }
  }

  _CCCL_API inline __value_func(__value_func&& __f) noexcept
  {
    if (__f.__f_ == nullptr)
    {
      __f_ = nullptr;
    }
    else if ((void*) __f.__f_ == &__f.__buf_)
    {
      __f_ = __as_base(&__buf_);
      __f.__f_->__clone(__f_);
    }
    else
    {
      __f_     = __f.__f_;
      __f.__f_ = nullptr;
    }
  }

  _CCCL_API inline ~__value_func()
  {
    if ((void*) __f_ == &__buf_)
    {
      __f_->destroy();
    }
    else if (__f_)
    {
      __f_->destroy_deallocate();
    }
  }

  _CCCL_API inline __value_func& operator=(__value_func&& __f)
  {
    *this = nullptr;
    if (__f.__f_ == nullptr)
    {
      __f_ = nullptr;
    }
    else if ((void*) __f.__f_ == &__f.__buf_)
    {
      __f_ = __as_base(&__buf_);
      __f.__f_->__clone(__f_);
    }
    else
    {
      __f_     = __f.__f_;
      __f.__f_ = nullptr;
    }
    return *this;
  }

  _CCCL_API inline __value_func& operator=(nullptr_t)
  {
    __func* __f = __f_;
    __f_        = nullptr;
    if ((void*) __f == &__buf_)
    {
      __f->destroy();
    }
    else if (__f)
    {
      __f->destroy_deallocate();
    }
    return *this;
  }

  _CCCL_API inline _Rp operator()(_ArgTypes&&... __args) const
  {
    if (__f_ == nullptr)
    {
      __throw_bad_function_call();
    }
    return (*__f_)(_CUDA_VSTD::forward<_ArgTypes>(__args)...);
  }

  _CCCL_API inline void swap(__value_func& __f) noexcept
  {
    if (&__f == this)
    {
      return;
    }
    if ((void*) __f_ == &__buf_ && (void*) __f.__f_ == &__f.__buf_)
    {
      typename aligned_storage<sizeof(__buf_)>::type __tempbuf;
      __func* __t = __as_base(&__tempbuf);
      __f_->__clone(__t);
      __f_->destroy();
      __f_ = nullptr;
      __f.__f_->__clone(__as_base(&__buf_));
      __f.__f_->destroy();
      __f.__f_ = nullptr;
      __f_     = __as_base(&__buf_);
      __t->__clone(__as_base(&__f.__buf_));
      __t->destroy();
      __f.__f_ = __as_base(&__f.__buf_);
    }
    else if ((void*) __f_ == &__buf_)
    {
      __f_->__clone(__as_base(&__f.__buf_));
      __f_->destroy();
      __f_     = __f.__f_;
      __f.__f_ = __as_base(&__f.__buf_);
    }
    else if ((void*) __f.__f_ == &__f.__buf_)
    {
      __f.__f_->__clone(__as_base(&__buf_));
      __f.__f_->destroy();
      __f.__f_ = __f_;
      __f_     = __as_base(&__buf_);
    }
    else
    {
      _CUDA_VSTD::swap(__f_, __f.__f_);
    }
  }

  _CCCL_API inline explicit operator bool() const noexcept
  {
    return __f_ != nullptr;
  }

#  ifndef _CCCL_NO_RTTI
  _CCCL_API inline const type_info& target_type() const noexcept
  {
    if (__f_ == nullptr)
    {
      return typeid(void);
    }
    return __f_->target_type();
  }

  template <typename _Tp>
  _CCCL_API inline const _Tp* target() const noexcept
  {
    if (__f_ == nullptr)
    {
      return nullptr;
    }
    return (const _Tp*) __f_->target(typeid(_Tp));
  }
#  endif // _CCCL_NO_RTTI
};

// Storage for a functor object, to be used with __policy to manage copy and
// destruction.
union __policy_storage
{
  mutable char __small[sizeof(void*) * 2];
  void* __large;
};

// True if _Fun can safely be held in __policy_storage.__small.
template <typename _Fun>
struct __use_small_storage
    : public integral_constant<
        bool,
        sizeof(_Fun) <= sizeof(__policy_storage) && alignof(_Fun) <= alignof(__policy_storage)
          && is_trivially_copy_constructible<_Fun>::value && is_trivially_destructible<_Fun>::value>
{};

// Policy contains information about how to copy, destroy, and move the
// underlying functor. You can think of it as a vtable of sorts.
struct __policy
{
  // Used to copy or destroy __large values. null for trivial objects.
  void* (*const __clone)(const void*);
  void (*const __destroy)(void*);

  // True if this is the null policy (no value).
  const bool __is_null;

  // The target type. May be null if RTTI is disabled.
  const type_info* const __type_info;

  // Returns a pointer to a static policy object suitable for the functor
  // type.
  template <typename _Fun>
  _CCCL_API inline static const __policy* __create()
  {
    return __choose_policy<_Fun>(__use_small_storage<_Fun>());
  }

  _CCCL_API inline static const __policy* __create_empty()
  {
    static const constexpr __policy __policy_ = {
      nullptr,
      nullptr,
      true,
#  ifndef _CCCL_NO_RTTI
      &typeid(void)
#  else
      nullptr
#  endif
    };
    return &__policy_;
  }

private:
  template <typename _Fun>
  static void* __large_clone(const void* __s)
  {
    const _Fun* __f = static_cast<const _Fun*>(__s);
    return __f->__clone();
  }

  template <typename _Fun>
  static void __large_destroy(void* __s)
  {
    _Fun::__destroy_and_delete(static_cast<_Fun*>(__s));
  }

  template <typename _Fun>
  _CCCL_API inline static const __policy* __choose_policy(/* is_small = */ false_type)
  {
    static const constexpr __policy __policy_ = {
      &__large_clone<_Fun>,
      &__large_destroy<_Fun>,
      false,
#  ifndef _CCCL_NO_RTTI
      &typeid(typename _Fun::_Target)
#  else
      nullptr
#  endif
    };
    return &__policy_;
  }

  template <typename _Fun>
  _CCCL_API inline static const __policy* __choose_policy(/* is_small = */ true_type)
  {
    static const constexpr __policy __policy_ = {
      nullptr,
      nullptr,
      false,
#  ifndef _CCCL_NO_RTTI
      &typeid(typename _Fun::_Target)
#  else
      nullptr
#  endif
    };
    return &__policy_;
  }
};

// Used to choose between perfect forwarding or pass-by-value. Pass-by-value is
// faster for types that can be passed in registers.
template <typename _Tp>
using __fast_forward = conditional_t<is_scalar<_Tp>::value, _Tp, _Tp&&>;

// __policy_invoker calls an instance of __alloc_func held in __policy_storage.

template <class _Fp>
struct __policy_invoker;

template <class _Rp, class... _ArgTypes>
struct __policy_invoker<_Rp(_ArgTypes...)>
{
  using __Call = _Rp (*)(const __policy_storage*, __fast_forward<_ArgTypes>...);

  __Call __call_;

  // Creates an invoker that throws bad_function_call.
  _CCCL_API inline __policy_invoker()
      : __call_(&__call_empty)
  {}

  // Creates an invoker that calls the given instance of __func.
  template <typename _Fun>
  _CCCL_API inline static __policy_invoker __create()
  {
    return __policy_invoker(&__call_impl<_Fun>);
  }

private:
  _CCCL_API inline explicit __policy_invoker(__Call __c)
      : __call_(__c)
  {}

  static _Rp __call_empty(const __policy_storage*, __fast_forward<_ArgTypes>...)
  {
    __throw_bad_function_call();
  }

  template <typename _Fun>
  static _Rp __call_impl(const __policy_storage* __buf, __fast_forward<_ArgTypes>... __args)
  {
    _Fun* __f = reinterpret_cast<_Fun*>(__use_small_storage<_Fun>::value ? &__buf->__small : __buf->__large);
    return (*__f)(_CUDA_VSTD::forward<_ArgTypes>(__args)...);
  }
};

// __policy_func uses a __policy and __policy_invoker to create a type-erased,
// copyable functor.

template <class _Fp>
class __policy_func;

template <class _Rp, class... _ArgTypes>
class __policy_func<_Rp(_ArgTypes...)>
{
  // Inline storage for small objects.
  __policy_storage __buf_;

  // Calls the value stored in __buf_. This could technically be part of
  // policy, but storing it here eliminates a level of indirection inside
  // operator().
  using __invoker = __function::__policy_invoker<_Rp(_ArgTypes...)>;
  __invoker __invoker_;

  // The policy that describes how to move / copy / destroy __buf_. Never
  // null, even if the function is empty.
  const __policy* __policy_;

public:
  _CCCL_API inline __policy_func()
      : __policy_(__policy::__create_empty())
  {}

  template <class _Fp, class _Alloc>
  _CCCL_API inline __policy_func(_Fp&& __f, const _Alloc& __a)
      : __policy_(__policy::__create_empty())
  {
    using _Fun           = __alloc_func<_Fp, _Alloc, _Rp(_ArgTypes...)>;
    using __alloc_traits = allocator_traits<_Alloc>;
    using _FunAlloc      = typename __rebind_alloc_helper<__alloc_traits, _Fun>::type;

    if (__function::__not_null(__f))
    {
      __invoker_ = __invoker::template __create<_Fun>();
      __policy_  = __policy::__create<_Fun>();

      _FunAlloc __af(__a);
      if (__use_small_storage<_Fun>())
      {
        ::new ((void*) &__buf_.__small) _Fun(_CUDA_VSTD::move(__f), _Alloc(__af));
      }
      else
      {
        using _Dp = __allocator_destructor<_FunAlloc>;
        unique_ptr<_Fun, _Dp> __hold(__af.allocate(1), _Dp(__af, 1));
        ::new ((void*) __hold.get()) _Fun(_CUDA_VSTD::move(__f), _Alloc(__af));
        __buf_.__large = __hold.release();
      }
    }
  }

  template <class _Fp, class = enable_if_t<!is_same<decay_t<_Fp>, __policy_func>::value>>
  _CCCL_API inline explicit __policy_func(_Fp&& __f)
      : __policy_(__policy::__create_empty())
  {
    using _Fun = __default_alloc_func<_Fp, _Rp(_ArgTypes...)>;

    if (__function::__not_null(__f))
    {
      __invoker_ = __invoker::template __create<_Fun>();
      __policy_  = __policy::__create<_Fun>();
      if (__use_small_storage<_Fun>())
      {
        ::new ((void*) &__buf_.__small) _Fun(_CUDA_VSTD::move(__f));
      }
      else
      {
        __builtin_new_allocator::__holder_t __hold = __builtin_new_allocator::__allocate_type<_Fun>(1);
        __buf_.__large                             = ::new ((void*) __hold.get()) _Fun(_CUDA_VSTD::move(__f));
        (void) __hold.release();
      }
    }
  }

  _CCCL_API inline __policy_func(const __policy_func& __f)
      : __buf_(__f.__buf_)
      , __invoker_(__f.__invoker_)
      , __policy_(__f.__policy_)
  {
    if (__policy_->__clone)
    {
      __buf_.__large = __policy_->__clone(__f.__buf_.__large);
    }
  }

  _CCCL_API inline __policy_func(__policy_func&& __f)
      : __buf_(__f.__buf_)
      , __invoker_(__f.__invoker_)
      , __policy_(__f.__policy_)
  {
    if (__policy_->__destroy)
    {
      __f.__policy_  = __policy::__create_empty();
      __f.__invoker_ = __invoker();
    }
  }

  _CCCL_API inline ~__policy_func()
  {
    if (__policy_->__destroy)
    {
      __policy_->__destroy(__buf_.__large);
    }
  }

  _CCCL_API inline __policy_func& operator=(__policy_func&& __f)
  {
    *this          = nullptr;
    __buf_         = __f.__buf_;
    __invoker_     = __f.__invoker_;
    __policy_      = __f.__policy_;
    __f.__policy_  = __policy::__create_empty();
    __f.__invoker_ = __invoker();
    return *this;
  }

  _CCCL_API inline __policy_func& operator=(nullptr_t)
  {
    const __policy* __p = __policy_;
    __policy_           = __policy::__create_empty();
    __invoker_          = __invoker();
    if (__p->__destroy)
    {
      __p->__destroy(__buf_.__large);
    }
    return *this;
  }

  _CCCL_API inline _Rp operator()(_ArgTypes&&... __args) const
  {
    return __invoker_.__call_(_CUDA_VSTD::addressof(__buf_), _CUDA_VSTD::forward<_ArgTypes>(__args)...);
  }

  _CCCL_API inline void swap(__policy_func& __f)
  {
    _CUDA_VSTD::swap(__invoker_, __f.__invoker_);
    _CUDA_VSTD::swap(__policy_, __f.__policy_);
    _CUDA_VSTD::swap(__buf_, __f.__buf_);
  }

  _CCCL_API inline explicit operator bool() const noexcept
  {
    return !__policy_->__is_null;
  }

#  ifndef _CCCL_NO_RTTI
  _CCCL_API inline const type_info& target_type() const noexcept
  {
    return *__policy_->__type_info;
  }

  template <typename _Tp>
  _CCCL_API inline const _Tp* target() const noexcept
  {
    if (__policy_->__is_null || typeid(_Tp) != *__policy_->__type_info)
    {
      return nullptr;
    }
    if (__policy_->__clone) // Out of line storage.
    {
      return reinterpret_cast<const _Tp*>(__buf_.__large);
    }
    else
    {
      return reinterpret_cast<const _Tp*>(&__buf_.__small);
    }
  }
#  endif // _CCCL_NO_RTTI
};

#  if defined(_LIBCUDACXX_HAS_BLOCKS_RUNTIME)

extern "C" void* _Block_copy(const void*);
extern "C" void _Block_release(const void*);

template <class _Rp1, class... _ArgTypes1, class _Alloc, class _Rp, class... _ArgTypes>
class __func<_Rp1 (^)(_ArgTypes1...), _Alloc, _Rp(_ArgTypes...)> : public __base<_Rp(_ArgTypes...)>
{
  using ...); = _Rp1 (^__block_type)(_ArgTypes1
  __block_type __f_;

public:
  _CCCL_API inline explicit __func(__block_type const& __f)
      : __f_(reinterpret_cast<__block_type>(__f ? _Block_copy(__f) : nullptr))
  {}

  // [TODO] add && to save on a retain

  _CCCL_API inline explicit __func(__block_type __f, const _Alloc& /* unused */)
      : __f_(reinterpret_cast<__block_type>(__f ? _Block_copy(__f) : nullptr))
  {}

  virtual __base<_Rp(_ArgTypes...)>* __clone() const
  {
    _CCCL_ASSERT(false,
                 "Block pointers are just pointers, so they should always fit into "
                 "std::function's small buffer optimization. This function should "
                 "never be invoked.");
    return nullptr;
  }

  virtual void __clone(__base<_Rp(_ArgTypes...)>* __p) const
  {
    ::new ((void*) __p) __func(__f_);
  }

  virtual void destroy() noexcept
  {
    if (__f_)
    {
      _Block_release(__f_);
    }
    __f_ = 0;
  }

  virtual void destroy_deallocate() noexcept
  {
    _CCCL_ASSERT(false,
                 "Block pointers are just pointers, so they should always fit into "
                 "std::function's small buffer optimization. This function should "
                 "never be invoked.");
  }

  virtual _Rp operator()(_ArgTypes&&... __arg)
  {
    return _CUDA_VSTD::__invoke(__f_, _CUDA_VSTD::forward<_ArgTypes>(__arg)...);
  }

#    ifndef _CCCL_NO_RTTI
  virtual const void* target(type_info const& __ti) const noexcept
  {
    if (__ti == typeid(__func::__block_type))
    {
      return &__f_;
    }
    return (const void*) nullptr;
  }

  virtual const type_info& target_type() const noexcept
  {
    return typeid(__func::__block_type);
  }
#    endif // _CCCL_NO_RTTI
};

#  endif // _LIBCUDACXX_HAS_EXTENSION_BLOCKS

} // namespace __function

template <class _Rp, class... _ArgTypes>
class _CCCL_TYPE_VISIBILITY_DEFAULT function<_Rp(_ArgTypes...)>
    : public __function::__maybe_derive_from_unary_function<_Rp(_ArgTypes...)>
    , public __function::__maybe_derive_from_binary_function<_Rp(_ArgTypes...)>
{
  using __func = __function::__policy_func<_Rp(_ArgTypes...)>;

  __func __f_;

  template <class _Fp, bool = _And<_IsNotSame<remove_cvref_t<_Fp>, function>, __invocable<_Fp, _ArgTypes...>>::value>
  struct __callable;
  template <class _Fp>
  struct __callable<_Fp, true>
  {
    static const bool value =
      is_void<_Rp>::value || __is_core_convertible<typename __invoke_of<_Fp, _ArgTypes...>::type, _Rp>::value;
  };
  template <class _Fp>
  struct __callable<_Fp, false>
  {
    static const bool value = false;
  };

  template <class _Fp>
  using _EnableIfLValueCallable = enable_if_t<__callable<_Fp&>::value>;

public:
  using result_type = _Rp;

  // construct/copy/destroy:
  _CCCL_API inline function() noexcept {}
  _CCCL_API inline function(nullptr_t) noexcept {}
  function(const function&);
  function(function&&) noexcept;
  template <class _Fp, class = _EnableIfLValueCallable<_Fp>>
  function(_Fp);

  function& operator=(const function&);
  function& operator=(function&&) noexcept;
  function& operator=(nullptr_t) noexcept;
  template <class _Fp, class = _EnableIfLValueCallable<decay_t<_Fp>>>
  function& operator=(_Fp&&);

  ~function();

  // function modifiers:
  void swap(function&) noexcept;

  // function capacity:
  _CCCL_API inline explicit operator bool() const noexcept
  {
    return static_cast<bool>(__f_);
  }

  // deleted overloads close possible hole in the type system
  template <class _R2, class... _ArgTypes2>
  bool operator==(const function<_R2(_ArgTypes2...)>&) const = delete;
  template <class _R2, class... _ArgTypes2>
  bool operator!=(const function<_R2(_ArgTypes2...)>&) const = delete;

public:
  // function invocation:
  _Rp operator()(_ArgTypes...) const;

#  ifndef _CCCL_NO_RTTI
  // function target access:
  const type_info& target_type() const noexcept;
  template <typename _Tp>
  _Tp* target() noexcept;
  template <typename _Tp>
  const _Tp* target() const noexcept;
#  endif // _CCCL_NO_RTTI
};

template <class _Rp, class... _Ap>
function(_Rp (*)(_Ap...)) -> function<_Rp(_Ap...)>;

template <class _Fp>
struct __strip_signature;

template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...)>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) volatile>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const volatile>
{
  using type = _Rp(_Ap...);
};

template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) &>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const&>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) volatile&>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const volatile&>
{
  using type = _Rp(_Ap...);
};

template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) noexcept>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const noexcept>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) volatile noexcept>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const volatile noexcept>
{
  using type = _Rp(_Ap...);
};

template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) & noexcept>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const & noexcept>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) volatile & noexcept>
{
  using type = _Rp(_Ap...);
};
template <class _Rp, class _Gp, class... _Ap>
struct __strip_signature<_Rp (_Gp::*)(_Ap...) const volatile & noexcept>
{
  using type = _Rp(_Ap...);
};

template <class _Fp, class _Stripped = typename __strip_signature<decltype(&_Fp::operator())>::type>
function(_Fp) -> function<_Stripped>;

template <class _Rp, class... _ArgTypes>
function<_Rp(_ArgTypes...)>::function(const function& __f)
    : __f_(__f.__f_)
{}

template <class _Rp, class... _ArgTypes>
function<_Rp(_ArgTypes...)>::function(function&& __f) noexcept
    : __f_(_CUDA_VSTD::move(__f.__f_))
{}

template <class _Rp, class... _ArgTypes>
template <class _Fp, class>
function<_Rp(_ArgTypes...)>::function(_Fp __f)
    : __f_(_CUDA_VSTD::move(__f))
{}

template <class _Rp, class... _ArgTypes>
function<_Rp(_ArgTypes...)>& function<_Rp(_ArgTypes...)>::operator=(const function& __f)
{
  function(__f).swap(*this);
  return *this;
}

template <class _Rp, class... _ArgTypes>
function<_Rp(_ArgTypes...)>& function<_Rp(_ArgTypes...)>::operator=(function&& __f) noexcept
{
  __f_ = _CUDA_VSTD::move(__f.__f_);
  return *this;
}

template <class _Rp, class... _ArgTypes>
function<_Rp(_ArgTypes...)>& function<_Rp(_ArgTypes...)>::operator=(nullptr_t) noexcept
{
  __f_ = nullptr;
  return *this;
}

template <class _Rp, class... _ArgTypes>
template <class _Fp, class>
function<_Rp(_ArgTypes...)>& function<_Rp(_ArgTypes...)>::operator=(_Fp&& __f)
{
  function(_CUDA_VSTD::forward<_Fp>(__f)).swap(*this);
  return *this;
}

template <class _Rp, class... _ArgTypes>
function<_Rp(_ArgTypes...)>::~function()
{}

template <class _Rp, class... _ArgTypes>
void function<_Rp(_ArgTypes...)>::swap(function& __f) noexcept
{
  __f_.swap(__f.__f_);
}

template <class _Rp, class... _ArgTypes>
_Rp function<_Rp(_ArgTypes...)>::operator()(_ArgTypes... __arg) const
{
  return __f_(_CUDA_VSTD::forward<_ArgTypes>(__arg)...);
}

#  ifndef _CCCL_NO_RTTI

template <class _Rp, class... _ArgTypes>
const type_info& function<_Rp(_ArgTypes...)>::target_type() const noexcept
{
  return __f_.target_type();
}

template <class _Rp, class... _ArgTypes>
template <typename _Tp>
_Tp* function<_Rp(_ArgTypes...)>::target() noexcept
{
  return (_Tp*) (__f_.template target<_Tp>());
}

template <class _Rp, class... _ArgTypes>
template <typename _Tp>
const _Tp* function<_Rp(_ArgTypes...)>::target() const noexcept
{
  return __f_.template target<_Tp>();
}

#  endif // _CCCL_NO_RTTI

template <class _Rp, class... _ArgTypes>
_CCCL_API inline bool operator==(const function<_Rp(_ArgTypes...)>& __f, nullptr_t) noexcept
{
  return !__f;
}

template <class _Rp, class... _ArgTypes>
_CCCL_API inline bool operator==(nullptr_t, const function<_Rp(_ArgTypes...)>& __f) noexcept
{
  return !__f;
}

template <class _Rp, class... _ArgTypes>
_CCCL_API inline bool operator!=(const function<_Rp(_ArgTypes...)>& __f, nullptr_t) noexcept
{
  return (bool) __f;
}

template <class _Rp, class... _ArgTypes>
_CCCL_API inline bool operator!=(nullptr_t, const function<_Rp(_ArgTypes...)>& __f) noexcept
{
  return (bool) __f;
}

template <class _Rp, class... _ArgTypes>
_CCCL_API inline void swap(function<_Rp(_ArgTypes...)>& __x, function<_Rp(_ArgTypes...)>& __y) noexcept
{
  return __x.swap(__y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // __cuda_std__

#endif // _LIBCUDACXX___FUNCTIONAL_FUNCTION_H
