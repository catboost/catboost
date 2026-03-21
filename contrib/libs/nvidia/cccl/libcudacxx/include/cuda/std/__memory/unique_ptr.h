// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_UNIQUE_PTR_H
#define _LIBCUDACXX___MEMORY_UNIQUE_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/compare_three_way.h>
#  include <cuda/std/__compare/compare_three_way_result.h>
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__functional/hash.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__memory/allocator_traits.h> // __pointer
#include <cuda/std/__memory/compressed_pair.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/dependent_type.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_extent.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT default_delete
{
  static_assert(!_CCCL_TRAIT(is_function, _Tp), "default_delete cannot be instantiated for function types");

  _CCCL_HIDE_FROM_ABI constexpr default_delete() noexcept = default;

  template <class _Up>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20
  default_delete(const default_delete<_Up>&, enable_if_t<_CCCL_TRAIT(is_convertible, _Up*, _Tp*), int> = 0) noexcept
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void operator()(_Tp* __ptr) const noexcept
  {
    static_assert(sizeof(_Tp) >= 0, "cannot delete an incomplete type");
    static_assert(!is_void<_Tp>::value, "cannot delete an incomplete type");
    delete __ptr;
  }
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT default_delete<_Tp[]>
{
  _CCCL_HIDE_FROM_ABI constexpr default_delete() noexcept = default;

  template <class _Up>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 default_delete(
    const default_delete<_Up[]>&, enable_if_t<_CCCL_TRAIT(is_convertible, _Up (*)[], _Tp (*)[]), int> = 0) noexcept
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Up>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 enable_if_t<_CCCL_TRAIT(is_convertible, _Up (*)[], _Tp (*)[]), void>
  operator()(_Up* __ptr) const noexcept
  {
    static_assert(sizeof(_Up) >= 0, "cannot delete an incomplete type");
    delete[] __ptr;
  }
};

template <class _Deleter>
struct __unique_ptr_deleter_sfinae
{
  static_assert(!_CCCL_TRAIT(is_reference, _Deleter), "incorrect specialization");
  typedef const _Deleter& __lval_ref_type;
  typedef _Deleter&& __good_rval_ref_type;
  typedef true_type __enable_rval_overload;
};

template <class _Deleter>
struct __unique_ptr_deleter_sfinae<_Deleter const&>
{
  typedef const _Deleter& __lval_ref_type;
  typedef const _Deleter&& __bad_rval_ref_type;
  typedef false_type __enable_rval_overload;
};

template <class _Deleter>
struct __unique_ptr_deleter_sfinae<_Deleter&>
{
  typedef _Deleter& __lval_ref_type;
  typedef _Deleter&& __bad_rval_ref_type;
  typedef false_type __enable_rval_overload;
};

#if defined(_LIBCUDACXX_ABI_ENABLE_UNIQUE_PTR_TRIVIAL_ABI)
#  define _LIBCUDACXX_UNIQUE_PTR_TRIVIAL_ABI __attribute__((__trivial_abi__))
#else
#  define _LIBCUDACXX_UNIQUE_PTR_TRIVIAL_ABI
#endif

template <class _Tp, class _Dp = default_delete<_Tp>>
class _LIBCUDACXX_UNIQUE_PTR_TRIVIAL_ABI _CCCL_TYPE_VISIBILITY_DEFAULT unique_ptr
{
public:
  typedef _Tp element_type;
  typedef _Dp deleter_type;
  typedef _CCCL_NODEBUG_ALIAS typename __pointer<_Tp, deleter_type>::type pointer;

  static_assert(!_CCCL_TRAIT(is_rvalue_reference, deleter_type),
                "the specified deleter type cannot be an rvalue reference");

private:
  __compressed_pair<pointer, deleter_type> __ptr_;

  struct __nat
  {
    int __for_bool_;
  };

  typedef _CCCL_NODEBUG_ALIAS __unique_ptr_deleter_sfinae<_Dp> _DeleterSFINAE;

  template <bool _Dummy>
  using _LValRefType _CCCL_NODEBUG_ALIAS = typename __dependent_type<_DeleterSFINAE, _Dummy>::__lval_ref_type;

  template <bool _Dummy>
  using _GoodRValRefType _CCCL_NODEBUG_ALIAS = typename __dependent_type<_DeleterSFINAE, _Dummy>::__good_rval_ref_type;

  template <bool _Dummy>
  using _BadRValRefType _CCCL_NODEBUG_ALIAS = typename __dependent_type<_DeleterSFINAE, _Dummy>::__bad_rval_ref_type;

  template <bool _Dummy, class _Deleter = typename __dependent_type<type_identity<deleter_type>, _Dummy>::type>
  using _EnableIfDeleterDefaultConstructible _CCCL_NODEBUG_ALIAS =
    typename enable_if<is_default_constructible<_Deleter>::value && !is_pointer<_Deleter>::value>::type;

  template <class _ArgType>
  using _EnableIfDeleterConstructible _CCCL_NODEBUG_ALIAS =
    typename enable_if<is_constructible<deleter_type, _ArgType>::value>::type;

  template <class _UPtr, class _Up>
  using _EnableIfMoveConvertible _CCCL_NODEBUG_ALIAS =
    typename enable_if<is_convertible<typename _UPtr::pointer, pointer>::value && !is_array<_Up>::value>::type;

  template <class _UDel>
  using _EnableIfDeleterConvertible _CCCL_NODEBUG_ALIAS =
    typename enable_if<(is_reference<_Dp>::value && is_same<_Dp, _UDel>::value)
                       || (!is_reference<_Dp>::value && is_convertible<_UDel, _Dp>::value)>::type;

  template <class _UDel>
  using _EnableIfDeleterAssignable = typename enable_if<is_assignable<_Dp&, _UDel&&>::value>::type;

public:
  template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
  _CCCL_API constexpr unique_ptr() noexcept
      : __ptr_(__value_init_tag(), __value_init_tag())
  {}

  template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
  _CCCL_API constexpr unique_ptr(nullptr_t) noexcept
      : __ptr_(__value_init_tag(), __value_init_tag())
  {}

  template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit unique_ptr(pointer __p) noexcept
      : __ptr_(__p, __value_init_tag())
  {}

  template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(pointer __p, _LValRefType<_Dummy> __d) noexcept
      : __ptr_(__p, __d)
  {}

  template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(pointer __p, _GoodRValRefType<_Dummy> __d) noexcept
      : __ptr_(__p, _CUDA_VSTD::move(__d))
  {
    static_assert(!is_reference<deleter_type>::value, "rvalue deleter bound to reference");
  }

  template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_BadRValRefType<_Dummy>>>
  _CCCL_API inline unique_ptr(pointer __p, _BadRValRefType<_Dummy> __d) = delete;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(unique_ptr&& __u) noexcept
      : __ptr_(__u.release(), _CUDA_VSTD::forward<deleter_type>(__u.get_deleter()))
  {}

  template <class _Up,
            class _Ep,
            class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
            class = _EnableIfDeleterConvertible<_Ep>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(unique_ptr<_Up, _Ep>&& __u) noexcept
      : __ptr_(__u.release(), _CUDA_VSTD::forward<_Ep>(__u.get_deleter()))
  {}

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr& operator=(unique_ptr&& __u) noexcept
  {
    reset(__u.release());
    __ptr_.second() = _CUDA_VSTD::forward<deleter_type>(__u.get_deleter());
    return *this;
  }

  template <class _Up,
            class _Ep,
            class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
            class = _EnableIfDeleterAssignable<_Ep>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr& operator=(unique_ptr<_Up, _Ep>&& __u) noexcept
  {
    reset(__u.release());
    __ptr_.second() = _CUDA_VSTD::forward<_Ep>(__u.get_deleter());
    return *this;
  }

  unique_ptr(unique_ptr const&)            = delete;
  unique_ptr& operator=(unique_ptr const&) = delete;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~unique_ptr()
  {
    reset();
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr& operator=(nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 add_lvalue_reference_t<_Tp> operator*() const
  {
    return *__ptr_.first();
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 pointer operator->() const noexcept
  {
    return __ptr_.first();
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 pointer get() const noexcept
  {
    return __ptr_.first();
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 deleter_type& get_deleter() noexcept
  {
    return __ptr_.second();
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 const deleter_type& get_deleter() const noexcept
  {
    return __ptr_.second();
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit operator bool() const noexcept
  {
    return __ptr_.first() != nullptr;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 pointer release() noexcept
  {
    pointer __t    = __ptr_.first();
    __ptr_.first() = pointer();
    return __t;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void reset(pointer __p = pointer()) noexcept
  {
    pointer __tmp  = __ptr_.first();
    __ptr_.first() = __p;
    if (__tmp)
    {
      __ptr_.second()(__tmp);
    }
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void swap(unique_ptr& __u) noexcept
  {
    __ptr_.swap(__u.__ptr_);
  }
};

template <class _Tp, class _Dp>
class _LIBCUDACXX_UNIQUE_PTR_TRIVIAL_ABI _CCCL_TYPE_VISIBILITY_DEFAULT unique_ptr<_Tp[], _Dp>
{
public:
  typedef _Tp element_type;
  typedef _Dp deleter_type;
  typedef typename __pointer<_Tp, deleter_type>::type pointer;

private:
  __compressed_pair<pointer, deleter_type> __ptr_;

  template <class _From>
  struct _CheckArrayPointerConversion : is_same<_From, pointer>
  {};

  template <class _FromElem>
  struct _CheckArrayPointerConversion<_FromElem*>
      : integral_constant<
          bool,
          is_same<_FromElem*, pointer>::value
            || (is_same<pointer, element_type*>::value && is_convertible<_FromElem (*)[], element_type (*)[]>::value)>
  {};

  typedef __unique_ptr_deleter_sfinae<_Dp> _DeleterSFINAE;

  template <bool _Dummy>
  using _LValRefType _CCCL_NODEBUG_ALIAS = typename __dependent_type<_DeleterSFINAE, _Dummy>::__lval_ref_type;

  template <bool _Dummy>
  using _GoodRValRefType _CCCL_NODEBUG_ALIAS = typename __dependent_type<_DeleterSFINAE, _Dummy>::__good_rval_ref_type;

  template <bool _Dummy>
  using _BadRValRefType _CCCL_NODEBUG_ALIAS = typename __dependent_type<_DeleterSFINAE, _Dummy>::__bad_rval_ref_type;

  template <bool _Dummy, class _Deleter = typename __dependent_type<type_identity<deleter_type>, _Dummy>::type>
  using _EnableIfDeleterDefaultConstructible _CCCL_NODEBUG_ALIAS =
    typename enable_if<is_default_constructible<_Deleter>::value && !is_pointer<_Deleter>::value>::type;

  template <class _ArgType>
  using _EnableIfDeleterConstructible _CCCL_NODEBUG_ALIAS =
    typename enable_if<is_constructible<deleter_type, _ArgType>::value>::type;

  template <class _Pp>
  using _EnableIfPointerConvertible _CCCL_NODEBUG_ALIAS =
    typename enable_if<_CheckArrayPointerConversion<_Pp>::value>::type;

  template <class _UPtr, class _Up, class _ElemT = typename _UPtr::element_type>
  using _EnableIfMoveConvertible _CCCL_NODEBUG_ALIAS = typename enable_if<
    is_array<_Up>::value && is_same<pointer, element_type*>::value && is_same<typename _UPtr::pointer, _ElemT*>::value
    && is_convertible<_ElemT (*)[], element_type (*)[]>::value>::type;

  template <class _UDel>
  using _EnableIfDeleterConvertible _CCCL_NODEBUG_ALIAS =
    typename enable_if<(is_reference<_Dp>::value && is_same<_Dp, _UDel>::value)
                       || (!is_reference<_Dp>::value && is_convertible<_UDel, _Dp>::value)>::type;

  template <class _UDel>
  using _EnableIfDeleterAssignable _CCCL_NODEBUG_ALIAS = typename enable_if<is_assignable<_Dp&, _UDel&&>::value>::type;

public:
  template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
  _CCCL_API constexpr unique_ptr() noexcept
      : __ptr_(__value_init_tag(), __value_init_tag())
  {}

  template <bool _Dummy = true, class = _EnableIfDeleterDefaultConstructible<_Dummy>>
  _CCCL_API constexpr unique_ptr(nullptr_t) noexcept
      : __ptr_(__value_init_tag(), __value_init_tag())
  {}

  template <class _Pp,
            bool _Dummy = true,
            class       = _EnableIfDeleterDefaultConstructible<_Dummy>,
            class       = _EnableIfPointerConvertible<_Pp>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit unique_ptr(_Pp __p) noexcept
      : __ptr_(__p, __value_init_tag())
  {}

  template <class _Pp,
            bool _Dummy = true,
            class       = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>,
            class       = _EnableIfPointerConvertible<_Pp>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(_Pp __p, _LValRefType<_Dummy> __d) noexcept
      : __ptr_(__p, __d)
  {}

  template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_LValRefType<_Dummy>>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(nullptr_t, _LValRefType<_Dummy> __d) noexcept
      : __ptr_(nullptr, __d)
  {}

  template <class _Pp,
            bool _Dummy = true,
            class       = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>,
            class       = _EnableIfPointerConvertible<_Pp>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(_Pp __p, _GoodRValRefType<_Dummy> __d) noexcept
      : __ptr_(__p, _CUDA_VSTD::move(__d))
  {
    static_assert(!is_reference<deleter_type>::value, "rvalue deleter bound to reference");
  }

  template <bool _Dummy = true, class = _EnableIfDeleterConstructible<_GoodRValRefType<_Dummy>>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(nullptr_t, _GoodRValRefType<_Dummy> __d) noexcept
      : __ptr_(nullptr, _CUDA_VSTD::move(__d))
  {
    static_assert(!is_reference<deleter_type>::value, "rvalue deleter bound to reference");
  }

  template <class _Pp,
            bool _Dummy = true,
            class       = _EnableIfDeleterConstructible<_BadRValRefType<_Dummy>>,
            class       = _EnableIfPointerConvertible<_Pp>>
  _CCCL_API inline unique_ptr(_Pp __p, _BadRValRefType<_Dummy> __d) = delete;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(unique_ptr&& __u) noexcept
      : __ptr_(__u.release(), _CUDA_VSTD::forward<deleter_type>(__u.get_deleter()))
  {}

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr& operator=(unique_ptr&& __u) noexcept
  {
    reset(__u.release());
    __ptr_.second() = _CUDA_VSTD::forward<deleter_type>(__u.get_deleter());
    return *this;
  }

  template <class _Up,
            class _Ep,
            class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
            class = _EnableIfDeleterConvertible<_Ep>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr(unique_ptr<_Up, _Ep>&& __u) noexcept
      : __ptr_(__u.release(), _CUDA_VSTD::forward<_Ep>(__u.get_deleter()))
  {}

  template <class _Up,
            class _Ep,
            class = _EnableIfMoveConvertible<unique_ptr<_Up, _Ep>, _Up>,
            class = _EnableIfDeleterAssignable<_Ep>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr& operator=(unique_ptr<_Up, _Ep>&& __u) noexcept
  {
    reset(__u.release());
    __ptr_.second() = _CUDA_VSTD::forward<_Ep>(__u.get_deleter());
    return *this;
  }

public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~unique_ptr()
  {
    reset();
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 unique_ptr& operator=(nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 add_lvalue_reference_t<_Tp> operator[](size_t __i) const
  {
    return __ptr_.first()[__i];
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 pointer get() const noexcept
  {
    return __ptr_.first();
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 deleter_type& get_deleter() noexcept
  {
    return __ptr_.second();
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 const deleter_type& get_deleter() const noexcept
  {
    return __ptr_.second();
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit operator bool() const noexcept
  {
    return __ptr_.first() != nullptr;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 pointer release() noexcept
  {
    pointer __t    = __ptr_.first();
    __ptr_.first() = pointer();
    return __t;
  }

  template <class _Pp, enable_if_t<_CheckArrayPointerConversion<_Pp>::value, int> = 0>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void reset(_Pp __p) noexcept
  {
    pointer __tmp  = __ptr_.first();
    __ptr_.first() = __p;
    if (__tmp)
    {
      __ptr_.second()(__tmp);
    }
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void reset(nullptr_t = nullptr) noexcept
  {
    pointer __tmp  = __ptr_.first();
    __ptr_.first() = nullptr;
    if (__tmp)
    {
      __ptr_.second()(__tmp);
    }
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void swap(unique_ptr& __u) noexcept
  {
    __ptr_.swap(__u.__ptr_);
  }
};

template <class _Tp, class _Dp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 enable_if_t<__is_swappable<_Dp>::value, void>
swap(unique_ptr<_Tp, _Dp>& __x, unique_ptr<_Tp, _Dp>& __y) noexcept
{
  __x.swap(__y);
}

template <class _T1, class _D1, class _T2, class _D2>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator==(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  return __x.get() == __y.get();
}

#if _CCCL_STD_VER <= 2017
template <class _T1, class _D1, class _T2, class _D2>
_CCCL_API inline

  bool
  operator!=(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  return !(__x == __y);
}
#endif

template <class _T1, class _D1, class _T2, class _D2>
_CCCL_API inline

  bool
  operator<(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  typedef typename unique_ptr<_T1, _D1>::pointer _P1;
  typedef typename unique_ptr<_T2, _D2>::pointer _P2;
  typedef typename common_type<_P1, _P2>::type _Vp;
  return less<_Vp>()(__x.get(), __y.get());
}

template <class _T1, class _D1, class _T2, class _D2>
_CCCL_API inline

  bool
  operator>(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  return __y < __x;
}

template <class _T1, class _D1, class _T2, class _D2>
_CCCL_API inline

  bool
  operator<=(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  return !(__y < __x);
}

template <class _T1, class _D1, class _T2, class _D2>
_CCCL_API inline

  bool
  operator>=(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  return !(__x < __y);
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  if _CCCL_STD_VER >= 2020
template <class _T1, class _D1, class _T2, class _D2>
  requires three_way_comparable_with<typename unique_ptr<_T1, _D1>::pointer, typename unique_ptr<_T2, _D2>::pointer>
_CCCL_API inline compare_three_way_result_t<typename unique_ptr<_T1, _D1>::pointer,
                                            typename unique_ptr<_T2, _D2>::pointer>
operator<=>(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
  return compare_three_way()(__x.get(), __y.get());
}
#  endif // _CCCL_STD_VER >= 2020
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator==(const unique_ptr<_T1, _D1>& __x, nullptr_t) noexcept
{
  return !__x;
}

#if _CCCL_STD_VER <= 2017
template <class _T1, class _D1>
_CCCL_API inline

  bool
  operator==(nullptr_t, const unique_ptr<_T1, _D1>& __x) noexcept
{
  return !__x;
}

template <class _T1, class _D1>
_CCCL_API inline

  bool
  operator!=(const unique_ptr<_T1, _D1>& __x, nullptr_t) noexcept
{
  return static_cast<bool>(__x);
}

template <class _T1, class _D1>
_CCCL_API inline

  bool
  operator!=(nullptr_t, const unique_ptr<_T1, _D1>& __x) noexcept
{
  return static_cast<bool>(__x);
}
#endif // _CCCL_STD_VER <= 2017

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator<(const unique_ptr<_T1, _D1>& __x, nullptr_t)
{
  typedef typename unique_ptr<_T1, _D1>::pointer _P1;
  return less<_P1>()(__x.get(), nullptr);
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator<(nullptr_t, const unique_ptr<_T1, _D1>& __x)
{
  typedef typename unique_ptr<_T1, _D1>::pointer _P1;
  return less<_P1>()(nullptr, __x.get());
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator>(const unique_ptr<_T1, _D1>& __x, nullptr_t)
{
  return nullptr < __x;
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator>(nullptr_t, const unique_ptr<_T1, _D1>& __x)
{
  return __x < nullptr;
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator<=(const unique_ptr<_T1, _D1>& __x, nullptr_t)
{
  return !(nullptr < __x);
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator<=(nullptr_t, const unique_ptr<_T1, _D1>& __x)
{
  return !(__x < nullptr);
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator>=(const unique_ptr<_T1, _D1>& __x, nullptr_t)
{
  return !(__x < nullptr);
}

template <class _T1, class _D1>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 bool operator>=(nullptr_t, const unique_ptr<_T1, _D1>& __x)
{
  return !(nullptr < __x);
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  if _CCCL_STD_VER >= 2020
template <class _T1, class _D1>
  requires three_way_comparable<typename unique_ptr<_T1, _D1>::pointer>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 compare_three_way_result_t<typename unique_ptr<_T1, _D1>::pointer>
operator<=>(const unique_ptr<_T1, _D1>& __x, nullptr_t)
{
  return compare_three_way()(__x.get(), static_cast<typename unique_ptr<_T1, _D1>::pointer>(nullptr));
}
#  endif // _CCCL_STD_VER >= 2020
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class _Tp>
struct __unique_if
{
  typedef unique_ptr<_Tp> __unique_single;
};

template <class _Tp>
struct __unique_if<_Tp[]>
{
  typedef unique_ptr<_Tp[]> __unique_array_unknown_bound;
};

template <class _Tp, size_t _Np>
struct __unique_if<_Tp[_Np]>
{
  typedef void __unique_array_known_bound;
};

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class... _Args>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 typename __unique_if<_Tp>::__unique_single make_unique(_Args&&... __args)
{
  return unique_ptr<_Tp>(new _Tp(_CUDA_VSTD::forward<_Args>(__args)...));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 typename __unique_if<_Tp>::__unique_array_unknown_bound make_unique(size_t __n)
{
  typedef remove_extent_t<_Tp> _Up;
  return unique_ptr<_Tp>(new _Up[__n]());
}

template <class _Tp, class... _Args>
_CCCL_API inline typename __unique_if<_Tp>::__unique_array_known_bound make_unique(_Args&&...) = delete;

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 typename __unique_if<_Tp>::__unique_single make_unique_for_overwrite()
{
  return unique_ptr<_Tp>(new _Tp);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 typename __unique_if<_Tp>::__unique_array_unknown_bound
make_unique_for_overwrite(size_t __n)
{
  return unique_ptr<_Tp>(new remove_extent_t<_Tp>[__n]);
}

template <class _Tp, class... _Args>
_CCCL_API inline typename __unique_if<_Tp>::__unique_array_known_bound make_unique_for_overwrite(_Args&&...) = delete;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash;

#ifndef __cuda_std__
template <class _Tp, class _Dp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash<unique_ptr<_Tp, _Dp>>
{
#  if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  _LIBCUDACXX_DEPRECATED typedef unique_ptr<_Tp, _Dp> argument_type;
  _LIBCUDACXX_DEPRECATED typedef size_t result_type;
#  endif

  _CCCL_API inline size_t operator()(const unique_ptr<_Tp, _Dp>& __ptr) const
  {
    typedef typename unique_ptr<_Tp, _Dp>::pointer pointer;
    return hash<pointer>()(__ptr.get());
  }
};
#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_UNIQUE_PTR_H
