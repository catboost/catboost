// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_POINTER_TRAITS_H
#define _LIBCUDACXX___MEMORY_POINTER_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class = void>
struct __has_element_type : false_type
{};

template <class _Tp>
struct __has_element_type<_Tp, void_t<typename _Tp::element_type>> : true_type
{};

template <class _Ptr, bool = __has_element_type<_Ptr>::value>
struct __pointer_traits_element_type;

template <class _Ptr>
struct __pointer_traits_element_type<_Ptr, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Ptr::element_type;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args>
struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Sp<_Tp, _Args...>::element_type;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args>
struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, false>
{
  using type _CCCL_NODEBUG_ALIAS = _Tp;
};

template <class _Tp, class = void>
struct __has_difference_type : false_type
{};

template <class _Tp>
struct __has_difference_type<_Tp, void_t<typename _Tp::difference_type>> : true_type
{};

template <class _Ptr, bool = __has_difference_type<_Ptr>::value>
struct __pointer_traits_difference_type
{
  using type _CCCL_NODEBUG_ALIAS = ptrdiff_t;
};

template <class _Ptr>
struct __pointer_traits_difference_type<_Ptr, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Ptr::difference_type;
};

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _Up>
struct __has_rebind
{
private:
  template <class _Xp>
  _CCCL_API inline static false_type __test(...);
  template <class _Xp>
  _CCCL_API inline static true_type __test(typename _Xp::template rebind<_Up>* = 0);

public:
  static const bool value = decltype(__test<_Tp>(0))::value;
};
_CCCL_SUPPRESS_DEPRECATED_POP

template <class _Tp, class _Up, bool = __has_rebind<_Tp, _Up>::value>
struct __pointer_traits_rebind
{
  using type _CCCL_NODEBUG_ALIAS = typename _Tp::template rebind<_Up>;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _Args...>, _Up, true>
{
  using type _CCCL_NODEBUG_ALIAS = typename _Sp<_Tp, _Args...>::template rebind<_Up>;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _Args...>, _Up, false>
{
  using type = _Sp<_Up, _Args...>;
};

template <class _Ptr, class = void>
struct __pointer_traits_impl
{};

template <class _Ptr>
struct __pointer_traits_impl<_Ptr, void_t<typename __pointer_traits_element_type<_Ptr>::type>>
{
  using pointer         = _Ptr;
  using element_type    = typename __pointer_traits_element_type<pointer>::type;
  using difference_type = typename __pointer_traits_difference_type<pointer>::type;

  template <class _Up>
  using rebind = typename __pointer_traits_rebind<pointer, _Up>::type;

private:
  struct __nat
  {};

public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static pointer
  pointer_to(conditional_t<is_void<element_type>::value, __nat, element_type>& __r)
  {
    return pointer::pointer_to(__r);
  }
};

template <class _Ptr, class = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pointer_traits : __pointer_traits_impl<_Ptr>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pointer_traits<_Tp*>
{
  using pointer         = _Tp*;
  using element_type    = _Tp;
  using difference_type = ptrdiff_t;

  template <class _Up>
  using rebind = _Up*;

private:
  struct __nat
  {};

public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static pointer
  pointer_to(conditional_t<is_void<element_type>::value, __nat, element_type>& __r) noexcept
  {
    return _CUDA_VSTD::addressof(__r);
  }
};

template <class _From, class _To>
struct __rebind_pointer
{
  using type = typename pointer_traits<_From>::template rebind<_To>;
};

// to_address

template <class _Pointer, class = void>
struct __to_address_helper;

template <class _Tp>
_CCCL_API constexpr _Tp* __to_address(_Tp* __p) noexcept
{
  static_assert(!is_function<_Tp>::value, "_Tp is a function type");
  return __p;
}

template <class _Pointer, class = void>
struct _HasToAddress : false_type
{};

template <class _Pointer>
struct _HasToAddress<_Pointer, decltype((void) pointer_traits<_Pointer>::to_address(declval<const _Pointer&>()))>
    : true_type
{};

template <class _Pointer, class = void>
struct _HasArrow : false_type
{};

template <class _Pointer>
struct _HasArrow<_Pointer, decltype((void) declval<const _Pointer&>().operator->())> : true_type
{};

template <class _Pointer>
struct _IsFancyPointer
{
  static const bool value = _HasArrow<_Pointer>::value || _HasToAddress<_Pointer>::value;
};

// enable_if is needed here to avoid instantiating checks for fancy pointers on raw pointers
template <class _Pointer, class = enable_if_t<_And<is_class<_Pointer>, _IsFancyPointer<_Pointer>>::value>>
_CCCL_API constexpr decay_t<decltype(__to_address_helper<_Pointer>::__call(declval<const _Pointer&>()))>
__to_address(const _Pointer& __p) noexcept
{
  return __to_address_helper<_Pointer>::__call(__p);
}

template <class _Pointer, class>
struct __to_address_helper
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr static decltype(_CUDA_VSTD::__to_address(declval<const _Pointer&>().operator->()))
  __call(const _Pointer& __p) noexcept
  {
    return _CUDA_VSTD::__to_address(__p.operator->());
  }
};

template <class _Pointer>
struct __to_address_helper<_Pointer, decltype((void) pointer_traits<_Pointer>::to_address(declval<const _Pointer&>()))>
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr static decltype(pointer_traits<_Pointer>::to_address(declval<const _Pointer&>()))
  __call(const _Pointer& __p) noexcept
  {
    return pointer_traits<_Pointer>::to_address(__p);
  }
};

template <class _Tp>
_CCCL_API constexpr auto to_address(_Tp* __p) noexcept
{
  return _CUDA_VSTD::__to_address(__p);
}

template <class _Pointer>
_CCCL_API constexpr auto to_address(const _Pointer& __p) noexcept -> decltype(_CUDA_VSTD::__to_address(__p))
{
  return _CUDA_VSTD::__to_address(__p);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_POINTER_TRAITS_H
