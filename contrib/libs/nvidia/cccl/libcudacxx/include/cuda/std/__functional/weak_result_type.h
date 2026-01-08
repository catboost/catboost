// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_WEAK_RESULT_TYPE_H
#define _LIBCUDACXX___FUNCTIONAL_WEAK_RESULT_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/unary_function.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __has_result_type
{
private:
  template <class _Up>
  _CCCL_API inline static false_type __test(...);
  template <class _Up>
  _CCCL_API inline static true_type __test(typename _Up::result_type* = 0);

public:
  static const bool value = decltype(__test<_Tp>(0))::value;
};

// __weak_result_type

template <class _Tp>
struct __derives_from_unary_function
{
private:
  struct __two
  {
    char __lx;
    char __lxx;
  };
  static _CCCL_API inline __two __test(...);
  template <class _Ap, class _Rp>
  static _CCCL_API inline __unary_function<_Ap, _Rp> __test(const volatile __unary_function<_Ap, _Rp>*);

public:
  static const bool value = !is_same<decltype(__test((_Tp*) 0)), __two>::value;
  using type              = decltype(__test((_Tp*) 0));
};

template <class _Tp>
struct __derives_from_binary_function
{
private:
  struct __two
  {
    char __lx;
    char __lxx;
  };
  static __two _CCCL_API inline __test(...);
  template <class _A1, class _A2, class _Rp>
  static _CCCL_API inline __binary_function<_A1, _A2, _Rp> __test(const volatile __binary_function<_A1, _A2, _Rp>*);

public:
  static const bool value = !is_same<decltype(__test((_Tp*) 0)), __two>::value;
  using type              = decltype(__test((_Tp*) 0));
};

template <class _Tp, bool = __derives_from_unary_function<_Tp>::value>
struct __maybe_derive_from_unary_function // bool is true
    : public __derives_from_unary_function<_Tp>::type
{};

template <class _Tp>
struct __maybe_derive_from_unary_function<_Tp, false>
{};

template <class _Tp, bool = __derives_from_binary_function<_Tp>::value>
struct __maybe_derive_from_binary_function // bool is true
    : public __derives_from_binary_function<_Tp>::type
{};

template <class _Tp>
struct __maybe_derive_from_binary_function<_Tp, false>
{};

template <class _Tp, bool = __has_result_type<_Tp>::value>
struct __weak_result_type_imp // bool is true
    : public __maybe_derive_from_unary_function<_Tp>
    , public __maybe_derive_from_binary_function<_Tp>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = typename _Tp::result_type;
#endif
};

template <class _Tp>
struct __weak_result_type_imp<_Tp, false>
    : public __maybe_derive_from_unary_function<_Tp>
    , public __maybe_derive_from_binary_function<_Tp>
{};

template <class _Tp>
struct __weak_result_type : public __weak_result_type_imp<_Tp>
{};

// 0 argument case

template <class _Rp>
struct __weak_result_type<_Rp()>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp>
struct __weak_result_type<_Rp (&)()>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp>
struct __weak_result_type<_Rp (*)()>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

// 1 argument case

template <class _Rp, class _A1>
struct __weak_result_type<_Rp(_A1)> : public __unary_function<_A1, _Rp>
{};

template <class _Rp, class _A1>
struct __weak_result_type<_Rp (&)(_A1)> : public __unary_function<_A1, _Rp>
{};

template <class _Rp, class _A1>
struct __weak_result_type<_Rp (*)(_A1)> : public __unary_function<_A1, _Rp>
{};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)()> : public __unary_function<_Cp*, _Rp>
{};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)() const> : public __unary_function<const _Cp*, _Rp>
{};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)() volatile> : public __unary_function<volatile _Cp*, _Rp>
{};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)() const volatile> : public __unary_function<const volatile _Cp*, _Rp>
{};

// 2 argument case

template <class _Rp, class _A1, class _A2>
struct __weak_result_type<_Rp(_A1, _A2)> : public __binary_function<_A1, _A2, _Rp>
{};

template <class _Rp, class _A1, class _A2>
struct __weak_result_type<_Rp (*)(_A1, _A2)> : public __binary_function<_A1, _A2, _Rp>
{};

template <class _Rp, class _A1, class _A2>
struct __weak_result_type<_Rp (&)(_A1, _A2)> : public __binary_function<_A1, _A2, _Rp>
{};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1)> : public __binary_function<_Cp*, _A1, _Rp>
{};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1) const> : public __binary_function<const _Cp*, _A1, _Rp>
{};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1) volatile> : public __binary_function<volatile _Cp*, _A1, _Rp>
{};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1) const volatile> : public __binary_function<const volatile _Cp*, _A1, _Rp>
{};

// 3 or more arguments

template <class _Rp, class _A1, class _A2, class _A3, class... _A4>
struct __weak_result_type<_Rp(_A1, _A2, _A3, _A4...)>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp, class _A1, class _A2, class _A3, class... _A4>
struct __weak_result_type<_Rp (&)(_A1, _A2, _A3, _A4...)>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp, class _A1, class _A2, class _A3, class... _A4>
struct __weak_result_type<_Rp (*)(_A1, _A2, _A3, _A4...)>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp, class _Cp, class _A1, class _A2, class... _A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...)>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp, class _Cp, class _A1, class _A2, class... _A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...) const>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp, class _Cp, class _A1, class _A2, class... _A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...) volatile>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Rp, class _Cp, class _A1, class _A2, class... _A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...) const volatile>
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using result_type _CCCL_NODEBUG_ALIAS _LIBCUDACXX_DEPRECATED = _Rp;
#endif
};

template <class _Tp, class... _Args>
struct __invoke_return
{
  using type = decltype(_CUDA_VSTD::__invoke(declval<_Tp>(), declval<_Args>()...));
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_WEAK_RESULT_TYPE_H
