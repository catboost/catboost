//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H
#define _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __numeric_type
{
  _CCCL_API inline static void __test(...);
#if _LIBCUDACXX_HAS_NVFP16()
  _CCCL_API inline static float __test(__half);
#endif // _LIBCUDACXX_HAS_NVBF16()
#if _LIBCUDACXX_HAS_NVBF16()
  _CCCL_API inline static float __test(__nv_bfloat16);
#endif // _LIBCUDACXX_HAS_NVFP16()
  _CCCL_API inline static float __test(float);
  _CCCL_API inline static double __test(char);
  _CCCL_API inline static double __test(int);
  _CCCL_API inline static double __test(unsigned);
  _CCCL_API inline static double __test(long);
  _CCCL_API inline static double __test(unsigned long);
  _CCCL_API inline static double __test(long long);
  _CCCL_API inline static double __test(unsigned long long);
  _CCCL_API inline static double __test(double);
  _CCCL_API inline static long double __test(long double);

  using type              = decltype(__test(declval<_Tp>()));
  static const bool value = _IsNotSame<type, void>::value;
};

template <>
struct __numeric_type<void>
{
  static const bool value = true;
};

template <class _A1, class _A2, class _A3>
struct __is_mixed_extended_floating_point
{
  static constexpr bool value = false;
};

#if _LIBCUDACXX_HAS_NVFP16() && _LIBCUDACXX_HAS_NVBF16()
template <class _A1>
struct __is_mixed_extended_floating_point<_A1, __half, __nv_bfloat16>
{
  static constexpr bool value = true;
};

template <class _A1>
struct __is_mixed_extended_floating_point<_A1, __nv_bfloat16, __half>
{
  static constexpr bool value = true;
};

template <class _A1>
struct __is_mixed_extended_floating_point<__half, _A1, __nv_bfloat16>
{
  static constexpr bool value = true;
};

template <class _A1>
struct __is_mixed_extended_floating_point<__nv_bfloat16, _A1, __half>
{
  static constexpr bool value = true;
};

template <class _A1>
struct __is_mixed_extended_floating_point<__half, __nv_bfloat16, _A1>
{
  static constexpr bool value = true;
};

template <class _A1>
struct __is_mixed_extended_floating_point<__nv_bfloat16, __half, _A1>
{
  static constexpr bool value = true;
};
#endif // _LIBCUDACXX_HAS_NVFP16() && _LIBCUDACXX_HAS_NVBF16()

template <class _A1,
          class _A2 = void,
          class _A3 = void,
          bool      = __numeric_type<_A1>::value && __numeric_type<_A2>::value && __numeric_type<_A3>::value
              && !__is_mixed_extended_floating_point<_A1, _A2, _A3>::value>
class __promote_imp
{
public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
  using __type1 = typename __promote_imp<_A1>::type;
  using __type2 = typename __promote_imp<_A2>::type;
  using __type3 = typename __promote_imp<_A3>::type;

public:
  using type              = decltype(__type1() + __type2() + __type3());
  static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
  using __type1 = typename __promote_imp<_A1>::type;
  using __type2 = typename __promote_imp<_A2>::type;

public:
  using type              = decltype(__type1() + __type2());
  static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
  using type              = typename __numeric_type<_A1>::type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3>
{};

template <class _A1, class _A2 = void, class _A3 = void>
using __promote_t _CCCL_NODEBUG_ALIAS = typename __promote<_A1, _A2, _A3>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H
