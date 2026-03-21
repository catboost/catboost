// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H
#define _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__functional/unary_function.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Arithmetic operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT plus : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x + __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(plus);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT plus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minus : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x - __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(minus);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT multiplies : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x * __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(multiplies);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT multiplies<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT divides : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x / __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(divides);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT divides<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT modulus : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x % __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(modulus);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT modulus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT negate : __unary_function<_Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _CCCL_API inline _Tp operator()(const _Tp& __x) const
  {
    return -__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(negate);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT negate<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  constexpr _CCCL_API inline auto operator()(_Tp&& __x) const noexcept(noexcept(-_CUDA_VSTD::forward<_Tp>(__x)))
    -> decltype(-_CUDA_VSTD::forward<_Tp>(__x))
  {
    return -_CUDA_VSTD::forward<_Tp>(__x);
  }
  using is_transparent = void;
};

// Bitwise operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_and : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x & __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_and);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_and<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_not : __unary_function<_Tp, _Tp>
{
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _CCCL_API inline _Tp operator()(const _Tp& __x) const
  {
    return ~__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_not);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_not<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  constexpr _CCCL_API inline auto operator()(_Tp&& __x) const noexcept(noexcept(~_CUDA_VSTD::forward<_Tp>(__x)))
    -> decltype(~_CUDA_VSTD::forward<_Tp>(__x))
  {
    return ~_CUDA_VSTD::forward<_Tp>(__x);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_or : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _CCCL_API inline _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x | __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_or);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_or<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_xor : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x ^ __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_xor);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_xor<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

// Comparison operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT equal_to : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x == __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(equal_to);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT equal_to<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT not_equal_to : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x != __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(not_equal_to);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT not_equal_to<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x < __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(less);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less_equal : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x <= __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(less_equal);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less_equal<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater_equal : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x >= __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(greater_equal);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater_equal<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x > __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(greater);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

// Logical operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_and : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x && __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_and);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_and<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_not : __unary_function<_Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x) const
  {
    return !__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_not);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_not<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  constexpr _CCCL_API inline auto operator()(_Tp&& __x) const noexcept(noexcept(!_CUDA_VSTD::forward<_Tp>(__x)))
    -> decltype(!_CUDA_VSTD::forward<_Tp>(__x))
  {
    return !_CUDA_VSTD::forward<_Tp>(__x);
  }
  using is_transparent = void;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_or : __binary_function<_Tp, _Tp, bool>
{
  using __result_type = bool; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x || __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_or);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_or<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u);
  }
  using is_transparent = void;
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H
