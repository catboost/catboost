//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ALIGNED_STORAGE_H
#define _LIBCUDACXX___TYPE_TRAITS_ALIGNED_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __align_type
{
  static const size_t value = _LIBCUDACXX_PREFERRED_ALIGNOF(_Tp);
  using type                = _Tp;
};

struct __struct_double
{
  long double __lx;
};
struct __struct_double4
{
  double __lx[4];
};

using __all_types =
  __type_list<__align_type<unsigned char>,
              __align_type<unsigned short>,
              __align_type<unsigned int>,
              __align_type<unsigned long>,
              __align_type<unsigned long long>,
              __align_type<double>,
              __align_type<long double>,
              __align_type<__struct_double>,
              __align_type<__struct_double4>,
              __align_type<int*>>;

template <size_t _Align>
struct _CCCL_ALIGNAS(_Align) __fallback_overaligned
{
  static const size_t value = _Align;
  using type                = __fallback_overaligned;
};

template <size_t _Align>
struct __has_align
{
  template <class _Ty>
  using __call = bool_constant<_Align == _Ty::value>;
};

template <class _TL, size_t _Align>
struct __find_pod
    : public __type_front<__type_find_if<__type_push_back<_TL, __fallback_overaligned<_Align>>, __has_align<_Align>>>
{};

_CCCL_HOST_DEVICE constexpr size_t __select_align_fn_2_(size_t __len, size_t __min, size_t __max)
{
  return __len < __max ? __min : __max;
}

_CCCL_HOST_DEVICE constexpr size_t __select_align_fn_(size_t __len, size_t __a1, size_t __a2)
{
  return __select_align_fn_2_(__len, __a2 < __a1 ? __a2 : __a1, __a1 < __a2 ? __a2 : __a1);
}

template <size_t _Len>
struct __select_align_fn
{
  template <class _State, class _Ty>
  using __call = integral_constant<size_t, __select_align_fn_(_Len, _State::value, _Ty::value)>;
};

template <class _TL, size_t _Len>
struct __find_max_align : public __type_fold_left<_TL, integral_constant<size_t, 0>, __select_align_fn<_Len>>
{};

template <size_t _Len, size_t _Align = __find_max_align<__all_types, _Len>::value>
struct _CCCL_TYPE_VISIBILITY_DEFAULT aligned_storage
{
  using _Aligner = typename __find_pod<__all_types, _Align>::type;
  union type
  {
    _Aligner __align;
    unsigned char __data[(_Len + _Align - 1) / _Align * _Align];
  };
};

template <size_t _Len, size_t _Align = __find_max_align<__all_types, _Len>::value>
using aligned_storage_t _CCCL_NODEBUG_ALIAS = typename aligned_storage<_Len, _Align>::type;

#define _CREATE_ALIGNED_STORAGE_SPECIALIZATION(n)               \
  template <size_t _Len>                                        \
  struct _CCCL_TYPE_VISIBILITY_DEFAULT aligned_storage<_Len, n> \
  {                                                             \
    struct _CCCL_ALIGNAS(n) type                                \
    {                                                           \
      unsigned char __lx[(_Len + n - 1) / n * n];               \
    };                                                          \
  }

_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x1);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x2);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x4);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x8);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x10);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x20);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x40);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x80);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x100);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x200);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x400);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x800);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x1000);
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x2000);
// PE/COFF does not support alignment beyond 8192 (=0x2000)
#if !defined(_WIN32)
_CREATE_ALIGNED_STORAGE_SPECIALIZATION(0x4000);
#endif // !defined(_WIN32)

#undef _CREATE_ALIGNED_STORAGE_SPECIALIZATION

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_ALIGNED_STORAGE_H
