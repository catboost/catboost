// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_MEM_FUN_REF_H
#define _LIBCUDACXX___FUNCTIONAL_MEM_FUN_REF_H

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

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class _Sp, class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED mem_fun_t : public __unary_function<_Tp*, _Sp>
{
  _Sp (_Tp::*__p_)();

public:
  _CCCL_API inline explicit mem_fun_t(_Sp (_Tp::*__p)())
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(_Tp* __p) const
  {
    return (__p->*__p_)();
  }
};

template <class _Sp, class _Tp, class _Ap>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED mem_fun1_t : public __binary_function<_Tp*, _Ap, _Sp>
{
  _Sp (_Tp::*__p_)(_Ap);

public:
  _CCCL_API inline explicit mem_fun1_t(_Sp (_Tp::*__p)(_Ap))
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(_Tp* __p, _Ap __x) const
  {
    return (__p->*__p_)(__x);
  }
};

template <class _Sp, class _Tp>
_LIBCUDACXX_DEPRECATED _CCCL_API inline mem_fun_t<_Sp, _Tp> mem_fun(_Sp (_Tp::*__f)())
{
  return mem_fun_t<_Sp, _Tp>(__f);
}

template <class _Sp, class _Tp, class _Ap>
_LIBCUDACXX_DEPRECATED _CCCL_API inline mem_fun1_t<_Sp, _Tp, _Ap> mem_fun(_Sp (_Tp::*__f)(_Ap))
{
  return mem_fun1_t<_Sp, _Tp, _Ap>(__f);
}

template <class _Sp, class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED mem_fun_ref_t : public __unary_function<_Tp, _Sp>
{
  _Sp (_Tp::*__p_)();

public:
  _CCCL_API inline explicit mem_fun_ref_t(_Sp (_Tp::*__p)())
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(_Tp& __p) const
  {
    return (__p.*__p_)();
  }
};

template <class _Sp, class _Tp, class _Ap>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED mem_fun1_ref_t : public __binary_function<_Tp, _Ap, _Sp>
{
  _Sp (_Tp::*__p_)(_Ap);

public:
  _CCCL_API inline explicit mem_fun1_ref_t(_Sp (_Tp::*__p)(_Ap))
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(_Tp& __p, _Ap __x) const
  {
    return (__p.*__p_)(__x);
  }
};

template <class _Sp, class _Tp>
_LIBCUDACXX_DEPRECATED _CCCL_API inline mem_fun_ref_t<_Sp, _Tp> mem_fun_ref(_Sp (_Tp::*__f)())
{
  return mem_fun_ref_t<_Sp, _Tp>(__f);
}

template <class _Sp, class _Tp, class _Ap>
_LIBCUDACXX_DEPRECATED _CCCL_API inline mem_fun1_ref_t<_Sp, _Tp, _Ap> mem_fun_ref(_Sp (_Tp::*__f)(_Ap))
{
  return mem_fun1_ref_t<_Sp, _Tp, _Ap>(__f);
}

template <class _Sp, class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED const_mem_fun_t : public __unary_function<const _Tp*, _Sp>
{
  _Sp (_Tp::*__p_)() const;

public:
  _CCCL_API inline explicit const_mem_fun_t(_Sp (_Tp::*__p)() const)
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(const _Tp* __p) const
  {
    return (__p->*__p_)();
  }
};

template <class _Sp, class _Tp, class _Ap>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED
const_mem_fun1_t : public __binary_function<const _Tp*, _Ap, _Sp>
{
  _Sp (_Tp::*__p_)(_Ap) const;

public:
  _CCCL_API inline explicit const_mem_fun1_t(_Sp (_Tp::*__p)(_Ap) const)
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(const _Tp* __p, _Ap __x) const
  {
    return (__p->*__p_)(__x);
  }
};

template <class _Sp, class _Tp>
_LIBCUDACXX_DEPRECATED _CCCL_API inline const_mem_fun_t<_Sp, _Tp> mem_fun(_Sp (_Tp::*__f)() const)
{
  return const_mem_fun_t<_Sp, _Tp>(__f);
}

template <class _Sp, class _Tp, class _Ap>
_LIBCUDACXX_DEPRECATED _CCCL_API inline const_mem_fun1_t<_Sp, _Tp, _Ap> mem_fun(_Sp (_Tp::*__f)(_Ap) const)
{
  return const_mem_fun1_t<_Sp, _Tp, _Ap>(__f);
}

template <class _Sp, class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED const_mem_fun_ref_t : public __unary_function<_Tp, _Sp>
{
  _Sp (_Tp::*__p_)() const;

public:
  _CCCL_API inline explicit const_mem_fun_ref_t(_Sp (_Tp::*__p)() const)
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(const _Tp& __p) const
  {
    return (__p.*__p_)();
  }
};

template <class _Sp, class _Tp, class _Ap>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED
const_mem_fun1_ref_t : public __binary_function<_Tp, _Ap, _Sp>
{
  _Sp (_Tp::*__p_)(_Ap) const;

public:
  _CCCL_API inline explicit const_mem_fun1_ref_t(_Sp (_Tp::*__p)(_Ap) const)
      : __p_(__p)
  {}
  _CCCL_API inline _Sp operator()(const _Tp& __p, _Ap __x) const
  {
    return (__p.*__p_)(__x);
  }
};

template <class _Sp, class _Tp>
_LIBCUDACXX_DEPRECATED _CCCL_API inline const_mem_fun_ref_t<_Sp, _Tp> mem_fun_ref(_Sp (_Tp::*__f)() const)
{
  return const_mem_fun_ref_t<_Sp, _Tp>(__f);
}

template <class _Sp, class _Tp, class _Ap>
_LIBCUDACXX_DEPRECATED _CCCL_API inline const_mem_fun1_ref_t<_Sp, _Tp, _Ap> mem_fun_ref(_Sp (_Tp::*__f)(_Ap) const)
{
  return const_mem_fun1_ref_t<_Sp, _Tp, _Ap>(__f);
}

_CCCL_SUPPRESS_DEPRECATED_POP

#endif // defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_MEM_FUN_REF_H
