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

#ifndef _LIBCUDACXX___NEW_ALLOCATE_H
#define _LIBCUDACXX___NEW_ALLOCATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__new/device_new.h>
#include <cuda/std/cstddef>

#if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION() && !_CCCL_COMPILER(NVRTC)
#  include <new> // for align_val_t
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOCATION() !_CCCL_COMPILER(NVRTC)

#if !defined(__cpp_sized_deallocation) || __cpp_sized_deallocation < 201309L
#  define _LIBCUDACXX_HAS_SIZED_DEALLOCATION() 0
#else
#  define _LIBCUDACXX_HAS_SIZED_DEALLOCATION() 1
#endif

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_API constexpr bool __is_overaligned_for_new(size_t __align) noexcept
{
#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
  return __align > __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else // ^^^ __STDCPP_DEFAULT_NEW_ALIGNMENT__ ^^^ / vvv !__STDCPP_DEFAULT_NEW_ALIGNMENT__ vvv
  return __align > alignof(max_align_t);
#endif // !__STDCPP_DEFAULT_NEW_ALIGNMENT__
}

template <class... _Args>
_CCCL_API inline void* __cccl_operator_new(_Args... __args)
{
  // Those builtins are not usable on device and the tests crash when using them
#if defined(_CCCL_BUILTIN_OPERATOR_NEW)
  return _CCCL_BUILTIN_OPERATOR_NEW(__args...);
#else // ^^^ _CCCL_BUILTIN_OPERATOR_NEW ^^^ / vvv !_CCCL_BUILTIN_OPERATOR_NEW vvv
  return ::operator new(__args...);
#endif // !_CCCL_BUILTIN_OPERATOR_NEW
}

template <class... _Args>
_CCCL_API inline void __cccl_operator_delete(_Args... __args)
{
  // Those builtins are not usable on device and the tests crash when using them
#if defined(_CCCL_BUILTIN_OPERATOR_DELETE)
  _CCCL_BUILTIN_OPERATOR_DELETE(__args...);
#else // ^^^ _CCCL_BUILTIN_OPERATOR_DELETE ^^^ / vvv !_CCCL_BUILTIN_OPERATOR_DELETE vvv
  ::operator delete(__args...);
#endif // !_CCCL_BUILTIN_OPERATOR_DELETE
}

#if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
using ::std::align_val_t;
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()

_CCCL_API inline void* __cccl_allocate(size_t __size, [[maybe_unused]] size_t __align)
{
#if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
  if (_CUDA_VSTD::__is_overaligned_for_new(__align))
  {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return _CUDA_VSTD::__cccl_operator_new(__size, __align_val);
  }
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
  return _CUDA_VSTD::__cccl_operator_new(__size);
}

template <class... _Args>
_CCCL_API inline void __do_deallocate_handle_size(void* __ptr, [[maybe_unused]] size_t __size, _Args... __args)
{
#if _LIBCUDACXX_HAS_SIZED_DEALLOCATION()
  return _CUDA_VSTD::__cccl_operator_delete(__ptr, __size, __args...);
#else // ^^^ _LIBCUDACXX_HAS_SIZED_DEALLOCATION() ^^^ / vvv !_LIBCUDACXX_HAS_SIZED_DEALLOCATION() vvv
  return _CUDA_VSTD::__cccl_operator_delete(__ptr, __args...);
#endif // !_LIBCUDACXX_HAS_SIZED_DEALLOCATION()
}

_CCCL_API inline void __cccl_deallocate(void* __ptr, size_t __size, [[maybe_unused]] size_t __align)
{
#if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
  if (_CUDA_VSTD::__is_overaligned_for_new(__align))
  {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return _CUDA_VSTD::__do_deallocate_handle_size(__ptr, __size, __align_val);
  }
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
  return _CUDA_VSTD::__do_deallocate_handle_size(__ptr, __size);
}

_CCCL_API inline void __cccl_deallocate_unsized(void* __ptr, [[maybe_unused]] size_t __align)
{
#if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
  if (_CUDA_VSTD::__is_overaligned_for_new(__align))
  {
    const align_val_t __align_val = static_cast<align_val_t>(__align);
    return _CUDA_VSTD::__cccl_operator_delete(__ptr, __align_val);
  }
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
  return _CUDA_VSTD::__cccl_operator_delete(__ptr);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NEW_ALLOCATE_H
