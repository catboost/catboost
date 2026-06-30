#ifndef _CUDA___UTILITY_NO_INIT_H
#define _CUDA___UTILITY_NO_INIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

struct _CCCL_TYPE_VISIBILITY_DEFAULT no_init_t
{
  _CCCL_HIDE_FROM_ABI explicit no_init_t() = default;
};

_CCCL_GLOBAL_CONSTANT no_init_t no_init{};

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_NO_INIT_H
