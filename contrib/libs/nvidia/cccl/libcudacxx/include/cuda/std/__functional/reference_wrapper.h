// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H
#define _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/weak_result_type.h>
#include <cuda/std/__fwd/reference_wrapper.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT reference_wrapper : public __weak_result_type<_Tp>
{
public:
  // types
  using type = _Tp;

private:
  type* __f_;

  static _CCCL_API inline void __fun(_Tp&) noexcept;
  static void __fun(_Tp&&) = delete;

public:
  template <class _Up,
            class = enable_if_t<!__is_same_uncvref<_Up, reference_wrapper>::value, decltype(__fun(declval<_Up>()))>>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 reference_wrapper(_Up&& __u) noexcept(noexcept(__fun(declval<_Up>())))
  {
    type& __f = static_cast<_Up&&>(__u);
    __f_      = _CUDA_VSTD::addressof(__f);
  }

  // access
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 operator type&() const noexcept
  {
    return *__f_;
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 type& get() const noexcept
  {
    return *__f_;
  }

  // invoke
  template <class... _ArgTypes>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 typename __invoke_of<type&, _ArgTypes...>::type
  operator()(_ArgTypes&&... __args) const noexcept(_CCCL_TRAIT(is_nothrow_invocable, _Tp&, _ArgTypes...))
  {
    return _CUDA_VSTD::__invoke(get(), _CUDA_VSTD::forward<_ArgTypes>(__args)...);
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE reference_wrapper(_Tp&) -> reference_wrapper<_Tp>;

template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 reference_wrapper<_Tp> ref(_Tp& __t) noexcept
{
  return reference_wrapper<_Tp>(__t);
}

template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 reference_wrapper<_Tp> ref(reference_wrapper<_Tp> __t) noexcept
{
  return __t;
}

template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 reference_wrapper<const _Tp> cref(const _Tp& __t) noexcept
{
  return reference_wrapper<const _Tp>(__t);
}

template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 reference_wrapper<const _Tp> cref(reference_wrapper<_Tp> __t) noexcept
{
  return __t;
}

template <class _Tp>
void ref(const _Tp&&) = delete;
template <class _Tp>
void cref(const _Tp&&) = delete;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_REFERENCE_WRAPPER_H
