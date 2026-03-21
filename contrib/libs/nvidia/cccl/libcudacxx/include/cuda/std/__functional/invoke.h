// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_INVOKE_H
#define _LIBCUDACXX___FUNCTIONAL_INVOKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_core_convertible.h>
#include <cuda/std/__type_traits/is_member_function_pointer.h>
#include <cuda/std/__type_traits/is_member_object_pointer.h>
#include <cuda/std/__type_traits/is_reference_wrapper.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

// TODO: Disentangle the type traits and _CUDA_VSTD::invoke properly

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __any
{
  _CCCL_API inline __any(...);
};

template <class _MP, bool _IsMemberFunctionPtr, bool _IsMemberObjectPtr>
struct __member_pointer_traits_imp
{};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...), true, false>
{
  using _ClassType  = _Class;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...), true, false>
{
  using _ClassType  = _Class;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) const, true, false>
{
  using _ClassType  = _Class const;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) const, true, false>
{
  using _ClassType  = _Class const;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) volatile, true, false>
{
  using _ClassType  = _Class volatile;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) volatile, true, false>
{
  using _ClassType  = _Class volatile;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) const volatile, true, false>
{
  using _ClassType  = _Class const volatile;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) const volatile, true, false>
{
  using _ClassType  = _Class const volatile;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) &, true, false>
{
  using _ClassType  = _Class&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) &, true, false>
{
  using _ClassType  = _Class&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) const&, true, false>
{
  using _ClassType  = _Class const&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) const&, true, false>
{
  using _ClassType  = _Class const&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) volatile&, true, false>
{
  using _ClassType  = _Class volatile&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) volatile&, true, false>
{
  using _ClassType  = _Class volatile&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) const volatile&, true, false>
{
  using _ClassType  = _Class const volatile&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) const volatile&, true, false>
{
  using _ClassType  = _Class const volatile&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) &&, true, false>
{
  using _ClassType  = _Class&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) &&, true, false>
{
  using _ClassType  = _Class&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) const&&, true, false>
{
  using _ClassType  = _Class const&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) const&&, true, false>
{
  using _ClassType  = _Class const&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) volatile&&, true, false>
{
  using _ClassType  = _Class volatile&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) volatile&&, true, false>
{
  using _ClassType  = _Class volatile&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param...) const volatile&&, true, false>
{
  using _ClassType  = _Class const volatile&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param...);
};

template <class _Rp, class _Class, class... _Param>
struct __member_pointer_traits_imp<_Rp (_Class::*)(_Param..., ...) const volatile&&, true, false>
{
  using _ClassType  = _Class const volatile&&;
  using _ReturnType = _Rp;
  using _FnType     = _Rp (*)(_Param..., ...);
};

template <class _Rp, class _Class>
struct __member_pointer_traits_imp<_Rp _Class::*, false, true>
{
  using _ClassType  = _Class;
  using _ReturnType = _Rp;
};

template <class _MP>
struct __member_pointer_traits
    : public __member_pointer_traits_imp<remove_cv_t<_MP>,
                                         is_member_function_pointer<_MP>::value,
                                         is_member_object_pointer<_MP>::value>
{
  //     typedef ... _ClassType;
  //     typedef ... _ReturnType;
  //     typedef ... _FnType;
};

template <class _DecayedFp>
struct __member_pointer_class_type
{};

template <class _Ret, class _ClassType>
struct __member_pointer_class_type<_Ret _ClassType::*>
{
  using type = _ClassType;
};

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = typename decay<_A0>::type,
          class _ClassT  = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet1 =
  enable_if_t<is_member_function_pointer<_DecayFp>::value && is_base_of<_ClassT, _DecayA0>::value>;

template <class _Fp, class _A0, class _DecayFp = decay_t<_Fp>, class _DecayA0 = typename decay<_A0>::type>
using __enable_if_bullet2 =
  enable_if_t<is_member_function_pointer<_DecayFp>::value && __cccl_is_reference_wrapper_v<_DecayA0>>;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = typename decay<_A0>::type,
          class _ClassT  = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet3 =
  enable_if_t<is_member_function_pointer<_DecayFp>::value && !is_base_of<_ClassT, _DecayA0>::value
              && !__cccl_is_reference_wrapper_v<_DecayA0>>;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = typename decay<_A0>::type,
          class _ClassT  = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet4 =
  enable_if_t<is_member_object_pointer<_DecayFp>::value && is_base_of<_ClassT, _DecayA0>::value>;

template <class _Fp, class _A0, class _DecayFp = decay_t<_Fp>, class _DecayA0 = typename decay<_A0>::type>
using __enable_if_bullet5 =
  enable_if_t<is_member_object_pointer<_DecayFp>::value && __cccl_is_reference_wrapper_v<_DecayA0>>;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = typename decay<_A0>::type,
          class _ClassT  = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet6 =
  enable_if_t<is_member_object_pointer<_DecayFp>::value && !is_base_of<_ClassT, _DecayA0>::value
              && !__cccl_is_reference_wrapper_v<_DecayA0>>;

// __invoke forward declarations

// fall back - none of the bullets

template <class... _Args>
_CCCL_API inline __nat __invoke(__any, _Args&&... __args);

// bullets 1, 2 and 3

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class... _Args, class = __enable_if_bullet1<_Fp, _A0>>
_CCCL_API constexpr decltype((_CUDA_VSTD::declval<_A0>().*_CUDA_VSTD::declval<_Fp>())(_CUDA_VSTD::declval<_Args>()...))
__invoke(_Fp&& __f,
         _A0&& __a0,
         _Args&&... __args) noexcept(noexcept((static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...)))
{
  return (static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class... _Args, class = __enable_if_bullet2<_Fp, _A0>>
_CCCL_API constexpr decltype((_CUDA_VSTD::declval<_A0>().get()
                              .*_CUDA_VSTD::declval<_Fp>())(_CUDA_VSTD::declval<_Args>()...))
__invoke(_Fp&& __f, _A0&& __a0, _Args&&... __args) noexcept(noexcept((__a0.get().*__f)(static_cast<_Args&&>(__args)...)))
{
  return (__a0.get().*__f)(static_cast<_Args&&>(__args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class... _Args, class = __enable_if_bullet3<_Fp, _A0>>
_CCCL_API constexpr decltype(((*_CUDA_VSTD::declval<_A0>())
                              .*_CUDA_VSTD::declval<_Fp>())(_CUDA_VSTD::declval<_Args>()...))
__invoke(_Fp&& __f,
         _A0&& __a0,
         _Args&&... __args) noexcept(noexcept(((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...)))
{
  return ((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...);
}

// bullets 4, 5 and 6

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class = __enable_if_bullet4<_Fp, _A0>>
_CCCL_API constexpr decltype(_CUDA_VSTD::declval<_A0>().*_CUDA_VSTD::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0) noexcept(noexcept(static_cast<_A0&&>(__a0).*__f))
{
  return static_cast<_A0&&>(__a0).*__f;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class = __enable_if_bullet5<_Fp, _A0>>
_CCCL_API constexpr decltype(_CUDA_VSTD::declval<_A0>().get().*_CUDA_VSTD::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0) noexcept(noexcept(__a0.get().*__f))
{
  return __a0.get().*__f;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class = __enable_if_bullet6<_Fp, _A0>>
_CCCL_API constexpr decltype((*_CUDA_VSTD::declval<_A0>()).*_CUDA_VSTD::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0) noexcept(noexcept((*static_cast<_A0&&>(__a0)).*__f))
{
  return (*static_cast<_A0&&>(__a0)).*__f;
}

// bullet 7

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class... _Args>
_CCCL_API constexpr decltype(_CUDA_VSTD::declval<_Fp>()(_CUDA_VSTD::declval<_Args>()...))
__invoke(_Fp&& __f, _Args&&... __args) noexcept(noexcept(static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...)))
{
  return static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...);
}

// __invocable
template <class _Ret, class _Fp, class... _Args>
struct __invocable_r
{
  template <class _XFp, class... _XArgs>
  _CCCL_API inline static decltype(_CUDA_VSTD::__invoke(_CUDA_VSTD::declval<_XFp>(), _CUDA_VSTD::declval<_XArgs>()...))
  __try_call(int);

  template <class _XFp, class... _XArgs>
  _CCCL_API inline static __nat __try_call(...);

  // FIXME: Check that _Ret, _Fp, and _Args... are all complete types, cv void,
  // or incomplete array types as required by the standard.
  using _Result = decltype(__try_call<_Fp, _Args...>(0));

  using type              = conditional_t<_IsNotSame<_Result, __nat>::value,
                                          conditional_t<is_void<_Ret>::value, true_type, __is_core_convertible<_Result, _Ret>>,
                                          false_type>;
  static const bool value = type::value;
};
template <class _Fp, class... _Args>
using __invocable = __invocable_r<void, _Fp, _Args...>;

template <bool _IsInvocable, bool _IsCVVoid, class _Ret, class _Fp, class... _Args>
struct __nothrow_invocable_r_imp
{
  static const bool value = false;
};

template <class _Ret, class _Fp, class... _Args>
struct __nothrow_invocable_r_imp<true, false, _Ret, _Fp, _Args...>
{
  using _ThisT = __nothrow_invocable_r_imp;

  template <class _Tp>
  _CCCL_API inline static void __test_noexcept(_Tp) noexcept;

  static const bool value =
    noexcept(_ThisT::__test_noexcept<_Ret>(_CUDA_VSTD::__invoke(declval<_Fp>(), _CUDA_VSTD::declval<_Args>()...)));
};

template <class _Ret, class _Fp, class... _Args>
struct __nothrow_invocable_r_imp<true, true, _Ret, _Fp, _Args...>
{
  static const bool value = noexcept(_CUDA_VSTD::__invoke(_CUDA_VSTD::declval<_Fp>(), _CUDA_VSTD::declval<_Args>()...));
};

template <class _Ret, class _Fp, class... _Args>
using __nothrow_invocable_r =
  __nothrow_invocable_r_imp<__invocable_r<_Ret, _Fp, _Args...>::value, is_void<_Ret>::value, _Ret, _Fp, _Args...>;

template <class _Fp, class... _Args>
using __nothrow_invocable = __nothrow_invocable_r_imp<__invocable<_Fp, _Args...>::value, true, void, _Fp, _Args...>;

template <class _Fp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __invoke_of //
    : public enable_if<__invocable<_Fp, _Args...>::value, typename __invocable_r<void, _Fp, _Args...>::_Result>
{
#if _CCCL_CUDA_COMPILER(NVCC) && defined(__CUDACC_EXTENDED_LAMBDA__) && !_CCCL_DEVICE_COMPILATION()
#  if _CCCL_CUDACC_BELOW(12, 3)
  static_assert(!__nv_is_extended_device_lambda_closure_type(_Fp),
                "Attempt to use an extended __device__ lambda in a context "
                "that requires querying its return type in host code. Use a "
                "named function object, an extended __host__ __device__ lambda, or "
                "cuda::proclaim_return_type instead.");
#  else // ^^^ _CCCL_CUDACC_BELOW(12, 3) ^^^ / vvv _CCCL_CUDACC_AT_LEAST(12, 3) vvv
  static_assert(
    !__nv_is_extended_device_lambda_closure_type(_Fp) || __nv_is_extended_host_device_lambda_closure_type(_Fp)
      || __nv_is_extended_device_lambda_with_preserved_return_type(_Fp),
    "Attempt to use an extended __device__ lambda in a context "
    "that requires querying its return type in host code. Use a "
    "named function object, an extended __host__ __device__ lambda, "
    "cuda::proclaim_return_type, or an extended __device__ lambda "
    "with a trailing return type instead ([] __device__ (...) -> RETURN_TYPE {...}).");
#  endif // _CCCL_CUDACC_AT_LEAST(12, 3)
#endif
};

template <class _Ret, bool = is_void<_Ret>::value>
struct __invoke_void_return_wrapper
{
  template <class... _Args>
  _CCCL_API static constexpr _Ret __call(_Args&&... __args)
  {
    return _CUDA_VSTD::__invoke(_CUDA_VSTD::forward<_Args>(__args)...);
  }
};

template <class _Ret>
struct __invoke_void_return_wrapper<_Ret, true>
{
  template <class... _Args>
  _CCCL_API static constexpr void __call(_Args&&... __args)
  {
    _CUDA_VSTD::__invoke(_CUDA_VSTD::forward<_Args>(__args)...);
  }
};

// is_invocable

template <class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_invocable : integral_constant<bool, __invocable<_Fn, _Args...>::value>
{};

template <class _Ret, class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_invocable_r : integral_constant<bool, __invocable_r<_Ret, _Fn, _Args...>::value>
{};

template <class _Fn, class... _Args>
inline constexpr bool is_invocable_v = is_invocable<_Fn, _Args...>::value;

template <class _Ret, class _Fn, class... _Args>
inline constexpr bool is_invocable_r_v = is_invocable_r<_Ret, _Fn, _Args...>::value;

// is_nothrow_invocable

template <class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_invocable : integral_constant<bool, __nothrow_invocable<_Fn, _Args...>::value>
{};

template <class _Ret, class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_invocable_r : integral_constant<bool, __nothrow_invocable_r<_Ret, _Fn, _Args...>::value>
{};

template <class _Fn, class... _Args>
inline constexpr bool is_nothrow_invocable_v = is_nothrow_invocable<_Fn, _Args...>::value;

template <class _Ret, class _Fn, class... _Args>
inline constexpr bool is_nothrow_invocable_r_v = is_nothrow_invocable_r<_Ret, _Fn, _Args...>::value;

template <class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT invoke_result : __invoke_of<_Fn, _Args...>
{};

template <class _Fn, class... _Args>
using invoke_result_t = typename invoke_result<_Fn, _Args...>::type;

template <class _Fn, class... _Args>
_CCCL_API constexpr invoke_result_t<_Fn, _Args...>
invoke(_Fn&& __f, _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_invocable, _Fn, _Args...))
{
  return _CUDA_VSTD::__invoke(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)...);
}

_CCCL_TEMPLATE(class _Ret, class _Fn, class... _Args)
_CCCL_REQUIRES(_CCCL_TRAIT(is_invocable_r, _Ret, _Fn, _Args...))
_CCCL_API constexpr _Ret invoke_r(_Fn&& __f,
                                  _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_invocable_r, _Ret, _Fn, _Args...))
{
  return __invoke_void_return_wrapper<_Ret>::__call(
    _CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)...);
}

/// The type of intermediate accumulator (according to P2322R6)
template <typename Invocable, typename InputT, typename InitT = InputT>
using __accumulator_t = typename decay<typename _CUDA_VSTD::__invoke_of<Invocable, InitT, InputT>::type>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_INVOKE_H
