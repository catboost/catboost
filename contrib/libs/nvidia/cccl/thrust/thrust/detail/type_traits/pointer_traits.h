/*
 *  Copyright 2008-2020 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/detail/type_traits/is_thrust_pointer.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename Ptr>
struct pointer_element;

template <template <typename...> class Ptr, typename FirstArg, typename... Args>
struct pointer_element<Ptr<FirstArg, Args...>>
{
  using type = FirstArg;
};

template <typename T>
struct pointer_element<T*>
{
  using type = T;
};

template <typename Ptr>
struct pointer_difference
{
  using type = typename Ptr::difference_type;
};

template <typename T>
struct pointer_difference<T*>
{
  using type = ::cuda::std::ptrdiff_t;
};

template <typename Ptr, typename T>
struct rebind_pointer;

template <typename T, typename U>
struct rebind_pointer<T*, U>
{
  using type = U*;
};

// Rebind generic fancy pointers.
template <template <typename, typename...> class Ptr, typename OldT, typename... Tail, typename T>
struct rebind_pointer<Ptr<OldT, Tail...>, T>
{
  using type = Ptr<T, Tail...>;
};

// Rebind `thrust::pointer`-like things with `thrust::reference`-like references.
template <template <typename, typename, typename, typename...> class Ptr,
          typename OldT,
          typename Tag,
          template <typename...> class Ref,
          typename... RefTail,
          typename... PtrTail,
          typename T>
struct rebind_pointer<Ptr<OldT, Tag, Ref<OldT, RefTail...>, PtrTail...>, T>
{
  //  static_assert(::cuda::std::is_same<OldT, Tag>::value, "0");
  using type = Ptr<T, Tag, Ref<T, RefTail...>, PtrTail...>;
};

// Rebind `thrust::pointer`-like things with `thrust::reference`-like references
// and templated derived types.
template <template <typename, typename, typename, typename...> class Ptr,
          typename OldT,
          typename Tag,
          template <typename...> class Ref,
          typename... RefTail,
          template <typename...> class DerivedPtr,
          typename... DerivedPtrTail,
          typename T>
struct rebind_pointer<Ptr<OldT, Tag, Ref<OldT, RefTail...>, DerivedPtr<OldT, DerivedPtrTail...>>, T>
{
  //  static_assert(::cuda::std::is_same<OldT, Tag>::value, "1");
  using type = Ptr<T, Tag, Ref<T, RefTail...>, DerivedPtr<T, DerivedPtrTail...>>;
};

// Rebind `thrust::pointer`-like things with native reference types.
template <template <typename, typename, typename, typename...> class Ptr,
          typename OldT,
          typename Tag,
          typename... PtrTail,
          typename T>
struct rebind_pointer<Ptr<OldT, Tag, typename ::cuda::std::add_lvalue_reference<OldT>::type, PtrTail...>, T>
{
  //  static_assert(::cuda::std::is_same<OldT, Tag>::value, "2");
  using type = Ptr<T, Tag, typename ::cuda::std::add_lvalue_reference<T>::type, PtrTail...>;
};

// Rebind `thrust::pointer`-like things with native reference types and templated
// derived types.
template <template <typename, typename, typename, typename...> class Ptr,
          typename OldT,
          typename Tag,
          template <typename...> class DerivedPtr,
          typename... DerivedPtrTail,
          typename T>
struct rebind_pointer<
  Ptr<OldT, Tag, typename ::cuda::std::add_lvalue_reference<OldT>::type, DerivedPtr<OldT, DerivedPtrTail...>>,
  T>
{
  //  static_assert(::cuda::std::is_same<OldT, Tag>::value, "3");
  using type = Ptr<T, Tag, typename ::cuda::std::add_lvalue_reference<T>::type, DerivedPtr<T, DerivedPtrTail...>>;
};

namespace pointer_traits_detail
{

template <typename Void>
struct capture_address
{
  template <typename T>
  _CCCL_HOST_DEVICE capture_address(T& r)
      : m_addr(&r)
  {}

  inline _CCCL_HOST_DEVICE Void* operator&() const
  {
    return m_addr;
  }

  Void* m_addr;
};

// metafunction to compute the type of pointer_to's parameter below
template <typename T>
struct pointer_to_param
    : thrust::detail::eval_if<::cuda::std::is_void<T>::value,
                              ::cuda::std::type_identity<capture_address<T>>,
                              ::cuda::std::add_lvalue_reference<T>>
{};

} // namespace pointer_traits_detail

template <typename Ptr>
struct pointer_traits
{
  using pointer         = Ptr;
  using reference       = typename Ptr::reference;
  using element_type    = typename pointer_element<Ptr>::type;
  using difference_type = typename pointer_difference<Ptr>::type;

  template <typename U>
  struct rebind
  {
    using other = typename rebind_pointer<Ptr, U>::type;
  };

  _CCCL_HOST_DEVICE inline static pointer
  pointer_to(typename pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    // XXX this is supposed to be pointer::pointer_to(&r); (i.e., call a static member function of pointer called
    // pointer_to)
    //     assume that pointer has a constructor from raw pointer instead

    return pointer(&r);
  }

  // thrust additions follow
  using raw_pointer = typename pointer_raw_pointer<Ptr>::type;

  _CCCL_HOST_DEVICE inline static raw_pointer get(pointer ptr)
  {
    return ptr.get();
  }
};

template <typename T>
struct pointer_traits<T*>
{
  using pointer         = T*;
  using reference       = T&;
  using element_type    = T;
  using difference_type = typename pointer_difference<T*>::type;

  template <typename U>
  struct rebind
  {
    using other = U*;
  };

  _CCCL_HOST_DEVICE inline static pointer
  pointer_to(typename pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    return &r;
  }

  // thrust additions follow
  using raw_pointer = typename pointer_raw_pointer<T*>::type;

  _CCCL_HOST_DEVICE inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

template <>
struct pointer_traits<void*>
{
  using pointer         = void*;
  using reference       = void;
  using element_type    = void;
  using difference_type = pointer_difference<void*>::type;

  template <typename U>
  struct rebind
  {
    using other = U*;
  };

  _CCCL_HOST_DEVICE inline static pointer pointer_to(pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    return &r;
  }

  // thrust additions follow
  using raw_pointer = pointer_raw_pointer<void*>::type;

  _CCCL_HOST_DEVICE inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

template <>
struct pointer_traits<const void*>
{
  using pointer         = const void*;
  using reference       = const void;
  using element_type    = const void;
  using difference_type = pointer_difference<const void*>::type;

  template <typename U>
  struct rebind
  {
    using other = U*;
  };

  _CCCL_HOST_DEVICE inline static pointer pointer_to(pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    return &r;
  }

  // thrust additions follow
  using raw_pointer = pointer_raw_pointer<const void*>::type;

  _CCCL_HOST_DEVICE inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

template <typename FromPtr, typename ToPtr>
struct is_pointer_system_convertible : ::cuda::std::is_convertible<iterator_system_t<FromPtr>, iterator_system_t<ToPtr>>
{};

template <typename FromPtr, typename ToPtr>
struct is_pointer_convertible
    : ::cuda::std::_And<
        ::cuda::std::is_convertible<typename pointer_element<FromPtr>::type*, typename pointer_element<ToPtr>::type*>,
        is_pointer_system_convertible<FromPtr, ToPtr>>
{};

template <typename FromPtr, typename ToPtr>
struct is_void_pointer_system_convertible
    : ::cuda::std::_And<::cuda::std::is_same<typename pointer_element<FromPtr>::type, void>,
                        is_pointer_system_convertible<FromPtr, ToPtr>>
{};

// avoid inspecting traits of the arguments if they aren't known to be pointers
template <typename FromPtr, typename ToPtr>
struct lazy_is_pointer_convertible
    : thrust::detail::eval_if<is_thrust_pointer_v<FromPtr> && is_thrust_pointer_v<ToPtr>,
                              is_pointer_convertible<FromPtr, ToPtr>,
                              ::cuda::std::type_identity<thrust::detail::false_type>>
{};

template <typename FromPtr, typename ToPtr>
struct lazy_is_void_pointer_system_convertible
    : thrust::detail::eval_if<is_thrust_pointer_v<FromPtr> && is_thrust_pointer_v<ToPtr>,
                              is_void_pointer_system_convertible<FromPtr, ToPtr>,
                              ::cuda::std::type_identity<thrust::detail::false_type>>
{};

template <typename FromPtr, typename ToPtr, typename T = void>
struct enable_if_pointer_is_convertible
    : ::cuda::std::enable_if<lazy_is_pointer_convertible<FromPtr, ToPtr>::type::value, T>
{};

template <typename FromPtr, typename ToPtr, typename T = void>
struct enable_if_void_pointer_is_system_convertible
    : ::cuda::std::enable_if<lazy_is_void_pointer_system_convertible<FromPtr, ToPtr>::type::value, T>
{};

} // namespace detail
THRUST_NAMESPACE_END
