/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file functional.h
 *  \brief Function objects and tools for manipulating them
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

#include <thrust/detail/functional/actor.h>

#include <cuda/functional>
#include <cuda/std/functional>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup predefined_function_objects Predefined Function Objects
 *  \ingroup function_objects
 */

/*! \addtogroup arithmetic_operations Arithmetic Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

template <class T = void>
using divides CCCL_DEPRECATED_BECAUSE("Use cuda::std::divides instead") = ::cuda::std::divides<T>;
template <class T = void>
using minus CCCL_DEPRECATED_BECAUSE("Use cuda::std::minus instead") = ::cuda::std::minus<T>;
template <class T = void>
using modulus CCCL_DEPRECATED_BECAUSE("Use cuda::std::modulus instead") = ::cuda::std::modulus<T>;
template <class T = void>
using multiplies CCCL_DEPRECATED_BECAUSE("Use cuda::std::multiplies instead") = ::cuda::std::multiplies<T>;
template <class T = void>
using negate CCCL_DEPRECATED_BECAUSE("Use cuda::std::negate instead") = ::cuda::std::negate<T>;
template <class T = void>
using plus CCCL_DEPRECATED_BECAUSE("Use cuda::std::plus instead") = ::cuda::std::plus<T>;

/*! \p square is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>square<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x*x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x*x</tt> must be defined and must have a return type that is
 * convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>square</tt> to square
 *  the elements of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                    thrust::square<float>());
 *  // V2 is now {1, 4, 9, ..., 1000000}
 *  \endcode
 */
template <typename T = void>
struct square
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T operator()(const T& x) const
  {
    return x * x;
  }
};

template <>
struct square<void>
{
  using is_transparent = void;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T>
  _CCCL_HOST_DEVICE constexpr T operator()(const T& x) const noexcept(noexcept(x * x))
  {
    return x * x;
  }
};

/*! \}
 */

/*! \addtogroup comparison_operations Comparison Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

//! deprecated [since 3.1]
template <class T = void>
using equal_to CCCL_DEPRECATED_BECAUSE("Use cuda::std::equal_to instead") = ::cuda::std::equal_to<T>;
//! deprecated [since 3.1]
template <class T = void>
using greater CCCL_DEPRECATED_BECAUSE("Use cuda::std::greater instead") = ::cuda::std::greater<T>;
//! deprecated [since 3.1]
template <class T = void>
using greater_equal CCCL_DEPRECATED_BECAUSE("Use cuda::std::greater_equal instead") = ::cuda::std::greater_equal<T>;
//! deprecated [since 3.1]
template <class T = void>
using less CCCL_DEPRECATED_BECAUSE("Use cuda::std::less instead") = ::cuda::std::less<T>;
//! deprecated [since 3.1]
template <class T = void>
using less_equal CCCL_DEPRECATED_BECAUSE("Use cuda::std::less_equal instead") = ::cuda::std::less_equal<T>;
//! deprecated [since 3.1]
template <class T = void>
using not_equal_to CCCL_DEPRECATED_BECAUSE("Use cuda::std::not_equal_to instead") = ::cuda::std::not_equal_to<T>;

/*! \}
 */

/*! \addtogroup logical_operations Logical Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

//! deprecated [since 3.1]
template <class T = void>
using logical_and CCCL_DEPRECATED_BECAUSE("Use cuda::std::logical_and instead") = ::cuda::std::logical_and<T>;
//! deprecated [since 3.1]
template <class T = void>
using logical_not CCCL_DEPRECATED_BECAUSE("Use cuda::std::logical_not instead") = ::cuda::std::logical_not<T>;
//! deprecated [since 3.1]
template <class T = void>
using logical_or CCCL_DEPRECATED_BECAUSE("Use cuda::std::logical_or instead") = ::cuda::std::logical_or<T>;

/*! \}
 */

/*! \addtogroup bitwise_operations Bitwise Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

//! deprecated [since 3.1]
template <class T = void>
using bit_and CCCL_DEPRECATED_BECAUSE("Use cuda::std::bit_and instead") = ::cuda::std::bit_and<T>;
//! deprecated [since 3.1]
template <class T = void>
using bit_or CCCL_DEPRECATED_BECAUSE("Use cuda::std::bit_or instead") = ::cuda::std::bit_or<T>;
//! deprecated [since 3.1]
template <class T = void>
using bit_xor CCCL_DEPRECATED_BECAUSE("Use cuda::std::bit_xor instead") = ::cuda::std::bit_xor<T>;

/*! \}
 */

/*! \addtogroup generalized_identity_operations Generalized Identity Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

//! deprecated [since 3.1]
template <class T = void>
using maximum CCCL_DEPRECATED_BECAUSE("Use cuda::maximum instead") = ::cuda::maximum<T>;
//! deprecated [since 3.1]
template <class T = void>
using minimum CCCL_DEPRECATED_BECAUSE("Use cuda::minimum instead") = ::cuda::minimum<T>;

/*! \p project1st is a function object that takes two arguments and returns
 *  its first argument; the second argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project1st<int> pj1;
 *  assert(x == pj1(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project2nd
 */
template <typename T1 = void, typename T2 = void>
struct project1st
{
  /*! Function call operator. The return value is <tt>lhs</tt>.
   */
  _CCCL_HOST_DEVICE constexpr const T1& operator()(const T1& lhs, const T2& /*rhs*/) const
  {
    return lhs;
  }
};

template <>
struct project1st<void, void>
{
  using is_transparent = void;
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&&) const noexcept(noexcept(THRUST_FWD(t1)))
    -> decltype(THRUST_FWD(t1))
  {
    return THRUST_FWD(t1);
  }
};

/*! \p project2nd is a function object that takes two arguments and returns
 *  its second argument; the first argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project2nd<int> pj2;
 *  assert(y == pj2(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project1st
 */
template <typename T1 = void, typename T2 = void>
struct project2nd
{
  /*! Function call operator. The return value is <tt>rhs</tt>.
   */
  _CCCL_HOST_DEVICE constexpr const T2& operator()(const T1& /*lhs*/, const T2& rhs) const
  {
    return rhs;
  }
}; // end project2nd

template <>
struct project2nd<void, void>
{
  using is_transparent = void;
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE constexpr auto operator()(T1&&, T2&& t2) const noexcept(noexcept(THRUST_FWD(t2)))
    -> decltype(THRUST_FWD(t2))
  {
    return THRUST_FWD(t2);
  }
};

/*! \}
 */

// odds and ends

/*! \addtogroup function_object_adaptors
 *  \{
 */

//! deprecated [since 3.1]
#ifdef _CCCL_DOXYGEN_INVOKED
using ::cuda::std::not_fn;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
_CCCL_TEMPLATE(class _Fn)
_CCCL_REQUIRES(::cuda::std::is_constructible_v<::cuda::std::decay_t<_Fn>, _Fn>
                 _CCCL_AND ::cuda::std::is_move_constructible_v<::cuda::std::decay_t<_Fn>>)
CCCL_DEPRECATED_BECAUSE("Use cuda::std::not_fn instead")
[[nodiscard]] _CCCL_API constexpr auto not_fn(_Fn&& __f)
{
  return ::cuda::std::not_fn(::cuda::std::forward<_Fn>(__f));
}
#endif // !_CCCL_DOXYGEN_INVOKED
/*! \}
 */

/*! \addtogroup placeholder_objects Placeholder Objects
 *  \ingroup function_objects
 *  \{
 */

/*! \namespace thrust::placeholders
 *  \brief Facilities for constructing simple functions inline.
 *
 *  Objects in the \p thrust::placeholders namespace may be used to create simple arithmetic functions inline
 *  in an algorithm invocation. Combining placeholders such as \p _1 and \p _2 with arithmetic operations such as \c +
 *  creates an unnamed function object which applies the operation to their arguments.
 *
 *  The type of placeholder objects is implementation-defined.
 *
 *  The following code snippet demonstrates how to use the placeholders \p _1 and \p _2 with \p thrust::transform
 *  to implement the SAXPY computation:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<float> x(4), y(4);
 *    x[0] = 1;
 *    x[1] = 2;
 *    x[2] = 3;
 *    x[3] = 4;
 *
 *    y[0] = 1;
 *    y[1] = 1;
 *    y[2] = 1;
 *    y[3] = 1;
 *
 *    float a = 2.0f;
 *
 *    using namespace thrust::placeholders;
 *
 *    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
 *      a * _1 + _2
 *    );
 *
 *    // y is now {3, 5, 7, 9}
 *  }
 *  \endcode
 */
namespace placeholders
{

/*! \p thrust::placeholders::_1 is the placeholder for the first function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<0>::type _1;

/*! \p thrust::placeholders::_2 is the placeholder for the second function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<1>::type _2;

/*! \p thrust::placeholders::_3 is the placeholder for the third function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<2>::type _3;

/*! \p thrust::placeholders::_4 is the placeholder for the fourth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<3>::type _4;

/*! \p thrust::placeholders::_5 is the placeholder for the fifth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<4>::type _5;

/*! \p thrust::placeholders::_6 is the placeholder for the sixth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<5>::type _6;

/*! \p thrust::placeholders::_7 is the placeholder for the seventh function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<6>::type _7;

/*! \p thrust::placeholders::_8 is the placeholder for the eighth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<7>::type _8;

/*! \p thrust::placeholders::_9 is the placeholder for the ninth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<8>::type _9;

/*! \p thrust::placeholders::_10 is the placeholder for the tenth function parameter.
 */
_CCCL_GLOBAL_CONSTANT thrust::detail::functional::placeholder<9>::type _10;

} // namespace placeholders

/*! \} // placeholder_objects
 */

THRUST_NAMESPACE_END

#include <thrust/detail/functional/operators.h>
#include <thrust/detail/type_traits/is_commutative.h>
