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

//! \file thrust/iterator/transform_output_iterator.h
//! \brief An output iterator which adapts another output iterator by applying a function to the result of its
//! dereference before writing it.

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_output_iterator.h>

THRUST_NAMESPACE_BEGIN

template <typename UnaryFunction, typename OutputIterator>
class transform_output_iterator;

namespace detail
{
// Proxy reference that uses Unary Function to transform the rhs of assignment
// operator before writing the result to OutputIterator
template <typename UnaryFunction, typename OutputIterator>
class transform_output_iterator_proxy
{
public:
  _CCCL_HOST_DEVICE transform_output_iterator_proxy(const OutputIterator& out, UnaryFunction fun)
      : out(out)
      , fun(fun)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <typename T>
  _CCCL_HOST_DEVICE transform_output_iterator_proxy operator=(const T& x)
  {
    *out = fun(x);
    return *this;
  }

private:
  OutputIterator out;
  UnaryFunction fun;
};

// Compute the iterator_adaptor instantiation to be used for transform_output_iterator
template <typename UnaryFunction, typename OutputIterator>
struct transform_output_iterator_base
{
  using type =
    iterator_adaptor<transform_output_iterator<UnaryFunction, OutputIterator>,
                     OutputIterator,
                     use_default,
                     use_default,
                     use_default,
                     transform_output_iterator_proxy<UnaryFunction, OutputIterator>>;
};

// Register transform_output_iterator_proxy with 'is_proxy_reference' from type_traits to enable its use with
// algorithms.
template <class UnaryFunction, class OutputIterator>
inline constexpr bool is_proxy_reference_v<transform_output_iterator_proxy<UnaryFunction, OutputIterator>> = true;
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p transform_output_iterator is a special kind of output iterator which transforms a value written upon dereference.
//! This iterator is useful for transforming an output from algorithms without explicitly storing the intermediate
//! result in the memory and applying subsequent transformation, thereby avoiding wasting memory capacity and bandwidth.
//! Using \p transform_iterator facilitates kernel fusion by deferring execution of transformation until the value is
//! written while saving both memory capacity and bandwidth.
//!
//! The following code snippet demonstrated how to create a \p transform_output_iterator which applies \c sqrtf to the
//! assigning value.
//!
//! \code
//! #include <thrust/iterator/transform_output_iterator.h>
//! #include <thrust/device_vector.h>
//!
//! struct square_root
//! {
//!   __host__ __device__
//!   float operator()(float x) const
//!   {
//!     return sqrtf(x);
//!   }
//! };
//!
//! int main()
//! {
//!   thrust::device_vector<float> v(4);
//!
//!   using FloatIterator = thrust::device_vector<float>::iterator;
//!   thrust::transform_output_iterator<square_root, FloatIterator> iter(v.begin(), square_root());
//!
//!   iter[0] =  1.0f;    // stores sqrtf( 1.0f)
//!   iter[1] =  4.0f;    // stores sqrtf( 4.0f)
//!   iter[2] =  9.0f;    // stores sqrtf( 9.0f)
//!   iter[3] = 16.0f;    // stores sqrtf(16.0f)
//!   // iter[4] is an out-of-bounds error
//!
//!   v[0]; // returns 1.0f;
//!   v[1]; // returns 2.0f;
//!   v[2]; // returns 3.0f;
//!   v[3]; // returns 4.0f;
//!
//! }
//! \endcode
//!
//! \see make_transform_output_iterator
template <typename UnaryFunction, typename OutputIterator>
class transform_output_iterator : public detail::transform_output_iterator_base<UnaryFunction, OutputIterator>::type
{
  //! \cond

public:
  using super_t = typename detail::transform_output_iterator_base<UnaryFunction, OutputIterator>::type;

  friend class iterator_core_access;
  //! \endcond

  transform_output_iterator() = default;

  //! This constructor takes as argument an \c OutputIterator and an \c UnaryFunction and copies them to a new \p
  //! transform_output_iterator
  //!
  //! \param out An \c OutputIterator pointing to the output range whereto the result of \p transform_output_iterator's
  //!            \c UnaryFunction will be written.
  //! \param fun An \c UnaryFunction used to transform the objects assigned to this \p transform_output_iterator.
  _CCCL_HOST_DEVICE transform_output_iterator(OutputIterator const& out, UnaryFunction fun)
      : super_t(out)
      , fun(fun)
  {}

  //! \cond

private:
  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    return detail::transform_output_iterator_proxy<UnaryFunction, OutputIterator>(this->base_reference(), fun);
  }

  UnaryFunction fun;
  //! \endcond
};

//! \p make_transform_output_iterator creates a \p transform_output_iterator from an \c OutputIterator and \c
//! UnaryFunction.
//!
//! \param out The \c OutputIterator pointing to the output range of the newly created \p transform_output_iterator
//! \param fun The \c UnaryFunction transform the object before assigning it to \c out by the newly created \p
//! transform_output_iterator
//! \see transform_output_iterator
template <typename UnaryFunction, typename OutputIterator>
transform_output_iterator<UnaryFunction, OutputIterator> _CCCL_HOST_DEVICE
make_transform_output_iterator(OutputIterator out, UnaryFunction fun)
{
  return transform_output_iterator<UnaryFunction, OutputIterator>(out, fun);
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
