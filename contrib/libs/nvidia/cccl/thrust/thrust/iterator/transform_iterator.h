/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/iterator/transform_iterator.h
 *  \brief An iterator which adapts another iterator by applying a function to the result of its dereference
 */

/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

template <class UnaryFunction, class Iterator, class Reference, class Value>
class transform_iterator;

namespace detail
{

template <class UnaryFunc, class Iterator>
struct transform_iterator_reference
{
  // by default, dereferencing the iterator yields the same as the function.
  using type = decltype(::cuda::std::declval<UnaryFunc>()(::cuda::std::declval<it_value_t<Iterator>>()));
};

// for certain function objects, we need to tweak the reference type. Notably, identity functions must decay to values.
// See the implementation of transform_iterator<...>::dereference() for several comments on why this is necessary.
template <class Iterator>
struct transform_iterator_reference<::cuda::std::identity, Iterator>
{
  using type = it_value_t<Iterator>;
};
template <typename Eval, class Iterator>
struct transform_iterator_reference<functional::actor<Eval>, Iterator>
{
  using type = ::cuda::std::remove_reference_t<decltype(::cuda::std::declval<functional::actor<Eval>>()(
    ::cuda::std::declval<it_value_t<Iterator>>()))>;
};

// Type function to compute the iterator_adaptor instantiation to be used for transform_iterator
template <class UnaryFunc, class Iterator, class Reference, class Value>
struct make_transform_iterator_base
{
private:
  using reference  = replace_if_use_default<Reference, transform_iterator_reference<UnaryFunc, Iterator>>;
  using value_type = replace_if_use_default<Value, ::cuda::std::remove_cvref<reference>>;

public:
  using type =
    iterator_adaptor<transform_iterator<UnaryFunc, Iterator, Reference, Value>,
                     Iterator,
                     value_type,
                     use_default,
                     typename ::cuda::std::iterator_traits<Iterator>::iterator_category,
                     reference>;
};

} // namespace detail

//! \addtogroup iterators
//! \{

//!! \addtogroup fancyiterator Fancy Iterators
//!  \ingroup iterators
//!  \{

//! \p transform_iterator is an iterator which represents a pointer into a range of values after transformation by a
//! function. This iterator is useful for creating a range filled with the result of applying an operation to another
//! range without either explicitly storing it in memory, or explicitly executing the transformation. Using \p
//! transform_iterator facilitates kernel fusion by deferring the execution of a transformation until the value is
//! needed while saving both memory capacity and bandwidth.
//!
//! The following code snippet demonstrates how to create a \p transform_iterator which represents the result of \c
//! sqrtf applied to the contents of a \p device_vector.
//!
//! \code
//! #include <thrust/iterator/transform_iterator.h>
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
//!   thrust::device_vector<float> v{1.0f, 4.0f, 9.0f, 16.0f};
//!
//!   using FloatIterator = thrust::device_vector<float>::iterator;
//!
//!   thrust::transform_iterator<square_root, FloatIterator> iter(v.begin(), square_root());
//!
//!   *iter;   // returns 1.0f
//!   iter[0]; // returns 1.0f;
//!   iter[1]; // returns 2.0f;
//!   iter[2]; // returns 3.0f;
//!   iter[3]; // returns 4.0f;
//!
//!   // iter[4] is an out-of-bounds error
//! }
//! \endcode
//!
//! This next example demonstrates how to use a \p transform_iterator with the \p thrust::reduce function to compute the
//! sum of squares of a sequence. We will create temporary \p transform_iterators with the \p make_transform_iterator
//! function in order to avoid explicitly specifying their type:
//!
//! \code
//! #include <thrust/iterator/transform_iterator.h>
//! #include <thrust/device_vector.h>
//! #include <thrust/reduce.h>
//! #include <iostream>
//!
//! struct square
//! {
//!   __host__ __device__
//!   float operator()(float x) const
//!   {
//!     return x * x;
//!   }
//! };
//!
//! int main()
//! {
//!   // initialize a device array
//!   thrust::device_vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
//!
//!   float sum_of_squares =
//!    thrust::reduce(thrust::make_transform_iterator(v.begin(), square()),
//!                   thrust::make_transform_iterator(v.end(),   square()));
//!
//!   std::cout << "sum of squares: " << sum_of_squares << std::endl;
//!   return 0;
//! }
//! \endcode
//!
//! The following example illustrates how to use the third template argument to explicitly specify the return type of
//! the function.
//!
//! \code
//! #include <thrust/iterator/transform_iterator.h>
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
//!   thrust::device_vector<float> v{1.0f, 4.0f, 9.0f, 16.0f};
//!
//!   using FloatIterator = thrust::device_vector<float>::iterator;
//!
//!   // note: float result_type is specified explicitly
//!   thrust::transform_iterator<square_root, FloatIterator, float> iter(v.begin(), square_root());
//!
//!   *iter;   // returns 1.0f
//!   iter[0]; // returns 1.0f;
//!   iter[1]; // returns 2.0f;
//!   iter[2]; // returns 3.0f;
//!   iter[3]; // returns 4.0f;
//!
//!   // iter[4] is an out-of-bounds error
//! }
//! \endcode
//!
//! \see make_transform_iterator
template <class AdaptableUnaryFunction, class Iterator, class Reference = use_default, class Value = use_default>
class transform_iterator
    : public detail::make_transform_iterator_base<AdaptableUnaryFunction, Iterator, Reference, Value>::type
{
  //! \cond

public:
  using super_t =
    typename detail::make_transform_iterator_base<AdaptableUnaryFunction, Iterator, Reference, Value>::type;

  friend class iterator_core_access;
  //! \endcond

  transform_iterator() = default;

  transform_iterator(transform_iterator const&) = default;

  //! This constructor takes as arguments an \c Iterator and an \c AdaptableUnaryFunction and copies them to a new \p
  //! transform_iterator.
  //!
  //! \param x An \c Iterator pointing to the input to this \p transform_iterator's \c AdaptableUnaryFunction.
  //! \param f An \c AdaptableUnaryFunction used to transform the objects pointed to by \p x.
  _CCCL_HOST_DEVICE transform_iterator(Iterator const& x, AdaptableUnaryFunction f)
      : super_t(x)
      , m_f(f)
  {}

  //! This explicit constructor copies the value of a given \c Iterator and creates this \p transform_iterator's \c
  //! AdaptableUnaryFunction using its null constructor.
  //!
  //! \param x An \c Iterator to copy.
  _CCCL_HOST_DEVICE explicit transform_iterator(Iterator const& x)
      : super_t(x)
  {}

  //! This copy constructor creates a new \p transform_iterator from another \p transform_iterator.
  //!
  //! \param other The \p transform_iterator to copy.
  template <typename OtherAdaptableUnaryFunction, typename OtherIterator, typename OtherReference, typename OtherValue>
  _CCCL_HOST_DEVICE transform_iterator(
    const transform_iterator<OtherAdaptableUnaryFunction, OtherIterator, OtherReference, OtherValue>& other,
    detail::enable_if_convertible_t<OtherIterator, Iterator>*                             = 0,
    detail::enable_if_convertible_t<OtherAdaptableUnaryFunction, AdaptableUnaryFunction>* = 0)
      : super_t(other.base())
      , m_f(other.functor())
  {}

  _CCCL_HOST_DEVICE transform_iterator& operator=(transform_iterator const& other)
  {
    super_t::operator=(other);
    if constexpr (_CCCL_TRAIT(::cuda::std::is_copy_assignable, AdaptableUnaryFunction))
    {
      m_f = other.m_f;
    }
    else if constexpr (_CCCL_TRAIT(::cuda::std::is_copy_constructible, AdaptableUnaryFunction))
    {
      ::cuda::std::__destroy_at(&m_f);
      ::cuda::std::__construct_at(&m_f, other.m_f);
    }
    else
    {
      static_assert(_CCCL_TRAIT(::cuda::std::is_copy_constructible, AdaptableUnaryFunction),
                    "Cannot use thrust::transform_iterator with a functor that is neither copy constructible nor "
                    "copy assignable");
    }
    return *this;
  }

  //! This method returns a copy of this \p transform_iterator's \c AdaptableUnaryFunction.
  //! \return A copy of this \p transform_iterator's \c AdaptableUnaryFunction.
  _CCCL_HOST_DEVICE AdaptableUnaryFunction functor() const
  {
    return m_f;
  }

  //! \cond

private:
  // MSVC 2013 and 2015 incorrectly warning about returning a reference to
  // a local/temporary here.
  // See goo.gl/LELTNp

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    // TODO(bgruber): we should ideally do as `std::ranges::transform_view::iterator` does:
    // `return std::invoke(m_f, *this->base());` and return `decltype(auto)`. However, `*this->base()` may return a
    // wrapped reference (`device_reference<T>`), which is a temporary value. If `m_f` forwards this value, e.g. as a
    // `device_reference<T>&&` if `m_f` is `identity<void>`, (and `super_t::reference` is thus deduced as
    // `device_reference<T>&&` as well), we return a dangling reference. So we cannot do as
    // `std::ranges::transform_view::iterator` does.

    // Interestingly, C++20 ranges have the same bug. The following program crashes because the transform iterator also
    // returns a reference to an expired temporary (given by the iota iterator upon dereferencing)
    //   for (auto e : std::views::iota(10) | std::views::transform(std::identity{}))
    //     std::cout << e << '\n';
    // See: https://godbolt.org/z/jrKcnMqhK

    // The workaround is to create a temporary to allow iterators with wrapped/proxy references to convert to their
    // value type before calling m_f. This also loads values from a different memory space (cf. `device_reference`).
    // Note that this disallows mutable operations through m_f.
    detail::it_value_t<Iterator> const& x = *this->base();
    // FIXME(bgruber): x may be a reference to a temporary (e.g. if the base iterator is a counting_iterator). If `m_f`
    // does not produce an independent copy and super_t::reference is a reference, we return a dangling reference (e.g.
    // for any `[thrust|::cuda::std]::identity` functor).
    return m_f(x);
  }

  // tag this as mutable per Dave Abrahams in this thread:
  // http://lists.boost.org/Archives/boost/2004/05/65332.php
  mutable AdaptableUnaryFunction m_f;

  //! \endcond
};

//! \p make_transform_iterator creates a \p transform_iterator from an \c Iterator and \c AdaptableUnaryFunction.
//!
//! \param it The \c Iterator pointing to the input range of the newly created \p transform_iterator.
//! \param fun The \c AdaptableUnaryFunction used to transform the range pointed to by \p it in the newly created \p
//!            transform_iterator.
//! \return A new \p transform_iterator which transforms the range at \p it by \p fun.
//! \see transform_iterator
template <class AdaptableUnaryFunction, class Iterator>
inline _CCCL_HOST_DEVICE transform_iterator<AdaptableUnaryFunction, Iterator>
make_transform_iterator(Iterator it, AdaptableUnaryFunction fun)
{
  return transform_iterator<AdaptableUnaryFunction, Iterator>(it, fun);
} // end make_transform_iterator

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
