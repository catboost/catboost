/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Random-access iterator types
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_type.cuh>

#include <thrust/iterator/iterator_facade.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif // !_CCCL_COMPILER(NVRTC)

CUB_NAMESPACE_BEGIN

/**
 * @brief A random-access input wrapper for pairing dereferenced values with their corresponding
 *        indices (forming \p KeyValuePair tuples).
 *
 * @par Overview
 * - ArgIndexInputIterator wraps a random access input iterator @p itr of type @p InputIteratorT.
 *   Dereferencing an ArgIndexInputIterator at offset @p i produces a @p KeyValuePair value whose
 *   @p key field is @p i and whose @p value field is <tt>itr[i]</tt>.
 * - Can be used with any data type.
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions.  Wrapped host memory can only be dereferenced on the host, and wrapped
 *   device memory can only be dereferenced on the device.
 * - Compatible with Thrust API v1.7 or newer.
 *
 * @par Snippet
 * The code snippet below illustrates the use of @p ArgIndexInputIterator to
 * dereference an array of doubles
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/iterator/arg_index_input_iterator.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * double *d_in;         // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::ArgIndexInputIterator<double*> itr(d_in);
 *
 * // Within device code:
 * cub::ArgIndexInputIterator<double*>::value_type tup = *itr;
 * printf("%f @ %ld\n",
 *   tup.value,
 *   tup.key);   // 8.0 @ 0
 *
 * itr = itr + 6;
 * tup = *itr;
 * printf("%f @ %ld\n",
 *   tup.value,
 *   tup.key);   // 9.0 @ 6
 *
 * @endcode
 *
 * @tparam InputIteratorT
 *   The value type of the wrapped input iterator
 *
 * @tparam OffsetT
 *   The difference type of this iterator (Default: @p ptrdiff_t)
 *
 * @tparam OutputValueT
 *   The paired value type of the <offset,value> tuple (Default: value type of input iterator)
 */
template <typename InputIteratorT,
          typename OffsetT      = ptrdiff_t,
          typename OutputValueT = detail::it_value_t<InputIteratorT>>
class ArgIndexInputIterator
{
public:
  // Required iterator traits

  /// My own type
  using self_type = ArgIndexInputIterator;

  /// Type to express the result of subtracting one iterator from another
  using difference_type = OffsetT;

  /// The type of the element the iterator can point to
  using value_type = KeyValuePair<difference_type, OutputValueT>;

  /// The type of a pointer to an element the iterator can point to
  using pointer = value_type*;

  /// The type of a reference to an element the iterator can point to
  using reference = value_type;

  /// The iterator category
  using iterator_category = typename THRUST_NS_QUALIFIER::detail::iterator_facade_category<
    THRUST_NS_QUALIFIER::any_system_tag,
    THRUST_NS_QUALIFIER::random_access_traversal_tag,
    value_type,
    reference>::type;

private:
  InputIteratorT itr;
  difference_type offset;

public:
  /**
   * @param itr
   *   Input iterator to wrap
   *
   * @param offset
   *   OffsetT (in items) from @p itr denoting the position of the iterator
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ArgIndexInputIterator(InputIteratorT itr, difference_type offset = 0)
      : itr(itr)
      , offset(offset)
  {}

  /// Postfix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++(int)
  {
    self_type retval = *this;
    offset++;
    return retval;
  }

  /// Prefix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++()
  {
    offset++;
    return *this;
  }

  /// Indirection
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator*() const
  {
    value_type retval;
    retval.value = itr[offset];
    retval.key   = offset;
    return retval;
  }

  /// Addition
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator+(Distance n) const
  {
    self_type retval(itr, offset + n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator+=(Distance n)
  {
    offset += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator-(Distance n) const
  {
    self_type retval(itr, offset - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator-=(Distance n)
  {
    offset -= n;
    return *this;
  }

  /// Distance
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE difference_type operator-(self_type other) const
  {
    return offset - other.offset;
  }

  /// Array subscript
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator[](Distance n) const
  {
    self_type offset = (*this) + n;
    return *offset;
  }

  /// Structure dereference
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE pointer operator->()
  {
    return &(*(*this));
  }

  /// Equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const self_type& rhs)
  {
    return ((itr == rhs.itr) && (offset == rhs.offset));
  }

  /// Not equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const self_type& rhs)
  {
    return ((itr != rhs.itr) || (offset != rhs.offset));
  }

  /// Normalize
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void normalize()
  {
    itr += offset;
    offset = 0;
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const self_type& /*itr*/)
  {
    return os;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

CUB_NAMESPACE_END
