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
 * @file
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

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>

#include <thrust/iterator/iterator_facade.h>

#include <cuda/std/type_traits>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif // !_CCCL_COMPILER(NVRTC)

CUB_NAMESPACE_BEGIN

/**
 * @brief A random-access input wrapper for dereferencing array values through texture cache.
 *        Uses newer Kepler-style texture objects.
 *
 * @par Overview
 * - TexObjInputIterator wraps a native device pointer of type <tt>ValueType*</tt>. References
 *   to elements are to be loaded through texture cache.
 * - Can be used to load any data type from memory through texture cache.
 * - Can be manipulated and exchanged within and between host and device
 *   functions, can only be constructed within host functions, and can only be
 *   dereferenced within device functions.
 * - With regard to nested/dynamic parallelism, TexObjInputIterator iterators may only be
 *   created by the host thread, but can be used by any descendant kernel.
 * - Compatible with Thrust API v1.7 or newer.
 *
 * @par Snippet
 * The code snippet below illustrates the use of @p TexObjInputIterator to
 * dereference a device array of doubles through texture cache.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/iterator/tex_obj_input_iterator.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * int num_items;   // e.g., 7
 * double *d_in;    // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::TexObjInputIterator<double> itr;
 * itr.BindTexture(d_in, sizeof(double) * num_items);
 * ...
 *
 * // Within device code:
 * printf("%f\n", itr[0]);      // 8.0
 * printf("%f\n", itr[1]);      // 6.0
 * printf("%f\n", itr[6]);      // 9.0
 *
 * ...
 * itr.UnbindTexture();
 *
 * @endcode
 *
 * @tparam T
 *   The value type of this iterator
 *
 * @tparam OffsetT
 *   The difference type of this iterator (Default: @p ptrdiff_t)
 */
template <typename T, typename OffsetT = ptrdiff_t>
class TexObjInputIterator
{
public:
  // Required iterator traits

  /// My own type
  using self_type = TexObjInputIterator;

  /// Type to express the result of subtracting one iterator from another
  using difference_type = OffsetT;

  /// The type of the element the iterator can point to
  using value_type = T;

  /// The type of a pointer to an element the iterator can point to
  using pointer = T*;

  /// The type of a reference to an element the iterator can point to
  using reference = T;

  /// The iterator category
  using iterator_category = typename THRUST_NS_QUALIFIER::detail::iterator_facade_category<
    THRUST_NS_QUALIFIER::device_system_tag,
    THRUST_NS_QUALIFIER::random_access_traversal_tag,
    value_type,
    reference>::type;

private:
  // Largest texture word we can use in device
  using TextureWord = typename UnitWord<T>::TextureWord;

  // Number of texture words per T
  enum
  {
    TEXTURE_MULTIPLE = sizeof(T) / sizeof(TextureWord)
  };

private:
  T* ptr;
  difference_type tex_offset;
  cudaTextureObject_t tex_obj;

public:
  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TexObjInputIterator()
      : ptr(nullptr)
      , tex_offset(0)
      , tex_obj(0)
  {}

#if !_CCCL_COMPILER(NVRTC)
  /**
   * @brief Use this iterator to bind @p ptr with a texture reference
   *
   * @param ptr
   *   Native pointer to wrap that is aligned to cudaDeviceProp::textureAlignment
   *
   * @param bytes
   *   Number of bytes in the range
   *
   * @param tex_offset
   *   OffsetT (in items) from @p ptr denoting the position of the iterator
   */
  template <typename QualifiedT>
  cudaError_t BindTexture(QualifiedT* ptr, size_t bytes, size_t tex_offset = 0)
  {
    this->ptr        = const_cast<::cuda::std::remove_cv_t<QualifiedT>*>(ptr);
    this->tex_offset = static_cast<difference_type>(tex_offset);

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<TextureWord>();
    cudaResourceDesc res_desc;
    cudaTextureDesc tex_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    res_desc.resType                = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr      = this->ptr;
    res_desc.res.linear.desc        = channel_desc;
    res_desc.res.linear.sizeInBytes = bytes;
    tex_desc.readMode               = cudaReadModeElementType;
    return CubDebug(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
  }

  /// Unbind this iterator from its texture reference
  cudaError_t UnbindTexture()
  {
    return CubDebug(cudaDestroyTextureObject(tex_obj));
  }
#endif // !_CCCL_COMPILER(NVRTC)

  /// Postfix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++(int)
  {
    self_type retval = *this;
    tex_offset++;
    return retval;
  }

  /// Prefix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++()
  {
    tex_offset++;
    return *this;
  }

  /// Indirection
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator*() const
  {
    NV_IF_TARGET(NV_IS_HOST, (return ptr[tex_offset];), (return this->device_deref();));
  }

  /// Addition
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator+(Distance n) const
  {
    self_type retval;
    retval.ptr        = ptr;
    retval.tex_obj    = tex_obj;
    retval.tex_offset = tex_offset + n;
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator+=(Distance n)
  {
    tex_offset += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator-(Distance n) const
  {
    self_type retval;
    retval.ptr        = ptr;
    retval.tex_obj    = tex_obj;
    retval.tex_offset = tex_offset - n;
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator-=(Distance n)
  {
    tex_offset -= n;
    return *this;
  }

  /// Distance
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE difference_type operator-(self_type other) const
  {
    return tex_offset - other.tex_offset;
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
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const self_type& rhs) const
  {
    return ((ptr == rhs.ptr) && (tex_offset == rhs.tex_offset) && (tex_obj == rhs.tex_obj));
  }

  /// Not equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const self_type& rhs) const
  {
    return ((ptr != rhs.ptr) || (tex_offset != rhs.tex_offset) || (tex_obj != rhs.tex_obj));
  }

#if !_CCCL_COMPILER(NVRTC)
  /// ostream operator
  friend ::std::ostream& operator<<(::std::ostream& os, const self_type& itr)
  {
    os << "cub::TexObjInputIterator( ptr=" << itr.ptr << ", offset=" << itr.tex_offset << ", tex_obj=" << itr.tex_obj
       << " )";
    return os;
  }
#endif // !_CCCL_COMPILER(NVRTC)

private:
  // This is hoisted out of operator* because #pragma can't be used inside of
  // NV_IF_TARGET
  _CCCL_DEVICE _CCCL_FORCEINLINE reference device_deref() const
  {
    // Move array of uninitialized words, then alias and assign to return
    // value
    TextureWord words[TEXTURE_MULTIPLE];

    const auto tex_idx_base = tex_offset * TEXTURE_MULTIPLE;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < TEXTURE_MULTIPLE; ++i)
    {
      words[i] = tex1Dfetch<TextureWord>(tex_obj, tex_idx_base + i);
    }

    // Load from words
    return *reinterpret_cast<T*>(words);
  }
};

CUB_NAMESPACE_END
