/*
 *  Copyright 2021 NVIDIA Corporation
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

#include <cub/util_namespace.cuh>


CUB_NAMESPACE_BEGIN

namespace detail
{


/**
 * @brief It's a double-buffer storage wrapper for multi-pass stream
 *        transformations that require more than one storage array for
 *        streaming intermediate results back and forth.
 *
 * Many multi-pass computations require a pair of "ping-pong" storage buffers
 * (e.g., one for reading from and the other for writing to, and then
 * vice-versa for the subsequent pass). This structure wraps a set of device
 * buffers.
 *
 * Unlike `cub::DoubleBuffer` this class doesn't provide a "selector" member
 * to track which buffer is "current". The main reason for this class existence
 * is the performance difference. Since `cub::DoubleBuffer` relies on the
 * runtime variable to index pointers arrays, they are placed in the local
 * memory instead of registers. Local memory accesses significantly affect
 * performance. On the contrary, this class swaps pointer, so all operations
 * can be performed in registers.
 */
template <typename T>
class device_double_buffer
{
  /// Pair of device buffer pointers
  T *m_current_buffer {};
  T *m_alternate_buffer {};

public:
  /**
   * @param d_current
   *   The currently valid buffer
   *
   * @param d_alternate
   *   Alternate storage buffer of the same size as @p d_current
   */
  __host__ __device__ __forceinline__ device_double_buffer(T *current,
                                                           T *alternate)
      : m_current_buffer(current)
      , m_alternate_buffer(alternate)
  {}

  /// \brief Return pointer to the currently valid buffer
  __host__ __device__ __forceinline__ T *current() const
  {
    return m_current_buffer;
  }

  /// \brief Return pointer to the currently invalid buffer
  __host__ __device__ __forceinline__ T *alternate() const
  {
    return m_alternate_buffer;
  }

  __host__ __device__ void swap()
  {
    T *tmp             = m_current_buffer;
    m_current_buffer   = m_alternate_buffer;
    m_alternate_buffer = tmp;
  }
};


} // namespace detail

CUB_NAMESPACE_END
