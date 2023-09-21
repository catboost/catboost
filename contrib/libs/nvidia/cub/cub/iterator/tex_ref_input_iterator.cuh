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
#pragma clang system_header


#include <cub/config.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>

#include <cstddef>

CUB_NAMESPACE_BEGIN

/**
 * \addtogroup UtilIterator
 * @{
 */

/**
 * \brief A random-access input wrapper for dereferencing array values through texture cache.
 *
 * \deprecated [Since 1.13.0] The CUDA texture management APIs used by
 * TexRefInputIterator are deprecated. Use cub::TexObjInputIterator instead.
 *
 * \par Overview
 * - TexRefInputIterator wraps a native device pointer of type <tt>ValueType*</tt>. References
 *   to elements are to be loaded through texture cache.
 * - Can be used to load any data type from memory through texture cache.
 * - Can be manipulated and exchanged within and between host and device
 *   functions, can only be constructed within host functions, and can only be
 *   dereferenced within device functions.
 * - The \p UNIQUE_ID template parameter is used to statically name the underlying texture
 *   reference.  Only one TexRefInputIterator instance can be bound at any given time for a
 *   specific combination of (1) data type \p T, (2) \p UNIQUE_ID, (3) host
 *   thread, and (4) compilation .o unit.
 * - With regard to nested/dynamic parallelism, TexRefInputIterator iterators may only be
 *   created by the host thread and used by a top-level kernel (i.e. the one which is launched
 *   from the host).
 * - Compatible with Thrust API v1.7 or newer.
 *
 * \par Snippet
 * The code snippet below illustrates the use of \p TexRefInputIterator to
 * dereference a device array of doubles through texture cache.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/iterator/tex_ref_input_iterator.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * int num_items;   // e.g., 7
 * double *d_in;    // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::TexRefInputIterator<double, __LINE__> itr;
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
 * \endcode
 *
 * \tparam T                    The value type of this iterator
 * \tparam UNIQUE_ID            A globally-unique identifier (within the compilation unit) to name the underlying texture reference
 * \tparam OffsetT              The difference type of this iterator (Default: \p ptrdiff_t)
 */
template <
    typename    T,
    int         /*UNIQUE_ID*/,
    typename    OffsetT = std::ptrdiff_t>
using TexRefInputIterator CUB_DEPRECATED = cub::TexObjInputIterator<T, OffsetT>;

/** @} */       // end group UtilIterator

CUB_NAMESPACE_END
