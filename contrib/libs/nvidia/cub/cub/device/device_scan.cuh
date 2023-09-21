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
 * cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <stdio.h>
#include <iterator>

#include "../config.cuh"
#include "../thread/thread_operators.cuh"
#include "dispatch/dispatch_scan.cuh"
#include "dispatch/dispatch_scan_by_key.cuh"

CUB_NAMESPACE_BEGIN


/**
 * \brief DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within device-accessible memory. ![](device_scan.png)
 * \ingroup SingleModule
 *
 * \par Overview
 * Given a sequence of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output sequence where each element is computed to be the reduction
 * of the elements occurring earlier in the input sequence.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction incorporates the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not incorporated into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par
 * As of CUB 1.0.1 (2013), CUB's device-wide scan APIs have implemented our <em>"decoupled look-back"</em> algorithm
 * for performing global prefix scan with only a single pass through the
 * input data, as described in our 2016 technical report [1].  The central
 * idea is to leverage a small, constant factor of redundant work in order to overlap the latencies
 * of global prefix propagation with local computation.  As such, our algorithm requires only
 * ~2<em>n</em> data movement (<em>n</em> inputs are read, <em>n</em> outputs are written), and typically
 * proceeds at "memcpy" speeds. Our algorithm supports inplace operations.
 *
 * \par
 * [1] [Duane Merrill and Michael Garland.  "Single-pass Parallel Prefix Scan with Decoupled Look-back", <em>NVIDIA Technical Report NVR-2016-002</em>, 2016.](https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back)
 *
 * \par Usage Considerations
 * \cdp_class{DeviceScan}
 *
 * \par Performance
 * \linear_performance{prefix scan}
 *
 * \par
 * The following chart illustrates DeviceScan::ExclusiveSum
 * performance across different CUDA architectures for \p int32 keys.
 * \plots_below
 *
 * \image html scan_int32.png
 *
 */
struct DeviceScan
{
    /******************************************************************//**
     * \name Exclusive scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes a device-wide exclusive prefix sum.  The value of 0 is applied as the initial value, and is assigned to *d_out.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Performance
     * The following charts illustrate saturated exclusive sum performance across different
     * CUDA architectures for \p int32 and \p int64 items, respectively.
     *
     * \image html scan_int32.png
     * \image html scan_int64.png
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix sum
     * cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        InputIteratorT,
        typename        OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveSum(
        void            *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Random-access iterator to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Random-access iterator to the output sequence of data items
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        using OffsetT = int;

        // The output value type -- used as the intermediate accumulator
        // Use the input value type per https://wg21.link/P0571
        using OutputT = cub::detail::value_t<InputIteratorT>;

        // Initial value
        OutputT init_value = 0;

        return DispatchScan<InputIteratorT, OutputIteratorT, Sum, detail::InputValue<OutputT>, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            Sum(),
            detail::InputValue<OutputT>(init_value),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.  The \p init_value value is applied as the initial value, and is assigned to *d_out.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix min-scan of an \p int device vector
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     * #include <climits>       // for INT_MAX
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements for exclusive prefix scan
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) INT_MAX, num_items);
     *
     * // Allocate temporary storage for exclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix min-scan
     * cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) INT_MAX, num_items);
     *
     * // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
     *
     * \endcode
     *
     * \tparam InputIteratorT   <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT  <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam InitValueT       <b>[inferred]</b> Type of the \p init_value used Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        InputIteratorT,
        typename        OutputIteratorT,
        typename        ScanOpT,
        typename        InitValueT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveScan(
        void            *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Random-access iterator to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Random-access iterator to the output sequence of data items
        ScanOpT         scan_op,                            ///< [in] Binary scan functor
        InitValueT      init_value,                         ///< [in] Initial value to seed the exclusive scan (and is assigned to *d_out)
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, detail::InputValue<InitValueT>, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            detail::InputValue<InitValueT>(init_value),
            num_items,
            stream,
            debug_synchronous);
    }

    template <
        typename        InputIteratorT,
        typename        OutputIteratorT,
        typename        ScanOpT,
        typename        InitValueT,
        typename        InitValueIterT=InitValueT*>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveScan(
        void                                    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                                  &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT                          d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT                         d_out,                              ///< [out] Pointer to the output sequence of data items
        ScanOpT                                 scan_op,                            ///< [in] Binary scan functor
        FutureValue<InitValueT, InitValueIterT> init_value,                         ///< [in] Initial value to seed the exclusive scan (and is assigned to *d_out)
        int                                     num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t                            stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                                    debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, detail::InputValue<InitValueT>, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            detail::InputValue<InitValueT>(init_value),
            num_items,
            stream,
            debug_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide inclusive prefix sum.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix sum
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage for inclusive prefix sum
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix sum
     * cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out <-- [8, 14, 21, 26, 29, 29, 38]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t InclusiveSum(
        void*               d_temp_storage,                 ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&             temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                           ///< [in] Random-access iterator to the input sequence of data items
        OutputIteratorT     d_out,                          ///< [out] Random-access iterator to the output sequence of data items
        int                 num_items,                      ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t        stream             = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, Sum, NullType, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            Sum(),
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide inclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix min-scan of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     * #include <climits>       // for INT_MAX
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix scan
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // Allocate temporary storage for inclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix min-scan
     * cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // d_out <-- [8, 6, 6, 5, 3, 0, 0]
     *
     * \endcode
     *
     * \tparam InputIteratorT   <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT  <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        InputIteratorT,
        typename        OutputIteratorT,
        typename        ScanOpT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t InclusiveScan(
        void            *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Random-access iterator to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Random-access iterator to the output sequence of data items
        ScanOpT         scan_op,                            ///< [in] Binary scan functor
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream             = 0,             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous  = false)         ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, NullType, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }

    /**
     * \brief Computes a device-wide exclusive prefix sum-by-key with key equality
     * defined by \p equality_op .  The value of 0 is applied as the initial value,
     * and is assigned to the beginning of each segment in \p d_values_out .
     *
     * \par
     * - Supports non-commutative sum operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix sum-by-key of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int num_items;      // e.g., 7
     * int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
     * int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix sum
     * cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items);
     *
     * // d_values_out <-- [0, 8, 0, 7, 12, 0, 0]
     *
     * \endcode
     *
     * \tparam KeysInputIteratorT      <b>[inferred]</b> Random-access input iterator type for reading scan keys inputs \iterator
     * \tparam ValuesInputIteratorT    <b>[inferred]</b> Random-access input iterator type for reading scan values inputs \iterator
     * \tparam ValuesOutputIteratorT   <b>[inferred]</b> Random-access output iterator type for writing scan values outputs \iterator
     * \tparam EqualityOpT             <b>[inferred]</b> Functor type having member <tt>T operator()(const T &a, const T &b)</tt> for binary operations that defines the equality of keys
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        KeysInputIteratorT,
        typename        ValuesInputIteratorT,
        typename        ValuesOutputIteratorT,
        typename        EqualityOpT = Equality>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveSumByKey(
        void                  *d_temp_storage,              ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                &temp_storage_bytes,          ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,                    ///< [in] Random-access input iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,                  ///< [in] Random-access input iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,                 ///< [out] Random-access output iterator to the output sequence of value items
        int                   num_items,                    ///< [in] Total number of input items (i.e., the length of \p d_keys_in and \p d_values_in)
        EqualityOpT           equality_op = EqualityOpT(),  ///< [in] Binary functor that defines the equality of keys. Default is cub::Equality().
        cudaStream_t          stream=0,                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous=false)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        using OffsetT = int;

        // The output value type -- used as the intermediate accumulator
        // Use the input value type per https://wg21.link/P0571
        using OutputT = cub::detail::value_t<ValuesInputIteratorT>;

        // Initial value
        OutputT init_value = 0;

        return DispatchScanByKey<
            KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT, EqualityOpT, Sum, OutputT, OffsetT>
        ::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_values_in,
            d_values_out,
            equality_op,
            Sum(),
            init_value,
            num_items,
            stream,
            debug_synchronous);
    }

    /**
     * \brief Computes a device-wide exclusive prefix scan-by-key using the specified binary \p scan_op functor.
     * The key equality is defined by \p equality_op .  The \p init_value value is applied as the initial value,
     * and is assigned to the beginning of each segment in \p d_values_out .
     *
     * \par
     * - Supports non-commutative scan operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix min-scan-by-key of an \p int device vector
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     * #include <climits>       // for INT_MAX
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // CustomEqual functor
     * struct CustomEqual
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return a == b;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
     * int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op;
     * CustomEqual  equality_op;
     * ...
     *
     * // Determine temporary device storage requirements for exclusive prefix scan
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveScanByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, min_op, (int) INT_MAX, num_items, equality_op);
     *
     * // Allocate temporary storage for exclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix min-scan
     * cub::DeviceScan::ExclusiveScanByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, min_op, (int) INT_MAX, num_items, equality_op);
     *
     * // d_values_out <-- [2147483647, 8, 2147483647, 7, 5, 2147483647, 0]
     *
     * \endcode
     *
     * \tparam KeysInputIteratorT      <b>[inferred]</b> Random-access input iterator type for reading scan keys inputs \iterator
     * \tparam ValuesInputIteratorT    <b>[inferred]</b> Random-access input iterator type for reading scan values inputs \iterator
     * \tparam ValuesOutputIteratorT   <b>[inferred]</b> Random-access output iterator type for writing scan values outputs \iterator
     * \tparam ScanOp                  <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam InitValueT              <b>[inferred]</b> Type of the \p init_value value used in Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam EqualityOpT             <b>[inferred]</b> Functor type having member <tt>T operator()(const T &a, const T &b)</tt> for binary operations that defines the equality of keys
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        KeysInputIteratorT,
        typename        ValuesInputIteratorT,
        typename        ValuesOutputIteratorT,
        typename        ScanOpT,
        typename        InitValueT,
        typename        EqualityOpT = Equality>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveScanByKey(
        void                  *d_temp_storage,              ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                &temp_storage_bytes,          ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,                    ///< [in] Random-access input iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,                  ///< [in] Random-access input iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,                 ///< [out] Random-access output iterator to the output sequence of value items
        ScanOpT               scan_op,                      ///< [in] Binary scan functor
        InitValueT            init_value,                   ///< [in] Initial value to seed the exclusive scan (and is assigned to the beginning of each segment in \p d_values_out)
        int                   num_items,                    ///< [in] Total number of input items (i.e., the length of \p d_keys_in and \p d_values_in)
        EqualityOpT           equality_op = EqualityOpT(),  ///< [in] Binary functor that defines the equality of keys. Default is cub::Equality().
        cudaStream_t          stream=0,                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous=false)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScanByKey<
            KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT, EqualityOpT, ScanOpT, InitValueT, OffsetT>
        ::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_values_in,
            d_values_out,
            equality_op,
            scan_op,
            init_value,
            num_items,
            stream,
            debug_synchronous);
    }

    /**
     * \brief Computes a device-wide inclusive prefix sum-by-key with key equality defined by \p equality_op .
     *
     * \par
     * - Supports non-commutative sum operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix sum-by-key of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int num_items;      // e.g., 7
     * int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
     * int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix sum
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items);
     *
     * // Allocate temporary storage for inclusive prefix sum
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix sum
     * cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items);
     *
     * // d_out <-- [8, 14, 7, 12, 15, 0, 9]
     *
     * \endcode
     *
     * \tparam KeysInputIteratorT      <b>[inferred]</b> Random-access input iterator type for reading scan keys inputs \iterator
     * \tparam ValuesInputIteratorT    <b>[inferred]</b> Random-access input iterator type for reading scan values inputs \iterator
     * \tparam ValuesOutputIteratorT   <b>[inferred]</b> Random-access output iterator type for writing scan values outputs \iterator
     * \tparam EqualityOpT             <b>[inferred]</b> Functor type having member <tt>T operator()(const T &a, const T &b)</tt> for binary operations that defines the equality of keys
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        KeysInputIteratorT,
        typename        ValuesInputIteratorT,
        typename        ValuesOutputIteratorT,
        typename        EqualityOpT = Equality>
    CUB_RUNTIME_FUNCTION
    static cudaError_t InclusiveSumByKey(
        void                  *d_temp_storage,              ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                &temp_storage_bytes,          ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,                    ///< [in] Random-access input iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,                  ///< [in] Random-access input iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,                 ///< [out] Random-access output iterator to the output sequence of value items
        int                   num_items,                    ///< [in] Total number of input items (i.e., the length of \p d_keys_in and \p d_values_in)
        EqualityOpT           equality_op = EqualityOpT(),  ///< [in] Binary functor that defines the equality of keys. Default is cub::Equality().
        cudaStream_t          stream=0,                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous=false)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScanByKey<
            KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT, EqualityOpT, Sum, NullType, OffsetT>
        ::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_values_in,
            d_values_out,
            equality_op,
            Sum(),
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }

    /**
     * \brief Computes a device-wide inclusive prefix scan-by-key using the specified binary \p scan_op functor.
     * The key equality is defined by \p equality_op .
     *
     * \par
     * - Supports non-commutative scan operators.
     * - Results are not deterministic for pseudo-associative operators (e.g.,
     *   addition of floating-point types). Results for pseudo-associative
     *   operators may vary from run to run. Additional details can be found in
     *   the [decoupled look-back] description.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix min-scan-by-key of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     * #include <climits>       // for INT_MAX
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // CustomEqual functor
     * struct CustomEqual
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return a == b;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
     * int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op;
     * CustomEqual  equality_op;
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix scan
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveScanByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
     *
     * // Allocate temporary storage for inclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix min-scan
     * cub::DeviceScan::InclusiveScanByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
     *
     * // d_out <-- [8, 6, 7, 5, 3, 0, 0]
     *
     * \endcode
     *
     * \tparam KeysInputIteratorT      <b>[inferred]</b> Random-access input iterator type for reading scan keys inputs \iterator
     * \tparam ValuesInputIteratorT    <b>[inferred]</b> Random-access input iterator type for reading scan values inputs \iterator
     * \tparam ValuesOutputIteratorT   <b>[inferred]</b> Random-access output iterator type for writing scan values outputs \iterator
     * \tparam ScanOp                  <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam EqualityOpT             <b>[inferred]</b> Functor type having member <tt>T operator()(const T &a, const T &b)</tt> for binary operations that defines the equality of keys
     *
     * [decoupled look-back]: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
     */
    template <
        typename        KeysInputIteratorT,
        typename        ValuesInputIteratorT,
        typename        ValuesOutputIteratorT,
        typename        ScanOpT,
        typename        EqualityOpT = Equality>
    CUB_RUNTIME_FUNCTION
    static cudaError_t InclusiveScanByKey(
        void                  *d_temp_storage,              ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                &temp_storage_bytes,          ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,                    ///< [in] Random-access input iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,                  ///< [in] Random-access input iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,                 ///< [out] Random-access output iterator to the output sequence of value items
        ScanOpT               scan_op,                      ///< [in] Binary scan functor
        int                   num_items,                    ///< [in] Total number of input items (i.e., the length of \p d_keys_in and \p d_values_in)
        EqualityOpT           equality_op = EqualityOpT(),  ///< [in] Binary functor that defines the equality of keys. Default is cub::Equality().
        cudaStream_t          stream=0,                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous=false)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScanByKey<
            KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT, EqualityOpT, ScanOpT, NullType, OffsetT>
        ::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_values_in,
            d_values_out,
            equality_op,
            scan_op,
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }

    //@}  end member group

};

/**
 * \example example_device_scan.cu
 */

CUB_NAMESPACE_END
