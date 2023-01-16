/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if 0

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>

#if 1
#  define BS_SIMPLE
#endif

namespace thrust
{
namespace cuda_cub {

namespace __binary_search {

  template <class HaystackIt, class NeedlesIt>
  struct lbf
  {
    typedef typename iterator_traits<HaystackIt>::difference_type result_type;
    typedef typename iterator_traits<NeedlesIt>::value_type T;

    template <class It, class CompareOp>
    THRUST_DEVICE_FUNCTION result_type
    operator()(It begin, It end, T const& value, CompareOp comp)
    {
      return system::detail::generic::scalar::lower_bound(begin,
                                                          end,
                                                          value,
                                                          comp) -
             begin;
    }
  };    // struct lbf

  template<class HaystackIt, class NeedlesIt>
  struct ubf
  {
    typedef typename iterator_traits<HaystackIt>::difference_type result_type;
    typedef typename iterator_traits<NeedlesIt>::value_type T;

    template <class It, class CompareOp>
    THRUST_DEVICE_FUNCTION result_type
    operator()(It begin, It end, T const& value, CompareOp comp)
    {
      return system::detail::generic::scalar::upper_bound(begin,
                                                          end,
                                                          value,
                                                          comp) -
             begin;
    }
  };    // struct ubf

  template<class HaystackIt, class NeedlesIt>
  struct bsf
  {
    typedef bool result_type;
    typedef typename iterator_traits<NeedlesIt>::value_type T;

    template <class It, class CompareOp>
    THRUST_DEVICE_FUNCTION bool
    operator()(It begin, It end, T const& value, CompareOp comp)
    {
      HaystackIt iter = system::detail::generic::scalar::lower_bound(begin,
                                                                     end,
                                                                     value,
                                                                     comp);

      detail::wrapped_function<CompareOp, bool> wrapped_comp(comp);

      return iter != end && !wrapped_comp(value, *iter);
    }
  };    // struct bsf

  template <class KeysIt1,
            class KeysIt2,
            class Size,
            class BinaryPred>
  THRUST_DEVICE_FUNCTION Size
  merge_path(KeysIt1    keys1,
             KeysIt2    keys2,
             Size       keys1_count,
             Size       keys2_count,
             Size       diag,
             BinaryPred binary_pred)
  {
    typedef typename iterator_traits<KeysIt1>::value_type key1_type;
    typedef typename iterator_traits<KeysIt2>::value_type key2_type;

    Size keys1_begin = thrust::max<Size>(0, diag - keys2_count);
    Size keys1_end   = thrust::min<Size>(diag, keys1_count);

    while (keys1_begin < keys1_end)
    {
      Size      mid  = (keys1_begin + keys1_end) >> 1;
      key1_type key1 = keys1[mid];
      key2_type key2 = keys2[diag - 1 - mid];
      bool      pred = binary_pred(key2, key1);
      if (pred)
      {
        keys1_end = mid;
      }
      else
      {
        keys1_begin = mid + 1;
      }
    }
    return keys1_begin;
  }

  template <class It, class T2, class CompareOp, int ITEMS_PER_THREAD>
  THRUST_DEVICE_FUNCTION void
  serial_merge(It  keys_shared,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T2 (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
  {
    int keys1_end = keys1_beg + keys1_count;
    int keys2_end = keys2_beg + keys2_count;

    typedef typename iterator_value<It>::type key_type;

    key_type key1 = keys_shared[keys1_beg];
    key_type key2 = keys_shared[keys2_beg];


#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      bool p = (keys2_beg < keys2_end) &&
               ((keys1_beg >= keys1_end) ||
                compare_op(key2,key1));

      output[ITEM]  = p ? key2 : key1;
      indices[ITEM] = p ? keys2_beg++ : keys1_beg++;

      if (p)
      {
        key2 = keys_shared[keys2_beg];
      }
      else
      {
        key1 = keys_shared[keys1_beg];
      }
    }
  }

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS      = _BLOCK_THREADS,
      ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE     = _BLOCK_THREADS * _ITEMS_PER_THREAD
    };

    static const cub::BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  };    // PtxPolicy

  template <class Arch, class T>
  struct Tuning;

  template<class T>
  struct Tuning<sm30,T>
  {
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 7,
      ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(3, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_STORE_TRANSPOSE>
        type;
  };

  template<class T>
  struct Tuning<sm52,T>
  {
    const static int INPUT_SIZE = sizeof(T);

    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 7,
      ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>
        type;
  };

  template <class NeedlesIt,
            class HaystackIt,
            class Size,
            class OutputIt,
            class CompareOp,
            class SearchOp>
  struct VectorizedBinarySearchAgent
  {
    typedef typename iterator_traits<NeedlesIt>::value_type  needle_type;
    typedef typename iterator_traits<HaystackIt>::value_type haystack_type;
    typedef typename SearchOp::result_type                   result_type;

    template <class Arch>
    struct PtxPlan : Tuning<Arch, needle_type>::type
    {
      typedef Tuning<Arch,needle_type> tuning;

      typedef typename core::LoadIterator<PtxPlan, NeedlesIt>::type  NeedlesLoadIt;
      typedef typename core::LoadIterator<PtxPlan, HaystackIt>::type HaystackLoadIt;

      typedef typename core::BlockLoad<PtxPlan, NeedlesLoadIt>::type BlockLoadNeedles;

      typedef typename core::BlockStore<PtxPlan, OutputIt, result_type>::type BlockStoreResult;

      union TempStorage
      {
        typename BlockLoadNeedles::TempStorage load_needles;
        typename BlockStoreResult::TempStorage store_result;

#ifndef BS_SIMPLE
        core::uninitialized_array<needle_type, PtxPlan::ITEMS_PER_TILE + 1> needles_shared;
        core::uninitialized_array<result_type, PtxPlan::ITEMS_PER_TILE>     result_shared;
        core::uninitialized_array<int, PtxPlan::ITEMS_PER_TILE>             indices_shared;
#endif
      };    // union TempStorage
    };

    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::NeedlesLoadIt    NeedlesLoadIt;
    typedef typename ptx_plan::HaystackLoadIt   HaystackLoadIt;
    typedef typename ptx_plan::BlockLoadNeedles BlockLoadNeedles;
    typedef typename ptx_plan::BlockStoreResult BlockStoreResult;
    typedef typename ptx_plan::TempStorage     TempStorage;

    enum
    {
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS,
      ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE
    };

    struct impl
    {
      TempStorage&   storage;
      NeedlesLoadIt  needles_load_it;
      HaystackLoadIt haystack_load_it;
      Size           needles_count;
      Size           haystack_size;
      OutputIt       result;
      CompareOp      compare_op;
      SearchOp       search_op;

      THRUST_DEVICE_FUNCTION
      void stable_odd_even_sort(needle_type (&needles)[ITEMS_PER_THREAD],
                                int (&indices)[ITEMS_PER_THREAD])
      {
#pragma unroll
        for (int I = 0; I < ITEMS_PER_THREAD; ++I)
        {
#pragma unroll
          for (int J = 1 & I; J < ITEMS_PER_THREAD - 1; J += 2)
          {
            if (compare_op(needles[J + 1], needles[J]))
            {
              using thrust::swap;
              swap(needles[J], needles[J + 1]);
              swap(indices[J], indices[J + 1]);
            }
          }    // inner loop
        }      // outer loop
      }

      THRUST_DEVICE_FUNCTION void
      block_mergesort(int tid,
                      int count,
                      needle_type (&needles_loc)[ITEMS_PER_THREAD],
                      int (&indices_loc)[ITEMS_PER_THREAD])
      {
        using core::sync_threadblock;

        // stable sort items in a single thread
        //
        stable_odd_even_sort(needles_loc,indices_loc);

        // each thread has  sorted keys_loc
        // merge sort keys_loc in shared memory
        //
#pragma unroll
        for (int coop = 2; coop <= BLOCK_THREADS; coop *= 2)
        {
          sync_threadblock();

          // store keys in shmem
          //
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = ITEMS_PER_THREAD * threadIdx.x + ITEM;
            storage.needles_shared[idx] = needles_loc[ITEM];
          }

          sync_threadblock();

          int  indices[ITEMS_PER_THREAD];

          int list  = ~(coop - 1) & tid;
          int start = ITEMS_PER_THREAD * list;
          int size  = ITEMS_PER_THREAD * (coop >> 1);

          int diag = min(count, ITEMS_PER_THREAD * ((coop - 1) & tid));

          int keys1_beg = min(count, start);
          int keys1_end = min(count, keys1_beg + size);
          int keys2_beg = keys1_end;
          int keys2_end = min(count, keys2_beg + size);

          int keys1_count = keys1_end - keys1_beg;
          int keys2_count = keys2_end - keys2_beg;

          int partition_diag = merge_path(&storage.needles_shared[keys1_beg],
                                          &storage.needles_shared[keys2_beg],
                                          keys1_count,
                                          keys2_count,
                                          diag,
                                          compare_op);

          int keys1_beg_loc   = keys1_beg + partition_diag;
          int keys1_end_loc   = keys1_end;
          int keys2_beg_loc   = keys2_beg + diag - partition_diag;
          int keys2_end_loc   = keys2_end;
          int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
          int keys2_count_loc = keys2_end_loc - keys2_beg_loc;
          serial_merge(&storage.needles_shared[0],
                       keys1_beg_loc,
                       keys2_beg_loc,
                       keys1_count_loc,
                       keys2_count_loc,
                       needles_loc,
                       indices,
                       compare_op);


          sync_threadblock();

#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = ITEMS_PER_THREAD * threadIdx.x + ITEM;
            storage.indices_shared[idx] = indices_loc[ITEM];
          }

          sync_threadblock();

#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            indices_loc[ITEM] = storage.indices_shared[indices[ITEM]];
          }
        }
      }    // func block_merge_sort

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(int  tid,
                   Size tile_idx,
                   Size tile_base,
                   int  num_remaining)
      {
        using core::sync_threadblock;

        needle_type needles_loc[ITEMS_PER_THREAD];
        BlockLoadNeedles(storage.load_needles)
            .Load(needles_load_it + tile_base, needles_loc, num_remaining);

#ifdef BS_SIMPLE

        result_type results_loc[ITEMS_PER_THREAD];
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          results_loc[ITEM] = search_op(haystack_load_it,
                                        haystack_load_it + haystack_size,
                                        needles_loc[ITEM],
                                        compare_op);
        }


#else

        if (IS_LAST_TILE)
        {
          needle_type max_value = needles_loc[0];
#pragma unroll
          for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            if (ITEMS_PER_THREAD * tid + ITEM < num_remaining)
            {
              max_value = compare_op(max_value, needles_loc[ITEM])
                            ? needles_loc[ITEM]
                            : max_value;
            }
            else
            {
              needles_loc[ITEM] = max_value;
            }
          }
        }

        sync_threadblock();

        int indices_loc[ITEMS_PER_THREAD];
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int idx = ITEMS_PER_THREAD*threadIdx.x + ITEM;
          indices_loc[ITEM] = idx;
        }

        if (IS_LAST_TILE)
        {
          block_mergesort(tid,
                          num_remaining,
                          needles_loc,
                          indices_loc);
        }
        else
        {
          block_mergesort(tid,
                          ITEMS_PER_TILE,
                          needles_loc,
                          indices_loc);
        }

        sync_threadblock();

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int idx = indices_loc[ITEM];
          storage.result_shared[idx] =
              search_op(haystack_load_it,
                        haystack_load_it + haystack_size,
                        needles_loc[ITEM],
                        compare_op);
        }

        sync_threadblock();

        result_type results_loc[ITEMS_PER_THREAD];
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int idx = ITEMS_PER_THREAD*threadIdx.x + ITEM;
          results_loc[ITEM] = storage.result_shared[idx];
        }

        sync_threadblock();
#endif

        BlockStoreResult(storage.store_result)
            .Store(result + tile_base, results_loc, num_remaining);
      }

      THRUST_DEVICE_FUNCTION
      impl(TempStorage& storage_,
           NeedlesIt    needles_it_,
           HaystackIt   haystack_it_,
           Size         needles_count_,
           Size         haystack_size_,
           OutputIt     result_,
           CompareOp    compare_op_,
           SearchOp     search_op_)
          : storage(storage_),
            needles_load_it(core::make_load_iterator(ptx_plan(), needles_it_)),
            haystack_load_it(core::make_load_iterator(ptx_plan(), haystack_it_)),
            needles_count(needles_count_),
            haystack_size(haystack_size_),
            result(result_),
            compare_op(compare_op_),
            search_op(search_op_)
      {
        int  tid           = threadIdx.x;
        Size tile_idx      = blockIdx.x;
        Size num_tiles     = gridDim.x;
        Size tile_base     = tile_idx * ITEMS_PER_TILE;
        int  items_in_tile = min<int>(needles_count - tile_base, ITEMS_PER_TILE);
        if (tile_idx < num_tiles - 1)
        {
          consume_tile<false>(tid, tile_idx, tile_base, ITEMS_PER_TILE);
        }
        else
        {
          consume_tile<true>(tid, tile_idx, tile_base, items_in_tile);
        }
      }
    };    // struct impl


    THRUST_AGENT_ENTRY(NeedlesIt  needles_it,
                       HaystackIt haystack_it,
                       Size       needles_count,
                       Size       haystack_size,
                       OutputIt   result,
                       CompareOp  compare_op,
                       SearchOp   search_op,
                       char*      shmem)
    {
      TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

      impl(storage,
           needles_it,
           haystack_it,
           needles_count,
           haystack_size,
           result,
           compare_op,
           search_op);
    }
  };    // struct VectorizedBinarySearchAgent

  template <class NeedlesIt,
            class HaystackIt,
            class Size,
            class OutputIt,
            class CompareOp,
            class SearchOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_pass(void*        d_temp_storage,
            size_t&      temp_storage_size,
            NeedlesIt    needles_it,
            HaystackIt   haystack_it,
            Size         needles_count,
            Size         haystack_size,
            OutputIt     result,
            CompareOp    compare_op,
            SearchOp     search_op,
            cudaStream_t stream,
            bool         debug_sync)
  {
    if (needles_count == 0)
      return cudaErrorNotSupported;

    cudaError_t status = cudaSuccess;

    using core::AgentPlan;
    using core::AgentLauncher;


    typedef AgentLauncher<
        VectorizedBinarySearchAgent<NeedlesIt,
                                    HaystackIt,
                                    Size,
                                    OutputIt,
                                    CompareOp,
                                    SearchOp> >
        search_agent;

    AgentPlan search_plan = search_agent::get_plan(stream);

    temp_storage_size = 1;
    if (d_temp_storage == NULL)
    {
      return status;
    }

    search_agent sa(search_plan, needles_count, stream, "binary_search::search_agent", debug_sync);
    sa.launch(needles_it,
              haystack_it,
              needles_count,
              haystack_size,
              result,
              compare_op,
              search_op);

    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
  }

  template <typename Derived,
            typename NeedlesIt,
            typename HaystackIt,
            typename OutputIt,
            typename CompareOp,
            typename SearchOp>
  OutputIt THRUST_RUNTIME_FUNCTION
  doit(execution_policy<Derived>& policy,
       HaystackIt                 haystack_begin,
       HaystackIt                 haystack_end,
       NeedlesIt                  needles_begin,
       NeedlesIt                  needles_end,
       OutputIt                   result,
       CompareOp                  compare_op,
       SearchOp                   search_op)
  {
    typedef typename iterator_traits<NeedlesIt>::difference_type size_type;

    size_type needles_count = thrust::distance(needles_begin, needles_end);
    size_type haystack_size = thrust::distance(haystack_begin, haystack_end);

    if (needles_count == 0)
      return result;

    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    cudaError status;
    status = doit_pass(NULL,
                       storage_size,
                       needles_begin,
                       haystack_begin,
                       needles_count,
                       haystack_size,
                       result,
                       compare_op,
                       search_op,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "binary_search: failed on 1st call");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = doit_pass(ptr,
                       storage_size,
                       needles_begin,
                       haystack_begin,
                       needles_count,
                       haystack_size,
                       result,
                       compare_op,
                       search_op,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "binary_search: failed on 2nt call");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "binary_search: failed to synchronize");

    return result + needles_count;
  }

  struct less
  {
    template <typename T1, typename T2>
    THRUST_DEVICE_FUNCTION bool
    operator()(const T1& lhs, const T2& rhs) const
    {
      return lhs < rhs;
    }
  };
}    // namespace __binary_search

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt,
          class CompareOp>
OutputIt __host__ __device__
lower_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result,
            CompareOp                  compare_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __binary_search::doit(policy,
                                first,
                                last,
                                values_first,
                                values_last,
                                result,
                                compare_op,
                                __binary_search::lbf<HaystackIt, NeedlesIt>());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::lower_bound(cvt_to_seq(derived_cast(policy)),
                              first,
                              last,
                              values_first,
                              values_last,
                              result);
#endif
  }
  return ret;
}


template <class Derived,
          class HaystackIt,
          class NeedlesIt,
          class OutputIt>
OutputIt __host__ __device__
lower_bound(execution_policy<Derived>& policy,
            HaystackIt                 first,
            HaystackIt                 last,
            NeedlesIt                  values_first,
            NeedlesIt                  values_last,
            OutputIt                   result)
{
  return cuda_cub::lower_bound(policy,
                               first,
                               last,
                               values_first,
                               values_last,
                               result,
                               __binary_search::less());
}

}    // namespace cuda_cub
} // end namespace thrust
#endif

#endif
