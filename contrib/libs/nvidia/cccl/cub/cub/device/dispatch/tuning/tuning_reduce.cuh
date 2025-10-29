/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace reduce
{
template <typename PolicyT, typename = void>
struct ReducePolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION ReducePolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ReducePolicyWrapper<StaticPolicyT,
                           _CUDA_VSTD::void_t<typename StaticPolicyT::ReducePolicy,
                                              typename StaticPolicyT::SingleTilePolicy,
                                              typename StaticPolicyT::SegmentedReducePolicy>> : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION ReducePolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(Reduce)
  CUB_DEFINE_SUB_POLICY_GETTER(SingleTile)
  CUB_DEFINE_SUB_POLICY_GETTER(SegmentedReduce)

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"ReducePolicy">()          = Reduce().EncodedPolicy(),
                  key<"SingleTilePolicy">()      = SingleTile().EncodedPolicy(),
                  key<"SegmentedReducePolicy">() = SegmentedReduce().EncodedPolicy()>();
  }
#endif
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION ReducePolicyWrapper<PolicyT> MakeReducePolicyWrapper(PolicyT policy)
{
  return ReducePolicyWrapper<PolicyT>{policy};
}
enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class op_type
{
  plus,
  min_or_max,
  unknown
};
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
template <class AccumT>
constexpr accum_size classify_accum_size()
{
  return sizeof(AccumT) == 1 ? accum_size::_1
       : sizeof(AccumT) == 2 ? accum_size::_2
       : sizeof(AccumT) == 4 ? accum_size::_4
       : sizeof(AccumT) == 8 ? accum_size::_8
       : sizeof(AccumT) == 16
         ? accum_size::_16
         : accum_size::unknown;
}
template <class OffsetT>
constexpr offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

template <typename Op>
struct is_plus
{
  static constexpr bool value = false;
};

template <typename T>
struct is_plus<::cuda::std::plus<T>>
{
  static constexpr bool value = true;
};
template <typename Op>
struct is_min_or_max
{
  static constexpr bool value = false;
};
template <typename T>
struct is_min_or_max<::cuda::minimum<T>>
{
  static constexpr bool value = true;
};
template <typename T>
struct is_min_or_max<::cuda::maximum<T>>
{
  static constexpr bool value = true;
};

template <class ScanOpT>
constexpr op_type classify_op()
{
  return is_plus<ScanOpT>::value
         ? op_type::plus
         : (is_min_or_max<ScanOpT>::value ? op_type::min_or_max : op_type::unknown);
}

template <class AccumT,
          class OffsetT,
          op_type OpTypeT        = classify_op<OffsetT>(),
          offset_size OffsetSize = classify_offset_size<OffsetT>(),
          accum_size AccumSize   = classify_accum_size<AccumT>()>
struct sm100_tuning;

// sum

// Tunings for offset size 4/8 and accum size 1/2/4 all showed no significant improvement during verification

template <class T, class OffsetT>
struct sm100_tuning<T, OffsetT, op_type::plus, offset_size::_4, accum_size::_8>
{
  // ipt_15.tpb_512.ipv_2 1.019887   1.0  1.017636  1.058036
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

template <class T, class OffsetT>
struct sm100_tuning<T, OffsetT, op_type::plus, offset_size::_8, accum_size::_8>
{
  // ipt_15.tpb_512.ipv_1 1.019414  1.000000  1.017218  1.057143
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 1;
};

template <class OffsetT>
struct sm100_tuning<float, OffsetT, op_type::plus, offset_size::_4, accum_size::_4>
{
  // ipt_16.tpb_512.ipv_2 1.061295  1.000000  1.065478  1.167139
  static constexpr int items              = 16;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

template <class OffsetT>
struct sm100_tuning<double, OffsetT, op_type::plus, offset_size::_4, accum_size::_8>
{
  // ipt_16.tpb_640.ipv_1 1.017834  1.000000  1.015835  1.057092
  static constexpr int items              = 16;
  static constexpr int threads            = 640;
  static constexpr int items_per_vec_load = 1;
};

// For min or max, verification showed the benefits were too small (within noise)

template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy600>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy600
    template <typename Tuning>
    static auto select_agent_policy(int)
      -> AgentReducePolicy<Tuning::threads,
                           Tuning::items,
                           AccumT,
                           Tuning::items_per_vec_load,
                           BLOCK_REDUCE_WARP_REDUCTIONS,
                           LOAD_LDG>;
    // use Policy600 as DefaultPolicy
    template <typename Tuning>
    static auto select_agent_policy(long) -> typename Policy600::ReducePolicy;

    using ReducePolicy =
      decltype(select_agent_policy<sm100_tuning<AccumT,
                                                OffsetT,
                                                classify_op<ReductionOpT>(),
                                                classify_offset_size<OffsetT>(),
                                                classify_accum_size<AccumT>()>>(0));

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy1000;
};

} // namespace reduce

namespace rfa
{
/**
 * @tparam AccumT
 *   Accumulator data type
 *
 * OffsetT
 *   Signed integer type for global offsets
 *
 * ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 */
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  //---------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //---------------------------------------------------------------------------

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B
    // items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;
  };

  /// SM60
  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy600;
};
} // namespace rfa

namespace fixed_size_segmented_reduce
{
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
  private:
    static constexpr int items_per_vec_load = 4;

    static constexpr int small_threads_per_warp  = 1;
    static constexpr int medium_threads_per_warp = 32;

    static constexpr int nominal_4b_large_threads_per_block = 256;

    static constexpr int nominal_4b_small_items_per_thread  = 16;
    static constexpr int nominal_4b_medium_items_per_thread = 16;
    static constexpr int nominal_4b_large_items_per_thread  = 16;

  public:
    using ReducePolicy =
      cub::AgentReducePolicy<nominal_4b_large_threads_per_block,
                             nominal_4b_large_items_per_thread,
                             AccumT,
                             items_per_vec_load,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             cub::LOAD_LDG>;

    using SmallReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 small_threads_per_warp,
                                 nominal_4b_small_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;

    using MediumReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 medium_threads_per_warp,
                                 nominal_4b_medium_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;
  };

  using MaxPolicy = Policy500;
};
} // namespace fixed_size_segmented_reduce
} // namespace detail

CUB_NAMESPACE_END
