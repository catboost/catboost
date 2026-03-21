/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/functional>
#include <cuda/std/functional>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace scan
{
enum class keep_rejects
{
  no,
  yes
};
enum class primitive_accum
{
  no,
  yes
};
enum class primitive_op
{
  no,
  yes
};
enum class op_type
{
  plus,
  unknown
};
enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class value_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
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
constexpr primitive_accum is_primitive_accum()
{
  return is_primitive<AccumT>::value ? primitive_accum::yes : primitive_accum::no;
}

template <class ScanOpT>
constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ScanOpT>::value ? primitive_op::yes : primitive_op::no;
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

template <class ScanOpT>
constexpr op_type classify_op()
{
  return is_plus<ScanOpT>::value ? op_type::plus : op_type::unknown;
}

template <class ValueT>
constexpr value_size classify_value_size()
{
  return sizeof(ValueT) == 1 ? value_size::_1
       : sizeof(ValueT) == 2 ? value_size::_2
       : sizeof(ValueT) == 4 ? value_size::_4
       : sizeof(ValueT) == 8 ? value_size::_8
       : sizeof(ValueT) == 16
         ? value_size::_16
         : value_size::unknown;
}

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

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm80_tuning;

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 14;
  using delay_constructor                              = fixed_delay_constructor_t<368, 725>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2>
{
  static constexpr int threads                         = 352;
  static constexpr int items                           = 16;
  using delay_constructor                              = fixed_delay_constructor_t<488, 1040>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 12;
  using delay_constructor                              = fixed_delay_constructor_t<268, 1180>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 22;
  using delay_constructor                              = fixed_delay_constructor_t<716, 785>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<float, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 8;
  using delay_constructor                              = fixed_delay_constructor_t<724, 1050>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads                         = 384;
  static constexpr int items                           = 12;
  using delay_constructor                              = fixed_delay_constructor_t<388, 1100>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{
  static constexpr int threads                         = 640;
  static constexpr int items                           = 24;
  using delay_constructor                              = no_delay_constructor_t<1200>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
};

template <>
struct sm80_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
    : sm80_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{};
#endif

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm90_tuning;

template <class AccumT, int Threads, int Items, int L2B, int L2W>
struct sm90_tuning_vals
{
  static constexpr int threads = Threads;
  static constexpr int items   = Items;
  using delay_constructor      = fixed_delay_constructor_t<L2B, L2W>;
  // same logic as default policy:
  static constexpr bool large_values = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm load_algorithm =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;
};

// clang-format off
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1> : sm90_tuning_vals<T, 192, 22, 168, 1140> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2> : sm90_tuning_vals<T, 512, 12, 376, 1125> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4> : sm90_tuning_vals<T, 128, 24, 648, 1245> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8> : sm90_tuning_vals<T, 224, 24, 632, 1290> {};

template <> struct sm90_tuning<float,  primitive_op::yes, primitive_accum::yes, accum_size::_4> : sm90_tuning_vals<float,  128, 24, 688, 1140> {};
template <> struct sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8> : sm90_tuning_vals<double, 224, 24, 576, 1215> {};

#if _CCCL_HAS_INT128()
template <> struct sm90_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16> : sm90_tuning_vals<__int128_t, 576, 21, 860, 630> {};
template <>
struct sm90_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
    : sm90_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{};
#endif
// clang-format on

template <class ValueT,
          class AccumT,
          class OffsetT,
          op_type OpTypeT,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          offset_size OffsetSize               = classify_offset_size<OffsetT>(),
          value_size ValueSize                 = classify_value_size<ValueT>()>
struct sm100_tuning;

// sum
template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_1>
{
  // ipt_18.tpb_512.ns_768.dcid_7.l2w_820.trp_1.ld_0 1.188818  1.005682  1.173041  1.305288
  static constexpr int items                           = 18;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<768, 820>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_1>
{
  // ipt_14.tpb_384.ns_228.dcid_7.l2w_775.trp_1.ld_1 1.107210  1.000000  1.100637  1.307692
  static constexpr int items                           = 14;
  static constexpr int threads                         = 384;
  using delay_constructor                              = exponential_backon_constructor_t<228, 775>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_2>
{
  // ipt_13.tpb_512.ns_1384.dcid_7.l2w_720.trp_1.ld_0 1.128443  1.002841  1.119688  1.307692
  static constexpr int items                           = 13;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<1384, 720>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// todo(gonidelis): Regresses for large inputs. Find better tuning.
// template <class ValueT, class AccumT, class OffsetT>
// struct sm100_tuning<ValueT,
//                     AccumT,
//                     OffsetT,
//                     op_type::plus,
//                     primitive_value::yes,
//                     primitive_accum::yes,
//                     offset_size::_8,
//                     value_size::_2>
// {
//   // ipt_13.tpb_288.ns_1520.dcid_5.l2w_895.trp_1.ld_1 1.080934  0.983509  1.077724  1.305288
//   static constexpr int items                           = 13;
//   static constexpr int threads                         = 288;
//   using delay_constructor                              = exponential_backon_jitter_window_constructor_t<1520, 895>;
//   static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
// };

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_4>
{
  // ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148442  0.997167  1.139902  1.462651
  static constexpr int items                           = 22;
  static constexpr int threads                         = 384;
  using delay_constructor                              = exponential_backon_jitter_constructor_t<1904, 830>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_4>
{
  // ipt_19.tpb_416.ns_956.dcid_7.l2w_550.trp_1.ld_1 1.146142  0.994350  1.137459  1.455636
  static constexpr int items                           = 19;
  static constexpr int threads                         = 416;
  using delay_constructor                              = exponential_backon_constructor_t<956, 550>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_8>
{
  // ipt_23.tpb_416.ns_772.dcid_5.l2w_710.trp_1.ld_0 1.089468  1.015581  1.085630  1.264583
  static constexpr int items                           = 23;
  static constexpr int threads                         = 416;
  using delay_constructor                              = exponential_backon_jitter_window_constructor_t<772, 710>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_8>
{
  // ipt_22.tpb_320.ns_328.dcid_2.l2w_965.trp_1.ld_0 1.080133  1.000000  1.075577  1.248963
  static constexpr int items                           = 22;
  static constexpr int threads                         = 320;
  using delay_constructor                              = exponential_backoff_constructor_t<328, 965>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for i128, float and double.
// template <class OffsetT> struct sm100_tuning<float, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8,
// accum_size::_4>;
// Default explicitly so it doesn't pick up the sm100<I64, I64> tuning.
template <class AccumT, class OffsetT>
struct sm100_tuning<double, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_8>
    : sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{};

#if _CCCL_HAS_INT128()
// template <class OffsetT> struct sm100_tuning<__int128_t, OffsetT, op_type::plus, primitive_accum::no,
// offset_size::_8, accum_size::_16> : tuning<576, 21, 860, 630> {}; template <class OffsetT> struct
// sm100_tuning<__uint128_t, OffsetT, op_type::plus, primitive_accum::no, offset_size::_8, accum_size::_16>
//     : sm100_tuning<__int128_t, OffsetT, op_type::plus, primitive_accum::no, offset_size::_8, accum_size::_16>
// {};
#endif

template <typename PolicyT, typename = void, typename = void>
struct ScanPolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION ScanPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ScanPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<decltype(StaticPolicyT::ScanPolicyT::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION ScanPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr auto Scan()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::ScanPolicyT());
  }

  CUB_RUNTIME_FUNCTION static constexpr CacheLoadModifier LoadModifier()
  {
    return StaticPolicyT::ScanPolicyT::LOAD_MODIFIER;
  }

  CUB_RUNTIME_FUNCTION constexpr void CheckLoadModifier()
  {
    static_assert(LoadModifier() != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");
  }
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION ScanPolicyWrapper<PolicyT> MakeScanPolicyWrapper(PolicyT policy)
{
  return ScanPolicyWrapper<PolicyT>{policy};
}

template <typename InputValueT, typename OutputValueT, typename AccumT, typename OffsetT, typename ScanOpT>
struct policy_hub
{
  // For large values, use timesliced loads/stores to fit shared memory.
  static constexpr bool large_values = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm scan_transposed_load =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm scan_transposed_store =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, BLOCK_SCAN_RAKING>;
  };
  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    // Titan X: 32.47B items/s @ 48M 32-bit T
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, scan_transposed_store, BLOCK_SCAN_WARP_SCANS>;
  };

  struct DefaultPolicy
  {
    using ScanPolicyT =
      AgentScanPolicy<128, 15, AccumT, scan_transposed_load, LOAD_DEFAULT, scan_transposed_store, BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy600
      : DefaultPolicy
      , ChainedPolicy<600, Policy600, Policy520>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentScanPolicy<Tuning::threads,
                       Tuning::items,
                       AccumT,
                       Tuning::load_algorithm,
                       LOAD_DEFAULT,
                       Tuning::store_algorithm,
                       BLOCK_SCAN_WARP_SCANS,
                       cub::detail::MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                       typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy::ScanPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy600>
  {
    using ScanPolicyT = decltype(select_agent_policy<sm80_tuning<AccumT, is_primitive_op<ScanOpT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ScanPolicyT = decltype(select_agent_policy<sm90_tuning<AccumT, is_primitive_op<ScanOpT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists that matches a benchmark, otherwise pick Policy900
    template <typename Tuning,
              typename IVT,
              // In the tuning benchmarks the Initial-, Input- and OutputType are the same. Let's check that the
              // accumulator type's size matches what we used during the benchmark since that has an impact (The
              // tunings also check later that it's a primitive type, so arithmetic impact is also comparable to the
              // benchmark). Input- and OutputType only impact loading and storing data (all arithmetic is done in the
              // accumulator type), so let's check that they are the same size and dispatch the size in the tunings.
              ::cuda::std::enable_if_t<sizeof(AccumT) == sizeof(::cuda::std::__accumulator_t<ScanOpT, IVT, IVT>)
                                         && sizeof(IVT) == sizeof(OutputValueT),
                                       int> = 0>
    static auto select_agent_policy100(int)
      -> AgentScanPolicy<Tuning::threads,
                         Tuning::items,
                         AccumT,
                         Tuning::load_algorithm,
                         Tuning::load_modifier,
                         Tuning::store_algorithm,
                         BLOCK_SCAN_WARP_SCANS,
                         MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                         typename Tuning::delay_constructor>;
    template <typename Tuning, typename IVT>
    static auto select_agent_policy100(long) -> typename Policy900::ScanPolicyT;

    using ScanPolicyT =
      decltype(select_agent_policy100<sm100_tuning<InputValueT, AccumT, OffsetT, classify_op<ScanOpT>()>, InputValueT>(
        0));
  };

  using MaxPolicy = Policy1000;
};
} // namespace scan
} // namespace detail

CUB_NAMESPACE_END
