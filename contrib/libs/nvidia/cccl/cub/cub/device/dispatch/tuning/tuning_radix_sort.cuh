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

#include <cub/agent/agent_radix_sort_downsweep.cuh>
#include <cub/agent/agent_radix_sort_histogram.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/agent/agent_scan.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace radix
{
// sm90 default
template <size_t KeySize, size_t ValueSize, size_t OffsetSize>
struct sm90_small_key_tuning
{
  static constexpr int threads = 384;
  static constexpr int items   = 23;
};

// clang-format off

// keys
template <> struct sm90_small_key_tuning<1,  0, 4> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<1,  0, 8> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<2,  0, 4> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<2,  0, 8> { static constexpr int threads = 512; static constexpr int items = 19; };

// pairs  8:xx
template <> struct sm90_small_key_tuning<1,  1, 4> { static constexpr int threads = 512; static constexpr int items = 15; };
template <> struct sm90_small_key_tuning<1,  1, 8> { static constexpr int threads = 448; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<1,  2, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<1,  2, 8> { static constexpr int threads = 512; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<1,  4, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<1,  4, 8> { static constexpr int threads = 512; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<1,  8, 4> { static constexpr int threads = 384; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<1,  8, 8> { static constexpr int threads = 384; static constexpr int items = 18; };
template <> struct sm90_small_key_tuning<1, 16, 4> { static constexpr int threads = 512; static constexpr int items = 22; };
template <> struct sm90_small_key_tuning<1, 16, 8> { static constexpr int threads = 512; static constexpr int items = 22; };

// pairs 16:xx
template <> struct sm90_small_key_tuning<2,  1, 4> { static constexpr int threads = 384; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<2,  1, 8> { static constexpr int threads = 384; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<2,  2, 4> { static constexpr int threads = 384; static constexpr int items = 15; };
template <> struct sm90_small_key_tuning<2,  2, 8> { static constexpr int threads = 448; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<2,  4, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<2,  4, 8> { static constexpr int threads = 512; static constexpr int items = 12; };
template <> struct sm90_small_key_tuning<2,  8, 4> { static constexpr int threads = 384; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<2,  8, 8> { static constexpr int threads = 512; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<2, 16, 4> { static constexpr int threads = 512; static constexpr int items = 21; };
template <> struct sm90_small_key_tuning<2, 16, 8> { static constexpr int threads = 576; static constexpr int items = 22; };
// clang-format on

// sm100 default
template <typename ValueT, size_t KeySize, size_t ValueSize, size_t OffsetSize>
struct sm100_small_key_tuning : sm90_small_key_tuning<KeySize, ValueSize, OffsetSize>
{};

// clang-format off

// keys

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  0, 4> : sm90_small_key_tuning<1, 0, 4> {};

// ipt_20.tpb_512 1.013282  0.967525  1.015764  1.047982
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  0, 4> { static constexpr int threads = 512; static constexpr int items = 20; };

// ipt_21.tpb_512 1.002873  0.994608  1.004196  1.019301
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  0, 4> { static constexpr int threads = 512; static constexpr int items = 21; };

// ipt_14.tpb_320 1.256020  1.000000  1.228182  1.486711
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  0, 4> { static constexpr int threads = 320; static constexpr int items = 14; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 16,  0, 4> : sm90_small_key_tuning<16, 0, 4> {};

// ipt_20.tpb_512 1.089698  0.979276  1.079822  1.199378
template <> struct sm100_small_key_tuning<float, 4,  0, 4> { static constexpr int threads = 512; static constexpr int items = 20; };

// ipt_18.tpb_288 1.049258  0.985085  1.042400  1.107771
template <> struct sm100_small_key_tuning<double, 8,  0, 4> { static constexpr int threads = 288; static constexpr int items = 18; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  0, 8> : sm90_small_key_tuning<1, 0, 8> {};

// ipt_20.tpb_384 1.038445  1.015608  1.037620  1.068105
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  0, 8> { static constexpr int threads = 384; static constexpr int items = 20; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  0, 8> : sm90_small_key_tuning<4, 0, 8> {};

// ipt_18.tpb_320 1.248354  1.000000  1.220666  1.446929
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  0, 8> { static constexpr int threads = 320; static constexpr int items = 18; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 16,  0, 8> : sm90_small_key_tuning<16, 0, 8> {};

// ipt_20.tpb_512 1.021557  0.981437  1.018920  1.039977
template <> struct sm100_small_key_tuning<float, 4,  0, 8> { static constexpr int threads = 512; static constexpr int items = 20; };

// ipt_21.tpb_256 1.068590  0.986635  1.059704  1.144921
template <> struct sm100_small_key_tuning<double, 8,  0, 8> { static constexpr int threads = 256; static constexpr int items = 21; };

// pairs 1-byte key

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  1, 4> : sm90_small_key_tuning<1, 1, 4> {};

// ipt_18.tpb_512 1.011463  0.978807  1.010106  1.024056
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  2, 4> { static constexpr int threads = 512; static constexpr int items = 18; };

// ipt_18.tpb_512 1.008207  0.980377  1.007132  1.022155
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  4, 4> { static constexpr int threads = 512; static constexpr int items = 18; };

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  8, 4> { static constexpr int threads = 288; static constexpr int items = 16; };

// ipt_21.tpb_576 1.044274  0.979145  1.038723  1.072068
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  16, 4> { static constexpr int threads = 576; static constexpr int items = 21; };

// ipt_20.tpb_384 1.008881  0.968750  1.006846  1.026910
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  1, 8> { static constexpr int threads = 384; static constexpr int items = 20; };

// ipt_22.tpb_256 1.015597  0.966038  1.011167  1.045921
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  2, 8> { static constexpr int threads = 256; static constexpr int items = 22; };

// ipt_15.tpb_384 1.029730  0.972699  1.029066  1.067894
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  4, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  8, 8> { static constexpr int threads = 256; static constexpr int items = 17; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  16, 8> : sm90_small_key_tuning<1, 16, 8> {};


// pairs 2-byte key

// ipt_20.tpb_448  1.031929  0.936849  1.023411  1.075172
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  1, 4> { static constexpr int threads = 448; static constexpr int items = 20; };

// ipt_23.tpb_384 1.104683  0.939335  1.087342  1.234988
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  2, 4> { static constexpr int threads = 384; static constexpr int items = 23; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  4, 4>  : sm90_small_key_tuning<2, 4, 4> {};

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  8, 4> { static constexpr int threads = 256; static constexpr int items = 17; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  16, 4> : sm90_small_key_tuning<2, 16, 4> {};

// ipt_15.tpb_384 1.093598  1.000000  1.088111  1.183369
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  1, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// ipt_15.tpb_576 1.040476  1.000333  1.037060  1.084850
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  2, 8> { static constexpr int threads = 576; static constexpr int items = 15; };

// ipt_18.tpb_512 1.096819  0.953488  1.082026  1.209533
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  4, 8> { static constexpr int threads = 512; static constexpr int items = 18; };

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  8, 8> { static constexpr int threads = 288; static constexpr int items = 16; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  16, 8> : sm90_small_key_tuning<2, 16, 8> {};


// pairs 4-byte key

// ipt_21.tpb_416 1.237956  1.001909  1.210882  1.469981
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  1, 4> { static constexpr int threads = 416; static constexpr int items = 21; };

// ipt_17.tpb_512 1.022121  1.012346  1.022439  1.038524
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  2, 4> { static constexpr int threads = 512; static constexpr int items = 17; };

// ipt_20.tpb_448 1.012688  0.999531  1.011865  1.028513
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  4, 4>  { static constexpr int threads = 448; static constexpr int items = 20; };

// ipt_15.tpb_384 1.006872  0.998651  1.008374  1.026118
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  8, 4> { static constexpr int threads = 384; static constexpr int items = 15; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  16, 4> : sm90_small_key_tuning<4, 16, 4> {};

// ipt_17.tpb_512 1.080000  0.927362  1.066211  1.172959
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  1, 8> { static constexpr int threads = 512; static constexpr int items = 17; };

// ipt_15.tpb_384 1.068529  1.000000  1.062277  1.135281
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  2, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// ipt_21.tpb_448  1.080642  0.927713  1.064758  1.191177
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  4, 8> { static constexpr int threads = 448; static constexpr int items = 21; };

// ipt_13.tpb_448 1.019046  0.991228  1.016971  1.039712
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  8, 8> { static constexpr int threads = 448; static constexpr int items = 13; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  16, 8> : sm90_small_key_tuning<4, 16, 8> {};

// pairs 8-byte key

// ipt_17.tpb_256 1.276445  1.025562  1.248511  1.496947
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  1, 4> { static constexpr int threads = 256; static constexpr int items = 17; };

// ipt_12.tpb_352 1.128086  1.040000  1.117960  1.207254
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  2, 4> { static constexpr int threads = 352; static constexpr int items = 12; };

// ipt_12.tpb_352 1.132699  1.040000  1.122676  1.207716
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  4, 4>  { static constexpr int threads = 352; static constexpr int items = 12; };

// ipt_18.tpb_256 1.266745  0.995432  1.237754  1.460538
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  8, 4> { static constexpr int threads = 256; static constexpr int items = 18; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  16, 4> : sm90_small_key_tuning<8, 16, 4> {};

// ipt_15.tpb_384 1.007343  0.997656  1.006929  1.047208
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  1, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// ipt_14.tpb_256 1.186477  1.012683  1.167150  1.332313
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  2, 8> { static constexpr int threads = 256; static constexpr int items = 14; };

// ipt_21.tpb_256 1.220607  1.000239  1.196400  1.390471
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  4, 8> { static constexpr int threads = 256; static constexpr int items = 21; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  8, 8> :  sm90_small_key_tuning<8, 8, 8> {};

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  16, 8> : sm90_small_key_tuning<8, 16, 8> {};
// clang-format on

template <typename PolicyT, typename = void>
struct RadixSortPolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION RadixSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct RadixSortPolicyWrapper<
  StaticPolicyT,
  ::cuda::std::void_t<typename StaticPolicyT::SingleTilePolicy,
                      typename StaticPolicyT::OnesweepPolicy,
                      typename StaticPolicyT::UpsweepPolicy,
                      typename StaticPolicyT::AltUpsweepPolicy,
                      typename StaticPolicyT::DownsweepPolicy,
                      typename StaticPolicyT::AltDownsweepPolicy,
                      typename StaticPolicyT::HistogramPolicy,
                      typename StaticPolicyT::ScanPolicy,
                      typename StaticPolicyT::ExclusiveSumPolicy,
                      typename StaticPolicyT::SegmentedPolicy,
                      typename StaticPolicyT::AltSegmentedPolicy>> : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION RadixSortPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr bool IsOnesweep()
  {
    return StaticPolicyT::ONESWEEP;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int RadixBits(PolicyT /*policy*/)
  {
    return PolicyT::RADIX_BITS;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int BlockThreads(PolicyT /*policy*/)
  {
    return PolicyT::BLOCK_THREADS;
  }

  CUB_DEFINE_SUB_POLICY_GETTER(SingleTile);
  CUB_DEFINE_SUB_POLICY_GETTER(Onesweep);
  CUB_DEFINE_SUB_POLICY_GETTER(Upsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(AltUpsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(Downsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(AltDownsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(Histogram);
  CUB_DEFINE_SUB_POLICY_GETTER(Scan);
  CUB_DEFINE_SUB_POLICY_GETTER(ExclusiveSum);
  CUB_DEFINE_SUB_POLICY_GETTER(Segmented);
  CUB_DEFINE_SUB_POLICY_GETTER(AltSegmented);
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION RadixSortPolicyWrapper<PolicyT> MakeRadixSortPolicyWrapper(PolicyT policy)
{
  return RadixSortPolicyWrapper<PolicyT>{policy};
}

/**
 * @brief Tuning policy for kernel specialization
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub
{
  //------------------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------------------

  // Whether this is a keys-only (or key-value) sort
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  // Dominant-sized key/value type
  using DominantT = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 3.5B 32b keys/s, 1.92B 32b pairs/s (TitanX)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 3.1B 32b segmented keys/s (TitanX)
      ONESWEEP               = false,
      ONESWEEP_RADIX_BITS    = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      160,
      39,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      31,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      256,
      11,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM60 (GP100)
  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 6.9B 32b keys/s (Quadro P100)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 5.9B 32b segmented keys/s (Quadro P100)
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t), // 10.0B 32b keys/s (GP100, 64M random keys)
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      OFFSET_64BIT ? 29 : 30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      25,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      192,
      OFFSET_64BIT ? 32 : 39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM61 (GP104)
  struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 3.4B 32b keys/s, 1.83B 32b pairs/s (1080)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 3.3B 32b segmented keys/s (1080)
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS    = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      384,
      31,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      35,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM62 (Tegra, less RF)
  struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
  {
    enum
    {
      PRIMARY_RADIX_BITS  = 5,
      ALT_RADIX_BITS      = PRIMARY_RADIX_BITS - 1,
      ONESWEEP            = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      ALT_RADIX_BITS>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy    = DownsweepPolicy;
    using AltSegmentedPolicy = AltDownsweepPolicy;
  };

  /// SM70 (GV100)
  struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 7.62B 32b keys/s (GV100)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 8.7B 32b segmented keys/s (GV100)
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t), // 15.8B 32b keys/s (V100-SXM2, 64M random keys)
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      sizeof(KeyT) == 4 && sizeof(ValueT) == 4 ? 46 : 23,
      DominantT,
      4,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      OFFSET_64BIT ? 46 : 47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy =
      AgentRadixSortUpsweepPolicy<256, OFFSET_64BIT ? 46 : 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM80
  struct Policy800 : ChainedPolicy<800, Policy800, Policy700>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      384,
      OFFSET_64BIT && sizeof(KeyT) == 4 && !KEYS_ONLY ? 17 : 21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  template <typename OnesweepSmallKeyPolicySizes>
  struct OnesweepSmallKeyTunedPolicy
  {
    static constexpr bool ONESWEEP           = true;
    static constexpr int ONESWEEP_RADIX_BITS = 8;

    using HistogramPolicy    = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

  private:
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5;
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int OFFSET_64BIT           = sizeof(OffsetT) == 8 ? 1 : 0;
    static constexpr int FLOAT_KEYS             = ::cuda::std::is_same_v<KeyT, float> ? 1 : 0;

    using OnesweepPolicyKey32 = AgentRadixSortOnesweepPolicy<
      384,
      KEYS_ONLY ? 20 - OFFSET_64BIT - FLOAT_KEYS
                : (sizeof(ValueT) < 8 ? (OFFSET_64BIT ? 17 : 23) : (OFFSET_64BIT ? 29 : 30)),
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepPolicyKey64 = AgentRadixSortOnesweepPolicy<
      384,
      sizeof(ValueT) < 8 ? 30 : 24,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepLargeKeyPolicy = ::cuda::std::_If<sizeof(KeyT) == 4, OnesweepPolicyKey32, OnesweepPolicyKey64>;

    using OnesweepSmallKeyPolicy = AgentRadixSortOnesweepPolicy<
      OnesweepSmallKeyPolicySizes::threads,
      OnesweepSmallKeyPolicySizes::items,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      8>;

  public:
    using OnesweepPolicy = ::cuda::std::_If<sizeof(KeyT) < 4, OnesweepSmallKeyPolicy, OnesweepLargeKeyPolicy>;

    // The Scan, Downsweep and Upsweep policies are never run on SM90, but we have to include them to prevent a
    // compilation error: When we compile e.g. for SM70 **and** SM90, the host compiler will reach calls to those
    // kernels, and instantiate them for MaxPolicy (which is Policy900) on the host, which will reach into the policies
    // below to set the launch bounds. The device compiler pass will also compile all kernels for SM70 **and** SM90,
    // even though only the Onesweep kernel is used on SM90.
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;

    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  struct Policy900
      : ChainedPolicy<900, Policy900, Policy800>
      , OnesweepSmallKeyTunedPolicy<sm90_small_key_tuning<sizeof(KeyT), KEYS_ONLY ? 0 : sizeof(ValueT), sizeof(OffsetT)>>
  {};

  struct Policy1000
      : ChainedPolicy<1000, Policy1000, Policy900>
      , OnesweepSmallKeyTunedPolicy<
          sm100_small_key_tuning<ValueT, sizeof(KeyT), KEYS_ONLY ? 0 : sizeof(ValueT), sizeof(OffsetT)>>
  {};

  using MaxPolicy = Policy1000;
};

} // namespace radix
} // namespace detail

CUB_NAMESPACE_END
