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

#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/agent/agent_rle.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/cmath>
#include <cuda/std/__algorithm_>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace rle
{
enum class primitive_key
{
  no,
  yes
};
enum class primitive_length
{
  no,
  yes
};
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class length_size
{
  _4,
  unknown
};

template <class T>
constexpr primitive_key is_primitive_key()
{
  return is_primitive<T>::value ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_length is_primitive_length()
{
  return is_primitive<T>::value ? primitive_length::yes : primitive_length::no;
}

template <class KeyT>
constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class LengthT>
constexpr length_size classify_length_size()
{
  return sizeof(LengthT) == 4 ? length_size::_4 : length_size::unknown;
}

namespace encode
{

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm80_tuning;

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<640>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<900>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1080>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1075>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<630>;
};

template <class LengthT>
struct sm80_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm90_tuning;

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<620>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<775>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<284, 480>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 19;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<515>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<428, 930>;
};

template <class LengthT>
struct sm90_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm100_tuning;

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  // ipt_14.tpb_256.trp_0.ld_1.ns_468.dcid_7.l2w_300 1.202228  1.126160  1.197973  1.307692
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backon_constructor_t<468, 300>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  // ipt_14.tpb_224.trp_0.ld_0.ns_376.dcid_7.l2w_420 1.123754  1.002404  1.113839  1.274882
  static constexpr int threads                       = 224;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_constructor_t<376, 420>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  // ipt_14.tpb_256.trp_0.ld_1.ns_956.dcid_7.l2w_70 1.134395  1.071951  1.137008  1.169419
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backon_constructor_t<956, 70>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_188.dcid_2.l2w_765 1.100140  1.020069  1.116462  1.345506
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<188, 765>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class LengthT>
// struct sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {
//   static constexpr int threads = 128;
//   static constexpr int items = 11;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor = detail::fixed_delay_constructor_t<428, 930>;
// };

// template <class LengthT>
// struct sm100_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
//     : sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {};
#endif

// this policy is passed to DispatchReduceByKey
template <class LengthT, class KeyT>
struct policy_hub
{
  static constexpr int max_input_bytes      = static_cast<int>((::cuda::std::max) (sizeof(KeyT), sizeof(LengthT)));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(LengthT);

  template <CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 6;
    static constexpr int items =
      (max_input_bytes <= 8)
        ? 6
        : ::cuda::std::clamp(
            ::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes), 1, nominal_4B_items_per_thread);
    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<128,
                             items,
                             BLOCK_LOAD_DIRECT,
                             LoadModifier,
                             BLOCK_SCAN_WARP_SCANS,
                             default_reduce_by_key_delay_constructor_t<LengthT, int>>;
  };

  struct Policy500
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentReduceByKeyPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              LOAD_DEFAULT,
                              BLOCK_SCAN_WARP_SCANS,
                              typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy<LOAD_DEFAULT>::ReduceByKeyPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    using ReduceByKeyPolicyT = decltype(select_agent_policy<sm80_tuning<LengthT, KeyT>>(0));
  };

  struct Policy860
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ReduceByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<LengthT, KeyT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> AgentReduceByKeyPolicy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                BLOCK_SCAN_WARP_SCANS,
                                typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy100(long) -> typename Policy900::ReduceByKeyPolicyT;

    using ReduceByKeyPolicyT = decltype(select_agent_policy100<sm100_tuning<LengthT, KeyT>>(0));
  };

  using MaxPolicy = Policy1000;
};
} // namespace encode

namespace non_trivial_runs
{

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm80_tuning;

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<630>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<1015>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<915>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<1065>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<1050>;
};

template <class LengthT>
struct sm80_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm90_tuning;

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<385>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<675>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<695>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<840>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::fixed_delay_constructor_t<484, 1150>;
};

template <class LengthT>
struct sm90_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm100_tuning;

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  // ipt_20.tpb_224.trp_1.ts_0.ld_1.ns_64.dcid_2.l2w_315 1.119878  1.003690  1.130067  1.338983
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<64, 315>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  // ipt_20.tpb_224.trp_1.ts_0.ld_0.ns_116.dcid_7.l2w_340 1.146528  1.072769  1.152390  1.333333
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_constructor_t<116, 340>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  // ipt_13.tpb_224.trp_0.ts_0.ld_0.ns_252.dcid_2.l2w_470 1.113202  1.003690  1.133114  1.349296
  static constexpr int threads                       = 224;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<252, 470>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  // ipt_15.tpb_256.trp_1.ts_0.ld_0.ns_28.dcid_2.l2w_520 1.114944  1.033189  1.122360  1.252083
  static constexpr int threads                       = 256;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<28, 520>;
};
// Fall back to Policy900 for double, because that one performs better than the above tuning (same key_size)
// TODO(bgruber): in C++20 put a requires(!::cuda::std::is_same_v<KeyT, double>) onto the above tuning and delete this
// one
template <class LengthT>
struct sm100_tuning<LengthT, double, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
    : sm90_tuning<LengthT, double, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class LengthT>
// struct sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {
//   static constexpr int threads = 288;
//   static constexpr int items = 9;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr bool store_with_time_slicing = false;
//   using delay_constructor = detail::fixed_delay_constructor_t<484, 1150>;
// };

// template <class LengthT>
// struct sm100_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
//     : sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {};
#endif

template <class LengthT, class KeyT>
struct policy_hub
{
  template <BlockLoadAlgorithm BlockLoad, typename DelayConstructorKey, CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 15;
    // TODO(bgruber): use clamp() in C++14
    static constexpr int ITEMS_PER_THREAD =
      _CUDA_VSTD::clamp(nominal_4B_items_per_thread * 4 / int{sizeof(KeyT)}, 1, nominal_4B_items_per_thread);
    using RleSweepPolicyT =
      AgentRlePolicy<96,
                     ITEMS_PER_THREAD,
                     BlockLoad,
                     LoadModifier,
                     true,
                     BLOCK_SCAN_WARP_SCANS,
                     default_reduce_by_key_delay_constructor_t<DelayConstructorKey, int>>;
  };

  struct Policy500
      : DefaultPolicy<BLOCK_LOAD_DIRECT, int, LOAD_LDG> // TODO(bgruber): I think we want `LengthT` instead of `int`
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentRlePolicy<Tuning::threads,
                      Tuning::items,
                      Tuning::load_algorithm,
                      LOAD_DEFAULT,
                      Tuning::store_with_time_slicing,
                      BLOCK_SCAN_WARP_SCANS,
                      typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) ->
    typename DefaultPolicy<BLOCK_LOAD_WARP_TRANSPOSE, LengthT, LOAD_DEFAULT>::RleSweepPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    using RleSweepPolicyT = decltype(select_agent_policy<sm80_tuning<LengthT, KeyT>>(0));
  };

  struct Policy860
      : DefaultPolicy<BLOCK_LOAD_DIRECT, int, LOAD_LDG> // TODO(bgruber): I think we want `LengthT` instead of `int`
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using RleSweepPolicyT = decltype(select_agent_policy<sm90_tuning<LengthT, KeyT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> AgentRlePolicy<Tuning::threads,
                        Tuning::items,
                        Tuning::load_algorithm,
                        Tuning::load_modifier,
                        Tuning::store_with_time_slicing,
                        BLOCK_SCAN_WARP_SCANS,
                        typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy100(long) -> typename Policy900::RleSweepPolicyT;

    using RleSweepPolicyT = decltype(select_agent_policy100<sm100_tuning<LengthT, KeyT>>(0));
  };

  using MaxPolicy = Policy1000;
};
} // namespace non_trivial_runs
} // namespace rle
} // namespace detail

CUB_NAMESPACE_END
