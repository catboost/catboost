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

#include <cub/agent/agent_select_if.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm_>

CUB_NAMESPACE_BEGIN

namespace detail::select
{

enum class may_alias
{
  no,
  yes
};

enum class flagged
{
  no,
  yes
};
enum class keep_rejects
{
  no,
  yes
};
enum class primitive
{
  no,
  yes
};
enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class input_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class distinct_partitions
{
  no,
  yes
};

template <class InputT, flagged, keep_rejects, offset_size OffsetSize, primitive, input_size InputSize>
struct sm80_tuning;

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 992;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<395>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 576;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<870>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1130>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<832, 1165>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 4;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1140>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// select::flagged
template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<735>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1155>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<124, 1115>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 6;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1130>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<464, 1025>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::if
template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<510>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1045>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1040>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<68, 1160>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<400, 1090>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::flagged
template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<595>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1105>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<912, 1025>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<884, 1130>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<400, 1090>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

template <class InputT, flagged, keep_rejects, offset_size OffsetSize, primitive, input_size InputSize>
struct sm90_tuning;

// select::if
template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<580>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<320, 605>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 17;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<76, 1150>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<380, 1140>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<460, 1145>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// select::flagged
template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<715>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<504, 765>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<415, 1125>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<360, 1170>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 3;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<284, 1130>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::if
template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<908, 995>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<500, 560>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<536, 1055>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<512, 1075>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<1616, 1115>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::flagged
template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<580, 850>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<388, 1055>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<72, 1165>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 6;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<532, 1180>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 160;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<720, 1105>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

template <class InputT,
          flagged,
          keep_rejects,
          offset_size OffsetSize,
          primitive,
          input_size InputSize,
          may_alias,
          distinct_partitions DistinctPartitions>
struct sm100_tuning;

// select::if
template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    DistinctPartitions>
{
  // trp_0.ld_0.ipt_22.tpb_384.ns_0.dcid_2.l2w_915 1.099232  0.980183  1.096778  1.545455
  static constexpr int threads                       = 384;
  static constexpr int nominal_4b_items              = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<0, 915>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::yes,
                    DistinctPartitions>
{
  // trp_1.ld_0.ipt_20.tpb_448.ns_596.dcid_6.l2w_295  1.214635  1.001421  1.207023  1.307692
  static constexpr int threads                       = 448;
  static constexpr int nominal_4b_items              = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_jitter_constructor_t<596, 295>;
};

// todo(gonidelis): for large input size select.unique regresses a lot and select.if regresses a bit.
// find better tuning.
// template <class Input, distinct_partitions DistinctPartitions>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::no,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_2,
//                     may_alias::no,
//                     DistinctPartitions>
// {
//   // trp_1.ld_0.ipt_20.tpb_256.ns_516.dcid_7.l2w_685 1.065598  0.937984  1.067343  1.452153
//   static constexpr int threads                       = 256;
//   static constexpr int nominal_4b_items              = 20;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = detail::exponential_backon_constructor_t<516, 685>;
// };

// template <class Input, distinct_partitions DistinctPartitions>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::no,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_2,
//                     may_alias::no,
//                     DistinctPartitions>
// {
//   // trp_1.ld_0.ipt_20.tpb_384.ns_1060.dcid_5.l2w_375 1.109871  0.973142  1.105415  1.459135
//   static constexpr int threads                       = 384;
//   static constexpr int nominal_4b_items              = 20;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor = detail::exponential_backon_jitter_window_constructor_t<1060, 375>;
// };

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    DistinctPartitions>
{
  // trp_1.ld_0.ipt_15.tpb_384.ns_1508.dcid_5.l2w_585 1.201993  0.920103  1.185134  1.441805
  static constexpr int threads                       = 384;
  static constexpr int nominal_4b_items              = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor = detail::exponential_backon_jitter_window_constructor_t<1508, 585>;
};

// todo(gonidelis): for large input size select.unique regresses a lot and select.if regresses a bit.
// find better tuning.
// template <class Input, distinct_partitions DistinctPartitions>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::no,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_4,
//                     may_alias::yes,
//                     DistinctPartitions>
// {
//   // trp_1.ld_0.ipt_19.tpb_512.ns_928.dcid_7.l2w_770 1.258815  1.000000  1.235251  1.444884
//   static constexpr int threads                       = 512;
//   static constexpr int nominal_4b_items              = 19;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = detail::exponential_backon_constructor_t<928, 770>;
// };

// template <class Input, may_alias MayAlias, distinct_partitions DistinctPartitions>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::no,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_8,
//                     may_alias::yes,
//                     MayAlias,
//                     DistinctPartitions>
// {
//   // trp_1.ld_0.ipt_23.tpb_384.ns_1140.dcid_7.l2w_520 1.081506  0.955298  1.088848  1.248971
//   static constexpr int threads                       = 384;
//   static constexpr int nominal_4b_items              = 23;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = detail::exponential_backon_constructor_t<1140, 520>;
// };

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <>
// struct sm100_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
// {
// // static constexpr int threads = 512;
// // static constexpr int nominal_4b_items   = 5;

// // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

// // using delay_constructor = detail::fixed_delay_constructor_t<460, 1145>;
// };

// template <>
// struct sm100_tuning<__uint128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
// {
// // static constexpr int threads = 512;
// // static constexpr int nominal_4b_items   = 5;

// // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

// // using delay_constructor = detail::fixed_delay_constructor_t<460, 1145>;
// };
#endif

// select::flagged
template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    DistinctPartitions>
{
  // trp_0.ld_0.ipt_20.tpb_896.ns_84.dcid_7.l2w_480 1.254262  0.846154  1.222437  1.462665
  static constexpr int threads                       = 896;
  static constexpr int nominal_4b_items              = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_constructor_t<84, 480>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::yes,
                    DistinctPartitions>
{
  // trp_0.ld_0.ipt_20.tpb_1024.ns_360.dcid_6.l2w_380 1.274174  0.748441  1.227123  1.610039
  static constexpr int threads                       = 1024;
  static constexpr int nominal_4b_items              = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_jitter_constructor_t<360, 380>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    DistinctPartitions>
{
  // trp_0.ld_0.ipt_22.tpb_256.ns_1292.dcid_5.l2w_750 1.283400  1.002841  1.267822  1.445913
  static constexpr int threads                       = 256;
  static constexpr int nominal_4b_items              = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor = detail::exponential_backon_jitter_window_constructor_t<1292, 750>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_2,
                    may_alias::yes,
                    DistinctPartitions>
{
  // trp_1.ld_0.ipt_20.tpb_448.ns_136.dcid_2.l2w_760 1.318819  0.994090  1.289173  1.551415
  static constexpr int threads                       = 448;
  static constexpr int nominal_4b_items              = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<136, 760>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    DistinctPartitions>
{
  // trp_0.ld_0.ipt_14.tpb_512.ns_844.dcid_6.l2w_675 1.207911  1.068001  1.208890  1.455636
  static constexpr int threads                       = 512;
  static constexpr int nominal_4b_items              = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_jitter_constructor_t<844, 675>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_4,
                    may_alias::yes,
                    DistinctPartitions>
{
  // trp_1.ld_0.ipt_14.tpb_384.ns_524.dcid_7.l2w_635 1.256212  1.004808  1.241086  1.373337
  static constexpr int threads                       = 384;
  static constexpr int nominal_4b_items              = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_constructor_t<524, 635>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_8,
                    may_alias::no,
                    DistinctPartitions>
{
  // trp_0.ld_1.ipt_22.tpb_320.ns_660.dcid_7.l2w_1030 1.162087  0.997167  1.154955  1.395010
  static constexpr int threads                       = 320;
  static constexpr int nominal_4b_items              = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backon_constructor_t<660, 1030>;
};

template <class Input, distinct_partitions DistinctPartitions>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::no,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_8,
                    may_alias::yes,
                    DistinctPartitions>
{
  // trp_1.ld_1.ipt_21.tpb_384.ns_1316.dcid_5.l2w_990 1.221365  1.019231  1.213141  1.372951
  static constexpr int threads                       = 384;
  static constexpr int nominal_4b_items              = 21;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor = detail::exponential_backon_jitter_window_constructor_t<1316, 990>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <>
// struct sm100_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
// {
// static constexpr int threads = 512;
// static constexpr int nominal_4b_items   = 3;

// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

// using delay_constructor = detail::fixed_delay_constructor_t<284, 1130>;
// };

// template <>
// struct sm100_tuning<__uint128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
// {
// static constexpr int threads = 512;
// static constexpr int nominal_4b_items   = 3;

// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

// using delay_constructor = detail::fixed_delay_constructor_t<284, 1130>;
// };
#endif

// partition::if
template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_15.tpb_608.ns_676.dcid_7.l2w_500 1.171303  1.042818  1.175890  1.456731
  static constexpr int nominal_4b_items              = 15;
  static constexpr int threads                       = 608;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<676, 500>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_22.tpb_320.ns_1756.dcid_6.l2w_615 1.206387  1.079118  1.202408  1.307692
  static constexpr int nominal_4b_items              = 22;
  static constexpr int threads                       = 320;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<1756, 615>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_1.ld_0.ipt_19.tpb_320.ns_716.dcid_5.l2w_570 1.177521  1.123348  1.177703  1.307692
  static constexpr int nominal_4b_items              = 19;
  static constexpr int threads                       = 320;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<716, 570>;
};

// TODO(gonidelis): Tuning Regresses for large input sizes. Find better tuning.
// template <class Input>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::yes,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_8,
//                     may_alias::no,
//                     distinct_partitions::yes>
// {
//   // trp_1.ld_0.ipt_20.tpb_416.ns_1672.dcid_7.l2w_1050 1.086221  0.977775  1.090731  1.257618
//   static constexpr int nominal_4b_items              = 20;
//   static constexpr int threads                       = 416;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = exponential_backon_constructor_t<1672, 1050>;
// };

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_22.tpb_576.ns_368.dcid_7.l2w_680 1.191750  0.990521  1.175654  1.433174
  static constexpr int nominal_4b_items              = 22;
  static constexpr int threads                       = 576;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<368, 680>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_1.ld_0.ipt_20.tpb_608.ns_516.dcid_7.l2w_635 1.244961  0.848558  1.212567  1.461538
  static constexpr int nominal_4b_items              = 20;
  static constexpr int threads                       = 608;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<516, 635>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_1.ld_0.ipt_18.tpb_608.ns_1712.dcid_5.l2w_825 1.255078  0.990588  1.231055  1.421176
  static constexpr int nominal_4b_items              = 18;
  static constexpr int threads                       = 608;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1712, 825>;
};

// todo(gonidelis): Tuning Regresses for large input sizes. Find better tuning.
// template <class Input>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::yes,
//                     offset_size::_8,
//                     primitive::yes,
//                     input_size::_8,
//                     may_alias::no,
//                     distinct_partitions::yes>
// {
//   // trp_1.ld_0.ipt_14.tpb_512.ns_1468.dcid_7.l2w_820 1.111830  1.011070  1.119481  1.245868
//   static constexpr int nominal_4b_items              = 14;
//   static constexpr int threads                       = 512;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = exponential_backon_jitter_constructor_t<1468, 820>;
// };

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_22.tpb_224.ns_68.dcid_2.l2w_990 1.151989  1.064433  1.146707  1.305288
  static constexpr int nominal_4b_items              = 22;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backoff_constructor_t<68, 990>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_22.tpb_320.ns_560.dcid_5.l2w_640 1.205538  1.080520  1.201709  1.307692
  static constexpr int nominal_4b_items              = 22;
  static constexpr int threads                       = 320;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<560, 640>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_1.ld_0.ipt_19.tpb_608.ns_724.dcid_5.l2w_970 1.196592  0.982227  1.177984  1.310843
  static constexpr int nominal_4b_items              = 19;
  static constexpr int threads                       = 608;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<724, 970>;
};

// template <class Input>
// struct sm100_tuning<Input,
//                     flagged::no,
//                     keep_rejects::yes,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_8,
//                     may_alias::no,
//                     distinct_partitions::no>
// {
//   // trp_1.ld_0.ipt_23.tpb_416.ns_1608.dcid_2.l2w_560 1.099752  0.977393  1.106477  1.259336
//   static constexpr int nominal_4b_items              = 23;
//   static constexpr int threads                       = 416;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = exponential_backoff_constructor_t<1608, 560>;
// };

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_20.tpb_608.ns_1016.dcid_6.l2w_545 1.239144  1.002404  1.225460  1.444711
  static constexpr int nominal_4b_items              = 20;
  static constexpr int threads                       = 608;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<1016, 545>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_1.ld_0.ipt_22.tpb_288.ns_124.dcid_2.l2w_690 1.202783  1.000000  1.183737  1.311755
  static constexpr int nominal_4b_items              = 22;
  static constexpr int threads                       = 288;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backoff_constructor_t<124, 690>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_1.ld_0.ipt_19.tpb_608.ns_1884.dcid_6.l2w_950 1.250302  0.988124  1.225191  1.392931
  static constexpr int nominal_4b_items              = 19;
  static constexpr int threads                       = 608;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<1884, 950>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::no,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_8,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_1.ld_0.ipt_23.tpb_416.ns_0.dcid_2.l2w_1200 1.156864  1.011990  1.152368  1.266667
  static constexpr int nominal_4b_items              = 23;
  static constexpr int threads                       = 416;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backoff_constructor_t<0, 1200>;
};

// partition::flagged
template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_20.tpb_448.ns_964.dcid_7.l2w_385 1.111204  1.036205  1.111986  1.275210
  static constexpr int nominal_4b_items              = 20;
  static constexpr int threads                       = 448;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<964, 385>;
};

// todo(gonidelis): Tunings regress for large input sizes. Find better tunings.
// template <class Input>
// struct sm100_tuning<Input,
//                     flagged::yes,
//                     keep_rejects::yes,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_2,
//                     may_alias::no,
//                     distinct_partitions::yes>
// {
//   // trp_0.ld_0.ipt_18.tpb_256.ns_300.dcid_6.l2w_820 1.107466  0.923750  1.126995  1.346591
//   static constexpr int nominal_4b_items              = 18;
//   static constexpr int threads                       = 256;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = exponential_backon_jitter_constructor_t<300, 820>;
// };

// template <class Input>
// struct sm100_tuning<Input,
//                     flagged::yes,
//                     keep_rejects::yes,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_4,
//                     may_alias::no,
//                     distinct_partitions::yes>
// {
//   // trp_0.ld_0.ipt_19.tpb_256.ns_1608.dcid_7.l2w_675 1.097548  0.964114  1.109189  1.283333
//   static constexpr int nominal_4b_items              = 19;
//   static constexpr int threads                       = 256;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = exponential_backon_constructor_t<1608, 675>;
// };

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_8,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_21.tpb_384.ns_300.dcid_7.l2w_580 1.239128  1.019324  1.238373  1.347458
  static constexpr int nominal_4b_items              = 21;
  static constexpr int threads                       = 384;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<300, 580>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_1.ipt_20.tpb_448.ns_240.dcid_6.l2w_845 1.097180  0.990453  1.091667  1.452153
  static constexpr int nominal_4b_items              = 20;
  static constexpr int threads                       = 448;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<240, 845>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_14.tpb_320.ns_1428.dcid_7.l2w_830 1.380164  1.133333  1.367514  1.628793
  static constexpr int nominal_4b_items              = 14;
  static constexpr int threads                       = 320;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1428, 830>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_14.tpb_640.ns_1204.dcid_5.l2w_635 1.155209  1.000000  1.143742  1.380659
  static constexpr int nominal_4b_items              = 14;
  static constexpr int threads                       = 640;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1204, 635>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_8,
                    may_alias::no,
                    distinct_partitions::yes>
{
  // trp_0.ld_0.ipt_19.tpb_384.ns_1016.dcid_7.l2w_875 1.227540  1.181818  1.223936  1.261954
  static constexpr int nominal_4b_items              = 19;
  static constexpr int threads                       = 384;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1016, 875>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_24.tpb_256.ns_2024.dcid_5.l2w_835 1.146782  1.001841  1.149438  1.439904
  static constexpr int nominal_4b_items              = 24;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<2024, 835>;
};

// todo(gonidelis): Tuning regresses for large input size. Find better tuning.
// template <class Input>
// struct sm100_tuning<Input,
//                     flagged::yes,
//                     keep_rejects::yes,
//                     offset_size::_4,
//                     primitive::yes,
//                     input_size::_2,
//                     may_alias::no,
//                     distinct_partitions::no>
// {
//   // trp_0.ld_0.ipt_18.tpb_256.ns_1832.dcid_5.l2w_590 1.128674  0.984403  1.150806  1.355932
//   static constexpr int nominal_4b_items              = 18;
//   static constexpr int threads                       = 256;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1832, 590>;
// };

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_11.tpb_448.ns_476.dcid_7.l2w_665 1.173664  1.035556  1.186114  1.393153
  static constexpr int nominal_4b_items              = 11;
  static constexpr int threads                       = 448;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<476, 665>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_4,
                    primitive::yes,
                    input_size::_8,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_20.tpb_384.ns_1420.dcid_5.l2w_525 (39_new/2.db)  1.157326  1.110920  1.162458  1.259336
  static constexpr int nominal_4b_items              = 20;
  static constexpr int threads                       = 384;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1420, 525>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_1,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_12.tpb_256.ns_0.dcid_5.l2w_850 1.150864  1.005760  1.157687  1.395833
  static constexpr int nominal_4b_items              = 12;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<0, 850>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_2,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_12.tpb_256.ns_1552.dcid_7.l2w_730 1.374892  1.171831  1.360076  1.513390
  static constexpr int nominal_4b_items              = 12;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1552, 730>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_4,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_14.tpb_352.ns_1444.dcid_5.l2w_655 1.183452  1.000000  1.177224  1.402083
  static constexpr int nominal_4b_items              = 14;
  static constexpr int threads                       = 352;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1444, 655>;
};

template <class Input>
struct sm100_tuning<Input,
                    flagged::yes,
                    keep_rejects::yes,
                    offset_size::_8,
                    primitive::yes,
                    input_size::_8,
                    may_alias::no,
                    distinct_partitions::no>
{
  // trp_0.ld_0.ipt_11.tpb_512.ns_536.dcid_2.l2w_845 1.248969  1.184659  1.251631  1.360795
  static constexpr int nominal_4b_items              = 11;
  static constexpr int threads                       = 512;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backoff_constructor_t<536, 845>;
};

template <class InputT>
constexpr primitive is_primitive()
{
  return detail::is_primitive_v<InputT> ? primitive::yes : primitive::no;
}

template <class FlagT>
constexpr flagged is_flagged()
{
  return ::cuda::std::is_same_v<FlagT, NullType> ? flagged::no : flagged::yes;
}

template <bool KeepRejects>
constexpr keep_rejects are_rejects_kept()
{
  return KeepRejects ? keep_rejects::yes : keep_rejects::no;
}

template <class InputT>
constexpr input_size classify_input_size()
{
  return sizeof(InputT) == 1 ? input_size::_1
       : sizeof(InputT) == 2 ? input_size::_2
       : sizeof(InputT) == 4 ? input_size::_4
       : sizeof(InputT) == 8 ? input_size::_8
       : sizeof(InputT) == 16
         ? input_size::_16
         : input_size::unknown;
}

template <class OffsetT>
constexpr offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

template <bool Alias>
constexpr may_alias should_alias()
{
  return Alias ? may_alias::yes : may_alias::no;
}

template <bool DistinctPartitions>
constexpr distinct_partitions is_distinct_partitions()
{
  return DistinctPartitions ? distinct_partitions::yes : distinct_partitions::no;
}

template <class InputT, class FlagT, class OffsetT, bool DistinctPartitions, SelectImpl Impl>
struct policy_hub
{
  static constexpr bool may_alias    = Impl == SelectImpl::SelectPotentiallyInPlace;
  static constexpr bool keep_rejects = Impl == SelectImpl::Partition;

  template <CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 10;
    static constexpr int items_per_thread =
      ::cuda::std::clamp(nominal_4B_items_per_thread * 4 / int{sizeof(InputT)}, 1, nominal_4B_items_per_thread);
    using SelectIfPolicyT =
      AgentSelectIfPolicy<128,
                          items_per_thread,
                          BLOCK_LOAD_DIRECT,
                          LoadModifier,
                          BLOCK_SCAN_WARP_SCANS,
                          detail::fixed_delay_constructor_t<350, 450>>;
  };

  struct Policy500
      : DefaultPolicy<may_alias ? LOAD_CA : LOAD_LDG>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentSelectIfPolicy<Tuning::threads,
                           Tuning::items,
                           Tuning::load_algorithm,
                           LOAD_DEFAULT,
                           BLOCK_SCAN_WARP_SCANS,
                           typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy<LOAD_DEFAULT>::SelectIfPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    using SelectIfPolicyT =
      decltype(select_agent_policy<sm80_tuning<InputT,
                                               is_flagged<FlagT>(),
                                               are_rejects_kept<keep_rejects>(),
                                               offset_size::_4, // before SM100, we only tuned for int32
                                               is_primitive<InputT>(),
                                               classify_input_size<InputT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy<may_alias ? LOAD_CA : LOAD_LDG>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using SelectIfPolicyT =
      decltype(select_agent_policy<sm90_tuning<InputT,
                                               is_flagged<FlagT>(),
                                               are_rejects_kept<keep_rejects>(),
                                               offset_size::_4, // before SM100, we only tuned for int32
                                               is_primitive<InputT>(),
                                               classify_input_size<InputT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> AgentSelectIfPolicy<Tuning::threads,
                             Nominal4BItemsToItems<InputT>(Tuning::nominal_4b_items),
                             Tuning::load_algorithm,
                             Tuning::load_modifier,
                             BLOCK_SCAN_WARP_SCANS,
                             typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy100(long) -> typename Policy900::SelectIfPolicyT;

    using SelectIfPolicyT =
      decltype(select_agent_policy100<sm100_tuning<InputT,
                                                   is_flagged<FlagT>(),
                                                   are_rejects_kept<keep_rejects>(),
                                                   classify_offset_size<OffsetT>(),
                                                   is_primitive<InputT>(),
                                                   classify_input_size<InputT>(),
                                                   should_alias<may_alias>(),
                                                   is_distinct_partitions<DistinctPartitions>()>>(0));
  };

  using MaxPolicy = Policy1000;
};
} // namespace detail::select

CUB_NAMESPACE_END
