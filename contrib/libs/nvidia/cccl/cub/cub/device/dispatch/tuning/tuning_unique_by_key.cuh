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

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::unique_by_key
{

enum class primitive_key
{
  no,
  yes
};
enum class primitive_val
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
enum class val_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class T>
constexpr primitive_key is_primitive_key()
{
  return is_primitive<T>::value ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_val is_primitive_val()
{
  return is_primitive<T>::value ? primitive_val::yes : primitive_val::no;
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

template <class ValueT>
constexpr val_size classify_val_size()
{
  return sizeof(ValueT) == 1 ? val_size::_1
       : sizeof(ValueT) == 2 ? val_size::_2
       : sizeof(ValueT) == 4 ? val_size::_4
       : sizeof(ValueT) == 8 ? val_size::_8
       : sizeof(ValueT) == 16
         ? val_size::_16
         : val_size::unknown;
}

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm80_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<835>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<765>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1155>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1065>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<248, 1200>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1020>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<328, 1080>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<535>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1185>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1115>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<320, 1115>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<24, 555>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<324, 1105>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<740, 1105>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<764, 1155>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<992, 1135>;
};

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm90_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<550>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<725>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1130>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1100>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<344, 1165>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<640>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<404, 710>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<525>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1200>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_2, val_size::_16>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<424, 1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<348, 580>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1045>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_4, val_size::_16>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1025>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<964, 1125>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads                       = 640;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1070>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1190>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1155>;
};

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm100_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  // ipt_12.tpb_512.trp_0.ld_0.ns_948.dcid_5.l2w_955 1.121279  1.000000  1.114566  1.43765
  static constexpr int threads                       = 512;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<948, 955>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_1228.dcid_7.l2w_320 1.151229  1.007229  1.151131  1.443520
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1228, 320>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_2016.dcid_7.l2w_620 1.165300  1.095238  1.164478  1.266667
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<2016, 620>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_1728.dcid_5.l2w_980 1.118716  0.997167  1.116537  1.400000
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1728, 980>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
// {
//   static constexpr int threads                       = 288;
//   static constexpr int items                         = 7;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = fixed_delay_constructor_t<344, 1165>;
// };
#endif

// 16-bit key
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_508.dcid_7.l2w_1020 1.171886  0.906530  1.157128  1.457933
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<508, 1020>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  // ipt_12.tpb_384.trp_0.ld_0.ns_928.dcid_7.l2w_605 1.166564  0.997579  1.154805  1.406709
  static constexpr int threads                       = 384;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<928, 605>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  // ipt_11.tpb_384.trp_0.ld_1.ns_1620.dcid_7.l2w_810 1.144483  1.011085  1.152798  1.393750
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = exponential_backon_constructor_t<1620, 810>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_1984.dcid_5.l2w_935 1.605554  1.177083  1.564488  1.946224
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1984, 935>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_2, val_size::_16>
// {
//   static constexpr int threads                       = 224;
//   static constexpr int items                         = 9;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = fixed_delay_constructor_t<424, 1055>;
// };
#endif

// 32-bit key
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_1136.dcid_7.l2w_605 1.148057  0.848558  1.133064  1.451074
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1136, 605>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  // ipt_11.tpb_384.trp_0.ld_0.ns_656.dcid_7.l2w_825 1.216312  1.090485  1.211800  1.535714
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<656, 825>;
};

// todo(gonidelis): tuning performs very well for medium input size, regresses for large input sizes.
// find better tuning.
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
    : sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  // // ipt_14.tpb_512.trp_0.ld_0.ns_408.dcid_7.l2w_960 1.136333  0.995833  1.144371  1.448687
  // static constexpr int threads                       = 512;
  // static constexpr int items                         = 14;
  // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  // static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  // using delay_constructor                            = exponential_backon_constructor_t<408, 960>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_1012.dcid_5.l2w_800 1.164713  1.014819  1.174307  1.526042
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1012, 800>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_4, val_size::_16>
// {
//   static constexpr int threads                       = 384;
//   static constexpr int items                         = 7;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = no_delay_constructor_t<1025>;
// };
#endif

// 64-bit key

// todo(gonidelis): tuning regresses for large input sizes. find better tuning.
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
    : sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  // // ipt_9.tpb_384.trp_0.ld_0.ns_1064.dcid_7.l2w_600 1.085831  0.972452  1.080521  1.397089
  // static constexpr int threads                       = 384;
  // static constexpr int items                         = 9;
  // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  // static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  // using delay_constructor                            = exponential_backon_constructor_t<1064, 600>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_864.dcid_5.l2w_1130 1.124095  0.985748  1.120262  1.391304
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<864, 1130>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_772.dcid_5.l2w_665 1.152243  1.019816  1.166636  1.517526
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<772, 665>;
};

// todo(gonidelis): tuning regresses for large input sizes. find better tuning.
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
    : sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  // // ipt_7.tpb_576.trp_0.ld_0.ns_1132.dcid_5.l2w_1115 1.120721  0.977642  1.131594  1.449407
  // static constexpr int threads                       = 576;
  // static constexpr int items                         = 7;
  // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  // static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  // using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1132, 1115>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
// {
//   static constexpr int threads                       = 256;
//   static constexpr int items                         = 9;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = no_delay_constructor_t<1155>;
// };
#endif

template <typename PolicyT, typename = void>
struct UniqueByKeyPolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION UniqueByKeyPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct UniqueByKeyPolicyWrapper<StaticPolicyT,
                                ::cuda::std::void_t<decltype(StaticPolicyT::UniqueByKeyPolicyT::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION UniqueByKeyPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr auto UniqueByKey()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::UniqueByKeyPolicyT());
  }
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION UniqueByKeyPolicyWrapper<PolicyT> MakeUniqueByKeyPolicyWrapper(PolicyT policy)
{
  return UniqueByKeyPolicyWrapper<PolicyT>{policy};
}

template <class KeyT, class ValueT>
struct policy_hub
{
  template <int Nominal4bItemsPerThread, int Threads>
  struct DefaultPolicy
  {
    static constexpr int items_per_thread = Nominal4BItemsToItems<KeyT>(Nominal4bItemsPerThread);
    using UniqueByKeyPolicyT =
      AgentUniqueByKeyPolicy<Threads,
                             items_per_thread,
                             BLOCK_LOAD_WARP_TRANSPOSE,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             detail::default_delay_constructor_t<int>>;
  };

  struct Policy500
      : DefaultPolicy<9, 128>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentUniqueByKeyPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              BLOCK_SCAN_WARP_SCANS,
                              typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy<11, 64>::UniqueByKeyPolicyT;

  struct Policy520
      : DefaultPolicy<11, 64>
      , ChainedPolicy<520, Policy520, Policy500>
  {};

  struct Policy800 : ChainedPolicy<800, Policy800, Policy520>
  {
    using UniqueByKeyPolicyT = decltype(select_agent_policy<sm80_tuning<KeyT, ValueT>>(0));
  };

  struct Policy860
      : DefaultPolicy<11, 64>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using UniqueByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<KeyT, ValueT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> AgentUniqueByKeyPolicy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                BLOCK_SCAN_WARP_SCANS,
                                typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy100(long) -> typename Policy900::UniqueByKeyPolicyT;

    using UniqueByKeyPolicyT = decltype(select_agent_policy100<sm100_tuning<KeyT, ValueT>>(0));
  };

  using MaxPolicy = Policy1000;
};

} // namespace detail::unique_by_key

CUB_NAMESPACE_END
