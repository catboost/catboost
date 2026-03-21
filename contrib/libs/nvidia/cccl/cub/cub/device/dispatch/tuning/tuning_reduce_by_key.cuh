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
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/cmath>
#include <cuda/std/__algorithm/clamp.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace reduce_by_key
{
enum class primitive_key
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
enum class key_size
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

template <class T>
constexpr primitive_key is_primitive_key()
{
  return detail::is_primitive<T>::value ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_accum is_primitive_accum()
{
  return detail::is_primitive<T>::value ? primitive_accum::yes : primitive_accum::no;
}

template <class ReductionOpT>
constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ReductionOpT>::value ? primitive_op::yes : primitive_op::no;
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

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm80_tuning;

// 8-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<975>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<840>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<760>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1070>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1175>;
};

// 16-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<620>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<640>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<905>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<810>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2, accum_size::_16>
{
  static constexpr int threads                       = 160;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1115>;
};

// 32-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1110>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1200>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1110>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1165>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4, accum_size::_16>
{
  static constexpr int threads                       = 160;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1100>;
};

// 64-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1175>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1075>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1040>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1080>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<430>;
};

// 128-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_1>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1105>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<755>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<535>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1035>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::no, key_size::_16, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1090>;
};

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm90_tuning;

// 8-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<720>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<865>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<735>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<580>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1100>;
};

// 16-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<985>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<276, 650>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<240, 765>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 19;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1190>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1175>;
};

// 32-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<404, 645>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1160>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1170>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1055>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1195>;
};

// 64-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1170>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<236, 1030>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<152, 560>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1030>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1125>;
};

// 128-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_1>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1080>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_2>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<320, 1005>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_4>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<232, 1100>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1195>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::no, key_size::_16, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1150>;
};

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm100_tuning;

// 8-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  // ipt_13.tpb_576.trp_0.ld_1.ns_2044.dcid_5.l2w_240 1.161888  0.848558  1.134941  1.299109
  static constexpr int items                         = 13;
  static constexpr int threads                       = 576;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<2044, 240>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  // ipt_10.tpb_224.trp_0.ld_0.ns_244.dcid_4.l2w_390 1.313932  1.260540  1.319588  1.427374
  static constexpr int items                         = 10;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_jitter_window_constructor_t<224, 390>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  // ipt_14.tpb_128.trp_0.ld_0.ns_248.dcid_2.l2w_285  1.118109  1.051534  1.134336  1.326788
  static constexpr int items                         = 14;
  static constexpr int threads                       = 128;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_constructor_t<248, 285>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  // ipt_19.tpb_128.trp_1.ld_0.ns_132.dcid_1.l2w_540 1.113820  1.002404  1.105014  1.202296
  static constexpr int items                         = 19;
  static constexpr int threads                       = 128;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<132, 540>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1,
// accum_size::_16>
// {
// static constexpr int threads                       = 128;
// static constexpr int items                         = 11;
// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
// using delay_constructor                            = detail::no_delay_constructor_t<1100>;
// };

// 16-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  // ipt_14.tpb_128.trp_1.ld_0.ns_164.dcid_2.l2w_290 1.239579  1.119705  1.239111  1.313112
  static constexpr int items                         = 14;
  static constexpr int threads                       = 128;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<164, 290>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  // ipt_14.tpb_256.trp_1.ld_0.ns_180.dcid_2.l2w_975 1.145635  1.012658  1.139956  1.251546
  static constexpr int items                         = 14;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<180, 975>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  // ipt_11.tpb_256.trp_0.ld_0.ns_224.dcid_2.l2w_550 1.066293  1.000109  1.073092  1.181818
  static constexpr int items                         = 11;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_constructor_t<224, 550>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  // ipt_10.tpb_160.trp_1.ld_0.ns_156.dcid_1.l2w_725 1.045007  1.002105  1.049690  1.141827
  static constexpr int items                         = 10;
  static constexpr int threads                       = 160;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<156, 725>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// I16, F32, I32 regresses, default it back.
template <class KeyT>
struct sm100_tuning<KeyT, float, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2,
// accum_size::_16>
// {
// static constexpr int threads                       = 128;
// static constexpr int items                         = 11;
// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
// using delay_constructor                            = detail::no_delay_constructor_t<1100>;
// };

// 32-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  // ipt_10.tpb_224.trp_0.ld_0.ns_324.dcid_2.l2w_285 1.157217  1.073724  1.166510  1.356940
  static constexpr int items                         = 10;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_constructor_t<324, 285>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  // ipt_11.tpb_256.trp_0.ld_0.ns_1984.dcid_5.l2w_115 1.214155  1.128842  1.214093  1.364476
  static constexpr int items                         = 11;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1984, 115>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  // ipt_14.tpb_224.trp_1.ld_0.ns_476.dcid_5.l2w_1005 1.187378  1.119705  1.185397  1.258420

  static constexpr int items                         = 14;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<476, 1005>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  // ipt_10.tpb_256.trp_1.ld_0.ns_1868.dcid_7.l2w_145 1.142915  1.020581  1.137459  1.237913
  static constexpr int items                         = 10;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_constructor_t<1868, 145>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4,
// accum_size::_16>
// {
// static constexpr int threads                       = 128;
// static constexpr int items                         = 11;
// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
// using delay_constructor                            = detail::no_delay_constructor_t<1100>;
// };

// 64-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_1940.dcid_5.l2w_460 1.157294  1.075650  1.153566  1.250729
  static constexpr int items                         = 9;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1940, 460>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  // ipt_11.tpb_224.trp_1.ld_1.ns_392.dcid_2.l2w_550 1.104034  1.007212  1.099543  1.220401
  static constexpr int items                         = 11;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<392, 550>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_244.dcid_2.l2w_475 1.130098  1.000000  1.130661  1.215722
  static constexpr int items                         = 9;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<244, 475>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_196.dcid_2.l2w_340 1.272056  1.142857  1.262499  1.352941
  static constexpr int items                         = 9;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<196, 340>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8,
// accum_size::_16>
// {
//   static constexpr int threads                       = 128;
//   static constexpr int items                         = 11;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = detail::no_delay_constructor_t<1125>;
// };

// todo(gonidelis): Add tunings for 128-bit key.
// 128-bit key
// template <class KeyT, class AccumT>
// struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_16,
// accum_size::_1>
// {
//   static constexpr int threads                       = 128;
//   static constexpr int items                         = 11;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = detail::no_delay_constructor_t<1125>;
// };

template <class ReductionOpT, class AccumT, class KeyT>
struct policy_hub
{
  static constexpr int max_input_bytes      = static_cast<int>((::cuda::std::max) (sizeof(KeyT), sizeof(AccumT)));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(AccumT);

  template <CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 6;
    static constexpr int items_per_thread =
      (max_input_bytes <= 8)
        ? 6
        : ::cuda::std::clamp(
            ::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes), 1, nominal_4B_items_per_thread);

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<128,
                             items_per_thread,
                             BLOCK_LOAD_DIRECT,
                             LoadModifier,
                             BLOCK_SCAN_WARP_SCANS,
                             default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };

  struct Policy500
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
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
    using ReduceByKeyPolicyT =
      decltype(select_agent_policy<sm80_tuning<KeyT, AccumT, is_primitive_op<ReductionOpT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ReduceByKeyPolicyT =
      decltype(select_agent_policy<sm90_tuning<KeyT, AccumT, is_primitive_op<ReductionOpT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick the default
    template <typename Tuning>
    static auto select_agent_policy(int)
      -> AgentReduceByKeyPolicy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                BLOCK_SCAN_WARP_SCANS,
                                typename Tuning::delay_constructor>;

    template <typename Tuning>
    static auto select_agent_policy(long) -> typename Policy900::ReduceByKeyPolicyT;

    using ReduceByKeyPolicyT =
      decltype(select_agent_policy<sm100_tuning<KeyT, AccumT, is_primitive_op<ReductionOpT>()>>(0));
  };
  using MaxPolicy = Policy1000;
};
} // namespace reduce_by_key
} // namespace detail

CUB_NAMESPACE_END
