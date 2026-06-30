/*
 * Copyright (c) 2017, Mellanox Technologies inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef VERBS_API_H
#define VERBS_API_H

#if UINTPTR_MAX == UINT32_MAX
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define RDMA_UAPI_PTR(_type, _name)                                            \
	union {                                                                \
		struct {                                                       \
			_type _name;                                           \
			__u32 _name##_reserved;                                \
		};                                                             \
		__aligned_u64 _name##_data_u64;                                \
	}
#else
#define RDMA_UAPI_PTR(_type, _name)                                            \
	union {                                                                \
		struct {                                                       \
			__u32 _name##_reserved;                                \
			_type _name;                                           \
		};                                                             \
		__aligned_u64 _name##_data_u64;                                \
	}
#endif
#elif UINTPTR_MAX == UINT64_MAX
#define RDMA_UAPI_PTR(_type, _name)                                            \
	union {                                                                \
		_type _name;                                                   \
		__aligned_u64 _name##_data_u64;                                \
	}
#else
#error "Pointer size not supported"
#endif

#include <infiniband/ib_user_ioctl_verbs.h>

#define ibv_flow_action_esp_keymat			ib_uverbs_flow_action_esp_keymat
#define IBV_FLOW_ACTION_ESP_KEYMAT_AES_GCM              IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM
#define ibv_flow_action_esp_keymat_aes_gcm_iv_algo      ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo
#define IBV_FLOW_ACTION_IV_ALGO_SEQ                     IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ
#define ibv_flow_action_esp_keymat_aes_gcm              ib_uverbs_flow_action_esp_keymat_aes_gcm
#define ibv_flow_action_esp_replay                      ib_uverbs_flow_action_esp_replay
#define IBV_FLOW_ACTION_ESP_REPLAY_NONE                 IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE
#define IBV_FLOW_ACTION_ESP_REPLAY_BMP                  IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP
#define ibv_flow_action_esp_replay_bmp                  ib_uverbs_flow_action_esp_replay_bmp
#define ibv_flow_action_esp_flags                       ib_uverbs_flow_action_esp_flags
#define IBV_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO         IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO
#define IBV_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD          IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD
#define IBV_FLOW_ACTION_ESP_FLAGS_TUNNEL                IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL
#define IBV_FLOW_ACTION_ESP_FLAGS_TRANSPORT             IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT
#define IBV_FLOW_ACTION_ESP_FLAGS_DECRYPT               IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT
#define IBV_FLOW_ACTION_ESP_FLAGS_ENCRYPT               IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT
#define IBV_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW        IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW
#define ibv_flow_action_esp_encap                       ib_uverbs_flow_action_esp_encap
#define ibv_flow_action_esp                             ib_uverbs_flow_action_esp

#define ibv_advise_mr_advice                            ib_uverbs_advise_mr_advice
#define IBV_ADVISE_MR_ADVICE_PREFETCH                   IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH
#define IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE             IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE

#define IBV_ADVISE_MR_FLAG_FLUSH                        IB_UVERBS_ADVISE_MR_FLAG_FLUSH

#define IBV_QPF_GRH_REQUIRED				IB_UVERBS_QPF_GRH_REQUIRED

#define IBV_ACCESS_OPTIONAL_RANGE			IB_UVERBS_ACCESS_OPTIONAL_RANGE
#define IBV_ACCESS_OPTIONAL_FIRST			IB_UVERBS_ACCESS_OPTIONAL_FIRST
#endif

