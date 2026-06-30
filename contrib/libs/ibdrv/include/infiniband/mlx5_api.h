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

#ifndef MLX5_API_H
#define MLX5_API_H

#include <infiniband/mlx5_user_ioctl_verbs.h>

#define mlx5dv_flow_action_flags			mlx5_ib_uapi_flow_action_flags
#define MLX5DV_FLOW_ACTION_FLAGS_REQUIRE_METADATA	MLX5_IB_UAPI_FLOW_ACTION_FLAGS_REQUIRE_METADATA
#define mlx5dv_flow_table_type				mlx5_ib_uapi_flow_table_type
#define MLX5DV_FLOW_TABLE_TYPE_NIC_RX			MLX5_IB_UAPI_FLOW_TABLE_TYPE_NIC_RX
#define MLX5DV_FLOW_TABLE_TYPE_NIC_TX			MLX5_IB_UAPI_FLOW_TABLE_TYPE_NIC_TX
#define MLX5DV_FLOW_TABLE_TYPE_FDB			MLX5_IB_UAPI_FLOW_TABLE_TYPE_FDB
#define MLX5DV_FLOW_TABLE_TYPE_RDMA_RX			MLX5_IB_UAPI_FLOW_TABLE_TYPE_RDMA_RX
#define MLX5DV_FLOW_TABLE_TYPE_RDMA_TX			MLX5_IB_UAPI_FLOW_TABLE_TYPE_RDMA_TX
#define mlx5dv_flow_action_packet_reformat_type		mlx5_ib_uapi_flow_action_packet_reformat_type
#define MLX5DV_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TUNNEL_TO_L2  MLX5_IB_UAPI_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TUNNEL_TO_L2
#define MLX5DV_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TO_L2_TUNNEL MLX5_IB_UAPI_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TO_L2_TUNNEL
#define MLX5DV_FLOW_ACTION_PACKET_REFORMAT_TYPE_L3_TUNNEL_TO_L2 MLX5_IB_UAPI_FLOW_ACTION_PACKET_REFORMAT_TYPE_L3_TUNNEL_TO_L2
#define MLX5DV_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TO_L3_TUNNEL MLX5_IB_UAPI_FLOW_ACTION_PACKET_REFORMAT_TYPE_L2_TO_L3_TUNNEL
#define mlx5dv_devx_async_cmd_hdr mlx5_ib_uapi_devx_async_cmd_hdr
#define mlx5dv_devx_async_event_hdr mlx5_ib_uapi_devx_async_event_hdr
#define mlx5dv_alloc_dm_type	mlx5_ib_uapi_dm_type
#define MLX5DV_DM_TYPE_MEMIC	MLX5_IB_UAPI_DM_TYPE_MEMIC
#define MLX5DV_DM_TYPE_STEERING_SW_ICM		MLX5_IB_UAPI_DM_TYPE_STEERING_SW_ICM
#define MLX5DV_DM_TYPE_HEADER_MODIFY_SW_ICM	MLX5_IB_UAPI_DM_TYPE_HEADER_MODIFY_SW_ICM
#define MLX5DV_DM_TYPE_HEADER_MODIFY_PATTERN_SW_ICM MLX5_IB_UAPI_DM_TYPE_HEADER_MODIFY_PATTERN_SW_ICM
#define mlx5dv_devx_create_event_channel_flags mlx5_ib_uapi_devx_create_event_channel_flags
#define MLX5DV_DEVX_CREATE_EVENT_CHANNEL_FLAGS_OMIT_EV_DATA MLX5_IB_UAPI_DEVX_CR_EV_CH_FLAGS_OMIT_DATA
#define MLX5DV_DEVX_PORT_VPORT MLX5_IB_UAPI_QUERY_PORT_VPORT
#define MLX5DV_DEVX_PORT_VPORT_VHCA_ID MLX5_IB_UAPI_QUERY_PORT_VPORT_VHCA_ID
#define MLX5DV_DEVX_PORT_ESW_OWNER_VHCA_ID MLX5_IB_UAPI_QUERY_PORT_ESW_OWNER_VHCA_ID
#define MLX5DV_DEVX_PORT_VPORT_ICM_RX MLX5_IB_UAPI_QUERY_PORT_VPORT_ICM_RX
#define MLX5DV_DEVX_PORT_VPORT_ICM_TX MLX5_IB_UAPI_QUERY_PORT_VPORT_ICM_TX
#define MLX5DV_DEVX_PORT_MATCH_REG_C_0 MLX5_IB_UAPI_QUERY_PORT_MATCH_REG_C_0
#define mlx5dv_devx_reg_32 mlx5_ib_uapi_devx_reg_32
#define MLX5DV_PP_ALLOC_FLAGS_DEDICATED_INDEX MLX5_IB_UAPI_PP_ALLOC_FLAGS_DEDICATED_INDEX
#define MLX5DV_UAR_ALLOC_TYPE_BF MLX5_IB_UAPI_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_NC MLX5_IB_UAPI_UAR_ALLOC_TYPE_NC

#endif
