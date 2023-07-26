/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
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

#ifndef INFINIBAND_VERBS_H
#define INFINIBAND_VERBS_H

#include <stdint.h>
#include <pthread.h>
#include <stddef.h>
#include <errno.h>
#include <string.h>
#include <linux/types.h>
#include <stdint.h>
#include <sys/types.h>
#include <infiniband/verbs_api.h>

#ifdef __cplusplus
#include <limits>
#endif

#if __GNUC__ >= 3
#  define __attribute_const __attribute__((const))
#else
#  define __attribute_const
#endif

#ifdef __cplusplus
extern "C" {
#endif


union ibv_gid {
	uint8_t			raw[16];
	struct {
		__be64	subnet_prefix;
		__be64	interface_id;
	} global;
};

enum ibv_gid_type {
	IBV_GID_TYPE_IB,
	IBV_GID_TYPE_ROCE_V1,
	IBV_GID_TYPE_ROCE_V2,
};

struct ibv_gid_entry {
	union ibv_gid gid;
	uint32_t gid_index;
	uint32_t port_num;
	uint32_t gid_type; /* enum ibv_gid_type */
	uint32_t ndev_ifindex;
};

#define vext_field_avail(type, fld, sz) (offsetof(type, fld) < (sz))

#ifdef __cplusplus
#define __VERBS_ABI_IS_EXTENDED ((void *)std::numeric_limits<uintptr_t>::max())
#else
#define __VERBS_ABI_IS_EXTENDED ((void *)UINTPTR_MAX)
#endif

enum ibv_node_type {
	IBV_NODE_UNKNOWN	= -1,
	IBV_NODE_CA 		= 1,
	IBV_NODE_SWITCH,
	IBV_NODE_ROUTER,
	IBV_NODE_RNIC,
	IBV_NODE_USNIC,
	IBV_NODE_USNIC_UDP,
	IBV_NODE_UNSPECIFIED,
};

enum ibv_transport_type {
	IBV_TRANSPORT_UNKNOWN	= -1,
	IBV_TRANSPORT_IB	= 0,
	IBV_TRANSPORT_IWARP,
	IBV_TRANSPORT_USNIC,
	IBV_TRANSPORT_USNIC_UDP,
	IBV_TRANSPORT_UNSPECIFIED,
};

enum ibv_device_cap_flags {
	IBV_DEVICE_RESIZE_MAX_WR	= 1,
	IBV_DEVICE_BAD_PKEY_CNTR	= 1 <<  1,
	IBV_DEVICE_BAD_QKEY_CNTR	= 1 <<  2,
	IBV_DEVICE_RAW_MULTI		= 1 <<  3,
	IBV_DEVICE_AUTO_PATH_MIG	= 1 <<  4,
	IBV_DEVICE_CHANGE_PHY_PORT	= 1 <<  5,
	IBV_DEVICE_UD_AV_PORT_ENFORCE	= 1 <<  6,
	IBV_DEVICE_CURR_QP_STATE_MOD	= 1 <<  7,
	IBV_DEVICE_SHUTDOWN_PORT	= 1 <<  8,
	IBV_DEVICE_INIT_TYPE		= 1 <<  9,
	IBV_DEVICE_PORT_ACTIVE_EVENT	= 1 << 10,
	IBV_DEVICE_SYS_IMAGE_GUID	= 1 << 11,
	IBV_DEVICE_RC_RNR_NAK_GEN	= 1 << 12,
	IBV_DEVICE_SRQ_RESIZE		= 1 << 13,
	IBV_DEVICE_N_NOTIFY_CQ		= 1 << 14,
	IBV_DEVICE_MEM_WINDOW           = 1 << 17,
	IBV_DEVICE_UD_IP_CSUM		= 1 << 18,
	IBV_DEVICE_XRC			= 1 << 20,
	IBV_DEVICE_MEM_MGT_EXTENSIONS	= 1 << 21,
	IBV_DEVICE_MEM_WINDOW_TYPE_2A	= 1 << 23,
	IBV_DEVICE_MEM_WINDOW_TYPE_2B	= 1 << 24,
	IBV_DEVICE_RC_IP_CSUM		= 1 << 25,
	IBV_DEVICE_RAW_IP_CSUM		= 1 << 26,
	IBV_DEVICE_MANAGED_FLOW_STEERING = 1 << 29
};

/*
 * Can't extended above ibv_device_cap_flags enum as in some systems/compilers
 * enum range is limited to 4 bytes.
 */
#define IBV_DEVICE_RAW_SCATTER_FCS (1ULL << 34)
#define IBV_DEVICE_PCI_WRITE_END_PADDING (1ULL << 36)

enum ibv_atomic_cap {
	IBV_ATOMIC_NONE,
	IBV_ATOMIC_HCA,
	IBV_ATOMIC_GLOB
};

struct ibv_alloc_dm_attr {
	size_t length;
	uint32_t log_align_req;
	uint32_t comp_mask;
};

struct ibv_dm {
	struct ibv_context *context;
	int (*memcpy_to_dm)(struct ibv_dm *dm, uint64_t dm_offset,
			    const void *host_addr, size_t length);
	int (*memcpy_from_dm)(void *host_addr, struct ibv_dm *dm,
			      uint64_t dm_offset, size_t length);
	uint32_t comp_mask;
};

struct ibv_device_attr {
	char			fw_ver[64];
	__be64			node_guid;
	__be64			sys_image_guid;
	uint64_t		max_mr_size;
	uint64_t		page_size_cap;
	uint32_t		vendor_id;
	uint32_t		vendor_part_id;
	uint32_t		hw_ver;
	int			max_qp;
	int			max_qp_wr;
	unsigned int		device_cap_flags;
	int			max_sge;
	int			max_sge_rd;
	int			max_cq;
	int			max_cqe;
	int			max_mr;
	int			max_pd;
	int			max_qp_rd_atom;
	int			max_ee_rd_atom;
	int			max_res_rd_atom;
	int			max_qp_init_rd_atom;
	int			max_ee_init_rd_atom;
	enum ibv_atomic_cap	atomic_cap;
	int			max_ee;
	int			max_rdd;
	int			max_mw;
	int			max_raw_ipv6_qp;
	int			max_raw_ethy_qp;
	int			max_mcast_grp;
	int			max_mcast_qp_attach;
	int			max_total_mcast_qp_attach;
	int			max_ah;
	int			max_fmr;
	int			max_map_per_fmr;
	int			max_srq;
	int			max_srq_wr;
	int			max_srq_sge;
	uint16_t		max_pkeys;
	uint8_t			local_ca_ack_delay;
	uint8_t			phys_port_cnt;
};

/* An extensible input struct for possible future extensions of the
 * ibv_query_device_ex verb. */
struct ibv_query_device_ex_input {
	uint32_t		comp_mask;
};

enum ibv_odp_transport_cap_bits {
	IBV_ODP_SUPPORT_SEND     = 1 << 0,
	IBV_ODP_SUPPORT_RECV     = 1 << 1,
	IBV_ODP_SUPPORT_WRITE    = 1 << 2,
	IBV_ODP_SUPPORT_READ     = 1 << 3,
	IBV_ODP_SUPPORT_ATOMIC   = 1 << 4,
	IBV_ODP_SUPPORT_SRQ_RECV = 1 << 5,
};

struct ibv_odp_caps {
	uint64_t general_caps;
	struct {
		uint32_t rc_odp_caps;
		uint32_t uc_odp_caps;
		uint32_t ud_odp_caps;
	} per_transport_caps;
};

enum ibv_odp_general_caps {
	IBV_ODP_SUPPORT = 1 << 0,
	IBV_ODP_SUPPORT_IMPLICIT = 1 << 1,
};

struct ibv_tso_caps {
	uint32_t max_tso;
	uint32_t supported_qpts;
};

/* RX Hash function flags */
enum ibv_rx_hash_function_flags {
	IBV_RX_HASH_FUNC_TOEPLITZ	= 1 << 0,
};

/*
 * RX Hash fields enable to set which incoming packet's field should
 * participates in RX Hash. Each flag represent certain packet's field,
 * when the flag is set the field that is represented by the flag will
 * participate in RX Hash calculation.
 * Note: *IPV4 and *IPV6 flags can't be enabled together on the same QP
 * and *TCP and *UDP flags can't be enabled together on the same QP.
*/
enum ibv_rx_hash_fields {
	IBV_RX_HASH_SRC_IPV4	= 1 << 0,
	IBV_RX_HASH_DST_IPV4	= 1 << 1,
	IBV_RX_HASH_SRC_IPV6	= 1 << 2,
	IBV_RX_HASH_DST_IPV6	= 1 << 3,
	IBV_RX_HASH_SRC_PORT_TCP	= 1 << 4,
	IBV_RX_HASH_DST_PORT_TCP	= 1 << 5,
	IBV_RX_HASH_SRC_PORT_UDP	= 1 << 6,
	IBV_RX_HASH_DST_PORT_UDP	= 1 << 7,
	IBV_RX_HASH_IPSEC_SPI		= 1 << 8,
	IBV_RX_HASH_INNER		= (1UL << 31),
};

struct ibv_rss_caps {
	uint32_t supported_qpts;
	uint32_t max_rwq_indirection_tables;
	uint32_t max_rwq_indirection_table_size;
	uint64_t rx_hash_fields_mask; /* enum ibv_rx_hash_fields */
	uint8_t  rx_hash_function; /* enum ibv_rx_hash_function_flags */
};

struct ibv_packet_pacing_caps {
	uint32_t qp_rate_limit_min;
	uint32_t qp_rate_limit_max; /* In kbps */
	uint32_t supported_qpts;
};

enum ibv_raw_packet_caps {
	IBV_RAW_PACKET_CAP_CVLAN_STRIPPING	= 1 << 0,
	IBV_RAW_PACKET_CAP_SCATTER_FCS		= 1 << 1,
	IBV_RAW_PACKET_CAP_IP_CSUM		= 1 << 2,
	IBV_RAW_PACKET_CAP_DELAY_DROP		= 1 << 3,
};

enum ibv_tm_cap_flags {
	IBV_TM_CAP_RC		    = 1 << 0,
};

struct ibv_tm_caps {
	/* Max size of rendezvous request header */
	uint32_t max_rndv_hdr_size;
	/* Max number of tagged buffers in a TM-SRQ matching list */
	uint32_t max_num_tags;
	/* From enum ibv_tm_cap_flags */
	uint32_t flags;
	/* Max number of outstanding list operations */
	uint32_t max_ops;
	/* Max number of SGEs in a tagged buffer */
	uint32_t max_sge;
};

struct ibv_cq_moderation_caps {
	uint16_t max_cq_count;
	uint16_t max_cq_period; /* in micro seconds */
};

enum ibv_pci_atomic_op_size {
	IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP = 1 << 0,
	IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP = 1 << 1,
	IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP = 1 << 2,
};

/*
 * Bitmask for supported operation sizes
 * Use enum ibv_pci_atomic_op_size
 */
struct ibv_pci_atomic_caps {
	uint16_t fetch_add;
	uint16_t swap;
	uint16_t compare_swap;
};

struct ibv_device_attr_ex {
	struct ibv_device_attr	orig_attr;
	uint32_t		comp_mask;
	struct ibv_odp_caps	odp_caps;
	uint64_t		completion_timestamp_mask;
	uint64_t		hca_core_clock;
	uint64_t		device_cap_flags_ex;
	struct ibv_tso_caps	tso_caps;
	struct ibv_rss_caps     rss_caps;
	uint32_t		max_wq_type_rq;
	struct ibv_packet_pacing_caps packet_pacing_caps;
	uint32_t		raw_packet_caps; /* Use ibv_raw_packet_caps */
	struct ibv_tm_caps	tm_caps;
	struct ibv_cq_moderation_caps  cq_mod_caps;
	uint64_t max_dm_size;
	struct ibv_pci_atomic_caps pci_atomic_caps;
	uint32_t xrc_odp_caps;
};

enum ibv_mtu {
	IBV_MTU_256  = 1,
	IBV_MTU_512  = 2,
	IBV_MTU_1024 = 3,
	IBV_MTU_2048 = 4,
	IBV_MTU_4096 = 5
};

enum ibv_port_state {
	IBV_PORT_NOP		= 0,
	IBV_PORT_DOWN		= 1,
	IBV_PORT_INIT		= 2,
	IBV_PORT_ARMED		= 3,
	IBV_PORT_ACTIVE		= 4,
	IBV_PORT_ACTIVE_DEFER	= 5
};

enum {
	IBV_LINK_LAYER_UNSPECIFIED,
	IBV_LINK_LAYER_INFINIBAND,
	IBV_LINK_LAYER_ETHERNET,
};

enum ibv_port_cap_flags {
	IBV_PORT_SM				= 1 <<  1,
	IBV_PORT_NOTICE_SUP			= 1 <<  2,
	IBV_PORT_TRAP_SUP			= 1 <<  3,
	IBV_PORT_OPT_IPD_SUP			= 1 <<  4,
	IBV_PORT_AUTO_MIGR_SUP			= 1 <<  5,
	IBV_PORT_SL_MAP_SUP			= 1 <<  6,
	IBV_PORT_MKEY_NVRAM			= 1 <<  7,
	IBV_PORT_PKEY_NVRAM			= 1 <<  8,
	IBV_PORT_LED_INFO_SUP			= 1 <<  9,
	IBV_PORT_SYS_IMAGE_GUID_SUP		= 1 << 11,
	IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP	= 1 << 12,
	IBV_PORT_EXTENDED_SPEEDS_SUP		= 1 << 14,
	IBV_PORT_CAP_MASK2_SUP			= 1 << 15,
	IBV_PORT_CM_SUP				= 1 << 16,
	IBV_PORT_SNMP_TUNNEL_SUP		= 1 << 17,
	IBV_PORT_REINIT_SUP			= 1 << 18,
	IBV_PORT_DEVICE_MGMT_SUP		= 1 << 19,
	IBV_PORT_VENDOR_CLASS_SUP		= 1 << 20,
	IBV_PORT_DR_NOTICE_SUP			= 1 << 21,
	IBV_PORT_CAP_MASK_NOTICE_SUP		= 1 << 22,
	IBV_PORT_BOOT_MGMT_SUP			= 1 << 23,
	IBV_PORT_LINK_LATENCY_SUP		= 1 << 24,
	IBV_PORT_CLIENT_REG_SUP			= 1 << 25,
	IBV_PORT_IP_BASED_GIDS			= 1 << 26
};

enum ibv_port_cap_flags2 {
	IBV_PORT_SET_NODE_DESC_SUP		= 1 << 0,
	IBV_PORT_INFO_EXT_SUP			= 1 << 1,
	IBV_PORT_VIRT_SUP			= 1 << 2,
	IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP	= 1 << 3,
	IBV_PORT_LINK_WIDTH_2X_SUP		= 1 << 4,
	IBV_PORT_LINK_SPEED_HDR_SUP		= 1 << 5,
};

struct ibv_port_attr {
	enum ibv_port_state	state;
	enum ibv_mtu		max_mtu;
	enum ibv_mtu		active_mtu;
	int			gid_tbl_len;
	uint32_t		port_cap_flags;
	uint32_t		max_msg_sz;
	uint32_t		bad_pkey_cntr;
	uint32_t		qkey_viol_cntr;
	uint16_t		pkey_tbl_len;
	uint16_t		lid;
	uint16_t		sm_lid;
	uint8_t			lmc;
	uint8_t			max_vl_num;
	uint8_t			sm_sl;
	uint8_t			subnet_timeout;
	uint8_t			init_type_reply;
	uint8_t			active_width;
	uint8_t			active_speed;
	uint8_t			phys_state;
	uint8_t			link_layer;
	uint8_t			flags;
	uint16_t		port_cap_flags2;
};

enum ibv_event_type {
	IBV_EVENT_CQ_ERR,
	IBV_EVENT_QP_FATAL,
	IBV_EVENT_QP_REQ_ERR,
	IBV_EVENT_QP_ACCESS_ERR,
	IBV_EVENT_COMM_EST,
	IBV_EVENT_SQ_DRAINED,
	IBV_EVENT_PATH_MIG,
	IBV_EVENT_PATH_MIG_ERR,
	IBV_EVENT_DEVICE_FATAL,
	IBV_EVENT_PORT_ACTIVE,
	IBV_EVENT_PORT_ERR,
	IBV_EVENT_LID_CHANGE,
	IBV_EVENT_PKEY_CHANGE,
	IBV_EVENT_SM_CHANGE,
	IBV_EVENT_SRQ_ERR,
	IBV_EVENT_SRQ_LIMIT_REACHED,
	IBV_EVENT_QP_LAST_WQE_REACHED,
	IBV_EVENT_CLIENT_REREGISTER,
	IBV_EVENT_GID_CHANGE,
	IBV_EVENT_WQ_FATAL,
};

struct ibv_async_event {
	union {
		struct ibv_cq  *cq;
		struct ibv_qp  *qp;
		struct ibv_srq *srq;
		struct ibv_wq  *wq;
		int		port_num;
	} element;
	enum ibv_event_type	event_type;
};

enum ibv_wc_status {
	IBV_WC_SUCCESS,
	IBV_WC_LOC_LEN_ERR,
	IBV_WC_LOC_QP_OP_ERR,
	IBV_WC_LOC_EEC_OP_ERR,
	IBV_WC_LOC_PROT_ERR,
	IBV_WC_WR_FLUSH_ERR,
	IBV_WC_MW_BIND_ERR,
	IBV_WC_BAD_RESP_ERR,
	IBV_WC_LOC_ACCESS_ERR,
	IBV_WC_REM_INV_REQ_ERR,
	IBV_WC_REM_ACCESS_ERR,
	IBV_WC_REM_OP_ERR,
	IBV_WC_RETRY_EXC_ERR,
	IBV_WC_RNR_RETRY_EXC_ERR,
	IBV_WC_LOC_RDD_VIOL_ERR,
	IBV_WC_REM_INV_RD_REQ_ERR,
	IBV_WC_REM_ABORT_ERR,
	IBV_WC_INV_EECN_ERR,
	IBV_WC_INV_EEC_STATE_ERR,
	IBV_WC_FATAL_ERR,
	IBV_WC_RESP_TIMEOUT_ERR,
	IBV_WC_GENERAL_ERR,
	IBV_WC_TM_ERR,
	IBV_WC_TM_RNDV_INCOMPLETE,
};
const char *ibv_wc_status_str(enum ibv_wc_status status);

enum ibv_wc_opcode {
	IBV_WC_SEND,
	IBV_WC_RDMA_WRITE,
	IBV_WC_RDMA_READ,
	IBV_WC_COMP_SWAP,
	IBV_WC_FETCH_ADD,
	IBV_WC_BIND_MW,
	IBV_WC_LOCAL_INV,
	IBV_WC_TSO,
/*
 * Set value of IBV_WC_RECV so consumers can test if a completion is a
 * receive by testing (opcode & IBV_WC_RECV).
 */
	IBV_WC_RECV			= 1 << 7,
	IBV_WC_RECV_RDMA_WITH_IMM,

	IBV_WC_TM_ADD,
	IBV_WC_TM_DEL,
	IBV_WC_TM_SYNC,
	IBV_WC_TM_RECV,
	IBV_WC_TM_NO_TAG,
	IBV_WC_DRIVER1,
};

enum {
	IBV_WC_IP_CSUM_OK_SHIFT	= 2
};

enum ibv_create_cq_wc_flags {
	IBV_WC_EX_WITH_BYTE_LEN		= 1 << 0,
	IBV_WC_EX_WITH_IMM		= 1 << 1,
	IBV_WC_EX_WITH_QP_NUM		= 1 << 2,
	IBV_WC_EX_WITH_SRC_QP		= 1 << 3,
	IBV_WC_EX_WITH_SLID		= 1 << 4,
	IBV_WC_EX_WITH_SL		= 1 << 5,
	IBV_WC_EX_WITH_DLID_PATH_BITS	= 1 << 6,
	IBV_WC_EX_WITH_COMPLETION_TIMESTAMP	= 1 << 7,
	IBV_WC_EX_WITH_CVLAN		= 1 << 8,
	IBV_WC_EX_WITH_FLOW_TAG		= 1 << 9,
	IBV_WC_EX_WITH_TM_INFO		= 1 << 10,
	IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK	= 1 << 11,
};

enum {
	IBV_WC_STANDARD_FLAGS = IBV_WC_EX_WITH_BYTE_LEN		|
				 IBV_WC_EX_WITH_IMM		|
				 IBV_WC_EX_WITH_QP_NUM		|
				 IBV_WC_EX_WITH_SRC_QP		|
				 IBV_WC_EX_WITH_SLID		|
				 IBV_WC_EX_WITH_SL		|
				 IBV_WC_EX_WITH_DLID_PATH_BITS
};

enum {
	IBV_CREATE_CQ_SUP_WC_FLAGS = IBV_WC_STANDARD_FLAGS |
				IBV_WC_EX_WITH_COMPLETION_TIMESTAMP |
				IBV_WC_EX_WITH_CVLAN |
				IBV_WC_EX_WITH_FLOW_TAG |
				IBV_WC_EX_WITH_TM_INFO |
				IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK
};

enum ibv_wc_flags {
	IBV_WC_GRH		= 1 << 0,
	IBV_WC_WITH_IMM		= 1 << 1,
	IBV_WC_IP_CSUM_OK	= 1 << IBV_WC_IP_CSUM_OK_SHIFT,
	IBV_WC_WITH_INV		= 1 << 3,
	IBV_WC_TM_SYNC_REQ	= 1 << 4,
	IBV_WC_TM_MATCH		= 1 << 5,
	IBV_WC_TM_DATA_VALID	= 1 << 6,
};

struct ibv_wc {
	uint64_t		wr_id;
	enum ibv_wc_status	status;
	enum ibv_wc_opcode	opcode;
	uint32_t		vendor_err;
	uint32_t		byte_len;
	/* When (wc_flags & IBV_WC_WITH_IMM): Immediate data in network byte order.
	 * When (wc_flags & IBV_WC_WITH_INV): Stores the invalidated rkey.
	 */
	union {
		__be32		imm_data;
		uint32_t	invalidated_rkey;
	};
	uint32_t		qp_num;
	uint32_t		src_qp;
	unsigned int		wc_flags;
	uint16_t		pkey_index;
	uint16_t		slid;
	uint8_t			sl;
	uint8_t			dlid_path_bits;
};

enum ibv_access_flags {
	IBV_ACCESS_LOCAL_WRITE		= 1,
	IBV_ACCESS_REMOTE_WRITE		= (1<<1),
	IBV_ACCESS_REMOTE_READ		= (1<<2),
	IBV_ACCESS_REMOTE_ATOMIC	= (1<<3),
	IBV_ACCESS_MW_BIND		= (1<<4),
	IBV_ACCESS_ZERO_BASED		= (1<<5),
	IBV_ACCESS_ON_DEMAND		= (1<<6),
	IBV_ACCESS_HUGETLB		= (1<<7),
	IBV_ACCESS_RELAXED_ORDERING	= IBV_ACCESS_OPTIONAL_FIRST,
};

struct ibv_mw_bind_info {
	struct ibv_mr	*mr;
	uint64_t	 addr;
	uint64_t	 length;
	unsigned int	 mw_access_flags; /* use ibv_access_flags */
};

struct ibv_pd {
	struct ibv_context     *context;
	uint32_t		handle;
};

struct ibv_td_init_attr {
	uint32_t comp_mask;
};

struct ibv_td {
	struct ibv_context     *context;
};

enum ibv_xrcd_init_attr_mask {
	IBV_XRCD_INIT_ATTR_FD	    = 1 << 0,
	IBV_XRCD_INIT_ATTR_OFLAGS   = 1 << 1,
	IBV_XRCD_INIT_ATTR_RESERVED = 1 << 2
};

struct ibv_xrcd_init_attr {
	uint32_t comp_mask;
	int	 fd;
	int	 oflags;
};

struct ibv_xrcd {
	struct ibv_context     *context;
};

enum ibv_rereg_mr_flags {
	IBV_REREG_MR_CHANGE_TRANSLATION	= (1 << 0),
	IBV_REREG_MR_CHANGE_PD		= (1 << 1),
	IBV_REREG_MR_CHANGE_ACCESS	= (1 << 2),
	IBV_REREG_MR_KEEP_VALID		= (1 << 3),
	IBV_REREG_MR_FLAGS_SUPPORTED	= ((IBV_REREG_MR_KEEP_VALID << 1) - 1)
};

struct ibv_mr {
	struct ibv_context     *context;
	struct ibv_pd	       *pd;
	void		       *addr;
	size_t			length;
	uint32_t		handle;
	uint32_t		lkey;
	uint32_t		rkey;
};

enum ibv_mw_type {
	IBV_MW_TYPE_1			= 1,
	IBV_MW_TYPE_2			= 2
};

struct ibv_mw {
	struct ibv_context     *context;
	struct ibv_pd	       *pd;
	uint32_t		rkey;
	uint32_t		handle;
	enum ibv_mw_type	type;
};

struct ibv_global_route {
	union ibv_gid		dgid;
	uint32_t		flow_label;
	uint8_t			sgid_index;
	uint8_t			hop_limit;
	uint8_t			traffic_class;
};

struct ibv_grh {
	__be32			version_tclass_flow;
	__be16			paylen;
	uint8_t			next_hdr;
	uint8_t			hop_limit;
	union ibv_gid		sgid;
	union ibv_gid		dgid;
};

enum ibv_rate {
	IBV_RATE_MAX      = 0,
	IBV_RATE_2_5_GBPS = 2,
	IBV_RATE_5_GBPS   = 5,
	IBV_RATE_10_GBPS  = 3,
	IBV_RATE_20_GBPS  = 6,
	IBV_RATE_30_GBPS  = 4,
	IBV_RATE_40_GBPS  = 7,
	IBV_RATE_60_GBPS  = 8,
	IBV_RATE_80_GBPS  = 9,
	IBV_RATE_120_GBPS = 10,
	IBV_RATE_14_GBPS  = 11,
	IBV_RATE_56_GBPS  = 12,
	IBV_RATE_112_GBPS = 13,
	IBV_RATE_168_GBPS = 14,
	IBV_RATE_25_GBPS  = 15,
	IBV_RATE_100_GBPS = 16,
	IBV_RATE_200_GBPS = 17,
	IBV_RATE_300_GBPS = 18,
	IBV_RATE_28_GBPS  = 19,
	IBV_RATE_50_GBPS  = 20,
	IBV_RATE_400_GBPS = 21,
	IBV_RATE_600_GBPS = 22,
};

/**
 * ibv_rate_to_mult - Convert the IB rate enum to a multiple of the
 * base rate of 2.5 Gbit/sec.  For example, IBV_RATE_5_GBPS will be
 * converted to 2, since 5 Gbit/sec is 2 * 2.5 Gbit/sec.
 * @rate: rate to convert.
 */
int  __attribute_const ibv_rate_to_mult(enum ibv_rate rate);

/**
 * mult_to_ibv_rate - Convert a multiple of 2.5 Gbit/sec to an IB rate enum.
 * @mult: multiple to convert.
 */
enum ibv_rate __attribute_const mult_to_ibv_rate(int mult);

/**
 * ibv_rate_to_mbps - Convert the IB rate enum to Mbit/sec.
 * For example, IBV_RATE_5_GBPS will return the value 5000.
 * @rate: rate to convert.
 */
int __attribute_const ibv_rate_to_mbps(enum ibv_rate rate);

/**
 * mbps_to_ibv_rate - Convert a Mbit/sec value to an IB rate enum.
 * @mbps: value to convert.
 */
enum ibv_rate __attribute_const mbps_to_ibv_rate(int mbps) __attribute_const;

struct ibv_ah_attr {
	struct ibv_global_route	grh;
	uint16_t		dlid;
	uint8_t			sl;
	uint8_t			src_path_bits;
	uint8_t			static_rate;
	uint8_t			is_global;
	uint8_t			port_num;
};

enum ibv_srq_attr_mask {
	IBV_SRQ_MAX_WR	= 1 << 0,
	IBV_SRQ_LIMIT	= 1 << 1
};

struct ibv_srq_attr {
	uint32_t		max_wr;
	uint32_t		max_sge;
	uint32_t		srq_limit;
};

struct ibv_srq_init_attr {
	void		       *srq_context;
	struct ibv_srq_attr	attr;
};

enum ibv_srq_type {
	IBV_SRQT_BASIC,
	IBV_SRQT_XRC,
	IBV_SRQT_TM,
};

enum ibv_srq_init_attr_mask {
	IBV_SRQ_INIT_ATTR_TYPE		= 1 << 0,
	IBV_SRQ_INIT_ATTR_PD		= 1 << 1,
	IBV_SRQ_INIT_ATTR_XRCD		= 1 << 2,
	IBV_SRQ_INIT_ATTR_CQ		= 1 << 3,
	IBV_SRQ_INIT_ATTR_TM		= 1 << 4,
	IBV_SRQ_INIT_ATTR_RESERVED	= 1 << 5,
};

struct ibv_tm_cap {
	uint32_t		max_num_tags;
	uint32_t		max_ops;
};

struct ibv_srq_init_attr_ex {
	void		       *srq_context;
	struct ibv_srq_attr	attr;

	uint32_t		comp_mask;
	enum ibv_srq_type	srq_type;
	struct ibv_pd	       *pd;
	struct ibv_xrcd	       *xrcd;
	struct ibv_cq	       *cq;
	struct ibv_tm_cap	tm_cap;
};

enum ibv_wq_type {
	IBV_WQT_RQ
};

enum ibv_wq_init_attr_mask {
	IBV_WQ_INIT_ATTR_FLAGS		= 1 << 0,
	IBV_WQ_INIT_ATTR_RESERVED	= 1 << 1,
};

enum ibv_wq_flags {
	IBV_WQ_FLAGS_CVLAN_STRIPPING		= 1 << 0,
	IBV_WQ_FLAGS_SCATTER_FCS		= 1 << 1,
	IBV_WQ_FLAGS_DELAY_DROP			= 1 << 2,
	IBV_WQ_FLAGS_PCI_WRITE_END_PADDING	= 1 << 3,
	IBV_WQ_FLAGS_RESERVED			= 1 << 4,
};

struct ibv_wq_init_attr {
	void		       *wq_context;
	enum ibv_wq_type	wq_type;
	uint32_t		max_wr;
	uint32_t		max_sge;
	struct	ibv_pd	       *pd;
	struct	ibv_cq	       *cq;
	uint32_t		comp_mask; /* Use ibv_wq_init_attr_mask */
	uint32_t		create_flags; /* use ibv_wq_flags */
};

enum ibv_wq_state {
	IBV_WQS_RESET,
	IBV_WQS_RDY,
	IBV_WQS_ERR,
	IBV_WQS_UNKNOWN
};

enum ibv_wq_attr_mask {
	IBV_WQ_ATTR_STATE	= 1 << 0,
	IBV_WQ_ATTR_CURR_STATE	= 1 << 1,
	IBV_WQ_ATTR_FLAGS	= 1 << 2,
	IBV_WQ_ATTR_RESERVED	= 1 << 3,
};

struct ibv_wq_attr {
	/* enum ibv_wq_attr_mask */
	uint32_t		attr_mask;
	/* Move the WQ to this state */
	enum	ibv_wq_state	wq_state;
	/* Assume this is the current WQ state */
	enum	ibv_wq_state	curr_wq_state;
	uint32_t		flags; /* Use ibv_wq_flags */
	uint32_t		flags_mask; /* Use ibv_wq_flags */
};

/*
 * Receive Work Queue Indirection Table.
 * It's used in order to distribute incoming packets between different
 * Receive Work Queues. Associating Receive WQs with different CPU cores
 * allows one to workload the traffic between different CPU cores.
 * The Indirection Table can contain only WQs of type IBV_WQT_RQ.
*/
struct ibv_rwq_ind_table {
	struct ibv_context *context;
	int ind_tbl_handle;
	int ind_tbl_num;
	uint32_t comp_mask;
};

enum ibv_ind_table_init_attr_mask {
	IBV_CREATE_IND_TABLE_RESERVED = (1 << 0)
};

/*
 * Receive Work Queue Indirection Table attributes
 */
struct ibv_rwq_ind_table_init_attr {
	uint32_t log_ind_tbl_size;
	/* Each entry is a pointer to a Receive Work Queue */
	struct ibv_wq **ind_tbl;
	uint32_t comp_mask;
};

enum ibv_qp_type {
	IBV_QPT_RC = 2,
	IBV_QPT_UC,
	IBV_QPT_UD,
	IBV_QPT_RAW_PACKET = 8,
	IBV_QPT_XRC_SEND = 9,
	IBV_QPT_XRC_RECV,
	IBV_QPT_DRIVER = 0xff,
};

struct ibv_qp_cap {
	uint32_t		max_send_wr;
	uint32_t		max_recv_wr;
	uint32_t		max_send_sge;
	uint32_t		max_recv_sge;
	uint32_t		max_inline_data;
};

struct ibv_qp_init_attr {
	void		       *qp_context;
	struct ibv_cq	       *send_cq;
	struct ibv_cq	       *recv_cq;
	struct ibv_srq	       *srq;
	struct ibv_qp_cap	cap;
	enum ibv_qp_type	qp_type;
	int			sq_sig_all;
};

enum ibv_qp_init_attr_mask {
	IBV_QP_INIT_ATTR_PD		= 1 << 0,
	IBV_QP_INIT_ATTR_XRCD		= 1 << 1,
	IBV_QP_INIT_ATTR_CREATE_FLAGS	= 1 << 2,
	IBV_QP_INIT_ATTR_MAX_TSO_HEADER = 1 << 3,
	IBV_QP_INIT_ATTR_IND_TABLE	= 1 << 4,
	IBV_QP_INIT_ATTR_RX_HASH	= 1 << 5,
	IBV_QP_INIT_ATTR_SEND_OPS_FLAGS = 1 << 6,
};

enum ibv_qp_create_flags {
	IBV_QP_CREATE_BLOCK_SELF_MCAST_LB	= 1 << 1,
	IBV_QP_CREATE_SCATTER_FCS		= 1 << 8,
	IBV_QP_CREATE_CVLAN_STRIPPING		= 1 << 9,
	IBV_QP_CREATE_SOURCE_QPN		= 1 << 10,
	IBV_QP_CREATE_PCI_WRITE_END_PADDING	= 1 << 11,
};

enum ibv_qp_create_send_ops_flags {
	IBV_QP_EX_WITH_RDMA_WRITE		= 1 << 0,
	IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM	= 1 << 1,
	IBV_QP_EX_WITH_SEND			= 1 << 2,
	IBV_QP_EX_WITH_SEND_WITH_IMM		= 1 << 3,
	IBV_QP_EX_WITH_RDMA_READ		= 1 << 4,
	IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP	= 1 << 5,
	IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD	= 1 << 6,
	IBV_QP_EX_WITH_LOCAL_INV		= 1 << 7,
	IBV_QP_EX_WITH_BIND_MW			= 1 << 8,
	IBV_QP_EX_WITH_SEND_WITH_INV		= 1 << 9,
	IBV_QP_EX_WITH_TSO			= 1 << 10,
};

struct ibv_rx_hash_conf {
	/* enum ibv_rx_hash_function_flags */
	uint8_t	rx_hash_function;
	uint8_t	rx_hash_key_len;
	uint8_t	*rx_hash_key;
	/* enum ibv_rx_hash_fields */
	uint64_t	rx_hash_fields_mask;
};

struct ibv_qp_init_attr_ex {
	void		       *qp_context;
	struct ibv_cq	       *send_cq;
	struct ibv_cq	       *recv_cq;
	struct ibv_srq	       *srq;
	struct ibv_qp_cap	cap;
	enum ibv_qp_type	qp_type;
	int			sq_sig_all;

	uint32_t		comp_mask;
	struct ibv_pd	       *pd;
	struct ibv_xrcd	       *xrcd;
	uint32_t                create_flags;
	uint16_t		max_tso_header;
	struct ibv_rwq_ind_table       *rwq_ind_tbl;
	struct ibv_rx_hash_conf	rx_hash_conf;
	uint32_t		source_qpn;
	/* See enum ibv_qp_create_send_ops_flags */
	uint64_t send_ops_flags;
};

enum ibv_qp_open_attr_mask {
	IBV_QP_OPEN_ATTR_NUM		= 1 << 0,
	IBV_QP_OPEN_ATTR_XRCD	        = 1 << 1,
	IBV_QP_OPEN_ATTR_CONTEXT	= 1 << 2,
	IBV_QP_OPEN_ATTR_TYPE		= 1 << 3,
	IBV_QP_OPEN_ATTR_RESERVED	= 1 << 4
};

struct ibv_qp_open_attr {
	uint32_t		comp_mask;
	uint32_t		qp_num;
	struct ibv_xrcd        *xrcd;
	void		       *qp_context;
	enum ibv_qp_type	qp_type;
};

enum ibv_qp_attr_mask {
	IBV_QP_STATE			= 1 << 	0,
	IBV_QP_CUR_STATE		= 1 << 	1,
	IBV_QP_EN_SQD_ASYNC_NOTIFY	= 1 << 	2,
	IBV_QP_ACCESS_FLAGS		= 1 << 	3,
	IBV_QP_PKEY_INDEX		= 1 << 	4,
	IBV_QP_PORT			= 1 << 	5,
	IBV_QP_QKEY			= 1 << 	6,
	IBV_QP_AV			= 1 << 	7,
	IBV_QP_PATH_MTU			= 1 << 	8,
	IBV_QP_TIMEOUT			= 1 << 	9,
	IBV_QP_RETRY_CNT		= 1 << 10,
	IBV_QP_RNR_RETRY		= 1 << 11,
	IBV_QP_RQ_PSN			= 1 << 12,
	IBV_QP_MAX_QP_RD_ATOMIC		= 1 << 13,
	IBV_QP_ALT_PATH			= 1 << 14,
	IBV_QP_MIN_RNR_TIMER		= 1 << 15,
	IBV_QP_SQ_PSN			= 1 << 16,
	IBV_QP_MAX_DEST_RD_ATOMIC	= 1 << 17,
	IBV_QP_PATH_MIG_STATE		= 1 << 18,
	IBV_QP_CAP			= 1 << 19,
	IBV_QP_DEST_QPN			= 1 << 20,
	/* These bits were supported on older kernels, but never exposed from
	   libibverbs:
	_IBV_QP_SMAC   			= 1 << 21,
	_IBV_QP_ALT_SMAC		= 1 << 22,
	_IBV_QP_VID    			= 1 << 23,
	_IBV_QP_ALT_VID 		= 1 << 24,
	*/
	IBV_QP_RATE_LIMIT		= 1 << 25,
};

enum ibv_qp_state {
	IBV_QPS_RESET,
	IBV_QPS_INIT,
	IBV_QPS_RTR,
	IBV_QPS_RTS,
	IBV_QPS_SQD,
	IBV_QPS_SQE,
	IBV_QPS_ERR,
	IBV_QPS_UNKNOWN
};

enum ibv_mig_state {
	IBV_MIG_MIGRATED,
	IBV_MIG_REARM,
	IBV_MIG_ARMED
};

struct ibv_qp_attr {
	enum ibv_qp_state	qp_state;
	enum ibv_qp_state	cur_qp_state;
	enum ibv_mtu		path_mtu;
	enum ibv_mig_state	path_mig_state;
	uint32_t		qkey;
	uint32_t		rq_psn;
	uint32_t		sq_psn;
	uint32_t		dest_qp_num;
	unsigned int		qp_access_flags;
	struct ibv_qp_cap	cap;
	struct ibv_ah_attr	ah_attr;
	struct ibv_ah_attr	alt_ah_attr;
	uint16_t		pkey_index;
	uint16_t		alt_pkey_index;
	uint8_t			en_sqd_async_notify;
	uint8_t			sq_draining;
	uint8_t			max_rd_atomic;
	uint8_t			max_dest_rd_atomic;
	uint8_t			min_rnr_timer;
	uint8_t			port_num;
	uint8_t			timeout;
	uint8_t			retry_cnt;
	uint8_t			rnr_retry;
	uint8_t			alt_port_num;
	uint8_t			alt_timeout;
	uint32_t		rate_limit;
};

struct ibv_qp_rate_limit_attr {
	uint32_t	rate_limit;  /* in kbps */
	uint32_t	max_burst_sz; /* total burst size in bytes */
	uint16_t	typical_pkt_sz; /* typical send packet size in bytes */
	uint32_t	comp_mask;
};

enum ibv_wr_opcode {
	IBV_WR_RDMA_WRITE,
	IBV_WR_RDMA_WRITE_WITH_IMM,
	IBV_WR_SEND,
	IBV_WR_SEND_WITH_IMM,
	IBV_WR_RDMA_READ,
	IBV_WR_ATOMIC_CMP_AND_SWP,
	IBV_WR_ATOMIC_FETCH_AND_ADD,
	IBV_WR_LOCAL_INV,
	IBV_WR_BIND_MW,
	IBV_WR_SEND_WITH_INV,
	IBV_WR_TSO,
	IBV_WR_DRIVER1,
};

enum ibv_send_flags {
	IBV_SEND_FENCE		= 1 << 0,
	IBV_SEND_SIGNALED	= 1 << 1,
	IBV_SEND_SOLICITED	= 1 << 2,
	IBV_SEND_INLINE		= 1 << 3,
	IBV_SEND_IP_CSUM	= 1 << 4
};

struct ibv_data_buf {
	void *addr;
	size_t length;
};

struct ibv_sge {
	uint64_t		addr;
	uint32_t		length;
	uint32_t		lkey;
};

struct ibv_send_wr {
	uint64_t		wr_id;
	struct ibv_send_wr     *next;
	struct ibv_sge	       *sg_list;
	int			num_sge;
	enum ibv_wr_opcode	opcode;
	unsigned int		send_flags;
	/* When opcode is *_WITH_IMM: Immediate data in network byte order.
	 * When opcode is *_INV: Stores the rkey to invalidate
	 */
	union {
		__be32			imm_data;
		uint32_t		invalidate_rkey;
	};
	union {
		struct {
			uint64_t	remote_addr;
			uint32_t	rkey;
		} rdma;
		struct {
			uint64_t	remote_addr;
			uint64_t	compare_add;
			uint64_t	swap;
			uint32_t	rkey;
		} atomic;
		struct {
			struct ibv_ah  *ah;
			uint32_t	remote_qpn;
			uint32_t	remote_qkey;
		} ud;
	} wr;
	union {
		struct {
			uint32_t    remote_srqn;
		} xrc;
	} qp_type;
	union {
		struct {
			struct ibv_mw	*mw;
			uint32_t		rkey;
			struct ibv_mw_bind_info	bind_info;
		} bind_mw;
		struct {
			void		       *hdr;
			uint16_t		hdr_sz;
			uint16_t		mss;
		} tso;
	};
};

struct ibv_recv_wr {
	uint64_t		wr_id;
	struct ibv_recv_wr     *next;
	struct ibv_sge	       *sg_list;
	int			num_sge;
};

enum ibv_ops_wr_opcode {
	IBV_WR_TAG_ADD,
	IBV_WR_TAG_DEL,
	IBV_WR_TAG_SYNC,
};

enum ibv_ops_flags {
	IBV_OPS_SIGNALED = 1 << 0,
	IBV_OPS_TM_SYNC  = 1 << 1,
};

struct ibv_ops_wr {
	uint64_t				wr_id;
	struct ibv_ops_wr		       *next;
	enum ibv_ops_wr_opcode			opcode;
	int					flags;
	struct {
		uint32_t			unexpected_cnt;
		uint32_t			handle;
		struct {
			uint64_t		recv_wr_id;
			struct ibv_sge	       *sg_list;
			int			num_sge;
			uint64_t		tag;
			uint64_t		mask;
		} add;
	} tm;
};

struct ibv_mw_bind {
	uint64_t		wr_id;
	unsigned int		send_flags;
	struct ibv_mw_bind_info bind_info;
};

struct ibv_srq {
	struct ibv_context     *context;
	void		       *srq_context;
	struct ibv_pd	       *pd;
	uint32_t		handle;

	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	uint32_t		events_completed;
};

/*
 * Work Queue. QP can be created without internal WQs "packaged" inside it,
 * this QP can be configured to use "external" WQ object as its
 * receive/send queue.
 * WQ associated (many to one) with Completion Queue it owns WQ properties
 * (PD, WQ size etc).
 * WQ of type IBV_WQT_RQ:
 * - Contains receive WQEs, in this case its PD serves as scatter as well.
 * - Exposes post receive function to be used to post a list of work
 *   requests (WRs) to its receive queue.
 */
struct ibv_wq {
	struct ibv_context     *context;
	void		       *wq_context;
	struct	ibv_pd	       *pd;
	struct	ibv_cq	       *cq;
	uint32_t		wq_num;
	uint32_t		handle;
	enum ibv_wq_state       state;
	enum ibv_wq_type	wq_type;
	int (*post_recv)(struct ibv_wq *current,
			 struct ibv_recv_wr *recv_wr,
			 struct ibv_recv_wr **bad_recv_wr);
	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	uint32_t		events_completed;
	uint32_t		comp_mask;
};

struct ibv_qp {
	struct ibv_context     *context;
	void		       *qp_context;
	struct ibv_pd	       *pd;
	struct ibv_cq	       *send_cq;
	struct ibv_cq	       *recv_cq;
	struct ibv_srq	       *srq;
	uint32_t		handle;
	uint32_t		qp_num;
	enum ibv_qp_state       state;
	enum ibv_qp_type	qp_type;

	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	uint32_t		events_completed;
};

struct ibv_qp_ex {
	struct ibv_qp qp_base;
	uint64_t comp_mask;

	uint64_t wr_id;
	/* bitmask from enum ibv_send_flags */
	unsigned int wr_flags;

	void (*wr_atomic_cmp_swp)(struct ibv_qp_ex *qp, uint32_t rkey,
				  uint64_t remote_addr, uint64_t compare,
				  uint64_t swap);
	void (*wr_atomic_fetch_add)(struct ibv_qp_ex *qp, uint32_t rkey,
				    uint64_t remote_addr, uint64_t add);
	void (*wr_bind_mw)(struct ibv_qp_ex *qp, struct ibv_mw *mw,
			   uint32_t rkey,
			   const struct ibv_mw_bind_info *bind_info);
	void (*wr_local_inv)(struct ibv_qp_ex *qp, uint32_t invalidate_rkey);
	void (*wr_rdma_read)(struct ibv_qp_ex *qp, uint32_t rkey,
			     uint64_t remote_addr);
	void (*wr_rdma_write)(struct ibv_qp_ex *qp, uint32_t rkey,
			      uint64_t remote_addr);
	void (*wr_rdma_write_imm)(struct ibv_qp_ex *qp, uint32_t rkey,
				  uint64_t remote_addr, __be32 imm_data);

	void (*wr_send)(struct ibv_qp_ex *qp);
	void (*wr_send_imm)(struct ibv_qp_ex *qp, __be32 imm_data);
	void (*wr_send_inv)(struct ibv_qp_ex *qp, uint32_t invalidate_rkey);
	void (*wr_send_tso)(struct ibv_qp_ex *qp, void *hdr, uint16_t hdr_sz,
			    uint16_t mss);

	void (*wr_set_ud_addr)(struct ibv_qp_ex *qp, struct ibv_ah *ah,
			       uint32_t remote_qpn, uint32_t remote_qkey);
	void (*wr_set_xrc_srqn)(struct ibv_qp_ex *qp, uint32_t remote_srqn);

	void (*wr_set_inline_data)(struct ibv_qp_ex *qp, void *addr,
				   size_t length);
	void (*wr_set_inline_data_list)(struct ibv_qp_ex *qp, size_t num_buf,
					const struct ibv_data_buf *buf_list);
	void (*wr_set_sge)(struct ibv_qp_ex *qp, uint32_t lkey, uint64_t addr,
			   uint32_t length);
	void (*wr_set_sge_list)(struct ibv_qp_ex *qp, size_t num_sge,
				const struct ibv_sge *sg_list);

	void (*wr_start)(struct ibv_qp_ex *qp);
	int (*wr_complete)(struct ibv_qp_ex *qp);
	void (*wr_abort)(struct ibv_qp_ex *qp);
};

struct ibv_qp_ex *ibv_qp_to_qp_ex(struct ibv_qp *qp);

static inline void ibv_wr_atomic_cmp_swp(struct ibv_qp_ex *qp, uint32_t rkey,
					 uint64_t remote_addr, uint64_t compare,
					 uint64_t swap)
{
	qp->wr_atomic_cmp_swp(qp, rkey, remote_addr, compare, swap);
}

static inline void ibv_wr_atomic_fetch_add(struct ibv_qp_ex *qp, uint32_t rkey,
					   uint64_t remote_addr, uint64_t add)
{
	qp->wr_atomic_fetch_add(qp, rkey, remote_addr, add);
}

static inline void ibv_wr_bind_mw(struct ibv_qp_ex *qp, struct ibv_mw *mw,
				  uint32_t rkey,
				  const struct ibv_mw_bind_info *bind_info)
{
	qp->wr_bind_mw(qp, mw, rkey, bind_info);
}

static inline void ibv_wr_local_inv(struct ibv_qp_ex *qp,
				    uint32_t invalidate_rkey)
{
	qp->wr_local_inv(qp, invalidate_rkey);
}

static inline void ibv_wr_rdma_read(struct ibv_qp_ex *qp, uint32_t rkey,
				    uint64_t remote_addr)
{
	qp->wr_rdma_read(qp, rkey, remote_addr);
}

static inline void ibv_wr_rdma_write(struct ibv_qp_ex *qp, uint32_t rkey,
				     uint64_t remote_addr)
{
	qp->wr_rdma_write(qp, rkey, remote_addr);
}

static inline void ibv_wr_rdma_write_imm(struct ibv_qp_ex *qp, uint32_t rkey,
					 uint64_t remote_addr, __be32 imm_data)
{
	qp->wr_rdma_write_imm(qp, rkey, remote_addr, imm_data);
}

static inline void ibv_wr_send(struct ibv_qp_ex *qp)
{
	qp->wr_send(qp);
}

static inline void ibv_wr_send_imm(struct ibv_qp_ex *qp, __be32 imm_data)
{
	qp->wr_send_imm(qp, imm_data);
}

static inline void ibv_wr_send_inv(struct ibv_qp_ex *qp,
				   uint32_t invalidate_rkey)
{
	qp->wr_send_inv(qp, invalidate_rkey);
}

static inline void ibv_wr_send_tso(struct ibv_qp_ex *qp, void *hdr,
				   uint16_t hdr_sz, uint16_t mss)
{
	qp->wr_send_tso(qp, hdr, hdr_sz, mss);
}

static inline void ibv_wr_set_ud_addr(struct ibv_qp_ex *qp, struct ibv_ah *ah,
				      uint32_t remote_qpn, uint32_t remote_qkey)
{
	qp->wr_set_ud_addr(qp, ah, remote_qpn, remote_qkey);
}

static inline void ibv_wr_set_xrc_srqn(struct ibv_qp_ex *qp,
				       uint32_t remote_srqn)
{
	qp->wr_set_xrc_srqn(qp, remote_srqn);
}

static inline void ibv_wr_set_inline_data(struct ibv_qp_ex *qp, void *addr,
					  size_t length)
{
	qp->wr_set_inline_data(qp, addr, length);
}

static inline void ibv_wr_set_inline_data_list(struct ibv_qp_ex *qp,
					       size_t num_buf,
					       const struct ibv_data_buf *buf_list)
{
	qp->wr_set_inline_data_list(qp, num_buf, buf_list);
}

static inline void ibv_wr_set_sge(struct ibv_qp_ex *qp, uint32_t lkey,
				  uint64_t addr, uint32_t length)
{
	qp->wr_set_sge(qp, lkey, addr, length);
}

static inline void ibv_wr_set_sge_list(struct ibv_qp_ex *qp, size_t num_sge,
				       const struct ibv_sge *sg_list)
{
	qp->wr_set_sge_list(qp, num_sge, sg_list);
}

static inline void ibv_wr_start(struct ibv_qp_ex *qp)
{
	qp->wr_start(qp);
}

static inline int ibv_wr_complete(struct ibv_qp_ex *qp)
{
	return qp->wr_complete(qp);
}

static inline void ibv_wr_abort(struct ibv_qp_ex *qp)
{
	qp->wr_abort(qp);
}

struct ibv_ece {
	/*
	 * Unique identifier of the provider vendor on the network.
	 * The providers will set IEEE OUI here to distinguish
	 * itself in non-homogenius network.
	 */
	uint32_t vendor_id;
	/*
	 * Provider specific attributes which are supported or
	 * needed to be enabled by ECE users.
	 */
	uint32_t options;
	uint32_t comp_mask;
};

struct ibv_comp_channel {
	struct ibv_context     *context;
	int			fd;
	int			refcnt;
};

struct ibv_cq {
	struct ibv_context     *context;
	struct ibv_comp_channel *channel;
	void		       *cq_context;
	uint32_t		handle;
	int			cqe;

	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	uint32_t		comp_events_completed;
	uint32_t		async_events_completed;
};

struct ibv_poll_cq_attr {
	uint32_t comp_mask;
};

struct ibv_wc_tm_info {
	uint64_t		tag;	 /* tag from TMH */
	uint32_t		priv;	 /* opaque user data from TMH */
};

struct ibv_cq_ex {
	struct ibv_context     *context;
	struct ibv_comp_channel *channel;
	void		       *cq_context;
	uint32_t		handle;
	int			cqe;

	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	uint32_t		comp_events_completed;
	uint32_t		async_events_completed;

	uint32_t		comp_mask;
	enum ibv_wc_status status;
	uint64_t wr_id;
	int (*start_poll)(struct ibv_cq_ex *current,
			     struct ibv_poll_cq_attr *attr);
	int (*next_poll)(struct ibv_cq_ex *current);
	void (*end_poll)(struct ibv_cq_ex *current);
	enum ibv_wc_opcode (*read_opcode)(struct ibv_cq_ex *current);
	uint32_t (*read_vendor_err)(struct ibv_cq_ex *current);
	uint32_t (*read_byte_len)(struct ibv_cq_ex *current);
	__be32 (*read_imm_data)(struct ibv_cq_ex *current);
	uint32_t (*read_qp_num)(struct ibv_cq_ex *current);
	uint32_t (*read_src_qp)(struct ibv_cq_ex *current);
	unsigned int (*read_wc_flags)(struct ibv_cq_ex *current);
	uint32_t (*read_slid)(struct ibv_cq_ex *current);
	uint8_t (*read_sl)(struct ibv_cq_ex *current);
	uint8_t (*read_dlid_path_bits)(struct ibv_cq_ex *current);
	uint64_t (*read_completion_ts)(struct ibv_cq_ex *current);
	uint16_t (*read_cvlan)(struct ibv_cq_ex *current);
	uint32_t (*read_flow_tag)(struct ibv_cq_ex *current);
	void (*read_tm_info)(struct ibv_cq_ex *current,
			     struct ibv_wc_tm_info *tm_info);
	uint64_t (*read_completion_wallclock_ns)(struct ibv_cq_ex *current);
};

static inline struct ibv_cq *ibv_cq_ex_to_cq(struct ibv_cq_ex *cq)
{
	return (struct ibv_cq *)cq;
}

enum ibv_cq_attr_mask {
	IBV_CQ_ATTR_MODERATE = 1 << 0,
	IBV_CQ_ATTR_RESERVED = 1 << 1,
};

struct ibv_moderate_cq {
	uint16_t cq_count;
	uint16_t cq_period; /* in micro seconds */
};

struct ibv_modify_cq_attr {
	uint32_t attr_mask;
	struct ibv_moderate_cq moderate;
};

static inline int ibv_start_poll(struct ibv_cq_ex *cq,
				    struct ibv_poll_cq_attr *attr)
{
	return cq->start_poll(cq, attr);
}

static inline int ibv_next_poll(struct ibv_cq_ex *cq)
{
	return cq->next_poll(cq);
}

static inline void ibv_end_poll(struct ibv_cq_ex *cq)
{
	cq->end_poll(cq);
}

static inline enum ibv_wc_opcode ibv_wc_read_opcode(struct ibv_cq_ex *cq)
{
	return cq->read_opcode(cq);
}

static inline uint32_t ibv_wc_read_vendor_err(struct ibv_cq_ex *cq)
{
	return cq->read_vendor_err(cq);
}

static inline uint32_t ibv_wc_read_byte_len(struct ibv_cq_ex *cq)
{
	return cq->read_byte_len(cq);
}

static inline __be32 ibv_wc_read_imm_data(struct ibv_cq_ex *cq)
{
	return cq->read_imm_data(cq);
}

static inline uint32_t ibv_wc_read_invalidated_rkey(struct ibv_cq_ex *cq)
{
#ifdef __CHECKER__
	return (__attribute__((force)) uint32_t)cq->read_imm_data(cq);
#else
	return cq->read_imm_data(cq);
#endif
}

static inline uint32_t ibv_wc_read_qp_num(struct ibv_cq_ex *cq)
{
	return cq->read_qp_num(cq);
}

static inline uint32_t ibv_wc_read_src_qp(struct ibv_cq_ex *cq)
{
	return cq->read_src_qp(cq);
}

static inline unsigned int ibv_wc_read_wc_flags(struct ibv_cq_ex *cq)
{
	return cq->read_wc_flags(cq);
}

static inline uint32_t ibv_wc_read_slid(struct ibv_cq_ex *cq)
{
	return cq->read_slid(cq);
}

static inline uint8_t ibv_wc_read_sl(struct ibv_cq_ex *cq)
{
	return cq->read_sl(cq);
}

static inline uint8_t ibv_wc_read_dlid_path_bits(struct ibv_cq_ex *cq)
{
	return cq->read_dlid_path_bits(cq);
}

static inline uint64_t ibv_wc_read_completion_ts(struct ibv_cq_ex *cq)
{
	return cq->read_completion_ts(cq);
}

static inline uint64_t ibv_wc_read_completion_wallclock_ns(struct ibv_cq_ex *cq)
{
	return cq->read_completion_wallclock_ns(cq);
}

static inline uint16_t ibv_wc_read_cvlan(struct ibv_cq_ex *cq)
{
	return cq->read_cvlan(cq);
}

static inline uint32_t ibv_wc_read_flow_tag(struct ibv_cq_ex *cq)
{
	return cq->read_flow_tag(cq);
}

static inline void ibv_wc_read_tm_info(struct ibv_cq_ex *cq,
				       struct ibv_wc_tm_info *tm_info)
{
	cq->read_tm_info(cq, tm_info);
}

static inline int ibv_post_wq_recv(struct ibv_wq *wq,
				   struct ibv_recv_wr *recv_wr,
				   struct ibv_recv_wr **bad_recv_wr)
{
	return wq->post_recv(wq, recv_wr, bad_recv_wr);
}

struct ibv_ah {
	struct ibv_context     *context;
	struct ibv_pd	       *pd;
	uint32_t		handle;
};

enum ibv_flow_flags {
	/* First bit is deprecated and can't be used */
	IBV_FLOW_ATTR_FLAGS_DONT_TRAP = 1 << 1,
	IBV_FLOW_ATTR_FLAGS_EGRESS = 1 << 2,
};

enum ibv_flow_attr_type {
	/* steering according to rule specifications */
	IBV_FLOW_ATTR_NORMAL		= 0x0,
	/* default unicast and multicast rule -
	 * receive all Eth traffic which isn't steered to any QP
	 */
	IBV_FLOW_ATTR_ALL_DEFAULT	= 0x1,
	/* default multicast rule -
	 * receive all Eth multicast traffic which isn't steered to any QP
	 */
	IBV_FLOW_ATTR_MC_DEFAULT	= 0x2,
	/* sniffer rule - receive all port traffic */
	IBV_FLOW_ATTR_SNIFFER		= 0x3,
};

enum ibv_flow_spec_type {
	IBV_FLOW_SPEC_ETH		= 0x20,
	IBV_FLOW_SPEC_IPV4		= 0x30,
	IBV_FLOW_SPEC_IPV6		= 0x31,
	IBV_FLOW_SPEC_IPV4_EXT		= 0x32,
	IBV_FLOW_SPEC_ESP		= 0x34,
	IBV_FLOW_SPEC_TCP		= 0x40,
	IBV_FLOW_SPEC_UDP		= 0x41,
	IBV_FLOW_SPEC_VXLAN_TUNNEL	= 0x50,
	IBV_FLOW_SPEC_GRE		= 0x51,
	IBV_FLOW_SPEC_MPLS		= 0x60,
	IBV_FLOW_SPEC_INNER		= 0x100,
	IBV_FLOW_SPEC_ACTION_TAG	= 0x1000,
	IBV_FLOW_SPEC_ACTION_DROP	= 0x1001,
	IBV_FLOW_SPEC_ACTION_HANDLE	= 0x1002,
	IBV_FLOW_SPEC_ACTION_COUNT	= 0x1003,
};

struct ibv_flow_eth_filter {
	uint8_t		dst_mac[6];
	uint8_t		src_mac[6];
	uint16_t	ether_type;
	/*
	 * same layout as 802.1q: prio 3, cfi 1, vlan id 12
	 */
	uint16_t	vlan_tag;
};

struct ibv_flow_spec_eth {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_eth_filter val;
	struct ibv_flow_eth_filter mask;
};

struct ibv_flow_ipv4_filter {
	uint32_t src_ip;
	uint32_t dst_ip;
};

struct ibv_flow_spec_ipv4 {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_ipv4_filter val;
	struct ibv_flow_ipv4_filter mask;
};

struct ibv_flow_ipv4_ext_filter {
	uint32_t src_ip;
	uint32_t dst_ip;
	uint8_t  proto;
	uint8_t  tos;
	uint8_t  ttl;
	uint8_t  flags;
};

struct ibv_flow_spec_ipv4_ext {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_ipv4_ext_filter val;
	struct ibv_flow_ipv4_ext_filter mask;
};

struct ibv_flow_ipv6_filter {
	uint8_t  src_ip[16];
	uint8_t  dst_ip[16];
	uint32_t flow_label;
	uint8_t  next_hdr;
	uint8_t  traffic_class;
	uint8_t  hop_limit;
};

struct ibv_flow_spec_ipv6 {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_ipv6_filter val;
	struct ibv_flow_ipv6_filter mask;
};

struct ibv_flow_esp_filter {
	uint32_t spi;
	uint32_t seq;
};

struct ibv_flow_spec_esp {
	enum ibv_flow_spec_type type;
	uint16_t size;
	struct ibv_flow_esp_filter val;
	struct ibv_flow_esp_filter mask;
};

struct ibv_flow_tcp_udp_filter {
	uint16_t dst_port;
	uint16_t src_port;
};

struct ibv_flow_spec_tcp_udp {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_tcp_udp_filter val;
	struct ibv_flow_tcp_udp_filter mask;
};

struct ibv_flow_gre_filter {
	/* c_ks_res0_ver field is bits 0-15 in offset 0 of a standard GRE header:
	 * bit 0 - checksum present bit.
	 * bit 1 - reserved. set to 0.
	 * bit 2 - key present bit.
	 * bit 3 - sequence number present bit.
	 * bits 4:12 - reserved. set to 0.
	 * bits 13:15 - GRE version.
	 */
	uint16_t c_ks_res0_ver;
	uint16_t protocol;
	uint32_t key;
};

struct ibv_flow_spec_gre {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_gre_filter val;
	struct ibv_flow_gre_filter mask;
};

struct ibv_flow_mpls_filter {
	/* The field includes the entire MPLS label:
	 * bits 0:19 - label value field.
	 * bits 20:22 - traffic class field.
	 * bits 23 - bottom of stack bit.
	 * bits 24:31 - ttl field.
	 */
	uint32_t label;
};

struct ibv_flow_spec_mpls {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_mpls_filter val;
	struct ibv_flow_mpls_filter mask;
};

struct ibv_flow_tunnel_filter {
	uint32_t tunnel_id;
};

struct ibv_flow_spec_tunnel {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_flow_tunnel_filter val;
	struct ibv_flow_tunnel_filter mask;
};

struct ibv_flow_spec_action_tag {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	uint32_t  tag_id;
};

struct ibv_flow_spec_action_drop {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
};

struct ibv_flow_spec_action_handle {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	const struct ibv_flow_action *action;
};

struct ibv_flow_spec_counter_action {
	enum ibv_flow_spec_type  type;
	uint16_t  size;
	struct ibv_counters *counters;
};

struct ibv_flow_spec {
	union {
		struct {
			enum ibv_flow_spec_type	type;
			uint16_t		size;
		} hdr;
		struct ibv_flow_spec_eth eth;
		struct ibv_flow_spec_ipv4 ipv4;
		struct ibv_flow_spec_tcp_udp tcp_udp;
		struct ibv_flow_spec_ipv4_ext ipv4_ext;
		struct ibv_flow_spec_ipv6 ipv6;
		struct ibv_flow_spec_esp esp;
		struct ibv_flow_spec_tunnel tunnel;
		struct ibv_flow_spec_gre gre;
		struct ibv_flow_spec_mpls mpls;
		struct ibv_flow_spec_action_tag flow_tag;
		struct ibv_flow_spec_action_drop drop;
		struct ibv_flow_spec_action_handle handle;
		struct ibv_flow_spec_counter_action flow_count;
	};
};

struct ibv_flow_attr {
	uint32_t comp_mask;
	enum ibv_flow_attr_type type;
	uint16_t size;
	uint16_t priority;
	uint8_t num_of_specs;
	uint8_t port;
	uint32_t flags;
	/* Following are the optional layers according to user request
	 * struct ibv_flow_spec_xxx [L2]
	 * struct ibv_flow_spec_yyy [L3/L4]
	 */
};

struct ibv_flow {
	uint32_t	   comp_mask;
	struct ibv_context *context;
	uint32_t	   handle;
};

struct ibv_flow_action {
	struct ibv_context *context;
};

enum ibv_flow_action_esp_mask {
	IBV_FLOW_ACTION_ESP_MASK_ESN    = 1UL << 0,
};

struct ibv_flow_action_esp_attr {
	struct ibv_flow_action_esp *esp_attr;

	enum ibv_flow_action_esp_keymat	keymat_proto;
	uint16_t		keymat_len;
	void			*keymat_ptr;

	enum ibv_flow_action_esp_replay replay_proto;
	uint16_t                replay_len;
	void                    *replay_ptr;

	struct ibv_flow_action_esp_encap *esp_encap;

	uint32_t		comp_mask; /* Use enum ibv_flow_action_esp_mask */
	uint32_t		esn;
};

struct ibv_device;
struct ibv_context;

/* Obsolete, never used, do not touch */
struct _ibv_device_ops {
	struct ibv_context *	(*_dummy1)(struct ibv_device *device, int cmd_fd);
	void			(*_dummy2)(struct ibv_context *context);
};

enum {
	IBV_SYSFS_NAME_MAX	= 64,
	IBV_SYSFS_PATH_MAX	= 256
};

struct ibv_device {
	struct _ibv_device_ops	_ops;
	enum ibv_node_type	node_type;
	enum ibv_transport_type	transport_type;
	/* Name of underlying kernel IB device, eg "mthca0" */
	char			name[IBV_SYSFS_NAME_MAX];
	/* Name of uverbs device, eg "uverbs0" */
	char			dev_name[IBV_SYSFS_NAME_MAX];
	/* Path to infiniband_verbs class device in sysfs */
	char			dev_path[IBV_SYSFS_PATH_MAX];
	/* Path to infiniband class device in sysfs */
	char			ibdev_path[IBV_SYSFS_PATH_MAX];
};

struct _compat_ibv_port_attr;
struct ibv_context_ops {
	void *(*_compat_query_device)(void);
	int (*_compat_query_port)(struct ibv_context *context,
				  uint8_t port_num,
				  struct _compat_ibv_port_attr *port_attr);
	void *(*_compat_alloc_pd)(void);
	void *(*_compat_dealloc_pd)(void);
	void *(*_compat_reg_mr)(void);
	void *(*_compat_rereg_mr)(void);
	void *(*_compat_dereg_mr)(void);
	struct ibv_mw *		(*alloc_mw)(struct ibv_pd *pd, enum ibv_mw_type type);
	int			(*bind_mw)(struct ibv_qp *qp, struct ibv_mw *mw,
					   struct ibv_mw_bind *mw_bind);
	int			(*dealloc_mw)(struct ibv_mw *mw);
	void *(*_compat_create_cq)(void);
	int			(*poll_cq)(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
	int			(*req_notify_cq)(struct ibv_cq *cq, int solicited_only);
	void *(*_compat_cq_event)(void);
	void *(*_compat_resize_cq)(void);
	void *(*_compat_destroy_cq)(void);
	void *(*_compat_create_srq)(void);
	void *(*_compat_modify_srq)(void);
	void *(*_compat_query_srq)(void);
	void *(*_compat_destroy_srq)(void);
	int			(*post_srq_recv)(struct ibv_srq *srq,
						 struct ibv_recv_wr *recv_wr,
						 struct ibv_recv_wr **bad_recv_wr);
	void *(*_compat_create_qp)(void);
	void *(*_compat_query_qp)(void);
	void *(*_compat_modify_qp)(void);
	void *(*_compat_destroy_qp)(void);
	int			(*post_send)(struct ibv_qp *qp, struct ibv_send_wr *wr,
					     struct ibv_send_wr **bad_wr);
	int			(*post_recv)(struct ibv_qp *qp, struct ibv_recv_wr *wr,
					     struct ibv_recv_wr **bad_wr);
	void *(*_compat_create_ah)(void);
	void *(*_compat_destroy_ah)(void);
	void *(*_compat_attach_mcast)(void);
	void *(*_compat_detach_mcast)(void);
	void *(*_compat_async_event)(void);
};

struct ibv_context {
	struct ibv_device      *device;
	struct ibv_context_ops	ops;
	int			cmd_fd;
	int			async_fd;
	int			num_comp_vectors;
	pthread_mutex_t		mutex;
	void		       *abi_compat;
};

enum ibv_cq_init_attr_mask {
	IBV_CQ_INIT_ATTR_MASK_FLAGS	= 1 << 0,
	IBV_CQ_INIT_ATTR_MASK_PD	= 1 << 1,
};

enum ibv_create_cq_attr_flags {
	IBV_CREATE_CQ_ATTR_SINGLE_THREADED = 1 << 0,
	IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN  = 1 << 1,
};

struct ibv_cq_init_attr_ex {
	/* Minimum number of entries required for CQ */
	uint32_t			cqe;
	/* Consumer-supplied context returned for completion events */
	void			*cq_context;
	/* Completion channel where completion events will be queued.
	 * May be NULL if completion events will not be used.
	 */
	struct ibv_comp_channel *channel;
	/* Completion vector used to signal completion events.
	 *  Must be < context->num_comp_vectors.
	 */
	uint32_t			comp_vector;
	 /* Or'ed bit of enum ibv_create_cq_wc_flags. */
	uint64_t		wc_flags;
	/* compatibility mask (extended verb). Or'd flags of
	 * enum ibv_cq_init_attr_mask
	 */
	uint32_t		comp_mask;
	/* create cq attr flags - one or more flags from
	 * enum ibv_create_cq_attr_flags
	 */
	uint32_t		flags;
	struct ibv_pd		*parent_domain;
};

enum ibv_parent_domain_init_attr_mask {
	IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS = 1 << 0,
	IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT = 1 << 1,
};

#define IBV_ALLOCATOR_USE_DEFAULT ((void *)-1)

struct ibv_parent_domain_init_attr {
	struct ibv_pd *pd; /* referance to a protection domain object, can't be NULL */
	struct ibv_td *td; /* referance to a thread domain object, or NULL */
	uint32_t comp_mask;
	void *(*alloc)(struct ibv_pd *pd, void *pd_context, size_t size,
		       size_t alignment, uint64_t resource_type);
	void (*free)(struct ibv_pd *pd, void *pd_context, void *ptr,
		     uint64_t resource_type);
	void *pd_context;
};

struct ibv_counters_init_attr {
	uint32_t	comp_mask;
};

struct ibv_counters {
	struct ibv_context	*context;
};

enum ibv_counter_description {
	IBV_COUNTER_PACKETS,
	IBV_COUNTER_BYTES,
};

struct ibv_counter_attach_attr {
	enum ibv_counter_description counter_desc;
	uint32_t index; /* Desired location index of the counter at the counters object */
	uint32_t comp_mask;
};

enum ibv_read_counters_flags {
	IBV_READ_COUNTERS_ATTR_PREFER_CACHED = 1 << 0,
};

enum ibv_values_mask {
	IBV_VALUES_MASK_RAW_CLOCK	= 1 << 0,
	IBV_VALUES_MASK_RESERVED	= 1 << 1
};

struct ibv_values_ex {
	uint32_t	comp_mask;
	struct timespec raw_clock;
};

struct verbs_context {
	/*  "grows up" - new fields go here */
	int (*query_port)(struct ibv_context *context, uint8_t port_num,
			  struct ibv_port_attr *port_attr,
			  size_t port_attr_len);
	int (*advise_mr)(struct ibv_pd *pd,
			 enum ibv_advise_mr_advice advice,
			 uint32_t flags,
			 struct ibv_sge *sg_list,
			 uint32_t num_sges);
	struct ibv_mr *(*alloc_null_mr)(struct ibv_pd *pd);
	int (*read_counters)(struct ibv_counters *counters,
			     uint64_t *counters_value,
			     uint32_t ncounters,
			     uint32_t flags);
	int (*attach_counters_point_flow)(struct ibv_counters *counters,
					  struct ibv_counter_attach_attr *attr,
					  struct ibv_flow *flow);
	struct ibv_counters *(*create_counters)(struct ibv_context *context,
						struct ibv_counters_init_attr *init_attr);
	int (*destroy_counters)(struct ibv_counters *counters);
	struct ibv_mr *(*reg_dm_mr)(struct ibv_pd *pd, struct ibv_dm *dm,
				    uint64_t dm_offset, size_t length,
				    unsigned int access);
	struct ibv_dm *(*alloc_dm)(struct ibv_context *context,
				   struct ibv_alloc_dm_attr *attr);
	int (*free_dm)(struct ibv_dm *dm);
	int (*modify_flow_action_esp)(struct ibv_flow_action *action,
				      struct ibv_flow_action_esp_attr *attr);
	int (*destroy_flow_action)(struct ibv_flow_action *action);
	struct ibv_flow_action *(*create_flow_action_esp)(struct ibv_context *context,
							  struct ibv_flow_action_esp_attr *attr);
	int (*modify_qp_rate_limit)(struct ibv_qp *qp,
				    struct ibv_qp_rate_limit_attr *attr);
	struct ibv_pd *(*alloc_parent_domain)(struct ibv_context *context,
					      struct ibv_parent_domain_init_attr *attr);
	int (*dealloc_td)(struct ibv_td *td);
	struct ibv_td *(*alloc_td)(struct ibv_context *context, struct ibv_td_init_attr *init_attr);
	int (*modify_cq)(struct ibv_cq *cq, struct ibv_modify_cq_attr *attr);
	int (*post_srq_ops)(struct ibv_srq *srq,
			    struct ibv_ops_wr *op,
			    struct ibv_ops_wr **bad_op);
	int (*destroy_rwq_ind_table)(struct ibv_rwq_ind_table *rwq_ind_table);
	struct ibv_rwq_ind_table *(*create_rwq_ind_table)(struct ibv_context *context,
							  struct ibv_rwq_ind_table_init_attr *init_attr);
	int (*destroy_wq)(struct ibv_wq *wq);
	int (*modify_wq)(struct ibv_wq *wq, struct ibv_wq_attr *wq_attr);
	struct ibv_wq * (*create_wq)(struct ibv_context *context,
				     struct ibv_wq_init_attr *wq_init_attr);
	int (*query_rt_values)(struct ibv_context *context,
			       struct ibv_values_ex *values);
	struct ibv_cq_ex *(*create_cq_ex)(struct ibv_context *context,
					  struct ibv_cq_init_attr_ex *init_attr);
	struct verbs_ex_private *priv;
	int (*query_device_ex)(struct ibv_context *context,
			       const struct ibv_query_device_ex_input *input,
			       struct ibv_device_attr_ex *attr,
			       size_t attr_size);
	int (*ibv_destroy_flow) (struct ibv_flow *flow);
	void (*ABI_placeholder2) (void); /* DO NOT COPY THIS GARBAGE */
	struct ibv_flow * (*ibv_create_flow) (struct ibv_qp *qp,
					      struct ibv_flow_attr *flow_attr);
	void (*ABI_placeholder1) (void); /* DO NOT COPY THIS GARBAGE */
	struct ibv_qp *(*open_qp)(struct ibv_context *context,
			struct ibv_qp_open_attr *attr);
	struct ibv_qp *(*create_qp_ex)(struct ibv_context *context,
			struct ibv_qp_init_attr_ex *qp_init_attr_ex);
	int (*get_srq_num)(struct ibv_srq *srq, uint32_t *srq_num);
	struct ibv_srq *	(*create_srq_ex)(struct ibv_context *context,
						 struct ibv_srq_init_attr_ex *srq_init_attr_ex);
	struct ibv_xrcd *	(*open_xrcd)(struct ibv_context *context,
					     struct ibv_xrcd_init_attr *xrcd_init_attr);
	int			(*close_xrcd)(struct ibv_xrcd *xrcd);
	uint64_t _ABI_placeholder3;
	size_t   sz;			/* Must be immediately before struct ibv_context */
	struct ibv_context context;	/* Must be last field in the struct */
};

static inline struct verbs_context *verbs_get_ctx(struct ibv_context *ctx)
{
	if (ctx->abi_compat != __VERBS_ABI_IS_EXTENDED)
		return NULL;

	/* open code container_of to not pollute the global namespace */
	return (struct verbs_context *)(((uint8_t *)ctx) -
					offsetof(struct verbs_context,
						 context));
}

#define verbs_get_ctx_op(ctx, op) ({ \
	struct verbs_context *__vctx = verbs_get_ctx(ctx); \
	(!__vctx || (__vctx->sz < sizeof(*__vctx) - offsetof(struct verbs_context, op)) || \
	 !__vctx->op) ? NULL : __vctx; })

/**
 * ibv_get_device_list - Get list of IB devices currently available
 * @num_devices: optional.  if non-NULL, set to the number of devices
 * returned in the array.
 *
 * Return a NULL-terminated array of IB devices.  The array can be
 * released with ibv_free_device_list().
 */
struct ibv_device **ibv_get_device_list(int *num_devices);

/*
 * When statically linking the user can set RDMA_STATIC_PROVIDERS to a comma
 * separated list of provider names to include in the static link, and this
 * machinery will cause those providers to be included statically.
 *
 * Linking will fail if this is set for dynamic linking.
 */
#ifdef RDMA_STATIC_PROVIDERS
#define _RDMA_STATIC_PREFIX_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11,     \
			     _12, _13, _14, _15, _16, _17, ...)                \
	&verbs_provider_##_1, &verbs_provider_##_2, &verbs_provider_##_3,      \
		&verbs_provider_##_4, &verbs_provider_##_5,                    \
		&verbs_provider_##_6, &verbs_provider_##_7,                    \
		&verbs_provider_##_8, &verbs_provider_##_9,                    \
		&verbs_provider_##_10, &verbs_provider_##_11,                  \
		&verbs_provider_##_12, &verbs_provider_##_13,                  \
		&verbs_provider_##_14, &verbs_provider_##_15,                  \
		&verbs_provider_##_16, &verbs_provider_##_17
#define _RDMA_STATIC_PREFIX(arg)                                               \
	_RDMA_STATIC_PREFIX_(arg, none, none, none, none, none, none, none,    \
			     none, none, none, none, none, none, none, none,   \
			     none)

struct verbs_devices_ops;
extern const struct verbs_device_ops verbs_provider_bnxt_re;
extern const struct verbs_device_ops verbs_provider_cxgb4;
extern const struct verbs_device_ops verbs_provider_efa;
extern const struct verbs_device_ops verbs_provider_hfi1verbs;
extern const struct verbs_device_ops verbs_provider_hns;
extern const struct verbs_device_ops verbs_provider_i40iw;
extern const struct verbs_device_ops verbs_provider_ipathverbs;
extern const struct verbs_device_ops verbs_provider_mlx4;
extern const struct verbs_device_ops verbs_provider_mlx5;
extern const struct verbs_device_ops verbs_provider_mthca;
extern const struct verbs_device_ops verbs_provider_ocrdma;
extern const struct verbs_device_ops verbs_provider_qedr;
extern const struct verbs_device_ops verbs_provider_rxe;
extern const struct verbs_device_ops verbs_provider_siw;
extern const struct verbs_device_ops verbs_provider_vmw_pvrdma;
extern const struct verbs_device_ops verbs_provider_all;
extern const struct verbs_device_ops verbs_provider_none;
void ibv_static_providers(void *unused, ...);

static inline struct ibv_device **__ibv_get_device_list(int *num_devices)
{
	ibv_static_providers(NULL, _RDMA_STATIC_PREFIX(RDMA_STATIC_PROVIDERS),
			     NULL);
	return ibv_get_device_list(num_devices);
}
#define ibv_get_device_list(num_devices) __ibv_get_device_list(num_devices)
#endif

/**
 * ibv_free_device_list - Free list from ibv_get_device_list()
 *
 * Free an array of devices returned from ibv_get_device_list().  Once
 * the array is freed, pointers to devices that were not opened with
 * ibv_open_device() are no longer valid.  Client code must open all
 * devices it intends to use before calling ibv_free_device_list().
 */
void ibv_free_device_list(struct ibv_device **list);

/**
 * ibv_get_device_name - Return kernel device name
 */
const char *ibv_get_device_name(struct ibv_device *device);

/**
 * ibv_get_device_index - Return kernel device index
 *
 * Available for the kernel with support of IB device query
 * over netlink interface. For the unsupported kernels, the
 * relevant -1 will be returned.
 */
int ibv_get_device_index(struct ibv_device *device);

/**
 * ibv_get_device_guid - Return device's node GUID
 */
__be64 ibv_get_device_guid(struct ibv_device *device);

/**
 * ibv_open_device - Initialize device for use
 */
struct ibv_context *ibv_open_device(struct ibv_device *device);

/**
 * ibv_close_device - Release device
 */
int ibv_close_device(struct ibv_context *context);

/**
 * ibv_import_device - Import device
 */
struct ibv_context *ibv_import_device(int cmd_fd);

/**
 * ibv_import_pd - Import a protetion domain
 */
struct ibv_pd *ibv_import_pd(struct ibv_context *context,
			     uint32_t pd_handle);

/**
 * ibv_unimport_pd - Unimport a protetion domain
 */
void ibv_unimport_pd(struct ibv_pd *pd);

/**
 * ibv_import_mr - Import a memory region
 */
struct ibv_mr *ibv_import_mr(struct ibv_pd *pd, uint32_t mr_handle);

/**
 * ibv_unimport_mr - Unimport a memory region
 */
void ibv_unimport_mr(struct ibv_mr *mr);

/**
 * ibv_get_async_event - Get next async event
 * @event: Pointer to use to return async event
 *
 * All async events returned by ibv_get_async_event() must eventually
 * be acknowledged with ibv_ack_async_event().
 */
int ibv_get_async_event(struct ibv_context *context,
			struct ibv_async_event *event);

/**
 * ibv_ack_async_event - Acknowledge an async event
 * @event: Event to be acknowledged.
 *
 * All async events which are returned by ibv_get_async_event() must
 * be acknowledged.  To avoid races, destroying an object (CQ, SRQ or
 * QP) will wait for all affiliated events to be acknowledged, so
 * there should be a one-to-one correspondence between acks and
 * successful gets.
 */
void ibv_ack_async_event(struct ibv_async_event *event);

/**
 * ibv_query_device - Get device properties
 */
int ibv_query_device(struct ibv_context *context,
		     struct ibv_device_attr *device_attr);

/**
 * ibv_query_port - Get port properties
 */
int ibv_query_port(struct ibv_context *context, uint8_t port_num,
		   struct _compat_ibv_port_attr *port_attr);

static inline int ___ibv_query_port(struct ibv_context *context,
				    uint8_t port_num,
				    struct ibv_port_attr *port_attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(context, query_port);

	if (!vctx) {
		int rc;

		memset(port_attr, 0, sizeof(*port_attr));

		rc = ibv_query_port(context, port_num,
				    (struct _compat_ibv_port_attr *)port_attr);
		return rc;
	}

	return vctx->query_port(context, port_num, port_attr,
				sizeof(*port_attr));
}

#define ibv_query_port(context, port_num, port_attr) \
	___ibv_query_port(context, port_num, port_attr)

/**
 * ibv_query_gid - Get a GID table entry
 */
int ibv_query_gid(struct ibv_context *context, uint8_t port_num,
		  int index, union ibv_gid *gid);

int _ibv_query_gid_ex(struct ibv_context *context, uint32_t port_num,
		     uint32_t gid_index, struct ibv_gid_entry *entry,
		     uint32_t flags, size_t entry_size);

/**
 * ibv_query_gid_ex - Read a GID table entry
 */
static inline int ibv_query_gid_ex(struct ibv_context *context,
				   uint32_t port_num, uint32_t gid_index,
				   struct ibv_gid_entry *entry, uint32_t flags)
{
	return _ibv_query_gid_ex(context, port_num, gid_index, entry, flags,
				 sizeof(*entry));
}

ssize_t _ibv_query_gid_table(struct ibv_context *context,
			     struct ibv_gid_entry *entries, size_t max_entries,
			     uint32_t flags, size_t entry_size);

/*
 * ibv_query_gid_table - Get all valid GID table entries
 */
static inline ssize_t ibv_query_gid_table(struct ibv_context *context,
					  struct ibv_gid_entry *entries,
					  size_t max_entries, uint32_t flags)
{
	return _ibv_query_gid_table(context, entries, max_entries, flags,
				    sizeof(*entries));
}

/**
 * ibv_query_pkey - Get a P_Key table entry
 */
int ibv_query_pkey(struct ibv_context *context, uint8_t port_num,
		   int index, __be16 *pkey);

/**
 * ibv_get_pkey_index - Translate a P_Key into a P_Key index
 */
int ibv_get_pkey_index(struct ibv_context *context, uint8_t port_num,
		       __be16 pkey);

/**
 * ibv_alloc_pd - Allocate a protection domain
 */
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);

/**
 * ibv_dealloc_pd - Free a protection domain
 */
int ibv_dealloc_pd(struct ibv_pd *pd);

static inline struct ibv_flow *ibv_create_flow(struct ibv_qp *qp,
					       struct ibv_flow_attr *flow)
{
	struct verbs_context *vctx = verbs_get_ctx_op(qp->context,
						      ibv_create_flow);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->ibv_create_flow(qp, flow);
}

static inline int ibv_destroy_flow(struct ibv_flow *flow_id)
{
	struct verbs_context *vctx = verbs_get_ctx_op(flow_id->context,
						      ibv_destroy_flow);
	if (!vctx)
		return EOPNOTSUPP;
	return vctx->ibv_destroy_flow(flow_id);
}

static inline struct ibv_flow_action *
ibv_create_flow_action_esp(struct ibv_context *ctx,
			   struct ibv_flow_action_esp_attr *esp)
{
	struct verbs_context *vctx = verbs_get_ctx_op(ctx,
						      create_flow_action_esp);

	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->create_flow_action_esp(ctx, esp);
}

static inline int
ibv_modify_flow_action_esp(struct ibv_flow_action *action,
			   struct ibv_flow_action_esp_attr *esp)
{
	struct verbs_context *vctx = verbs_get_ctx_op(action->context,
						      modify_flow_action_esp);

	if (!vctx)
		return EOPNOTSUPP;

	return vctx->modify_flow_action_esp(action, esp);
}

static inline int ibv_destroy_flow_action(struct ibv_flow_action *action)
{
	struct verbs_context *vctx = verbs_get_ctx_op(action->context,
						      destroy_flow_action);

	if (!vctx)
		return EOPNOTSUPP;

	return vctx->destroy_flow_action(action);
}

/**
 * ibv_open_xrcd - Open an extended connection domain
 */
static inline struct ibv_xrcd *
ibv_open_xrcd(struct ibv_context *context, struct ibv_xrcd_init_attr *xrcd_init_attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(context, open_xrcd);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}
	return vctx->open_xrcd(context, xrcd_init_attr);
}

/**
 * ibv_close_xrcd - Close an extended connection domain
 */
static inline int ibv_close_xrcd(struct ibv_xrcd *xrcd)
{
	struct verbs_context *vctx = verbs_get_ctx(xrcd->context);
	return vctx->close_xrcd(xrcd);
}

/**
 * ibv_reg_mr_iova2 - Register memory region with a virtual offset address
 *
 * This version will be called if ibv_reg_mr or ibv_reg_mr_iova were called
 * with at least one potential access flag from the IBV_OPTIONAL_ACCESS_RANGE
 * flags range The optional access flags will be masked if running over kernel
 * that does not support passing them.
 */
struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length,
				uint64_t iova, unsigned int access);

/**
 * ibv_reg_mr - Register a memory region
 */
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
			  int access);
/* use new ibv_reg_mr version only if access flags that require it are used */
__attribute__((__always_inline__)) static inline struct ibv_mr *
__ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, unsigned int access,
	     int is_access_const)
{
	if (is_access_const && (access & IBV_ACCESS_OPTIONAL_RANGE) == 0)
		return ibv_reg_mr(pd, addr, length, access);
	else
		return ibv_reg_mr_iova2(pd, addr, length, (uintptr_t)addr,
					access);
}

#define ibv_reg_mr(pd, addr, length, access)                                   \
	__ibv_reg_mr(pd, addr, length, access,                                 \
		     __builtin_constant_p(				       \
			     ((access) & IBV_ACCESS_OPTIONAL_RANGE) == 0))

/**
 * ibv_reg_mr_iova - Register a memory region with a virtual offset
 * address
 */
struct ibv_mr *ibv_reg_mr_iova(struct ibv_pd *pd, void *addr, size_t length,
			       uint64_t iova, int access);
/* use new ibv_reg_mr version only if access flags that require it are used */
__attribute__((__always_inline__)) static inline struct ibv_mr *
__ibv_reg_mr_iova(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova,
		  unsigned int access, int is_access_const)
{
	if (is_access_const && (access & IBV_ACCESS_OPTIONAL_RANGE) == 0)
		return ibv_reg_mr_iova(pd, addr, length, iova, access);
	else
		return ibv_reg_mr_iova2(pd, addr, length, iova, access);
}

#define ibv_reg_mr_iova(pd, addr, length, iova, access)                        \
	__ibv_reg_mr_iova(pd, addr, length, iova, access,                      \
			  __builtin_constant_p(                                \
				  ((access) & IBV_ACCESS_OPTIONAL_RANGE) == 0))

enum ibv_rereg_mr_err_code {
	/* Old MR is valid, invalid input */
	IBV_REREG_MR_ERR_INPUT = -1,
	/* Old MR is valid, failed via don't fork on new address range */
	IBV_REREG_MR_ERR_DONT_FORK_NEW = -2,
	/* New MR is valid, failed via do fork on old address range */
	IBV_REREG_MR_ERR_DO_FORK_OLD = -3,
	/* MR shouldn't be used, command error */
	IBV_REREG_MR_ERR_CMD = -4,
	/* MR shouldn't be used, command error, invalid fork state on new address range */
	IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW = -5,
};

/**
 * ibv_rereg_mr - Re-Register a memory region
 */
int ibv_rereg_mr(struct ibv_mr *mr, int flags,
		 struct ibv_pd *pd, void *addr,
		 size_t length, int access);
/**
 * ibv_dereg_mr - Deregister a memory region
 */
int ibv_dereg_mr(struct ibv_mr *mr);

/**
 * ibv_alloc_mw - Allocate a memory window
 */
static inline struct ibv_mw *ibv_alloc_mw(struct ibv_pd *pd,
					  enum ibv_mw_type type)
{
	struct ibv_mw *mw;

	if (!pd->context->ops.alloc_mw) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	mw = pd->context->ops.alloc_mw(pd, type);
	return mw;
}

/**
 * ibv_dealloc_mw - Free a memory window
 */
static inline int ibv_dealloc_mw(struct ibv_mw *mw)
{
	return mw->context->ops.dealloc_mw(mw);
}

/**
 * ibv_inc_rkey - Increase the 8 lsb in the given rkey
 */
static inline uint32_t ibv_inc_rkey(uint32_t rkey)
{
	const uint32_t mask = 0x000000ff;
	uint8_t newtag = (uint8_t)((rkey + 1) & mask);

	return (rkey & ~mask) | newtag;
}

/**
 * ibv_bind_mw - Bind a memory window to a region
 */
static inline int ibv_bind_mw(struct ibv_qp *qp, struct ibv_mw *mw,
			      struct ibv_mw_bind *mw_bind)
{
	if (mw->type != IBV_MW_TYPE_1)
		return EINVAL;

	return mw->context->ops.bind_mw(qp, mw, mw_bind);
}

/**
 * ibv_create_comp_channel - Create a completion event channel
 */
struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context);

/**
 * ibv_destroy_comp_channel - Destroy a completion event channel
 */
int ibv_destroy_comp_channel(struct ibv_comp_channel *channel);

/**
 * ibv_advise_mr - Gives advice about an address range in MRs
 * @pd - protection domain of all MRs for which the advice is for
 * @advice - type of advice
 * @flags - advice modifiers
 * @sg_list - an array of memory ranges
 * @num_sge - number of elements in the array
 */
static inline int ibv_advise_mr(struct ibv_pd *pd,
				enum ibv_advise_mr_advice advice,
				uint32_t flags,
				struct ibv_sge *sg_list,
				uint32_t num_sge)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(pd->context, advise_mr);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->advise_mr(pd, advice, flags, sg_list, num_sge);
}

/**
 * ibv_alloc_dm - Allocate device memory
 * @context - Context DM will be attached to
 * @attr - Attributes to allocate the DM with
 */
static inline
struct ibv_dm *ibv_alloc_dm(struct ibv_context *context,
			    struct ibv_alloc_dm_attr *attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(context, alloc_dm);

	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->alloc_dm(context, attr);
}

/**
 * ibv_free_dm - Free device allocated memory
 * @dm - The DM to free
 */
static inline
int ibv_free_dm(struct ibv_dm *dm)
{
	struct verbs_context *vctx = verbs_get_ctx_op(dm->context, free_dm);

	if (!vctx)
		return EOPNOTSUPP;

	return vctx->free_dm(dm);
}

/**
 * ibv_memcpy_to/from_dm - copy to/from device allocated memory
 * @dm - The DM to copy to/from
 * @dm_offset - Offset in bytes from beginning of DM to start copy to/form
 * @host_addr - Host memory address to copy to/from
 * @length - Number of bytes to copy
 */
static inline
int ibv_memcpy_to_dm(struct ibv_dm *dm, uint64_t dm_offset,
		     const void *host_addr, size_t length)
{
	return dm->memcpy_to_dm(dm, dm_offset, host_addr, length);
}

static inline
int ibv_memcpy_from_dm(void *host_addr, struct ibv_dm *dm,
		       uint64_t dm_offset, size_t length)
{
	return dm->memcpy_from_dm(host_addr, dm, dm_offset, length);
}

/*
 * ibv_alloc_null_mr - Allocate a null memory region.
 * @pd - The protection domain associated with the MR.
 */
static inline
struct ibv_mr *ibv_alloc_null_mr(struct ibv_pd *pd)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(pd->context, alloc_null_mr);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->alloc_null_mr(pd);
}

/**
 * ibv_reg_dm_mr - Register device memory as a memory region
 * @pd - The PD to associated this MR with
 * @dm - The DM to register
 * @dm_offset - Offset in bytes from beginning of DM to start registration from
 * @length - Number of bytes to register
 * @access - memory region access flags
 */
static inline
struct ibv_mr *ibv_reg_dm_mr(struct ibv_pd *pd, struct ibv_dm *dm,
			     uint64_t dm_offset,
			     size_t length, unsigned int access)
{
	struct verbs_context *vctx = verbs_get_ctx_op(pd->context, reg_dm_mr);

	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->reg_dm_mr(pd, dm, dm_offset, length, access);
}

/**
 * ibv_create_cq - Create a completion queue
 * @context - Context CQ will be attached to
 * @cqe - Minimum number of entries required for CQ
 * @cq_context - Consumer-supplied context returned for completion events
 * @channel - Completion channel where completion events will be queued.
 *     May be NULL if completion events will not be used.
 * @comp_vector - Completion vector used to signal completion events.
 *     Must be >= 0 and < context->num_comp_vectors.
 */
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
			     void *cq_context,
			     struct ibv_comp_channel *channel,
			     int comp_vector);

/**
 * ibv_create_cq_ex - Create a completion queue
 * @context - Context CQ will be attached to
 * @cq_attr - Attributes to create the CQ with
 */
static inline
struct ibv_cq_ex *ibv_create_cq_ex(struct ibv_context *context,
				   struct ibv_cq_init_attr_ex *cq_attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(context, create_cq_ex);

	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->create_cq_ex(context, cq_attr);
}

/**
 * ibv_resize_cq - Modifies the capacity of the CQ.
 * @cq: The CQ to resize.
 * @cqe: The minimum size of the CQ.
 *
 * Users can examine the cq structure to determine the actual CQ size.
 */
int ibv_resize_cq(struct ibv_cq *cq, int cqe);

/**
 * ibv_destroy_cq - Destroy a completion queue
 */
int ibv_destroy_cq(struct ibv_cq *cq);

/**
 * ibv_get_cq_event - Read next CQ event
 * @channel: Channel to get next event from.
 * @cq: Used to return pointer to CQ.
 * @cq_context: Used to return consumer-supplied CQ context.
 *
 * All completion events returned by ibv_get_cq_event() must
 * eventually be acknowledged with ibv_ack_cq_events().
 */
int ibv_get_cq_event(struct ibv_comp_channel *channel,
		     struct ibv_cq **cq, void **cq_context);

/**
 * ibv_ack_cq_events - Acknowledge CQ completion events
 * @cq: CQ to acknowledge events for
 * @nevents: Number of events to acknowledge.
 *
 * All completion events which are returned by ibv_get_cq_event() must
 * be acknowledged.  To avoid races, ibv_destroy_cq() will wait for
 * all completion events to be acknowledged, so there should be a
 * one-to-one correspondence between acks and successful gets.  An
 * application may accumulate multiple completion events and
 * acknowledge them in a single call to ibv_ack_cq_events() by passing
 * the number of events to ack in @nevents.
 */
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents);

/**
 * ibv_poll_cq - Poll a CQ for work completions
 * @cq:the CQ being polled
 * @num_entries:maximum number of completions to return
 * @wc:array of at least @num_entries of &struct ibv_wc where completions
 *   will be returned
 *
 * Poll a CQ for (possibly multiple) completions.  If the return value
 * is < 0, an error occurred.  If the return value is >= 0, it is the
 * number of completions returned.  If the return value is
 * non-negative and strictly less than num_entries, then the CQ was
 * emptied.
 */
static inline int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc)
{
	return cq->context->ops.poll_cq(cq, num_entries, wc);
}

/**
 * ibv_req_notify_cq - Request completion notification on a CQ.  An
 *   event will be added to the completion channel associated with the
 *   CQ when an entry is added to the CQ.
 * @cq: The completion queue to request notification for.
 * @solicited_only: If non-zero, an event will be generated only for
 *   the next solicited CQ entry.  If zero, any CQ entry, solicited or
 *   not, will generate an event.
 */
static inline int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only)
{
	return cq->context->ops.req_notify_cq(cq, solicited_only);
}

static inline int ibv_modify_cq(struct ibv_cq *cq, struct ibv_modify_cq_attr *attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(cq->context, modify_cq);

	if (!vctx)
		return EOPNOTSUPP;

	return vctx->modify_cq(cq, attr);
}
/**
 * ibv_create_srq - Creates a SRQ associated with the specified protection
 *   domain.
 * @pd: The protection domain associated with the SRQ.
 * @srq_init_attr: A list of initial attributes required to create the SRQ.
 *
 * srq_attr->max_wr and srq_attr->max_sge are read the determine the
 * requested size of the SRQ, and set to the actual values allocated
 * on return.  If ibv_create_srq() succeeds, then max_wr and max_sge
 * will always be at least as large as the requested values.
 */
struct ibv_srq *ibv_create_srq(struct ibv_pd *pd,
			       struct ibv_srq_init_attr *srq_init_attr);

static inline struct ibv_srq *
ibv_create_srq_ex(struct ibv_context *context,
		  struct ibv_srq_init_attr_ex *srq_init_attr_ex)
{
	struct verbs_context *vctx;
	uint32_t mask = srq_init_attr_ex->comp_mask;

	if (!(mask & ~(IBV_SRQ_INIT_ATTR_PD | IBV_SRQ_INIT_ATTR_TYPE)) &&
	    (mask & IBV_SRQ_INIT_ATTR_PD) &&
	    (!(mask & IBV_SRQ_INIT_ATTR_TYPE) ||
	     (srq_init_attr_ex->srq_type == IBV_SRQT_BASIC)))
		return ibv_create_srq(srq_init_attr_ex->pd,
				      (struct ibv_srq_init_attr *)srq_init_attr_ex);

	vctx = verbs_get_ctx_op(context, create_srq_ex);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}
	return vctx->create_srq_ex(context, srq_init_attr_ex);
}

/**
 * ibv_modify_srq - Modifies the attributes for the specified SRQ.
 * @srq: The SRQ to modify.
 * @srq_attr: On input, specifies the SRQ attributes to modify.  On output,
 *   the current values of selected SRQ attributes are returned.
 * @srq_attr_mask: A bit-mask used to specify which attributes of the SRQ
 *   are being modified.
 *
 * The mask may contain IBV_SRQ_MAX_WR to resize the SRQ and/or
 * IBV_SRQ_LIMIT to set the SRQ's limit and request notification when
 * the number of receives queued drops below the limit.
 */
int ibv_modify_srq(struct ibv_srq *srq,
		   struct ibv_srq_attr *srq_attr,
		   int srq_attr_mask);

/**
 * ibv_query_srq - Returns the attribute list and current values for the
 *   specified SRQ.
 * @srq: The SRQ to query.
 * @srq_attr: The attributes of the specified SRQ.
 */
int ibv_query_srq(struct ibv_srq *srq, struct ibv_srq_attr *srq_attr);

static inline int ibv_get_srq_num(struct ibv_srq *srq, uint32_t *srq_num)
{
	struct verbs_context *vctx = verbs_get_ctx_op(srq->context, get_srq_num);

	if (!vctx)
		return EOPNOTSUPP;

	return vctx->get_srq_num(srq, srq_num);
}

/**
 * ibv_destroy_srq - Destroys the specified SRQ.
 * @srq: The SRQ to destroy.
 */
int ibv_destroy_srq(struct ibv_srq *srq);

/**
 * ibv_post_srq_recv - Posts a list of work requests to the specified SRQ.
 * @srq: The SRQ to post the work request on.
 * @recv_wr: A list of work requests to post on the receive queue.
 * @bad_recv_wr: On an immediate failure, this parameter will reference
 *   the work request that failed to be posted on the QP.
 */
static inline int ibv_post_srq_recv(struct ibv_srq *srq,
				    struct ibv_recv_wr *recv_wr,
				    struct ibv_recv_wr **bad_recv_wr)
{
	return srq->context->ops.post_srq_recv(srq, recv_wr, bad_recv_wr);
}

static inline int ibv_post_srq_ops(struct ibv_srq *srq,
				   struct ibv_ops_wr *op,
				   struct ibv_ops_wr **bad_op)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(srq->context, post_srq_ops);
	if (!vctx) {
		*bad_op = op;
		return EOPNOTSUPP;
	}
	return vctx->post_srq_ops(srq, op, bad_op);
}

/**
 * ibv_create_qp - Create a queue pair.
 */
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
			     struct ibv_qp_init_attr *qp_init_attr);

static inline struct ibv_qp *
ibv_create_qp_ex(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_init_attr_ex)
{
	struct verbs_context *vctx;
	uint32_t mask = qp_init_attr_ex->comp_mask;

	if (mask == IBV_QP_INIT_ATTR_PD)
		return ibv_create_qp(qp_init_attr_ex->pd,
				     (struct ibv_qp_init_attr *)qp_init_attr_ex);

	vctx = verbs_get_ctx_op(context, create_qp_ex);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}
	return vctx->create_qp_ex(context, qp_init_attr_ex);
}

/**
 * ibv_alloc_td - Allocate a thread domain
 */
static inline struct ibv_td *ibv_alloc_td(struct ibv_context *context,
					  struct ibv_td_init_attr *init_attr)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(context, alloc_td);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->alloc_td(context, init_attr);
}

/**
 * ibv_dealloc_td - Free a thread domain
 */
static inline int ibv_dealloc_td(struct ibv_td *td)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(td->context, dealloc_td);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->dealloc_td(td);
}

/**
 * ibv_alloc_parent_domain - Allocate a parent domain
 */
static inline struct ibv_pd *
ibv_alloc_parent_domain(struct ibv_context *context,
			struct ibv_parent_domain_init_attr *attr)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(context, alloc_parent_domain);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->alloc_parent_domain(context, attr);
}

/**
 * ibv_query_rt_values_ex - Get current real time @values of a device.
 * @values - in/out - defines the attributes we need to query/queried.
 * (Or's bits of enum ibv_values_mask on values->comp_mask field)
 */
static inline int
ibv_query_rt_values_ex(struct ibv_context *context,
		       struct ibv_values_ex *values)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(context, query_rt_values);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->query_rt_values(context, values);
}

/**
 * ibv_query_device_ex - Get extended device properties
 */
static inline int
ibv_query_device_ex(struct ibv_context *context,
		    const struct ibv_query_device_ex_input *input,
		    struct ibv_device_attr_ex *attr)
{
	struct verbs_context *vctx;
	int ret;

	if (input && input->comp_mask)
		return EINVAL;

	vctx = verbs_get_ctx_op(context, query_device_ex);
	if (!vctx)
		goto legacy;

	ret = vctx->query_device_ex(context, input, attr, sizeof(*attr));
	if (ret == EOPNOTSUPP || ret == ENOSYS)
		goto legacy;

	return ret;

legacy:
	memset(attr, 0, sizeof(*attr));
	ret = ibv_query_device(context, &attr->orig_attr);

	return ret;
}

/**
 * ibv_open_qp - Open a shareable queue pair.
 */
static inline struct ibv_qp *
ibv_open_qp(struct ibv_context *context, struct ibv_qp_open_attr *qp_open_attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(context, open_qp);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}
	return vctx->open_qp(context, qp_open_attr);
}

/**
 * ibv_modify_qp - Modify a queue pair.
 */
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
		  int attr_mask);

/**
 * ibv_modify_qp_rate_limit - Modify a queue pair rate limit values
 * @qp - QP object to modify
 * @attr - Attributes to configure the rate limiting values of the QP
 */
static inline int
ibv_modify_qp_rate_limit(struct ibv_qp *qp,
			 struct ibv_qp_rate_limit_attr *attr)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(qp->context, modify_qp_rate_limit);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->modify_qp_rate_limit(qp, attr);
}

/**
 * ibv_query_qp - Returns the attribute list and current values for the
 *   specified QP.
 * @qp: The QP to query.
 * @attr: The attributes of the specified QP.
 * @attr_mask: A bit-mask used to select specific attributes to query.
 * @init_attr: Additional attributes of the selected QP.
 *
 * The qp_attr_mask may be used to limit the query to gathering only the
 * selected attributes.
 */
int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
		 int attr_mask,
		 struct ibv_qp_init_attr *init_attr);

/**
 * ibv_destroy_qp - Destroy a queue pair.
 */
int ibv_destroy_qp(struct ibv_qp *qp);

/*
 * ibv_create_wq - Creates a WQ associated with the specified protection
 * domain.
 * @context: ibv_context.
 * @wq_init_attr: A list of initial attributes required to create the
 * WQ. If WQ creation succeeds, then the attributes are updated to
 * the actual capabilities of the created WQ.
 *
 * wq_init_attr->max_wr and wq_init_attr->max_sge determine
 * the requested size of the WQ, and set to the actual values allocated
 * on return.
 * If ibv_create_wq() succeeds, then max_wr and max_sge will always be
 * at least as large as the requested values.
 *
 * Return Value
 * ibv_create_wq() returns a pointer to the created WQ, or NULL if the request
 * fails.
 */
static inline struct ibv_wq *ibv_create_wq(struct ibv_context *context,
					   struct ibv_wq_init_attr *wq_init_attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(context, create_wq);
	struct ibv_wq *wq;

	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	wq = vctx->create_wq(context, wq_init_attr);
	if (wq) {
		wq->wq_context = wq_init_attr->wq_context;
		wq->events_completed = 0;
		pthread_mutex_init(&wq->mutex, NULL);
		pthread_cond_init(&wq->cond, NULL);
	}

	return wq;
}

/*
 * ibv_modify_wq - Modifies the attributes for the specified WQ.
 * @wq: The WQ to modify.
 * @wq_attr: On input, specifies the WQ attributes to modify.
 *    wq_attr->attr_mask: A bit-mask used to specify which attributes of the WQ
 *    are being modified.
 * On output, the current values of selected WQ attributes are returned.
 *
 * Return Value
 * ibv_modify_wq() returns 0 on success, or the value of errno
 * on failure (which indicates the failure reason).
 *
*/
static inline int ibv_modify_wq(struct ibv_wq *wq, struct ibv_wq_attr *wq_attr)
{
	struct verbs_context *vctx = verbs_get_ctx_op(wq->context, modify_wq);

	if (!vctx)
		return EOPNOTSUPP;

	return vctx->modify_wq(wq, wq_attr);
}

/*
 * ibv_destroy_wq - Destroys the specified WQ.
 * @ibv_wq: The WQ to destroy.
 * Return Value
 * ibv_destroy_wq() returns 0 on success, or the value of errno
 * on failure (which indicates the failure reason).
*/
static inline int ibv_destroy_wq(struct ibv_wq *wq)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(wq->context, destroy_wq);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->destroy_wq(wq);
}

/*
 * ibv_create_rwq_ind_table - Creates a receive work queue Indirection Table
 * @context: ibv_context.
 * @init_attr: A list of initial attributes required to create the Indirection Table.
 * Return Value
 * ibv_create_rwq_ind_table returns a pointer to the created
 * Indirection Table, or NULL if the request fails.
 */
static inline struct ibv_rwq_ind_table *ibv_create_rwq_ind_table(struct ibv_context *context,
								 struct ibv_rwq_ind_table_init_attr *init_attr)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(context, create_rwq_ind_table);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->create_rwq_ind_table(context, init_attr);
}

/*
 * ibv_destroy_rwq_ind_table - Destroys the specified Indirection Table.
 * @rwq_ind_table: The Indirection Table to destroy.
 * Return Value
 * ibv_destroy_rwq_ind_table() returns 0 on success, or the value of errno
 * on failure (which indicates the failure reason).
*/
static inline int ibv_destroy_rwq_ind_table(struct ibv_rwq_ind_table *rwq_ind_table)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(rwq_ind_table->context, destroy_rwq_ind_table);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->destroy_rwq_ind_table(rwq_ind_table);
}

/**
 * ibv_post_send - Post a list of work requests to a send queue.
 *
 * If IBV_SEND_INLINE flag is set, the data buffers can be reused
 * immediately after the call returns.
 */
static inline int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
				struct ibv_send_wr **bad_wr)
{
	return qp->context->ops.post_send(qp, wr, bad_wr);
}

/**
 * ibv_post_recv - Post a list of work requests to a receive queue.
 */
static inline int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
				struct ibv_recv_wr **bad_wr)
{
	return qp->context->ops.post_recv(qp, wr, bad_wr);
}

/**
 * ibv_create_ah - Create an address handle.
 */
struct ibv_ah *ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr);

/**
 * ibv_init_ah_from_wc - Initializes address handle attributes from a
 *   work completion.
 * @context: Device context on which the received message arrived.
 * @port_num: Port on which the received message arrived.
 * @wc: Work completion associated with the received message.
 * @grh: References the received global route header.  This parameter is
 *   ignored unless the work completion indicates that the GRH is valid.
 * @ah_attr: Returned attributes that can be used when creating an address
 *   handle for replying to the message.
 */
int ibv_init_ah_from_wc(struct ibv_context *context, uint8_t port_num,
			struct ibv_wc *wc, struct ibv_grh *grh,
			struct ibv_ah_attr *ah_attr);

/**
 * ibv_create_ah_from_wc - Creates an address handle associated with the
 *   sender of the specified work completion.
 * @pd: The protection domain associated with the address handle.
 * @wc: Work completion information associated with a received message.
 * @grh: References the received global route header.  This parameter is
 *   ignored unless the work completion indicates that the GRH is valid.
 * @port_num: The outbound port number to associate with the address.
 *
 * The address handle is used to reference a local or global destination
 * in all UD QP post sends.
 */
struct ibv_ah *ibv_create_ah_from_wc(struct ibv_pd *pd, struct ibv_wc *wc,
				     struct ibv_grh *grh, uint8_t port_num);

/**
 * ibv_destroy_ah - Destroy an address handle.
 */
int ibv_destroy_ah(struct ibv_ah *ah);

/**
 * ibv_attach_mcast - Attaches the specified QP to a multicast group.
 * @qp: QP to attach to the multicast group.  The QP must be a UD QP.
 * @gid: Multicast group GID.
 * @lid: Multicast group LID in host byte order.
 *
 * In order to route multicast packets correctly, subnet
 * administration must have created the multicast group and configured
 * the fabric appropriately.  The port associated with the specified
 * QP must also be a member of the multicast group.
 */
int ibv_attach_mcast(struct ibv_qp *qp, const union ibv_gid *gid, uint16_t lid);

/**
 * ibv_detach_mcast - Detaches the specified QP from a multicast group.
 * @qp: QP to detach from the multicast group.
 * @gid: Multicast group GID.
 * @lid: Multicast group LID in host byte order.
 */
int ibv_detach_mcast(struct ibv_qp *qp, const union ibv_gid *gid, uint16_t lid);

/**
 * ibv_fork_init - Prepare data structures so that fork() may be used
 * safely.  If this function is not called or returns a non-zero
 * status, then libibverbs data structures are not fork()-safe and the
 * effect of an application calling fork() is undefined.
 */
int ibv_fork_init(void);

/**
 * ibv_node_type_str - Return string describing node_type enum value
 */
const char *ibv_node_type_str(enum ibv_node_type node_type);

/**
 * ibv_port_state_str - Return string describing port_state enum value
 */
const char *ibv_port_state_str(enum ibv_port_state port_state);

/**
 * ibv_event_type_str - Return string describing event_type enum value
 */
const char *ibv_event_type_str(enum ibv_event_type event);

#define ETHERNET_LL_SIZE 6
int ibv_resolve_eth_l2_from_gid(struct ibv_context *context,
				struct ibv_ah_attr *attr,
				uint8_t eth_mac[ETHERNET_LL_SIZE],
				uint16_t *vid);

static inline int ibv_is_qpt_supported(uint32_t caps, enum ibv_qp_type qpt)
{
	return !!(caps & (1 << qpt));
}

static inline struct ibv_counters *ibv_create_counters(struct ibv_context *context,
						       struct ibv_counters_init_attr *init_attr)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(context, create_counters);
	if (!vctx) {
		errno = EOPNOTSUPP;
		return NULL;
	}

	return vctx->create_counters(context, init_attr);
}

static inline int ibv_destroy_counters(struct ibv_counters *counters)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(counters->context, destroy_counters);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->destroy_counters(counters);
}

static inline int ibv_attach_counters_point_flow(struct ibv_counters *counters,
						 struct ibv_counter_attach_attr *attr,
						 struct ibv_flow *flow)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(counters->context, attach_counters_point_flow);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->attach_counters_point_flow(counters, attr, flow);
}

static inline int ibv_read_counters(struct ibv_counters *counters,
				    uint64_t *counters_value,
				    uint32_t ncounters,
				    uint32_t flags)
{
	struct verbs_context *vctx;

	vctx = verbs_get_ctx_op(counters->context, read_counters);
	if (!vctx)
		return EOPNOTSUPP;

	return vctx->read_counters(counters, counters_value, ncounters, flags);
}

#define IB_ROCE_UDP_ENCAP_VALID_PORT_MIN (0xC000)
#define IB_ROCE_UDP_ENCAP_VALID_PORT_MAX (0xFFFF)
#define IB_GRH_FLOWLABEL_MASK (0x000FFFFF)

static inline uint16_t ibv_flow_label_to_udp_sport(uint32_t fl)
{
	uint32_t fl_low = fl & 0x03FFF, fl_high = fl & 0xFC000;

	fl_low ^= fl_high >> 14;
	return (uint16_t)(fl_low | IB_ROCE_UDP_ENCAP_VALID_PORT_MIN);
}

/**
 * ibv_set_ece - Set ECE options
 */
int ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece);

/**
 * ibv_query_ece - Get accepted ECE options
 */
int ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece);
#ifdef __cplusplus
}
#endif

#  undef __attribute_const


#endif /* INFINIBAND_VERBS_H */
