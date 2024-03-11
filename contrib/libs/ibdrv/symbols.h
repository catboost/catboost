#pragma once

#define USE_DYNAMIC_OPEN

#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#undef ibv_reg_mr
#undef ibv_query_port

#ifdef __cplusplus
template <class T>
struct TId {
    typedef T R;
};

#define DOSTRUCT(name, type) TId<type>::R* name;
#else
#define DOSTRUCT(name, type) typeof(name)* name;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// verbs

#define DOVERBS(M) \
    M(ibv_modify_qp, int (struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask)) \
    M(ibv_create_ah, struct ibv_ah *(struct ibv_pd *pd, struct ibv_ah_attr *attr)) \
    M(ibv_create_cq, struct ibv_cq *(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector)) \
    M(ibv_destroy_ah, int (struct ibv_ah *ah)) \
    M(ibv_create_qp, struct ibv_qp *(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr)) \
    M(ibv_fork_init, int (void)) \
    M(ibv_open_device, struct ibv_context *(struct ibv_device *device)) \
    M(ibv_close_device, int (struct ibv_context *context)) \
    M(ibv_alloc_pd, struct ibv_pd *(struct ibv_context *context)) \
    M(ibv_dealloc_pd, int (struct ibv_pd *pd)) \
    M(ibv_free_device_list, void (struct ibv_device **list)) \
    M(ibv_query_device, int (struct ibv_context *context, struct ibv_device_attr *device_attr)) \
    M(ibv_get_device_list, struct ibv_device **(int *num_devices)) \
    M(ibv_destroy_qp, int (struct ibv_qp *qp)) \
    M(ibv_create_srq, struct ibv_srq *(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr)) \
    M(ibv_destroy_srq, int (struct ibv_srq *srq)) \
    M(ibv_init_ah_from_wc, int (struct ibv_context *context, uint8_t port_num, struct ibv_wc *wc, struct ibv_grh *grh, struct ibv_ah_attr *ah_attr)) \
    M(ibv_reg_mr, struct ibv_mr *(struct ibv_pd *pd, void *addr, size_t length, int access)) \
    M(ibv_reg_mr_iova2, struct ibv_mr *(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access)) \
    M(ibv_dereg_mr, int (struct ibv_mr *mr)) \
    M(ibv_query_pkey, int (struct ibv_context *context, uint8_t port_num, int index, __be16 *pkey)) \
    M(ibv_node_type_str, const char *(enum ibv_node_type node_type)) \
    M(ibv_destroy_cq, int (struct ibv_cq *cq)) \
    M(ibv_query_gid, int (struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid)) \
    M(ibv_query_port, int (struct ibv_context *context, uint8_t port_num, struct _compat_ibv_port_attr *port_attr)) \
    M(ibv_wc_status_str, const char *(enum ibv_wc_status status)) \
    M(ibv_get_device_name, const char *(struct ibv_device *device)) \
    M(ibv_get_async_event, int (struct ibv_context *context, struct ibv_async_event *event)) \
    M(ibv_event_type_str, const char *(enum ibv_event_type event)) \
    M(ibv_query_qp, int (struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr)) \
    M(ibv_resize_cq, int (struct ibv_cq *cq, int cqe)) \
    M(ibv_ack_async_event, void (struct ibv_async_event *event)) \
    M(ibv_create_comp_channel, struct ibv_comp_channel *(struct ibv_context *context)) \
    M(ibv_destroy_comp_channel, int (struct ibv_comp_channel *channel)) \
    M(ibv_get_cq_event, int (struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cq_context)) \
    M(ibv_ack_cq_events, void (struct ibv_cq *cq, unsigned int nevents)) \
    M(ibv_port_state_str, const char *(enum ibv_port_state port_state)) \
// DOVERBS

struct TInfinibandSymbols {
    DOVERBS(DOSTRUCT)
};

const struct TInfinibandSymbols* IBSym();

// rdma

#define DORDMA(M) \
    M(rdma_ack_cm_event, int (struct rdma_cm_event *event)) \
    M(rdma_get_cm_event, int (struct rdma_event_channel *channel, struct rdma_cm_event **event)) \
    M(rdma_create_qp, int (struct rdma_cm_id *id, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr)) \
    M(rdma_create_event_channel, struct rdma_event_channel * (void)) \
    M(rdma_create_id, int (struct rdma_event_channel *channel, struct rdma_cm_id **id, void *context, enum rdma_port_space ps)) \
    M(rdma_resolve_addr, int (struct rdma_cm_id *id, struct sockaddr *src_addr, struct sockaddr *dst_addr, int timeout_ms)) \
    M(rdma_resolve_route, int (struct rdma_cm_id *id, int timeout_ms)) \
    M(rdma_bind_addr, int (struct rdma_cm_id *id, struct sockaddr *addr)) \
    M(rdma_listen, int (struct rdma_cm_id *id, int backlog)) \
    M(rdma_accept, int (struct rdma_cm_id *id, struct rdma_conn_param *conn_param)) \
    M(rdma_connect, int (struct rdma_cm_id *id, struct rdma_conn_param *conn_param)) \
    M(rdma_disconnect, int (struct rdma_cm_id *id)) \
    M(rdma_set_option, int (struct rdma_cm_id *id, int level, int optname, void *optval, size_t optlen)) \
    M(rdma_destroy_id, int (struct rdma_cm_id *id)) \
    M(rdma_destroy_qp, void (struct rdma_cm_id *id)) \
    M(rdma_get_devices, struct ibv_context **(int *num_devices)) \
    M(rdma_free_devices, void (struct ibv_context **list)) \
    M(rdma_destroy_event_channel, void (struct rdma_event_channel *channel)) \
    M(rdma_reject, int (struct rdma_cm_id *id, const void *private_data, uint8_t private_data_len)) \
    M(rdma_get_dst_port, uint16_t (struct rdma_cm_id *id)) \
    M(rdma_get_src_port, uint16_t (struct rdma_cm_id *id)) \
    M(rdma_getaddrinfo, int (const char *node, const char *service, const struct rdma_addrinfo *hints, struct rdma_addrinfo **res)) \
    M(rdma_freeaddrinfo, void (struct rdma_addrinfo *res)) \
// DORDMA

struct TRdmaSymbols {
    DORDMA(DOSTRUCT)
};

const struct TRdmaSymbols* RDSym();

// mlx5

#define DOMLX5(M) \
    M(mlx5dv_alloc_var, struct mlx5dv_var *(struct ibv_context *context, uint32_t flags)) \
    M(mlx5dv_create_cq, struct ibv_cq_ex *(struct ibv_context *context, struct ibv_cq_init_attr_ex *cq_attr, struct mlx5dv_cq_init_attr *mlx5_cq_attr)) \
    M(mlx5dv_create_flow, struct ibv_flow *(struct mlx5dv_flow_matcher *matcher, struct mlx5dv_flow_match_parameters *match_value, size_t num_actions, struct mlx5dv_flow_action_attr actions_attr[])) \
    M(mlx5dv_create_flow_matcher, struct mlx5dv_flow_matcher *(struct ibv_context *context, struct mlx5dv_flow_matcher_attr *matcher_attr)) \
    M(mlx5dv_create_qp, struct ibv_qp *(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, struct mlx5dv_qp_init_attr *mlx5_qp_attr)) \
    M(mlx5dv_create_wq, struct ibv_wq *(struct ibv_context *context, struct ibv_wq_init_attr *wq_init_attr, struct mlx5dv_wq_init_attr *mlx5_wq_attr)) \
    M(mlx5dv_destroy_flow_matcher, int (struct mlx5dv_flow_matcher *matcher)) \
    M(mlx5dv_devx_alloc_uar, struct mlx5dv_devx_uar *(struct ibv_context *context, uint32_t flags)) \
    M(mlx5dv_devx_create_cmd_comp, struct mlx5dv_devx_cmd_comp *(struct ibv_context *context)) \
    M(mlx5dv_devx_create_event_channel, struct mlx5dv_devx_event_channel *(struct ibv_context *context, enum mlx5dv_devx_create_event_channel_flags flags)) \
    M(mlx5dv_devx_destroy_cmd_comp, void (struct mlx5dv_devx_cmd_comp *cmd_comp)) \
    M(mlx5dv_devx_destroy_event_channel, void (struct mlx5dv_devx_event_channel *event_channel)) \
    M(mlx5dv_devx_free_uar, void (struct mlx5dv_devx_uar *devx_uar)) \
    M(mlx5dv_devx_general_cmd, int (struct ibv_context *context, const void *in, size_t inlen, void *out, size_t outlen)) \
    M(mlx5dv_devx_get_async_cmd_comp, int (struct mlx5dv_devx_cmd_comp *cmd_comp, struct mlx5dv_devx_async_cmd_hdr *cmd_resp, size_t cmd_resp_len)) \
    M(mlx5dv_devx_get_event, ssize_t (struct mlx5dv_devx_event_channel *event_channel, struct mlx5dv_devx_async_event_hdr *event_data, size_t event_resp_len)) \
    M(mlx5dv_devx_obj_create, struct mlx5dv_devx_obj *(struct ibv_context *context, const void *in, size_t inlen, void *out, size_t outlen)) \
    M(mlx5dv_devx_obj_destroy, int (struct mlx5dv_devx_obj *obj)) \
    M(mlx5dv_devx_obj_modify, int (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, void *out, size_t outlen)) \
    M(mlx5dv_devx_obj_query, int (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, void *out, size_t outlen)) \
    M(mlx5dv_devx_obj_query_async, int (struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, size_t outlen, uint64_t wr_id, struct mlx5dv_devx_cmd_comp *cmd_comp)) \
    M(mlx5dv_devx_qp_query, int (struct ibv_qp *qp, const void *in, size_t inlen, void *out, size_t outlen)) \
    M(mlx5dv_devx_query_eqn, int (struct ibv_context *context, uint32_t vector, uint32_t *eqn)) \
    M(mlx5dv_devx_subscribe_devx_event, int (struct mlx5dv_devx_event_channel *event_channel, struct mlx5dv_devx_obj *obj, uint16_t events_sz, uint16_t events_num[], uint64_t cookie)) \
    M(mlx5dv_devx_subscribe_devx_event_fd, int (struct mlx5dv_devx_event_channel *event_channel, int fd, struct mlx5dv_devx_obj *obj, uint16_t event_num)) \
    M(mlx5dv_devx_umem_dereg, int (struct mlx5dv_devx_umem *umem)) \
    M(mlx5dv_devx_umem_reg, struct mlx5dv_devx_umem *(struct ibv_context *ctx, void *addr, size_t size, uint32_t access)) \
    M(mlx5dv_dr_action_create_aso, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, struct mlx5dv_devx_obj *devx_obj, uint32_t offset, uint32_t flags, uint8_t return_reg_c)) \
    M(mlx5dv_dr_action_create_default_miss, struct mlx5dv_dr_action *(void)) \
    M(mlx5dv_dr_action_create_dest_array, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, size_t num_dest, struct mlx5dv_dr_action_dest_attr *dests[])) \
    M(mlx5dv_dr_action_create_dest_devx_tir, struct mlx5dv_dr_action *(struct mlx5dv_devx_obj *devx_obj)) \
    M(mlx5dv_dr_action_create_dest_ib_port, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, uint32_t ib_port)) \
    M(mlx5dv_dr_action_create_dest_ibv_qp, struct mlx5dv_dr_action *(struct ibv_qp *ibqp)) \
    M(mlx5dv_dr_action_create_dest_table, struct mlx5dv_dr_action *(struct mlx5dv_dr_table *table)) \
    M(mlx5dv_dr_action_create_dest_vport, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, uint32_t vport)) \
    M(mlx5dv_dr_action_create_drop, struct mlx5dv_dr_action *(void)) \
    M(mlx5dv_dr_action_create_flow_counter, struct mlx5dv_dr_action *(struct mlx5dv_devx_obj *devx_obj, uint32_t offset)) \
    M(mlx5dv_dr_action_create_flow_meter, struct mlx5dv_dr_action *(struct mlx5dv_dr_flow_meter_attr *attr)) \
    M(mlx5dv_dr_action_create_flow_sampler, struct mlx5dv_dr_action *(struct mlx5dv_dr_flow_sampler_attr *attr)) \
    M(mlx5dv_dr_action_create_modify_header, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, uint32_t flags, size_t actions_sz, __be64 actions[])) \
    M(mlx5dv_dr_action_create_packet_reformat, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, uint32_t flags, enum mlx5dv_flow_action_packet_reformat_type reformat_type, size_t data_sz, void *data)) \
    M(mlx5dv_dr_action_create_pop_vlan, struct mlx5dv_dr_action *(void)) \
    M(mlx5dv_dr_action_create_push_vlan, struct mlx5dv_dr_action *(struct mlx5dv_dr_domain *domain, __be32 vlan_hdr)) \
    M(mlx5dv_dr_action_create_tag, struct mlx5dv_dr_action *(uint32_t tag_value)) \
    M(mlx5dv_dr_action_destroy, int (struct mlx5dv_dr_action *action)) \
    M(mlx5dv_dr_action_modify_flow_meter, int (struct mlx5dv_dr_action *action, struct mlx5dv_dr_flow_meter_attr *attr, __be64 modify_field_select)) \
    M(mlx5dv_dr_domain_create, struct mlx5dv_dr_domain *(struct ibv_context *ctx, enum mlx5dv_dr_domain_type type)) \
    M(mlx5dv_dr_domain_destroy, int (struct mlx5dv_dr_domain *domain)) \
    M(mlx5dv_dr_domain_set_reclaim_device_memory, void (struct mlx5dv_dr_domain *dmn, bool enable)) \
    M(mlx5dv_dr_domain_sync, int (struct mlx5dv_dr_domain *domain, uint32_t flags)) \
    M(mlx5dv_dr_matcher_create, struct mlx5dv_dr_matcher *(struct mlx5dv_dr_table *table, uint16_t priority, uint8_t match_criteria_enable, struct mlx5dv_flow_match_parameters *mask)) \
    M(mlx5dv_dr_matcher_destroy, int (struct mlx5dv_dr_matcher *matcher)) \
    M(mlx5dv_dr_rule_create, struct mlx5dv_dr_rule *(struct mlx5dv_dr_matcher *matcher, struct mlx5dv_flow_match_parameters *value, size_t num_actions, struct mlx5dv_dr_action *actions[])) \
    M(mlx5dv_dr_rule_destroy, int (struct mlx5dv_dr_rule *rule)) \
    M(mlx5dv_dr_table_create, struct mlx5dv_dr_table *(struct mlx5dv_dr_domain *domain, uint32_t level)) \
    M(mlx5dv_dr_table_destroy, int (struct mlx5dv_dr_table *table)) \
    M(mlx5dv_dump_dr_domain, int (FILE *fout, struct mlx5dv_dr_domain *domain)) \
    M(mlx5dv_free_var, void (struct mlx5dv_var *dv_var)) \
    M(mlx5dv_init_obj, int (struct mlx5dv_obj *obj, uint64_t obj_type)) \
    M(mlx5dv_open_device, struct ibv_context *(struct ibv_device *device, struct mlx5dv_context_attr *attr)) \
    M(mlx5dv_pp_alloc, struct mlx5dv_pp *(struct ibv_context *context, size_t pp_context_sz, const void *pp_context, uint32_t flags)) \
    M(mlx5dv_pp_free, void (struct mlx5dv_pp *pp)) \
    M(mlx5dv_query_device, int (struct ibv_context *ctx_in, struct mlx5dv_context *attrs_out)) \
    M(mlx5dv_query_devx_port, int (struct ibv_context *ctx, uint32_t port_num, struct mlx5dv_devx_port *mlx5_devx_port)) \
    M(mlx5dv_set_context_attr, int (struct ibv_context *context, enum mlx5dv_set_ctx_attr_type type, void *attr)) \
// DOMLX5

struct TMlx5Symbols {
    DOMLX5(DOSTRUCT)
};

const struct TMlx5Symbols* M5Sym();

#undef DOSTRUCT

#ifdef __cplusplus
}
#endif
