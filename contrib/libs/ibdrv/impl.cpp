#include "symbols.h"

#include <util/generic/yexception.h>

template <typename Method, typename... Args>
static auto Call(Method* m, Args&&... args) {
    Y_ENSURE(m);
    return m(std::forward<Args>(args)...);
}

// verbs

Y_HIDDEN
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
    return Call(IBSym()->ibv_modify_qp, qp, attr, attr_mask);
}

Y_HIDDEN
int ibv_query_pkey(struct ibv_context *context, uint8_t port_num,
		   int index, __be16 *pkey) {
    return Call(IBSym()->ibv_query_pkey, context, port_num, index, pkey);
}

Y_HIDDEN
const char *ibv_node_type_str(enum ibv_node_type node_type) {
    return Call(IBSym()->ibv_node_type_str, node_type);
}

Y_HIDDEN
struct ibv_ah *ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr) {
    return Call(IBSym()->ibv_create_ah, pd, attr);
}

Y_HIDDEN
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
    return Call(IBSym()->ibv_create_cq, context, cqe, cq_context, channel, comp_vector);
}

Y_HIDDEN
int ibv_destroy_ah(struct ibv_ah *ah) {
    return Call(IBSym()->ibv_destroy_ah, ah);
}

Y_HIDDEN
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
    return Call(IBSym()->ibv_create_qp, pd, qp_init_attr);
}

Y_HIDDEN
int ibv_fork_init() {
    return Call(IBSym()->ibv_fork_init);
}

Y_HIDDEN
struct ibv_context *ibv_open_device(struct ibv_device *device) {
    return Call(IBSym()->ibv_open_device, device);
}

Y_HIDDEN
int ibv_close_device(struct ibv_context *context) {
    return Call(IBSym()->ibv_close_device, context);
}

Y_HIDDEN
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context) {
    return Call(IBSym()->ibv_alloc_pd, context);
}

Y_HIDDEN
int ibv_dealloc_pd(struct ibv_pd *pd) {
    return Call(IBSym()->ibv_dealloc_pd, pd);
}

Y_HIDDEN
void ibv_free_device_list(struct ibv_device **list) {
    return Call(IBSym()->ibv_free_device_list, list);
}

Y_HIDDEN
int ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) {
    return Call(IBSym()->ibv_query_device, context, device_attr);
}

Y_HIDDEN
struct ibv_device **ibv_get_device_list(int *num_devices) {
    return Call(IBSym()->ibv_get_device_list, num_devices);
}

Y_HIDDEN
int ibv_destroy_qp(struct ibv_qp *qp) {
    return Call(IBSym()->ibv_destroy_qp, qp);
}

Y_HIDDEN
struct ibv_srq *ibv_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr) {
    return Call(IBSym()->ibv_create_srq, pd, srq_init_attr);
}

Y_HIDDEN
int ibv_destroy_srq(struct ibv_srq *srq) {
    return Call(IBSym()->ibv_destroy_srq, srq);
}

Y_HIDDEN
int ibv_init_ah_from_wc(struct ibv_context *context, uint8_t port_num, struct ibv_wc *wc, struct ibv_grh *grh, struct ibv_ah_attr *ah_attr) {
    return Call(IBSym()->ibv_init_ah_from_wc, context, port_num, wc, grh, ah_attr);
}

Y_HIDDEN
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
    return Call(IBSym()->ibv_reg_mr, pd, addr, length, access);
}

Y_HIDDEN
struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access) {
    auto symFunc = IBSym()->ibv_reg_mr_iova2;
    if (!symFunc) {
        // On old versions we don`t have optimized ibv_reg_mr_iova2 on machines, 
        // so fallback on ibv_reg_mr
        return Call(IBSym()->ibv_reg_mr, pd, addr, length, access);
    }
    return Call(symFunc, pd, addr, length, iova, access);
}

Y_HIDDEN
int ibv_dereg_mr(struct ibv_mr *mr) {
    return Call(IBSym()->ibv_dereg_mr, mr);
}

Y_HIDDEN
int ibv_destroy_cq(struct ibv_cq *cq) {
    return Call(IBSym()->ibv_destroy_cq, cq);
}

Y_HIDDEN
int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
    return Call(IBSym()->ibv_query_gid, context, port_num, index, gid);
}

Y_HIDDEN
int ibv_query_port(struct ibv_context *context, uint8_t port_num, struct _compat_ibv_port_attr *port_attr) {
    return Call(IBSym()->ibv_query_port, context, port_num, port_attr);
}

Y_HIDDEN
const char *ibv_wc_status_str(enum ibv_wc_status status) {
    return Call(IBSym()->ibv_wc_status_str, status);
}

Y_HIDDEN
const char *ibv_get_device_name(struct ibv_device *device) {
    return Call(IBSym()->ibv_get_device_name, device);
}

Y_HIDDEN
int ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) {
    return Call(IBSym()->ibv_get_async_event, context, event);
}

Y_HIDDEN
const char *ibv_event_type_str(enum ibv_event_type event) {
    return Call(IBSym()->ibv_event_type_str, event);
}

Y_HIDDEN
int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr) {
    return Call(IBSym()->ibv_query_qp, qp, attr, attr_mask, init_attr);
}

Y_HIDDEN
int ibv_resize_cq(struct ibv_cq *cq, int cqe) {
    return Call(IBSym()->ibv_resize_cq, cq, cqe);
}

Y_HIDDEN
void ibv_ack_async_event(struct ibv_async_event *event) {
    return Call(IBSym()->ibv_ack_async_event, event);
}

Y_HIDDEN
struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context) {
    return Call(IBSym()->ibv_create_comp_channel, context);
}

Y_HIDDEN
int ibv_destroy_comp_channel(struct ibv_comp_channel *channel) {
    return Call(IBSym()->ibv_destroy_comp_channel, channel);
}

Y_HIDDEN
int ibv_get_cq_event(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cq_context) {
    return Call(IBSym()->ibv_get_cq_event, channel, cq, cq_context);
}

Y_HIDDEN
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents) {
    return Call(IBSym()->ibv_ack_cq_events, cq, nevents);
}

Y_HIDDEN
const char *ibv_port_state_str(enum ibv_port_state port_state) {
    return Call(IBSym()->ibv_port_state_str, port_state);
}

// rdma

Y_HIDDEN
int rdma_ack_cm_event(struct rdma_cm_event *event) {
    return Call(RDSym()->rdma_ack_cm_event, event);
}

Y_HIDDEN
int rdma_get_cm_event(struct rdma_event_channel *channel, struct rdma_cm_event **event) {
    return Call(RDSym()->rdma_get_cm_event, channel, event);
}

Y_HIDDEN
int rdma_create_qp(struct rdma_cm_id *id, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
    return Call(RDSym()->rdma_create_qp, id, pd, qp_init_attr);
}

Y_HIDDEN
struct rdma_event_channel *rdma_create_event_channel() {
    return Call(RDSym()->rdma_create_event_channel);
}

Y_HIDDEN
int rdma_create_id(struct rdma_event_channel *channel, struct rdma_cm_id **id, void *context, enum rdma_port_space ps) {
    return Call(RDSym()->rdma_create_id, channel, id, context, ps);
}

Y_HIDDEN
int rdma_resolve_addr(struct rdma_cm_id *id, struct sockaddr *src_addr, struct sockaddr *dst_addr, int timeout_ms) {
    return Call(RDSym()->rdma_resolve_addr, id, src_addr, dst_addr, timeout_ms);
}

Y_HIDDEN
int rdma_resolve_route(struct rdma_cm_id *id, int timeout_ms) {
    return Call(RDSym()->rdma_resolve_route, id, timeout_ms);
}

Y_HIDDEN
int rdma_bind_addr(struct rdma_cm_id *id, struct sockaddr *addr) {
    return Call(RDSym()->rdma_bind_addr, id, addr);
}

Y_HIDDEN
int rdma_listen(struct rdma_cm_id *id, int backlog) {
    return Call(RDSym()->rdma_listen, id, backlog);
}

Y_HIDDEN
int rdma_accept(struct rdma_cm_id *id, struct rdma_conn_param *conn_param) {
    return Call(RDSym()->rdma_accept, id, conn_param);
}

Y_HIDDEN
int rdma_connect(struct rdma_cm_id *id, struct rdma_conn_param *conn_param) {
    return Call(RDSym()->rdma_connect, id, conn_param);
}

Y_HIDDEN
int rdma_disconnect(struct rdma_cm_id *id) {
    return Call(RDSym()->rdma_disconnect, id);
}

Y_HIDDEN
int rdma_set_option(struct rdma_cm_id *id, int level, int optname, void *optval, size_t optlen) {
    return Call(RDSym()->rdma_set_option, id, level, optname, optval, optlen);
}

Y_HIDDEN
int rdma_destroy_id(struct rdma_cm_id *id) {
    return Call(RDSym()->rdma_destroy_id, id);
}

Y_HIDDEN
void rdma_destroy_qp(struct rdma_cm_id *id) {
    return Call(RDSym()->rdma_destroy_qp, id);
}

Y_HIDDEN
struct ibv_context **rdma_get_devices(int *num_devices) {
    return Call(RDSym()->rdma_get_devices, num_devices);
}

Y_HIDDEN
void rdma_free_devices(struct ibv_context **list) {
    return Call(RDSym()->rdma_free_devices, list);
}

Y_HIDDEN
void rdma_destroy_event_channel(struct rdma_event_channel *channel) {
    return Call(RDSym()->rdma_destroy_event_channel, channel);
}

Y_HIDDEN
int rdma_reject(struct rdma_cm_id *id, const void *private_data, uint8_t private_data_len) {
    return Call(RDSym()->rdma_reject, id, private_data, private_data_len);
}

Y_HIDDEN
uint16_t rdma_get_dst_port(struct rdma_cm_id *id) {
    return Call(RDSym()->rdma_get_dst_port, id);
}

Y_HIDDEN
uint16_t rdma_get_src_port(struct rdma_cm_id *id) {
    return Call(RDSym()->rdma_get_src_port, id);
}

Y_HIDDEN
int rdma_getaddrinfo(const char *node, const char *service, const struct rdma_addrinfo *hints, struct rdma_addrinfo **res) {
    return Call(RDSym()->rdma_getaddrinfo, node, service, hints, res);
}

Y_HIDDEN
void rdma_freeaddrinfo(struct rdma_addrinfo *res) {
    return Call(RDSym()->rdma_freeaddrinfo, res);
}

// mlx5

Y_HIDDEN
struct mlx5dv_var *mlx5dv_alloc_var(struct ibv_context *context, uint32_t flags) {
    return Call(M5Sym()->mlx5dv_alloc_var, context, flags);
}

Y_HIDDEN
struct ibv_cq_ex *mlx5dv_create_cq(struct ibv_context *context, struct ibv_cq_init_attr_ex *cq_attr, struct mlx5dv_cq_init_attr *mlx5_cq_attr) {
    return Call(M5Sym()->mlx5dv_create_cq, context, cq_attr, mlx5_cq_attr);
}

Y_HIDDEN
struct ibv_flow *mlx5dv_create_flow(struct mlx5dv_flow_matcher *matcher, struct mlx5dv_flow_match_parameters *match_value, size_t num_actions, struct mlx5dv_flow_action_attr actions_attr[]) {
    return Call(M5Sym()->mlx5dv_create_flow, matcher, match_value, num_actions, actions_attr);
}

Y_HIDDEN
struct mlx5dv_flow_matcher *mlx5dv_create_flow_matcher(struct ibv_context *context, struct mlx5dv_flow_matcher_attr *matcher_attr) {
    return Call(M5Sym()->mlx5dv_create_flow_matcher, context, matcher_attr);
}

Y_HIDDEN
struct ibv_qp *mlx5dv_create_qp(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, struct mlx5dv_qp_init_attr *mlx5_qp_attr) {
    return Call(M5Sym()->mlx5dv_create_qp, context, qp_attr, mlx5_qp_attr);
}

Y_HIDDEN
struct ibv_wq *mlx5dv_create_wq(struct ibv_context *context, struct ibv_wq_init_attr *wq_init_attr, struct mlx5dv_wq_init_attr *mlx5_wq_attr) {
    return Call(M5Sym()->mlx5dv_create_wq, context, wq_init_attr, mlx5_wq_attr);
}

Y_HIDDEN
int mlx5dv_destroy_flow_matcher(struct mlx5dv_flow_matcher *matcher) {
    return Call(M5Sym()->mlx5dv_destroy_flow_matcher, matcher);
}

Y_HIDDEN
struct mlx5dv_devx_uar *mlx5dv_devx_alloc_uar(struct ibv_context *context, uint32_t flags) {
    return Call(M5Sym()->mlx5dv_devx_alloc_uar, context, flags);
}

Y_HIDDEN
struct mlx5dv_devx_cmd_comp *mlx5dv_devx_create_cmd_comp(struct ibv_context *context) {
    return Call(M5Sym()->mlx5dv_devx_create_cmd_comp, context);
}

Y_HIDDEN
struct mlx5dv_devx_event_channel *mlx5dv_devx_create_event_channel(struct ibv_context *context, enum mlx5dv_devx_create_event_channel_flags flags) {
    return Call(M5Sym()->mlx5dv_devx_create_event_channel, context, flags);
}

Y_HIDDEN
void mlx5dv_devx_destroy_cmd_comp(struct mlx5dv_devx_cmd_comp *cmd_comp) {
    return Call(M5Sym()->mlx5dv_devx_destroy_cmd_comp, cmd_comp);
}

Y_HIDDEN
void mlx5dv_devx_destroy_event_channel(struct mlx5dv_devx_event_channel *event_channel) {
    return Call(M5Sym()->mlx5dv_devx_destroy_event_channel, event_channel);
}

Y_HIDDEN
void mlx5dv_devx_free_uar(struct mlx5dv_devx_uar *devx_uar) {
    return Call(M5Sym()->mlx5dv_devx_free_uar, devx_uar);
}

Y_HIDDEN
int mlx5dv_devx_general_cmd(struct ibv_context *context, const void *in, size_t inlen, void *out, size_t outlen) {
    return Call(M5Sym()->mlx5dv_devx_general_cmd, context, in, inlen, out, outlen);
}

Y_HIDDEN
int mlx5dv_devx_get_async_cmd_comp(struct mlx5dv_devx_cmd_comp *cmd_comp, struct mlx5dv_devx_async_cmd_hdr *cmd_resp, size_t cmd_resp_len) {
    return Call(M5Sym()->mlx5dv_devx_get_async_cmd_comp, cmd_comp, cmd_resp, cmd_resp_len);
}

Y_HIDDEN
ssize_t mlx5dv_devx_get_event(struct mlx5dv_devx_event_channel *event_channel, struct mlx5dv_devx_async_event_hdr *event_data, size_t event_resp_len) {
    return Call(M5Sym()->mlx5dv_devx_get_event, event_channel, event_data, event_resp_len);
}

Y_HIDDEN
struct mlx5dv_devx_obj *mlx5dv_devx_obj_create(struct ibv_context *context, const void *in, size_t inlen, void *out, size_t outlen) {
    return Call(M5Sym()->mlx5dv_devx_obj_create, context, in, inlen, out, outlen);
}

Y_HIDDEN
int mlx5dv_devx_obj_destroy(struct mlx5dv_devx_obj *obj) {
    return Call(M5Sym()->mlx5dv_devx_obj_destroy, obj);
}

Y_HIDDEN
int mlx5dv_devx_obj_modify(struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, void *out, size_t outlen) {
    return Call(M5Sym()->mlx5dv_devx_obj_modify, obj, in, inlen, out, outlen);
}

Y_HIDDEN
int mlx5dv_devx_obj_query(struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, void *out, size_t outlen) {
    return Call(M5Sym()->mlx5dv_devx_obj_query, obj, in, inlen, out, outlen);
}

Y_HIDDEN
int mlx5dv_devx_obj_query_async(struct mlx5dv_devx_obj *obj, const void *in, size_t inlen, size_t outlen, uint64_t wr_id, struct mlx5dv_devx_cmd_comp *cmd_comp) {
    return Call(M5Sym()->mlx5dv_devx_obj_query_async, obj, in, inlen, outlen, wr_id, cmd_comp);
}

Y_HIDDEN
int mlx5dv_devx_qp_query(struct ibv_qp *qp, const void *in, size_t inlen, void *out, size_t outlen) {
    return Call(M5Sym()->mlx5dv_devx_qp_query, qp, in, inlen, out, outlen);
}

Y_HIDDEN
int mlx5dv_devx_query_eqn(struct ibv_context *context, uint32_t vector, uint32_t *eqn) {
    return Call(M5Sym()->mlx5dv_devx_query_eqn, context, vector, eqn);
}

Y_HIDDEN
int mlx5dv_devx_subscribe_devx_event(struct mlx5dv_devx_event_channel *event_channel, struct mlx5dv_devx_obj *obj, uint16_t events_sz, uint16_t events_num[], uint64_t cookie) {
    return Call(M5Sym()->mlx5dv_devx_subscribe_devx_event, event_channel, obj, events_sz, events_num, cookie);
}

Y_HIDDEN
int mlx5dv_devx_subscribe_devx_event_fd(struct mlx5dv_devx_event_channel *event_channel, int fd, struct mlx5dv_devx_obj *obj, uint16_t event_num) {
    return Call(M5Sym()->mlx5dv_devx_subscribe_devx_event_fd, event_channel, fd, obj, event_num);
}

Y_HIDDEN
int mlx5dv_devx_umem_dereg(struct mlx5dv_devx_umem *umem) {
    return Call(M5Sym()->mlx5dv_devx_umem_dereg, umem);
}

Y_HIDDEN
struct mlx5dv_devx_umem *mlx5dv_devx_umem_reg(struct ibv_context *ctx, void *addr, size_t size, uint32_t access) {
    return Call(M5Sym()->mlx5dv_devx_umem_reg, ctx, addr, size, access);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_aso(struct mlx5dv_dr_domain *domain, struct mlx5dv_devx_obj *devx_obj, uint32_t offset, uint32_t flags, uint8_t return_reg_c) {
    return Call(M5Sym()->mlx5dv_dr_action_create_aso, domain, devx_obj, offset, flags, return_reg_c);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_default_miss(void) {
    return Call(M5Sym()->mlx5dv_dr_action_create_default_miss);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_dest_array(struct mlx5dv_dr_domain *domain, size_t num_dest, struct mlx5dv_dr_action_dest_attr *dests[]) {
    return Call(M5Sym()->mlx5dv_dr_action_create_dest_array, domain, num_dest, dests);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_dest_devx_tir(struct mlx5dv_devx_obj *devx_obj) {
    return Call(M5Sym()->mlx5dv_dr_action_create_dest_devx_tir, devx_obj);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_dest_ib_port(struct mlx5dv_dr_domain *domain, uint32_t ib_port) {
    return Call(M5Sym()->mlx5dv_dr_action_create_dest_ib_port, domain, ib_port);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_dest_ibv_qp(struct ibv_qp *ibqp) {
    return Call(M5Sym()->mlx5dv_dr_action_create_dest_ibv_qp, ibqp);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_dest_table(struct mlx5dv_dr_table *table) {
    return Call(M5Sym()->mlx5dv_dr_action_create_dest_table, table);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_dest_vport(struct mlx5dv_dr_domain *domain, uint32_t vport) {
    return Call(M5Sym()->mlx5dv_dr_action_create_dest_vport, domain, vport);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_drop(void) {
    return Call(M5Sym()->mlx5dv_dr_action_create_drop);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_flow_counter(struct mlx5dv_devx_obj *devx_obj, uint32_t offset) {
    return Call(M5Sym()->mlx5dv_dr_action_create_flow_counter, devx_obj, offset);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_flow_meter(struct mlx5dv_dr_flow_meter_attr *attr) {
    return Call(M5Sym()->mlx5dv_dr_action_create_flow_meter, attr);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_flow_sampler(struct mlx5dv_dr_flow_sampler_attr *attr) {
    return Call(M5Sym()->mlx5dv_dr_action_create_flow_sampler, attr);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_modify_header(struct mlx5dv_dr_domain *domain, uint32_t flags, size_t actions_sz, __be64 actions[]) {
    return Call(M5Sym()->mlx5dv_dr_action_create_modify_header, domain, flags, actions_sz, actions);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_packet_reformat(struct mlx5dv_dr_domain *domain, uint32_t flags, enum mlx5dv_flow_action_packet_reformat_type reformat_type, size_t data_sz, void *data) {
    return Call(M5Sym()->mlx5dv_dr_action_create_packet_reformat, domain, flags, reformat_type, data_sz, data);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_pop_vlan(void) {
    return Call(M5Sym()->mlx5dv_dr_action_create_pop_vlan);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_push_vlan(struct mlx5dv_dr_domain *domain, __be32 vlan_hdr) {
    return Call(M5Sym()->mlx5dv_dr_action_create_push_vlan, domain, vlan_hdr);
}

Y_HIDDEN
struct mlx5dv_dr_action *mlx5dv_dr_action_create_tag(uint32_t tag_value) {
    return Call(M5Sym()->mlx5dv_dr_action_create_tag, tag_value);
}

Y_HIDDEN
int mlx5dv_dr_action_destroy(struct mlx5dv_dr_action *action) {
    return Call(M5Sym()->mlx5dv_dr_action_destroy, action);
}

Y_HIDDEN
int mlx5dv_dr_action_modify_flow_meter(struct mlx5dv_dr_action *action, struct mlx5dv_dr_flow_meter_attr *attr, __be64 modify_field_select) {
    return Call(M5Sym()->mlx5dv_dr_action_modify_flow_meter, action, attr, modify_field_select);
}

Y_HIDDEN
struct mlx5dv_dr_domain *mlx5dv_dr_domain_create(struct ibv_context *ctx, enum mlx5dv_dr_domain_type type) {
    return Call(M5Sym()->mlx5dv_dr_domain_create, ctx, type);
}

Y_HIDDEN
int mlx5dv_dr_domain_destroy(struct mlx5dv_dr_domain *domain) {
    return Call(M5Sym()->mlx5dv_dr_domain_destroy, domain);
}

Y_HIDDEN
void mlx5dv_dr_domain_set_reclaim_device_memory(struct mlx5dv_dr_domain *dmn, bool enable) {
    return Call(M5Sym()->mlx5dv_dr_domain_set_reclaim_device_memory, dmn, enable);
}

Y_HIDDEN
int mlx5dv_dr_domain_sync(struct mlx5dv_dr_domain *domain, uint32_t flags) {
    return Call(M5Sym()->mlx5dv_dr_domain_sync, domain, flags);
}

Y_HIDDEN
struct mlx5dv_dr_matcher *mlx5dv_dr_matcher_create(struct mlx5dv_dr_table *table, uint16_t priority, uint8_t match_criteria_enable, struct mlx5dv_flow_match_parameters *mask) {
    return Call(M5Sym()->mlx5dv_dr_matcher_create, table, priority, match_criteria_enable, mask);
}

Y_HIDDEN
int mlx5dv_dr_matcher_destroy(struct mlx5dv_dr_matcher *matcher) {
    return Call(M5Sym()->mlx5dv_dr_matcher_destroy, matcher);
}

Y_HIDDEN
struct mlx5dv_dr_rule *mlx5dv_dr_rule_create(struct mlx5dv_dr_matcher *matcher, struct mlx5dv_flow_match_parameters *value, size_t num_actions, struct mlx5dv_dr_action *actions[]) {
    return Call(M5Sym()->mlx5dv_dr_rule_create, matcher, value, num_actions, actions);
}

Y_HIDDEN
int mlx5dv_dr_rule_destroy(struct mlx5dv_dr_rule *rule) {
    return Call(M5Sym()->mlx5dv_dr_rule_destroy, rule);
}

Y_HIDDEN
struct mlx5dv_dr_table *mlx5dv_dr_table_create(struct mlx5dv_dr_domain *domain, uint32_t level) {
    return Call(M5Sym()->mlx5dv_dr_table_create, domain, level);
}

Y_HIDDEN
int mlx5dv_dr_table_destroy(struct mlx5dv_dr_table *table) {
    return Call(M5Sym()->mlx5dv_dr_table_destroy, table);
}

Y_HIDDEN
int mlx5dv_dump_dr_domain(FILE *fout, struct mlx5dv_dr_domain *domain) {
    return Call(M5Sym()->mlx5dv_dump_dr_domain, fout, domain);
}

Y_HIDDEN
void mlx5dv_free_var(struct mlx5dv_var *dv_var) {
    return Call(M5Sym()->mlx5dv_free_var, dv_var);
}

Y_HIDDEN
int mlx5dv_init_obj(struct mlx5dv_obj *obj, uint64_t obj_type) {
    return Call(M5Sym()->mlx5dv_init_obj, obj, obj_type);
}

Y_HIDDEN
struct ibv_context *mlx5dv_open_device(struct ibv_device *device, struct mlx5dv_context_attr *attr) {
    return Call(M5Sym()->mlx5dv_open_device, device, attr);
}

Y_HIDDEN
struct mlx5dv_pp *mlx5dv_pp_alloc(struct ibv_context *context, size_t pp_context_sz, const void *pp_context, uint32_t flags) {
    return Call(M5Sym()->mlx5dv_pp_alloc, context, pp_context_sz, pp_context, flags);
}

Y_HIDDEN
void mlx5dv_pp_free(struct mlx5dv_pp *pp) {
    return Call(M5Sym()->mlx5dv_pp_free, pp);
}

Y_HIDDEN
int mlx5dv_query_device(struct ibv_context *ctx_in, struct mlx5dv_context *attrs_out) {
    return Call(M5Sym()->mlx5dv_query_device, ctx_in, attrs_out);
}

Y_HIDDEN
int mlx5dv_query_devx_port(struct ibv_context *ctx, uint32_t port_num, struct mlx5dv_devx_port *mlx5_devx_port) {
    return Call(M5Sym()->mlx5dv_query_devx_port, ctx, port_num, mlx5_devx_port);
}

Y_HIDDEN
int mlx5dv_set_context_attr(struct ibv_context *context, enum mlx5dv_set_ctx_attr_type type, void *attr) {
    return Call(M5Sym()->mlx5dv_set_context_attr, context, type, attr);
}
