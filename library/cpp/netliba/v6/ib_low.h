#pragma once

#include "udp_address.h"

#if defined(_linux_)
#include <contrib/libs/ibdrv/include/infiniband/verbs.h>
#include <contrib/libs/ibdrv/include/rdma/rdma_cma.h>
#endif

namespace NNetliba {
#define CHECK_Z(x)                                                  \
    {                                                               \
        int rv = (x);                                               \
        if (rv != 0) {                                              \
            fprintf(stderr, "check_z failed, errno = %d\n", errno); \
            Y_ABORT_UNLESS(0, "check_z");                                 \
        }                                                           \
    }

    //////////////////////////////////////////////////////////////////////////
    const int MAX_SGE = 1;
    const size_t MAX_INLINE_DATA_SIZE = 16;
    const int MAX_OUTSTANDING_RDMA = 10;

#if defined(_linux_)
    class TIBContext: public TThrRefBase, TNonCopyable {
        ibv_context* Context;
        ibv_pd* ProtDomain;
        TMutex Lock;

        ~TIBContext() override {
            if (Context) {
                CHECK_Z(ibv_dealloc_pd(ProtDomain));
                CHECK_Z(ibv_close_device(Context));
            }
        }

    public:
        TIBContext(ibv_device* device) {
            Context = ibv_open_device(device);
            if (Context) {
                ProtDomain = ibv_alloc_pd(Context);
            }
        }
        bool IsValid() const {
            return Context != nullptr && ProtDomain != nullptr;
        }

        class TLock {
            TIntrusivePtr<TIBContext> Ptr;
            TGuard<TMutex> Guard;

        public:
            TLock(TPtrArg<TIBContext> ctx)
                : Ptr(ctx)
                , Guard(ctx->Lock)
            {
            }
            ibv_context* GetContext() {
                return Ptr->Context;
            }
            ibv_pd* GetProtDomain() {
                return Ptr->ProtDomain;
            }
        };
    };

    class TIBPort: public TThrRefBase, TNonCopyable {
        int Port;
        int LID;
        TIntrusivePtr<TIBContext> IBCtx;
        enum {
            MAX_GID = 16
        };
        ibv_gid MyGidArr[MAX_GID];

    public:
        TIBPort(TPtrArg<TIBContext> ctx, int port)
            : IBCtx(ctx)
        {
            ibv_port_attr portAttrs;
            TIBContext::TLock ibContext(IBCtx);
            CHECK_Z(ibv_query_port(ibContext.GetContext(), port, &portAttrs));
            Port = port;
            LID = portAttrs.lid;
            for (int i = 0; i < MAX_GID; ++i) {
                ibv_gid& dst = MyGidArr[i];
                Zero(dst);
                ibv_query_gid(ibContext.GetContext(), Port, i, &dst);
            }
        }
        int GetPort() const {
            return Port;
        }
        int GetLID() const {
            return LID;
        }
        TIBContext* GetCtx() {
            return IBCtx.Get();
        }
        void GetGID(ibv_gid* res) const {
            *res = MyGidArr[0];
        }
        int GetGIDIndex(const ibv_gid& arg) const {
            for (int i = 0; i < MAX_GID; ++i) {
                const ibv_gid& chk = MyGidArr[i];
                if (memcmp(&chk, &arg, sizeof(chk)) == 0) {
                    return i;
                }
            }
            return 0;
        }
        void GetAHAttr(ibv_wc* wc, ibv_grh* grh, ibv_ah_attr* res) {
            TIBContext::TLock ibContext(IBCtx);
            CHECK_Z(ibv_init_ah_from_wc(ibContext.GetContext(), Port, wc, grh, res));
        }
    };

    class TComplectionQueue: public TThrRefBase, TNonCopyable {
        ibv_cq* CQ;
        TIntrusivePtr<TIBContext> IBCtx;

        ~TComplectionQueue() override {
            if (CQ) {
                CHECK_Z(ibv_destroy_cq(CQ));
            }
        }

    public:
        TComplectionQueue(TPtrArg<TIBContext> ctx, int maxCQEcount)
            : IBCtx(ctx)
        {
            TIBContext::TLock ibContext(IBCtx);
            /*      ibv_cq_init_attr_ex attr;
        Zero(attr);
        attr.cqe = maxCQEcount;
        attr.cq_create_flags = 0;
        ibv_cq_ex *vcq = ibv_create_cq_ex(ibContext.GetContext(), &attr);
        if (vcq) {
            CQ = (ibv_cq*)vcq; // doubtful trick but that's life
        } else {*/
            // no completion channel
            // no completion vector
            CQ = ibv_create_cq(ibContext.GetContext(), maxCQEcount, nullptr, nullptr, 0);
            //       }
        }
        ibv_cq* GetCQ() {
            return CQ;
        }
        int Poll(ibv_wc* res, int bufSize) {
            Y_ASSERT(bufSize >= 1);
            //struct ibv_wc
            //{
            //    ui64 wr_id;
            //    enum ibv_wc_status status;
            //    enum ibv_wc_opcode opcode;
            //    ui32 vendor_err;
            //    ui32 byte_len;
            //    ui32 imm_data;/* network byte order */
            //    ui32 qp_num;
            //    ui32 src_qp;
            //    enum ibv_wc_flags wc_flags;
            //    ui16 pkey_index;
            //    ui16 slid;
            //    ui8 sl;
            //    ui8 dlid_path_bits;
            //};
            int rv = ibv_poll_cq(CQ, bufSize, res);
            if (rv < 0) {
                Y_ABORT_UNLESS(0, "ibv_poll_cq failed");
            }
            if (rv > 0) {
                //printf("Completed wr\n");
                //printf("  wr_id = %" PRIx64 "\n", wc.wr_id);
                //printf("  status = %d\n", wc.status);
                //printf("  opcode = %d\n", wc.opcode);
                //printf("  byte_len = %d\n", wc.byte_len);
                //printf("  imm_data = %d\n", wc.imm_data);
                //printf("  qp_num = %d\n", wc.qp_num);
                //printf("  src_qp = %d\n", wc.src_qp);
                //printf("  wc_flags = %x\n", wc.wc_flags);
                //printf("  slid = %d\n", wc.slid);
            }
            //rv = number_of_toggled_wc;
            return rv;
        }
    };

    //struct ibv_mr
    //{
    //    struct ibv_context *context;
    //    struct ibv_pd *pd;
    //    void *addr;
    //    size_t length;
    //    ui32 handle;
    //    ui32 lkey;
    //    ui32 rkey;
    //};
    class TMemoryRegion: public TThrRefBase, TNonCopyable {
        ibv_mr* MR;
        TIntrusivePtr<TIBContext> IBCtx;

        ~TMemoryRegion() override {
            if (MR) {
                CHECK_Z(ibv_dereg_mr(MR));
            }
        }

    public:
        TMemoryRegion(TPtrArg<TIBContext> ctx, size_t len)
            : IBCtx(ctx)
        {
            TIBContext::TLock ibContext(IBCtx);
            int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; // TODO: IBV_ACCESS_ALLOCATE_MR
            MR = ibv_reg_mr(ibContext.GetProtDomain(), 0, len, access);
            Y_ASSERT(MR);
        }
        ui32 GetLKey() const {
            static_assert(sizeof(ui32) == sizeof(MR->lkey), "expect sizeof(ui32) == sizeof(MR->lkey)");
            return MR->lkey;
        }
        ui32 GetRKey() const {
            static_assert(sizeof(ui32) == sizeof(MR->lkey), "expect sizeof(ui32) == sizeof(MR->lkey)");
            return MR->lkey;
        }
        char* GetData() {
            return MR ? (char*)MR->addr : nullptr;
        }
        bool IsCovered(const void* data, size_t len) const {
            size_t dataAddr = reinterpret_cast<size_t>(data) / sizeof(char);
            size_t bufAddr = reinterpret_cast<size_t>(MR->addr) / sizeof(char);
            return (dataAddr >= bufAddr) && (dataAddr + len <= bufAddr + MR->length);
        }
    };

    class TSharedReceiveQueue: public TThrRefBase, TNonCopyable {
        ibv_srq* SRQ;
        TIntrusivePtr<TIBContext> IBCtx;

        ~TSharedReceiveQueue() override {
            if (SRQ) {
                ibv_destroy_srq(SRQ);
            }
        }

    public:
        TSharedReceiveQueue(TPtrArg<TIBContext> ctx, int maxWR)
            : IBCtx(ctx)
        {
            ibv_srq_init_attr attr;
            Zero(attr);
            attr.srq_context = this;
            attr.attr.max_sge = MAX_SGE;
            attr.attr.max_wr = maxWR;

            TIBContext::TLock ibContext(IBCtx);
            SRQ = ibv_create_srq(ibContext.GetProtDomain(), &attr);
            Y_ASSERT(SRQ);
        }
        ibv_srq* GetSRQ() {
            return SRQ;
        }
        void PostReceive(TPtrArg<TMemoryRegion> mem, ui64 id, const void* buf, size_t len) {
            Y_ASSERT(mem->IsCovered(buf, len));
            ibv_recv_wr wr, *bad;
            ibv_sge sg;
            sg.addr = reinterpret_cast<ui64>(buf) / sizeof(char);
            sg.length = len;
            sg.lkey = mem->GetLKey();
            Zero(wr);
            wr.wr_id = id;
            wr.sg_list = &sg;
            wr.num_sge = 1;
            CHECK_Z(ibv_post_srq_recv(SRQ, &wr, &bad));
        }
    };

    inline void MakeAH(ibv_ah_attr* res, TPtrArg<TIBPort> port, int lid, int serviceLevel) {
        Zero(*res);
        res->dlid = lid;
        res->port_num = port->GetPort();
        res->sl = serviceLevel;
    }

    void MakeAH(ibv_ah_attr* res, TPtrArg<TIBPort> port, const TUdpAddress& remoteAddr, const TUdpAddress& localAddr, int serviceLevel);

    class TAddressHandle: public TThrRefBase, TNonCopyable {
        ibv_ah* AH;
        TIntrusivePtr<TIBContext> IBCtx;

        ~TAddressHandle() override {
            if (AH) {
                CHECK_Z(ibv_destroy_ah(AH));
            }
            AH = nullptr;
            IBCtx = nullptr;
        }

    public:
        TAddressHandle(TPtrArg<TIBContext> ctx, ibv_ah_attr* attr)
            : IBCtx(ctx)
        {
            TIBContext::TLock ibContext(IBCtx);
            AH = ibv_create_ah(ibContext.GetProtDomain(), attr);
            Y_ASSERT(AH != nullptr);
        }
        TAddressHandle(TPtrArg<TIBPort> port, int lid, int serviceLevel)
            : IBCtx(port->GetCtx())
        {
            ibv_ah_attr attr;
            MakeAH(&attr, port, lid, serviceLevel);
            TIBContext::TLock ibContext(IBCtx);
            AH = ibv_create_ah(ibContext.GetProtDomain(), &attr);
            Y_ASSERT(AH != nullptr);
        }
        TAddressHandle(TPtrArg<TIBPort> port, const TUdpAddress& remoteAddr, const TUdpAddress& localAddr, int serviceLevel)
            : IBCtx(port->GetCtx())
        {
            ibv_ah_attr attr;
            MakeAH(&attr, port, remoteAddr, localAddr, serviceLevel);
            TIBContext::TLock ibContext(IBCtx);
            AH = ibv_create_ah(ibContext.GetProtDomain(), &attr);
            Y_ASSERT(AH != nullptr);
        }
        ibv_ah* GetAH() {
            return AH;
        }
        bool IsValid() const {
            return AH != nullptr;
        }
    };

    // GRH + wc -> address_handle_attr
    //int ibv_init_ah_from_wc(struct ibv_context *context, ui8 port_num,
    //struct ibv_wc *wc, struct ibv_grh *grh,
    //struct ibv_ah_attr *ah_attr)
    //ibv_create_ah_from_wc(struct ibv_pd *pd, struct ibv_wc *wc, struct ibv_grh
    //                      *grh, ui8 port_num)

    class TQueuePair: public TThrRefBase, TNonCopyable {
    protected:
        ibv_qp* QP;
        int MyPSN; // start packet sequence number
        TIntrusivePtr<TIBContext> IBCtx;
        TIntrusivePtr<TComplectionQueue> CQ;
        TIntrusivePtr<TSharedReceiveQueue> SRQ;

        TQueuePair(TPtrArg<TIBContext> ctx, TPtrArg<TComplectionQueue> cq, TPtrArg<TSharedReceiveQueue> srq,
                   int sendQueueSize,
                   ibv_qp_type qp_type)
            : IBCtx(ctx)
            , CQ(cq)
            , SRQ(srq)
        {
            MyPSN = GetCycleCount() & 0xffffff; // should be random and different on different runs, 24bit

            ibv_qp_init_attr attr;
            Zero(attr);
            attr.qp_context = this; // not really useful
            attr.send_cq = cq->GetCQ();
            attr.recv_cq = cq->GetCQ();
            attr.srq = srq->GetSRQ();
            attr.cap.max_send_wr = sendQueueSize;
            attr.cap.max_recv_wr = 0; // we are using srq, no need for qp's rq
            attr.cap.max_send_sge = MAX_SGE;
            attr.cap.max_recv_sge = MAX_SGE;
            attr.cap.max_inline_data = MAX_INLINE_DATA_SIZE;
            attr.qp_type = qp_type;
            attr.sq_sig_all = 1; // inline sends need not be signaled, but if they are not work queue overflows

            TIBContext::TLock ibContext(IBCtx);
            QP = ibv_create_qp(ibContext.GetProtDomain(), &attr);
            Y_ASSERT(QP);

            //struct ibv_qp {
            //    struct ibv_context     *context;
            //    void         *qp_context;
            //    struct ibv_pd        *pd;
            //    struct ibv_cq        *send_cq;
            //    struct ibv_cq        *recv_cq;
            //    struct ibv_srq        *srq;
            //    ui32  handle;
            //    ui32  qp_num;
            //    enum ibv_qp_state       state;
            //    enum ibv_qp_type qp_type;

            //    pthread_mutex_t  mutex;
            //    pthread_cond_t  cond;
            //    ui32  events_completed;
            //};
            //qp_context  The value qp_context that was provided to ibv_create_qp()
            //qp_num  The number of this Queue Pair
            //state   The last known state of this Queue Pair. The actual state may be different from this state (in the RDMA device transitioned the state into other state)
            //qp_type  The Transport Service Type of this Queue Pair
        }
        ~TQueuePair() override {
            if (QP) {
                CHECK_Z(ibv_destroy_qp(QP));
            }
        }
        void FillSendAttrs(ibv_send_wr* wr, ibv_sge* sg,
                           ui64 localAddr, ui32 lKey, ui64 id, size_t len) {
            sg->addr = localAddr;
            sg->length = len;
            sg->lkey = lKey;
            Zero(*wr);
            wr->wr_id = id;
            wr->sg_list = sg;
            wr->num_sge = 1;
            if (len <= MAX_INLINE_DATA_SIZE) {
                wr->send_flags = IBV_SEND_INLINE;
            }
        }
        void FillSendAttrs(ibv_send_wr* wr, ibv_sge* sg,
                           TPtrArg<TMemoryRegion> mem, ui64 id, const void* data, size_t len) {
            ui64 localAddr = reinterpret_cast<ui64>(data) / sizeof(char);
            ui32 lKey = 0;
            if (mem) {
                Y_ASSERT(mem->IsCovered(data, len));
                lKey = mem->GetLKey();
            } else {
                Y_ASSERT(len <= MAX_INLINE_DATA_SIZE);
            }
            FillSendAttrs(wr, sg, localAddr, lKey, id, len);
        }

    public:
        int GetQPN() const {
            if (QP)
                return QP->qp_num;
            return 0;
        }
        int GetPSN() const {
            return MyPSN;
        }
        // we are using srq
        //void PostReceive(const TMemoryRegion &mem)
        //{
        //    ibv_recv_wr wr, *bad;
        //    ibv_sge sg;
        //    sg.addr = mem.Addr;
        //    sg.length = mem.Length;
        //    sg.lkey = mem.lkey;
        //    Zero(wr);
        //    wr.wr_id = 13;
        //    wr.sg_list = sg;
        //    wr.num_sge = 1;
        //    CHECK_Z(ibv_post_recv(QP, &wr, &bad));
        //}
    };

    class TRCQueuePair: public TQueuePair {
    public:
        TRCQueuePair(TPtrArg<TIBContext> ctx, TPtrArg<TComplectionQueue> cq, TPtrArg<TSharedReceiveQueue> srq, int sendQueueSize)
            : TQueuePair(ctx, cq, srq, sendQueueSize, IBV_QPT_RC)
        {
        }
        // SRQ should have receive posted
        void Init(const ibv_ah_attr& peerAddr, int peerQPN, int peerPSN) {
            Y_ASSERT(QP->qp_type == IBV_QPT_RC);
            ibv_qp_attr attr;
            //{
            //    enum ibv_qp_state qp_state;
            //    enum ibv_qp_state cur_qp_state;
            //    enum ibv_mtu path_mtu;
            //    enum ibv_mig_state path_mig_state;
            //    ui32 qkey;
            //    ui32 rq_psn;
            //    ui32 sq_psn;
            //    ui32 dest_qp_num;
            //    int qp_access_flags;
            //    struct ibv_qp_cap cap;
            //    struct ibv_ah_attr ah_attr;
            //    struct ibv_ah_attr alt_ah_attr;
            //    ui16 pkey_index;
            //    ui16 alt_pkey_index;
            //    ui8 en_sqd_async_notify;
            //    ui8 sq_draining;
            //    ui8 max_rd_atomic;
            //    ui8 max_dest_rd_atomic;
            //    ui8 min_rnr_timer;
            //    ui8 port_num;
            //    ui8 timeout;
            //    ui8 retry_cnt;
            //    ui8 rnr_retry;
            //    ui8 alt_port_num;
            //    ui8 alt_timeout;
            //};
            // RESET -> INIT
            Zero(attr);
            attr.qp_state = IBV_QPS_INIT;
            attr.pkey_index = 0;
            attr.port_num = peerAddr.port_num;
            // for connected QP
            attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
            CHECK_Z(ibv_modify_qp(QP, &attr,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));

            // INIT -> ReadyToReceive
            //PostReceive(mem);
            attr.qp_state = IBV_QPS_RTR;
            attr.path_mtu = IBV_MTU_512; // allows more fine grained VL arbitration
            // for connected QP
            attr.ah_attr = peerAddr;
            attr.dest_qp_num = peerQPN;
            attr.rq_psn = peerPSN;
            attr.max_dest_rd_atomic = MAX_OUTSTANDING_RDMA; // number of outstanding RDMA requests
            attr.min_rnr_timer = 12;                        // recommended
            CHECK_Z(ibv_modify_qp(QP, &attr,
                                  IBV_QP_STATE | IBV_QP_PATH_MTU |
                                      IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));

            // ReadyToReceive -> ReadyToTransmit
            attr.qp_state = IBV_QPS_RTS;
            // for connected QP
            attr.timeout = 14; // increased to 18 for sometime, 14 recommended
            //attr.retry_cnt = 0; // for debug purposes
            //attr.rnr_retry = 0; // for debug purposes
            attr.retry_cnt = 7; // release configuration
            attr.rnr_retry = 7; // release configuration (try forever)
            attr.sq_psn = MyPSN;
            attr.max_rd_atomic = MAX_OUTSTANDING_RDMA; // number of outstanding RDMA requests
            CHECK_Z(ibv_modify_qp(QP, &attr,
                                  IBV_QP_STATE |
                                      IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
        }
        void PostSend(TPtrArg<TMemoryRegion> mem, ui64 id, const void* data, size_t len) {
            ibv_send_wr wr, *bad;
            ibv_sge sg;
            FillSendAttrs(&wr, &sg, mem, id, data, len);
            wr.opcode = IBV_WR_SEND;
            //IBV_WR_RDMA_WRITE
            //IBV_WR_RDMA_WRITE_WITH_IMM
            //IBV_WR_SEND
            //IBV_WR_SEND_WITH_IMM
            //IBV_WR_RDMA_READ
            //wr.imm_data = xz;

            CHECK_Z(ibv_post_send(QP, &wr, &bad));
        }
        void PostRDMAWrite(ui64 remoteAddr, ui32 remoteKey,
                           TPtrArg<TMemoryRegion> mem, ui64 id, const void* data, size_t len) {
            ibv_send_wr wr, *bad;
            ibv_sge sg;
            FillSendAttrs(&wr, &sg, mem, id, data, len);
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = remoteAddr;
            wr.wr.rdma.rkey = remoteKey;

            CHECK_Z(ibv_post_send(QP, &wr, &bad));
        }
        void PostRDMAWrite(ui64 remoteAddr, ui32 remoteKey,
                           ui64 localAddr, ui32 localKey, ui64 id, size_t len) {
            ibv_send_wr wr, *bad;
            ibv_sge sg;
            FillSendAttrs(&wr, &sg, localAddr, localKey, id, len);
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = remoteAddr;
            wr.wr.rdma.rkey = remoteKey;

            CHECK_Z(ibv_post_send(QP, &wr, &bad));
        }
        void PostRDMAWriteImm(ui64 remoteAddr, ui32 remoteKey, ui32 immData,
                              TPtrArg<TMemoryRegion> mem, ui64 id, const void* data, size_t len) {
            ibv_send_wr wr, *bad;
            ibv_sge sg;
            FillSendAttrs(&wr, &sg, mem, id, data, len);
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wr.imm_data = immData;
            wr.wr.rdma.remote_addr = remoteAddr;
            wr.wr.rdma.rkey = remoteKey;

            CHECK_Z(ibv_post_send(QP, &wr, &bad));
        }
    };

    class TUDQueuePair: public TQueuePair {
        TIntrusivePtr<TIBPort> Port;

    public:
        TUDQueuePair(TPtrArg<TIBPort> port, TPtrArg<TComplectionQueue> cq, TPtrArg<TSharedReceiveQueue> srq, int sendQueueSize)
            : TQueuePair(port->GetCtx(), cq, srq, sendQueueSize, IBV_QPT_UD)
            , Port(port)
        {
        }
        // SRQ should have receive posted
        void Init(int qkey) {
            Y_ASSERT(QP->qp_type == IBV_QPT_UD);
            ibv_qp_attr attr;
            // RESET -> INIT
            Zero(attr);
            attr.qp_state = IBV_QPS_INIT;
            attr.pkey_index = 0;
            attr.port_num = Port->GetPort();
            // for unconnected qp
            attr.qkey = qkey;
            CHECK_Z(ibv_modify_qp(QP, &attr,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY));

            // INIT -> ReadyToReceive
            //PostReceive(mem);
            attr.qp_state = IBV_QPS_RTR;
            CHECK_Z(ibv_modify_qp(QP, &attr, IBV_QP_STATE));

            // ReadyToReceive -> ReadyToTransmit
            attr.qp_state = IBV_QPS_RTS;
            attr.sq_psn = 0;
            CHECK_Z(ibv_modify_qp(QP, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN));
        }
        void PostSend(TPtrArg<TAddressHandle> ah, int remoteQPN, int remoteQKey,
                      TPtrArg<TMemoryRegion> mem, ui64 id, const void* data, size_t len) {
            ibv_send_wr wr, *bad;
            ibv_sge sg;
            FillSendAttrs(&wr, &sg, mem, id, data, len);
            wr.opcode = IBV_WR_SEND;
            wr.wr.ud.ah = ah->GetAH();
            wr.wr.ud.remote_qpn = remoteQPN;
            wr.wr.ud.remote_qkey = remoteQKey;
            //IBV_WR_SEND_WITH_IMM
            //wr.imm_data = xz;

            CHECK_Z(ibv_post_send(QP, &wr, &bad));
        }
    };

    TIntrusivePtr<TIBPort> GetIBDevice();

#else
    //////////////////////////////////////////////////////////////////////////
    // stub for OS without IB support
    //////////////////////////////////////////////////////////////////////////
    enum ibv_wc_opcode {
        IBV_WC_SEND,
        IBV_WC_RDMA_WRITE,
        IBV_WC_RDMA_READ,
        IBV_WC_COMP_SWAP,
        IBV_WC_FETCH_ADD,
        IBV_WC_BIND_MW,
        IBV_WC_RECV = 1 << 7,
        IBV_WC_RECV_RDMA_WITH_IMM
    };

    enum ibv_wc_status {
        IBV_WC_SUCCESS,
        // lots of errors follow
    };
    //struct ibv_device;
    //struct ibv_pd;
    union ibv_gid {
        ui8 raw[16];
        struct {
            ui64 subnet_prefix;
            ui64 interface_id;
        } global;
    };

    struct ibv_wc {
        ui64 wr_id;
        enum ibv_wc_status status;
        enum ibv_wc_opcode opcode;
        ui32 imm_data; /* in network byte order */
        ui32 qp_num;
        ui32 src_qp;
    };
    struct ibv_grh {};
    struct ibv_ah_attr {
        ui8 sl;
    };
    //struct ibv_cq;
    class TIBContext: public TThrRefBase, TNonCopyable {
    public:
        bool IsValid() const {
            return false;
        }
        //ibv_context *GetContext() { return 0; }
        //ibv_pd *GetProtDomain() { return 0; }
    };

    class TIBPort: public TThrRefBase, TNonCopyable {
    public:
        TIBPort(TPtrArg<TIBContext>, int) {
        }
        int GetPort() const {
            return 1;
        }
        int GetLID() const {
            return 1;
        }
        TIBContext* GetCtx() {
            return 0;
        }
        void GetGID(ibv_gid* res) {
            Zero(*res);
        }
        void GetAHAttr(ibv_wc*, ibv_grh*, ibv_ah_attr*) {
        }
    };

    class TComplectionQueue: public TThrRefBase, TNonCopyable {
    public:
        TComplectionQueue(TPtrArg<TIBContext>, int) {
        }
        //ibv_cq *GetCQ() { return 0; }
        int Poll(ibv_wc*, int) {
            return 0;
        }
    };

    class TMemoryRegion: public TThrRefBase, TNonCopyable {
    public:
        TMemoryRegion(TPtrArg<TIBContext>, size_t) {
        }
        ui32 GetLKey() const {
            return 0;
        }
        ui32 GetRKey() const {
            return 0;
        }
        char* GetData() {
            return 0;
        }
        bool IsCovered(const void*, size_t) const {
            return false;
        }
    };

    class TSharedReceiveQueue: public TThrRefBase, TNonCopyable {
    public:
        TSharedReceiveQueue(TPtrArg<TIBContext>, int) {
        }
        //ibv_srq *GetSRQ() { return SRQ; }
        void PostReceive(TPtrArg<TMemoryRegion>, ui64, const void*, size_t) {
        }
    };

    inline void MakeAH(ibv_ah_attr*, TPtrArg<TIBPort>, int, int) {
    }

    class TAddressHandle: public TThrRefBase, TNonCopyable {
    public:
        TAddressHandle(TPtrArg<TIBContext>, ibv_ah_attr*) {
        }
        TAddressHandle(TPtrArg<TIBPort>, int, int) {
        }
        TAddressHandle(TPtrArg<TIBPort>, const TUdpAddress&, const TUdpAddress&, int) {
        }
        //ibv_ah *GetAH() { return AH; }
        bool IsValid() {
            return true;
        }
    };

    class TQueuePair: public TThrRefBase, TNonCopyable {
    public:
        int GetQPN() const {
            return 0;
        }
        int GetPSN() const {
            return 0;
        }
    };

    class TRCQueuePair: public TQueuePair {
    public:
        TRCQueuePair(TPtrArg<TIBContext>, TPtrArg<TComplectionQueue>, TPtrArg<TSharedReceiveQueue>, int) {
        }
        // SRQ should have receive posted
        void Init(const ibv_ah_attr&, int, int) {
        }
        void PostSend(TPtrArg<TMemoryRegion>, ui64, const void*, size_t) {
        }
        void PostRDMAWrite(ui64, ui32, TPtrArg<TMemoryRegion>, ui64, const void*, size_t) {
        }
        void PostRDMAWrite(ui64, ui32, ui64, ui32, ui64, size_t) {
        }
        void PostRDMAWriteImm(ui64, ui32, ui32, TPtrArg<TMemoryRegion>, ui64, const void*, size_t) {
        }
    };

    class TUDQueuePair: public TQueuePair {
        TIntrusivePtr<TIBPort> Port;

    public:
        TUDQueuePair(TPtrArg<TIBPort>, TPtrArg<TComplectionQueue>, TPtrArg<TSharedReceiveQueue>, int) {
        }
        // SRQ should have receive posted
        void Init(int) {
        }
        void PostSend(TPtrArg<TAddressHandle>, int, int, TPtrArg<TMemoryRegion>, ui64, const void*, size_t) {
        }
    };

    inline TIntrusivePtr<TIBPort> GetIBDevice() {
        return 0;
    }
#endif
}
