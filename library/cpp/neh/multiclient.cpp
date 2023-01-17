#include "multiclient.h"
#include "utils.h"

#include <library/cpp/containers/intrusive_rb_tree/rb_tree.h>

#include <atomic>

namespace {
    using namespace NNeh;

    struct TCompareDeadline {
        template <class T>
        static inline bool Compare(const T& l, const T& r) noexcept {
            return l.Deadline() < r.Deadline() || (l.Deadline() == r.Deadline() && &l < &r);
        }
    };

    class TMultiClient: public IMultiClient, public TThrRefBase {
        class TRequestSupervisor: public TRbTreeItem<TRequestSupervisor, TCompareDeadline>, public IOnRecv, public TThrRefBase, public TNonCopyable {
        private:
            TRequestSupervisor() {
            } //disable

        public:
            inline TRequestSupervisor(const TRequest& request, TMultiClient* mc) noexcept
                : MC_(mc)
                , Request_(request)
                , Maked_(0)
                , FinishOnMakeRequest_(0)
                , Handled_(0)
                , Dequeued_(false)
            {
            }

            inline TInstant Deadline() const noexcept {
                return Request_.Deadline;
            }

            //not thread safe (can be called at some time from TMultiClient::Request() and TRequestSupervisor::OnNotify())
            void OnMakeRequest(THandleRef h) noexcept {
                //request can be mark as maked only once, so only one/first call set handle
                if (AtomicCas(&Maked_, 1, 0)) {
                    H_.Swap(h);
                    //[paranoid mode on] make sure handle be initiated before return
                    AtomicSet(FinishOnMakeRequest_, 1);
                } else {
                    while (!AtomicGet(FinishOnMakeRequest_)) {
                        SpinLockPause();
                    }
                    //[paranoid mode off]
                }
            }

            void FillEvent(TEvent& ev) noexcept {
                ev.Hndl = H_;
                FillEventUserData(ev);
            }

            void FillEventUserData(TEvent& ev) noexcept {
                ev.UserData = Request_.UserData;
            }

            void ResetRequest() noexcept { //destroy keepaliving cross-ref TRequestSupervisor<->THandle
                H_.Drop();
            }

            //method OnProcessRequest() & OnProcessResponse() executed from Wait() context (thread)
            void OnEndProcessRequest() {
                Dequeued_ = true;
                if (Y_UNLIKELY(IsHandled())) {
                    ResetRequest(); //race - response already handled before processing request from queue
                } else {
                    MC_->RegisterRequest(this);
                }
            }

            void OnEndProcessResponse() {
                if (Y_LIKELY(Dequeued_)) {
                    UnLink();
                    ResetRequest();
                } //else request yet not dequeued/registered, so we not need unlink request
                  //(when we later dequeue request OnEndProcessRequest()...IsHandled() return true and we reset request)
            }

            //IOnRecv interface
            void OnNotify(THandle& h) override {
                if (Y_LIKELY(MarkAsHandled())) {
                    THandleRef hr(&h);
                    OnMakeRequest(hr); //fix race with receiving response before return control from NNeh::Request()
                    MC_->ScheduleResponse(this, hr);
                }
            }

            void OnRecv(THandle&) noexcept override {
                UnRef();
            }

            void OnEnd() noexcept override {
                UnRef();
            }
            //

            //request can be handled only once, so only one/first call MarkAsHandled() return true
            bool MarkAsHandled() noexcept {
                return AtomicCas(&Handled_, 1, 0);
            }

            bool IsHandled() const noexcept {
                return AtomicGet(Handled_);
            }

        private:
            TIntrusivePtr<TMultiClient> MC_;
            TRequest Request_;
            THandleRef H_;
            TAtomic Maked_;
            TAtomic FinishOnMakeRequest_;
            TAtomic Handled_;
            bool Dequeued_;
        };

        typedef TRbTree<TRequestSupervisor, TCompareDeadline> TRequestsSupervisors;
        typedef TIntrusivePtr<TRequestSupervisor> TRequestSupervisorRef;

    public:
        TMultiClient()
            : Interrupt_(false)
            , NearDeadline_(TInstant::Max().GetValue())
            , E_(::TSystemEvent::rAuto)
            , Shutdown_(false)
        {
        }

        struct TResetRequest {
            inline void operator()(TRequestSupervisor& rs) const noexcept {
                rs.ResetRequest();
            }
        };

        void Shutdown() {
            //reset THandleRef's for all exist supervisors and jobs queue (+prevent creating new)
            //- so we break crossref-chain, which prevent destroy this object THande->TRequestSupervisor->TMultiClient)
            Shutdown_ = true;
            RS_.ForEachNoOrder(TResetRequest());
            RS_.Clear();
            CleanQueue();
        }

    private:
        class IJob {
        public:
            virtual ~IJob() {
            }
            virtual bool Process(TEvent&) = 0;
            virtual void Cancel() = 0;
        };
        typedef TAutoPtr<IJob> TJobPtr;

        class TNewRequest: public IJob {
        public:
            TNewRequest(TRequestSupervisorRef& rs)
                : RS_(rs)
            {
            }

        private:
            bool Process(TEvent&) override {
                RS_->OnEndProcessRequest();
                return false;
            }

            void Cancel() override {
                RS_->ResetRequest();
            }

            TRequestSupervisorRef RS_;
        };

        class TNewResponse: public IJob {
        public:
            TNewResponse(TRequestSupervisor* rs, THandleRef& h) noexcept
                : RS_(rs)
                , H_(h)
            {
            }

        private:
            bool Process(TEvent& ev) override {
                ev.Type = TEvent::Response;
                ev.Hndl = H_;
                RS_->FillEventUserData(ev);
                RS_->OnEndProcessResponse();
                return true;
            }

            void Cancel() override {
                RS_->ResetRequest();
            }

            TRequestSupervisorRef RS_;
            THandleRef H_;
        };

    public:
        THandleRef Request(const TRequest& request) override {
            TIntrusivePtr<TRequestSupervisor> rs(new TRequestSupervisor(request, this));
            THandleRef h;
            try {
                rs->Ref();
                h = NNeh::Request(request.Msg, rs.Get());
                //accurately handle race when processing new request event
                //(we already can receive response (call OnNotify) before we schedule info about new request here)
            } catch (...) {
                rs->UnRef();
                throw;
            }
            rs->OnMakeRequest(h);
            ScheduleRequest(rs, h, request.Deadline);
            return h;
        }

        bool Wait(TEvent& ev, const TInstant deadline_ = TInstant::Max()) override {
            while (!Interrupt_) {
                TInstant deadline = deadline_;
                const TInstant now = TInstant::Now();
                if (deadline != TInstant::Max() && now >= deadline) {
                    break;
                }

                { //process jobs queue (requests/responses info)
                    TAutoPtr<IJob> j;
                    while (JQ_.Dequeue(&j)) {
                        if (j->Process(ev)) {
                            return true;
                        }
                    }
                }

                if (!RS_.Empty()) {
                    TRequestSupervisor* nearRS = &*RS_.Begin();
                    if (nearRS->Deadline() <= now) {
                        if (!nearRS->MarkAsHandled()) {
                            //race with notify, - now in queue must exist response job for this request
                            continue;
                        }
                        ev.Type = TEvent::Timeout;
                        nearRS->FillEvent(ev);
                        nearRS->ResetRequest();
                        nearRS->UnLink();
                        return true;
                    }
                    deadline = Min(nearRS->Deadline(), deadline);
                }

                if (SetNearDeadline(deadline)) {
                    continue; //update deadline to more far time, so need re-check queue for avoiding race
                }

                E_.WaitD(deadline);
            }
            Interrupt_ = false;
            return false;
        }

        void Interrupt() override {
            Interrupt_ = true;
            Signal();
        }

        size_t QueueSize() override {
            return JQ_.Size();
        }

    private:
        void Signal() {
            //TODO:try optimize - hack with skipping signaling if not have waiters (reduce mutex usage)
            E_.Signal();
        }

        void ScheduleRequest(TIntrusivePtr<TRequestSupervisor>& rs, const THandleRef& h, const TInstant& deadline) {
            TJobPtr j(new TNewRequest(rs));
            JQ_.Enqueue(j);
            if (!h->Signalled()) {
                if (deadline.GetValue() < GetNearDeadline_()) {
                    Signal();
                }
            }
        }

        void ScheduleResponse(TRequestSupervisor* rs, THandleRef& h) {
            TJobPtr j(new TNewResponse(rs, h));
            JQ_.Enqueue(j);
            if (Y_UNLIKELY(Shutdown_)) {
                CleanQueue();
            } else {
                Signal();
            }
        }

        //return true, if deadline re-installed to more late time
        bool SetNearDeadline(const TInstant& deadline) {
            bool deadlineMovedFurther = deadline.GetValue() > GetNearDeadline_();
            SetNearDeadline_(deadline.GetValue());
            return deadlineMovedFurther;
        }

        //used only from Wait()
        void RegisterRequest(TRequestSupervisor* rs) {
            if (rs->Deadline() != TInstant::Max()) {
                RS_.Insert(rs);
            } else {
                rs->ResetRequest(); //prevent blocking destruction 'endless' requests
            }
        }

        void CleanQueue() {
            TAutoPtr<IJob> j;
            while (JQ_.Dequeue(&j)) {
                j->Cancel();
            }
        }

    private:
        void SetNearDeadline_(const TInstant::TValue& v) noexcept {
            TGuard<TAdaptiveLock> g(NDLock_);
            NearDeadline_.store(v, std::memory_order_release);
        }

        TInstant::TValue GetNearDeadline_() const noexcept {
            TGuard<TAdaptiveLock> g(NDLock_);
            return NearDeadline_.load(std::memory_order_acquire);
        }

        NNeh::TAutoLockFreeQueue<IJob> JQ_;
        TAtomicBool Interrupt_;
        TRequestsSupervisors RS_;
        TAdaptiveLock NDLock_;
        std::atomic<TInstant::TValue> NearDeadline_;
        ::TSystemEvent E_;
        TAtomicBool Shutdown_;
    };

    class TMultiClientAutoShutdown: public IMultiClient {
    public:
        TMultiClientAutoShutdown()
            : MC_(new TMultiClient())
        {
        }

        ~TMultiClientAutoShutdown() override {
            MC_->Shutdown();
        }

        size_t QueueSize() override {
            return MC_->QueueSize();
        }

    private:
        THandleRef Request(const TRequest& req) override {
            return MC_->Request(req);
        }

        bool Wait(TEvent& ev, TInstant deadline = TInstant::Max()) override {
            return MC_->Wait(ev, deadline);
        }

        void Interrupt() override {
            return MC_->Interrupt();
        }

    private:
        TIntrusivePtr<TMultiClient> MC_;
    };
}

TMultiClientPtr NNeh::CreateMultiClient() {
    return new TMultiClientAutoShutdown();
}
