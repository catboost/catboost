#pragma once

#include "asio.h"
#include "poll_interrupter.h"

#include <library/cpp/neh/lfqueue.h>
#include <library/cpp/neh/pipequeue.h>

#include <library/cpp/dns/cache.h>

#include <util/generic/hash_set.h>
#include <util/network/iovec.h>
#include <util/network/pollerimpl.h>
#include <util/thread/lfqueue.h>
#include <util/thread/factory.h>

#ifdef DEBUG_ASIO
#define DBGOUT(args) Cout << args << Endl;
#else
#define DBGOUT(args)
#endif

namespace NAsio {
#if defined(_arm_)
    template <typename T>
    struct TLockFreeSequence {
        Y_NO_INLINE T& Get(size_t n) {
            with_lock (M) {
                return H[n];
            }
        }

        TMutex M;
        THashMap<size_t, T> H;
    };
#else
    //TODO: copypaste from neh, - need fix
    template <class T>
    class TLockFreeSequence {
    public:
        inline TLockFreeSequence() {
            memset((void*)T_, 0, sizeof(T_));
        }

        inline ~TLockFreeSequence() {
            for (size_t i = 0; i < Y_ARRAY_SIZE(T_); ++i) {
                delete[] T_[i];
            }
        }

        inline T& Get(size_t n) {
            const size_t i = GetValueBitCount(n + 1) - 1;

            return GetList(i)[n + 1 - (((size_t)1) << i)];
        }

    private:
        inline T* GetList(size_t n) {
            T* volatile* t = T_ + n;

            while (!*t) {
                TArrayHolder<T> nt(new T[((size_t)1) << n]);

                if (AtomicCas(t, nt.Get(), nullptr)) {
                    return nt.Release();
                }
            }

            return *t;
        }

    private:
        T* volatile T_[sizeof(size_t) * 8];
    };
#endif

    struct TOperationCompare {
        template <class T>
        static inline bool Compare(const T& l, const T& r) noexcept {
            return l.DeadLine() < r.DeadLine() || (l.DeadLine() == r.DeadLine() && &l < &r);
        }
    };

    //async operation, execute in contex TIOService()::Run() thread-executor
    //usualy used for call functors/callbacks
    class TOperation: public TRbTreeItem<TOperation, TOperationCompare>, public IHandlingContext {
    public:
        TOperation(TInstant deadline = TInstant::Max())
            : D_(deadline)
            , Speculative_(false)
            , RequiredRepeatExecution_(false)
            , ND_(deadline)
        {
        }

        //register this operation in svc.impl.
        virtual void AddOp(TIOService::TImpl&) = 0;

        //return false, if operation not completed
        virtual bool Execute(int errorCode = 0) = 0;

        void ContinueUseHandler(TDeadline deadline) override {
            RequiredRepeatExecution_ = true;
            ND_ = deadline;
        }

        virtual void Finalize() = 0;

        inline TInstant Deadline() const noexcept {
            return D_;
        }

        inline TInstant DeadLine() const noexcept {
            return D_;
        }

        inline bool Speculative() const noexcept {
            return Speculative_;
        }

        inline bool IsRequiredRepeat() const noexcept {
            return RequiredRepeatExecution_;
        }

        inline void PrepareReExecution() noexcept {
            RequiredRepeatExecution_ = false;
            D_ = ND_;
        }

    protected:
        TInstant D_;
        bool Speculative_;             //if true, operation will be runned immediately after dequeue (even without wating any event)
                                       //as sample used for optimisation writing, - obviously in buffers exist space for write
        bool RequiredRepeatExecution_; //set to true, if required re-exec operation
        TInstant ND_;                  //new deadline (for re-exec operation)
    };

    typedef TAutoPtr<TOperation> TOperationPtr;

    class TNoneOperation: public TOperation {
    public:
        TNoneOperation(TInstant deadline = TInstant::Max())
            : TOperation(deadline)
        {
        }

        void AddOp(TIOService::TImpl&) override {
            Y_ASSERT(0);
        }

        void Finalize() override {
        }
    };

    class TPollFdEventHandler;

    //descriptor use operation
    class TFdOperation: public TOperation {
    public:
        enum TPollType {
            PollRead,
            PollWrite
        };

        TFdOperation(SOCKET fd, TPollType pt, TInstant deadline = TInstant::Max())
            : TOperation(deadline)
            , Fd_(fd)
            , PT_(pt)
            , PH_(nullptr)
        {
            Y_ASSERT(Fd() != INVALID_SOCKET);
        }

        inline SOCKET Fd() const noexcept {
            return Fd_;
        }

        inline bool IsPollRead() const noexcept {
            return PT_ == PollRead;
        }

        void AddOp(TIOService::TImpl& srv) override;

        void Finalize() override;

    protected:
        SOCKET Fd_;
        TPollType PT_;

    public:
        TAutoPtr<TPollFdEventHandler>* PH_;
    };

    typedef TAutoPtr<TFdOperation> TFdOperationPtr;

    class TPollFdEventHandler {
    public:
        TPollFdEventHandler(SOCKET fd, TIOService::TImpl& srv)
            : Fd_(fd)
            , HandledEvents_(0)
            , Srv_(srv)
        {
        }

        virtual ~TPollFdEventHandler() {
            Y_ASSERT(ReadOperations_.size() == 0);
            Y_ASSERT(WriteOperations_.size() == 0);
        }

        inline void AddReadOp(TFdOperationPtr op) {
            ReadOperations_.push_back(op);
        }

        inline void AddWriteOp(TFdOperationPtr op) {
            WriteOperations_.push_back(op);
        }

        virtual void OnFdEvent(int status, ui16 filter) {
            DBGOUT("PollEvent(fd=" << Fd_ << ", " << status << ", " << filter << ")");
            if (status) {
                ExecuteOperations(ReadOperations_, status);
                ExecuteOperations(WriteOperations_, status);
            } else {
                if (filter & CONT_POLL_READ) {
                    ExecuteOperations(ReadOperations_, status);
                }
                if (filter & CONT_POLL_WRITE) {
                    ExecuteOperations(WriteOperations_, status);
                }
            }
        }

        typedef TVector<TFdOperationPtr> TFdOperations;

        void ExecuteOperations(TFdOperations& oprs, int errorCode);

        //return true if filter handled events changed and require re-configure events poller
        virtual bool FixHandledEvents() noexcept {
            DBGOUT("TPollFdEventHandler::FixHandledEvents()");
            ui16 filter = 0;

            if (WriteOperations_.size()) {
                filter |= CONT_POLL_WRITE;
            }
            if (ReadOperations_.size()) {
                filter |= CONT_POLL_READ;
            }

            if (Y_LIKELY(HandledEvents_ == filter)) {
                return false;
            }

            HandledEvents_ = filter;
            return true;
        }

        inline bool FinishOp(TFdOperations& oprs, TFdOperation* op) noexcept {
            for (TFdOperations::iterator it = oprs.begin(); it != oprs.end(); ++it) {
                if (it->Get() == op) {
                    FinishedOperations_.push_back(*it);
                    oprs.erase(it);
                    return true;
                }
            }
            return false;
        }

        void DelOp(TFdOperation* op);

        inline SOCKET Fd() const noexcept {
            return Fd_;
        }

        inline ui16 HandledEvents() const noexcept {
            return HandledEvents_;
        }

        inline void AddHandlingEvent(ui16 ev) noexcept {
            HandledEvents_ |= ev;
        }

        inline void DestroyFinishedOperations() {
            FinishedOperations_.clear();
        }

        TIOService::TImpl& GetServiceImpl() const noexcept {
            return Srv_;
        }

    protected:
        SOCKET Fd_;
        ui16 HandledEvents_;
        TIOService::TImpl& Srv_;

    private:
        TVector<TFdOperationPtr> ReadOperations_;
        TVector<TFdOperationPtr> WriteOperations_;
        // we can't immediatly destroy finished operations, this can cause closing used socket descriptor Fd_
        // (on cascade deletion operation object-handler), but later we use Fd_ for modify handled events at poller,
        // so we collect here finished operations and destroy it only after update poller, -
        // call FixHandledEvents(TPollFdEventHandlerPtr&)
        TVector<TFdOperationPtr> FinishedOperations_;
    };

    //additional descriptor for poller, used for interrupt current poll wait
    class TInterrupterHandler: public TPollFdEventHandler {
    public:
        TInterrupterHandler(TIOService::TImpl& srv, TPollInterrupter& pi)
            : TPollFdEventHandler(pi.Fd(), srv)
            , PI_(pi)
        {
            HandledEvents_ = CONT_POLL_READ;
        }

        ~TInterrupterHandler() override {
            DBGOUT("~TInterrupterHandler");
        }

        void OnFdEvent(int status, ui16 filter) override;

        bool FixHandledEvents() noexcept override {
            DBGOUT("TInterrupterHandler::FixHandledEvents()");
            return false;
        }

    private:
        TPollInterrupter& PI_;
    };

    namespace {
        inline TAutoPtr<IPollerFace> CreatePoller() {
            try {
#if defined(_linux_)
                return IPollerFace::Construct(TStringBuf("epoll"));
#endif
#if defined(_freebsd_) || defined(_darwin_)
                return IPollerFace::Construct(TStringBuf("kqueue"));
#endif
            } catch (...) {
                Cdbg << CurrentExceptionMessage() << Endl;
            }
            return IPollerFace::Default();
        }
    }

    //some equivalent TContExecutor
    class TIOService::TImpl: public TNonCopyable {
    public:
        typedef TAutoPtr<TPollFdEventHandler> TEvh;
        typedef TLockFreeSequence<TEvh> TEventHandlers;

        class TTimer {
        public:
            typedef THashSet<TOperation*> TOperations;

            TTimer(TIOService::TImpl& srv)
                : Srv_(srv)
            {
            }

            virtual ~TTimer() {
                FailOperations(ECANCELED);
            }

            void AddOp(TOperation* op) {
                THolder<TOperation> tmp(op);
                Operations_.insert(op);
                Y_UNUSED(tmp.Release());
                Srv_.RegisterOpDeadline(op);
                Srv_.IncTimersOp();
            }

            void DelOp(TOperation* op) {
                TOperations::iterator it = Operations_.find(op);
                if (it != Operations_.end()) {
                    Srv_.DecTimersOp();
                    delete op;
                    Operations_.erase(it);
                }
            }

            inline void FailOperations(int ec) {
                for (auto operation : Operations_) {
                    try {
                        operation->Execute(ec); //throw ?
                    } catch (...) {
                    }
                    Srv_.DecTimersOp();
                    delete operation;
                }
                Operations_.clear();
            }

            TIOService::TImpl& GetIOServiceImpl() const noexcept {
                return Srv_;
            }

        protected:
            TIOService::TImpl& Srv_;
            THashSet<TOperation*> Operations_;
        };

        class TTimers: public THashSet<TTimer*> {
        public:
            ~TTimers() {
                for (auto it : *this) {
                    delete it;
                }
            }
        };

        TImpl()
            : P_(CreatePoller())
            , DeadlinesQueue_(*this)
        {
        }

        ~TImpl() {
            TOperationPtr op;

            while (OpQueue_.Dequeue(&op)) { //cancel all enqueued operations
                try {
                    op->Execute(ECANCELED);
                } catch (...) {
                }
                op.Destroy();
            }
        }

        //similar TContExecutor::Execute() or io_service::run()
        //process event loop (exit if none to do (no timers or event handlers))
        void Run();

        //enqueue functor fo call in Run() eventloop (thread safing)
        inline void Post(TCompletionHandler h) {
            class TFuncOperation: public TNoneOperation {
            public:
                TFuncOperation(TCompletionHandler completionHandler)
                    : TNoneOperation()
                    , H_(std::move(completionHandler))
                {
                    Speculative_ = true;
                }

            private:
                //return false, if operation not completed
                bool Execute(int errorCode) override {
                    Y_UNUSED(errorCode);
                    H_();
                    return true;
                }

                TCompletionHandler H_;
            };

            ScheduleOp(new TFuncOperation(std::move(h)));
        }

        //cancel all current operations (handlers be called with errorCode == ECANCELED)
        void Abort();
        bool HasAbort() {
            return AtomicGet(HasAbort_);
        }

        inline void ScheduleOp(TOperationPtr op) { //throw std::bad_alloc
            Y_ASSERT(!Aborted_);
            Y_ASSERT(!!op);
            OpQueue_.Enqueue(op);
            Interrupt();
        }

        inline void Interrupt() noexcept {
            AtomicSet(NeedCheckOpQueue_, 1);
            if (AtomicAdd(IsWaiting_, 0) == 1) {
                I_.Interrupt();
            }
        }

        inline void UpdateOpDeadline(TOperation* op) {
            TInstant oldDeadline = op->Deadline();
            op->PrepareReExecution();

            if (oldDeadline == op->Deadline()) {
                return;
            }

            if (oldDeadline != TInstant::Max()) {
                op->UnLink();
            }
            if (op->Deadline() != TInstant::Max()) {
                DeadlinesQueue_.Register(op);
            }
        }

        inline size_t GetOpQueueSize() noexcept {
            return OpQueue_.Size();
        }

        void SyncRegisterTimer(TTimer* t) {
            Timers_.insert(t);
        }

        inline void SyncUnregisterAndDestroyTimer(TTimer* t) {
            Timers_.erase(t);
            delete t;
        }

        inline void IncTimersOp() noexcept {
            ++TimersOpCnt_;
        }

        inline void DecTimersOp() noexcept {
            --TimersOpCnt_;
        }

        inline void WorkStarted() {
            AtomicIncrement(OutstandingWork_);
        }

        inline void WorkFinished() {
            if (AtomicDecrement(OutstandingWork_) == 0) {
                Interrupt();
            }
        }

    private:
        void ProcessAbort();

        inline TEvh& EnsureGetEvh(SOCKET fd) {
            TEvh& evh = Evh_.Get(fd);
            if (!evh) {
                evh.Reset(new TPollFdEventHandler(fd, *this));
            }
            return evh;
        }

        inline void OnTimeoutOp(TOperation* op) {
            DBGOUT("OnTimeoutOp");
            try {
                op->Execute(ETIMEDOUT); //throw ?
            } catch (...) {
                op->Finalize();
                throw;
            }

            if (op->IsRequiredRepeat()) {
                //operation not completed
                UpdateOpDeadline(op);
            } else {
                //destroy operation structure
                op->Finalize();
            }
        }

    public:
        inline void FixHandledEvents(TEvh& evh) {
            if (!!evh) {
                if (evh->FixHandledEvents()) {
                    if (!evh->HandledEvents()) {
                        DelEventHandler(evh);
                        evh.Destroy();
                    } else {
                        ModEventHandler(evh);
                        evh->DestroyFinishedOperations();
                    }
                } else {
                    evh->DestroyFinishedOperations();
                }
            }
        }

    private:
        inline TEvh& GetHandlerForOp(TFdOperation* op) {
            TEvh& evh = EnsureGetEvh(op->Fd());
            op->PH_ = &evh;
            return evh;
        }

        void ProcessOpQueue() {
            if (!AtomicGet(NeedCheckOpQueue_)) {
                return;
            }
            AtomicSet(NeedCheckOpQueue_, 0);

            TOperationPtr op;

            while (OpQueue_.Dequeue(&op)) {
                if (op->Speculative()) {
                    if (op->Execute(Y_UNLIKELY(Aborted_) ? ECANCELED : 0)) {
                        op.Destroy();
                        continue; //operation completed
                    }

                    if (!op->IsRequiredRepeat()) {
                        op->PrepareReExecution();
                    }
                }
                RegisterOpDeadline(op.Get());
                op.Get()->AddOp(*this); // ... -> AddOp()
                Y_UNUSED(op.Release());
            }
        }

        inline void RegisterOpDeadline(TOperation* op) {
            if (op->DeadLine() != TInstant::Max()) {
                DeadlinesQueue_.Register(op);
            }
        }

    public:
        inline void AddOp(TFdOperation* op) {
            DBGOUT("AddOp<Fd>(" << op->Fd() << ")");
            TEvh& evh = GetHandlerForOp(op);
            if (op->IsPollRead()) {
                evh->AddReadOp(op);
                EnsureEventHandled(evh, CONT_POLL_READ);
            } else {
                evh->AddWriteOp(op);
                EnsureEventHandled(evh, CONT_POLL_WRITE);
            }
        }

    private:
        inline void EnsureEventHandled(TEvh& evh, ui16 ev) {
            if (!evh->HandledEvents()) {
                evh->AddHandlingEvent(ev);
                AddEventHandler(evh);
            } else {
                if ((evh->HandledEvents() & ev) == 0) {
                    evh->AddHandlingEvent(ev);
                    ModEventHandler(evh);
                }
            }
        }

    public:
        //cancel all current operations for socket
        //method MUST be called from Run() thread-executor
        void CancelFdOp(SOCKET fd) {
            TEvh& evh = Evh_.Get(fd);
            if (!evh) {
                return;
            }

            OnFdEvent(evh, ECANCELED, CONT_POLL_READ | CONT_POLL_WRITE);
        }

    private:
        //helper for fixing handled events even in case exception
        struct TExceptionProofFixerHandledEvents {
            TExceptionProofFixerHandledEvents(TIOService::TImpl& srv, TEvh& iEvh)
                : Srv_(srv)
                , Evh_(iEvh)
            {
            }

            ~TExceptionProofFixerHandledEvents() {
                Srv_.FixHandledEvents(Evh_);
            }

            TIOService::TImpl& Srv_;
            TEvh& Evh_;
        };

        inline void OnFdEvent(TEvh& evh, int status, ui16 filter) {
            TExceptionProofFixerHandledEvents fixer(*this, evh);
            Y_UNUSED(fixer);
            evh->OnFdEvent(status, filter);
        }

        inline void AddEventHandler(TEvh& evh) {
            if (evh->Fd() > MaxFd_) {
                MaxFd_ = evh->Fd();
            }
            SetEventHandler(&evh, evh->Fd(), evh->HandledEvents());
            ++FdEventHandlersCnt_;
        }

        inline void ModEventHandler(TEvh& evh) {
            SetEventHandler(&evh, evh->Fd(), evh->HandledEvents());
        }

        inline void DelEventHandler(TEvh& evh) {
            SetEventHandler(&evh, evh->Fd(), 0);
            --FdEventHandlersCnt_;
        }

        inline void SetEventHandler(void* h, int fd, ui16 flags) {
            DBGOUT("SetEventHandler(" << fd << ", " << flags << ")");
            P_->Set(h, fd, flags);
        }

        //exception safe call DelEventHandler
        struct TInterrupterKeeper {
            TInterrupterKeeper(TImpl& srv, TEvh& iEvh)
                : Srv_(srv)
                , Evh_(iEvh)
            {
                Srv_.AddEventHandler(Evh_);
            }

            ~TInterrupterKeeper() {
                Srv_.DelEventHandler(Evh_);
            }

            TImpl& Srv_;
            TEvh& Evh_;
        };

        TAutoPtr<IPollerFace> P_;
        TPollInterrupter I_;
        TAtomic IsWaiting_ = 0;
        TAtomic NeedCheckOpQueue_ = 0;
        TAtomic OutstandingWork_ = 0;

        NNeh::TAutoLockFreeQueue<TOperation> OpQueue_;

        TEventHandlers Evh_; //i/o event handlers
        TTimers Timers_;     //timeout event handlers

        size_t FdEventHandlersCnt_ = 0; //i/o event handlers counter
        size_t TimersOpCnt_ = 0;        //timers op counter
        SOCKET MaxFd_ = 0;              //max used descriptor num
        TAtomic HasAbort_ = 0;
        bool Aborted_ = false;

        class TDeadlinesQueue {
        public:
            TDeadlinesQueue(TIOService::TImpl& srv)
                : Srv_(srv)
            {
            }

            inline void Register(TOperation* op) {
                Deadlines_.Insert(op);
            }

            TInstant NextDeadline() {
                TDeadlines::TIterator it = Deadlines_.Begin();

                while (it != Deadlines_.End()) {
                    if (it->DeadLine() > TInstant::Now()) {
                        DBGOUT("TDeadlinesQueue::NewDeadline:" << (it->DeadLine().GetValue() - TInstant::Now().GetValue()));
                        return it->DeadLine();
                    }

                    TOperation* op = &*(it++);
                    Srv_.OnTimeoutOp(op);
                }

                return Deadlines_.Empty() ? TInstant::Max() : Deadlines_.Begin()->DeadLine();
            }

        private:
            typedef TRbTree<TOperation, TOperationCompare> TDeadlines;
            TDeadlines Deadlines_;
            TIOService::TImpl& Srv_;
        };

        TDeadlinesQueue DeadlinesQueue_;
    };
}
