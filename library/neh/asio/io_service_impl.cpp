#include "io_service_impl.h"

#include <library/cpp/coroutine/engine/poller.h>

using namespace NAsio;

void TFdOperation::AddOp(TIOService::TImpl& srv) {
    srv.AddOp(this);
}

void TFdOperation::Finalize() {
    (*PH_)->DelOp(this);
}

void TPollFdEventHandler::ExecuteOperations(TFdOperations& oprs, int errorCode) {
    TFdOperations::iterator it = oprs.begin();

    try {
        while (it != oprs.end()) {
            TFdOperation* op = it->Get();

            if (op->Execute(errorCode)) { // throw ?
                if (op->IsRequiredRepeat()) {
                    Srv_.UpdateOpDeadline(op);
                    ++it; //operation completed, but want be repeated
                } else {
                    FinishedOperations_.push_back(*it);
                    it = oprs.erase(it);
                }
            } else {
                ++it; //operation not completed
            }
        }
    } catch (...) {
        if (it != oprs.end()) {
            FinishedOperations_.push_back(*it);
            oprs.erase(it);
        }
        throw;
    }
}

void TPollFdEventHandler::DelOp(TFdOperation* op) {
    TAutoPtr<TPollFdEventHandler>& evh = *op->PH_;

    if (op->IsPollRead()) {
        Y_ASSERT(FinishOp(ReadOperations_, op));
    } else {
        Y_ASSERT(FinishOp(WriteOperations_, op));
    }
    Srv_.FixHandledEvents(evh); //alarm, - 'this' can be destroyed here!
}

void TInterrupterHandler::OnFdEvent(int status, ui16 filter) {
    if (!status && (filter & CONT_POLL_READ)) {
        PI_.Reset();
    }
}

void TIOService::TImpl::Run() {
    TEvh& iEvh = Evh_.Get(I_.Fd());
    iEvh.Reset(new TInterrupterHandler(*this, I_));

    TInterrupterKeeper ik(*this, iEvh);
    Y_UNUSED(ik);
    IPollerFace::TEvents evs;
    AtomicSet(NeedCheckOpQueue_, 1);
    TInstant deadline;

    while (Y_LIKELY(!Aborted_ && (AtomicGet(OutstandingWork_) || FdEventHandlersCnt_ > 1 || TimersOpCnt_ || AtomicGet(NeedCheckOpQueue_)))) {
        //while
        //  expected work (external flag)
        //  or have event handlers (exclude interrupter)
        //  or have not completed timer operation
        //  or have any operation in queues

        AtomicIncrement(IsWaiting_);
        if (!AtomicGet(NeedCheckOpQueue_)) {
            P_->Wait(evs, deadline);
        }
        AtomicDecrement(IsWaiting_);

        if (evs.size()) {
            for (IPollerFace::TEvents::const_iterator iev = evs.begin(); iev != evs.end() && !Aborted_; ++iev) {
                const IPollerFace::TEvent& ev = *iev;
                TEvh& evh = *(TEvh*)ev.Data;

                if (!evh) {
                    continue; //op. cancel (see ProcessOpQueue) can destroy evh
                }

                int status = ev.Status;
                if (ev.Status == EIO) {
                    int error = status;
                    if (GetSockOpt(evh->Fd(), SOL_SOCKET, SO_ERROR, error) == 0) {
                        status = error;
                    }
                }

                OnFdEvent(evh, status, ev.Filter); //here handle fd events
                //immediatly after handling events for one descriptor check op. queue
                //often queue can contain another operation for this fd (next async read as sample)
                //so we can optimize redundant epoll_ctl (or similar) calls
                ProcessOpQueue();
            }

            evs.clear();
        } else {
            ProcessOpQueue();
        }

        deadline = DeadlinesQueue_.NextDeadline(); //here handle timeouts/process timers
    }
}

void TIOService::TImpl::Abort() {
    class TAbortOperation: public TNoneOperation {
    public:
        TAbortOperation(TIOService::TImpl& srv)
            : TNoneOperation()
            , Srv_(srv)
        {
            Speculative_ = true;
        }

    private:
        bool Execute(int errorCode) override {
            Y_UNUSED(errorCode);
            Srv_.ProcessAbort();
            return true;
        }

        TIOService::TImpl& Srv_;
    };
    AtomicSet(HasAbort_, 1);
    ScheduleOp(new TAbortOperation(*this));
}

void TIOService::TImpl::ProcessAbort() {
    Aborted_ = true;

    for (int fd = 0; fd <= MaxFd_; ++fd) {
        TEvh& evh = Evh_.Get(fd);
        if (!!evh && evh->Fd() != I_.Fd()) {
            OnFdEvent(evh, ECANCELED, CONT_POLL_READ | CONT_POLL_WRITE);
        }
    }

    for (auto t : Timers_) {
        t->FailOperations(ECANCELED);
    }

    TOperationPtr op;
    while (OpQueue_.Dequeue(&op)) { //cancel all enqueued operations
        try {
            op->Execute(ECANCELED);
        } catch (...) {
        }
        op.Destroy();
    }
}
