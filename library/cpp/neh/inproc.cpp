#include "inproc.h"

#include "details.h"
#include "neh.h"
#include "location.h"
#include "utils.h"
#include "factory.h"

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/singleton.h>
#include <util/stream/output.h>
#include <util/string/cast.h>

using namespace NNeh;

namespace {
    const TString canceled = "canceled";

    struct TInprocHandle: public TNotifyHandle {
        inline TInprocHandle(const TMessage& msg, IOnRecv* r, TStatCollector* sc) noexcept
            : TNotifyHandle(r, msg, sc)
            , Canceled_(false)
            , NotifyCnt_(0)
        {
        }

        bool MessageSendedCompletely() const noexcept override {
            return true;
        }

        void Cancel() noexcept override {
            THandle::Cancel(); //inform stat collector
            Canceled_ = true;
            try {
                if (MarkReplied()) {
                    NotifyError(new TError(canceled, TError::Cancelled));
                }
            } catch (...) {
                Cdbg << "inproc canc. " << CurrentExceptionMessage() << Endl;
            }
        }

        inline void SendReply(const TString& resp) {
            if (MarkReplied()) {
                NotifyResponse(resp);
            }
        }

        inline void SendError(const TString& details) {
            if (MarkReplied()) {
                NotifyError(new TError{details, TError::ProtocolSpecific, 1});
            }
        }

        void Disable() {
            F_ = nullptr;
            MarkReplied();
        }

        inline bool Canceled() const noexcept {
            return Canceled_;
        }

    private:
        //return true when mark first reply
        inline bool MarkReplied() {
            return AtomicAdd(NotifyCnt_, 1) == 1;
        }

    private:
        TAtomicBool Canceled_;
        TAtomic NotifyCnt_;
    };

    typedef TIntrusivePtr<TInprocHandle> TInprocHandleRef;

    class TInprocLocation: public TParsedLocation {
    public:
        TInprocLocation(const TStringBuf& addr)
            : TParsedLocation(addr)
        {
            Service.Split('?', InprocService, InprocId);
        }

        TStringBuf InprocService;
        TStringBuf InprocId;
    };

    class TRequest: public IRequest {
    public:
        TRequest(const TInprocHandleRef& hndl)
            : Location(hndl->Message().Addr)
            , Handle_(hndl)
        {
        }

        TStringBuf Scheme() const override {
            return TStringBuf("inproc");
        }

        TString RemoteHost() const override {
            return TString();
        }

        TStringBuf Service() const override {
            return Location.InprocService;
        }

        TStringBuf Data() const override {
            return Handle_->Message().Data;
        }

        TStringBuf RequestId() const override {
            return Location.InprocId;
        }

        bool Canceled() const override {
            return Handle_->Canceled();
        }

        void SendReply(TData& data) override {
            Handle_->SendReply(TString(data.data(), data.size()));
        }

        void SendError(TResponseError, const TString& details) override {
            Handle_->SendError(details);
        }

        const TMessage Request;
        const TInprocLocation Location;

    private:
        TInprocHandleRef Handle_;
    };

    class TInprocRequester: public IRequester {
    public:
        TInprocRequester(IOnRequest*& rqcb)
            : RegisteredCallback_(rqcb)
        {
        }

        ~TInprocRequester() override {
            RegisteredCallback_ = nullptr;
        }

    private:
        IOnRequest*& RegisteredCallback_;
    };

    class TInprocRequesterStg: public IProtocol {
    public:
        inline TInprocRequesterStg() {
            V_.resize(1 + (size_t)Max<ui16>());
        }

        IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
            IOnRequest*& rqcb = Find(loc);

            if (!rqcb) {
                rqcb = cb;
            } else if (rqcb != cb) {
                ythrow yexception() << "shit happen - already registered";
            }

            return new TInprocRequester(rqcb);
        }

        THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
            TInprocHandleRef hndl(new TInprocHandle(msg, fallback, !ss ? nullptr : new TStatCollector(ss)));
            try {
                TAutoPtr<TRequest> req(new TRequest(hndl));

                if (IOnRequest* cb = Find(req->Location)) {
                    cb->OnRequest(req.Release());
                } else {
                    throw yexception() << TStringBuf("not found inproc location");
                }
            } catch (...) {
                hndl->Disable();
                throw;
            }

            return THandleRef(hndl.Get());
        }

        TStringBuf Scheme() const noexcept override {
            return TStringBuf("inproc");
        }

    private:
        static inline ui16 Id(const TParsedLocation& loc) {
            return loc.GetPort();
        }

        inline IOnRequest*& Find(const TParsedLocation& loc) {
            return Find(Id(loc));
        }

        inline IOnRequest*& Find(ui16 id) {
            return V_[id];
        }

    private:
        TVector<IOnRequest*> V_;
    };
}

IProtocol* NNeh::InProcProtocol() {
    return Singleton<TInprocRequesterStg>();
}
