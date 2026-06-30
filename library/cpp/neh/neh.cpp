#include "neh.h"

#include "details.h"
#include "factory.h"

#include <util/generic/list.h>
#include <util/generic/hash_set.h>
#include <util/digest/numeric.h>
#include <util/string/cast.h>

using namespace NNeh;

namespace {
    class TMultiRequester: public IMultiRequester {
        struct TOps {
            template <class T>
            inline bool operator()(const T& l, const T& r) const noexcept {
                return l.Get() == r.Get();
            }

            template <class T>
            inline size_t operator()(const T& t) const noexcept {
                return NumericHash(t.Get());
            }
        };

        struct TOnComplete {
            TMultiRequester* Parent;
            bool Signalled;

            inline TOnComplete(TMultiRequester* parent)
                : Parent(parent)
                , Signalled(false)
            {
            }

            inline void operator()(TWaitHandle* wh) {
                THandleRef req(static_cast<THandle*>(wh));

                Signalled = true;
                Parent->OnComplete(req);
            }
        };

    public:
        void Add(const THandleRef& req) override {
            Reqs_.insert(req);
            req->Register(WaitQueue_);
        }

        void Del(const THandleRef& req) override {
            Reqs_.erase(req);
        }

        bool Wait(THandleRef& req, TInstant deadLine) override {
            while (Complete_.empty()) {
                if (Reqs_.empty()) {
                    return false;
                }
                TOnComplete cb(this);
                WaitForMultipleObj(*WaitQueue_, deadLine, cb);
                if (!cb.Signalled) {
                    return false;
                }
            }

            req = *Complete_.begin();
            Complete_.pop_front();

            return true;
        }

        bool IsEmpty() const override {
            return Reqs_.empty() && Complete_.empty();
        }

        inline void OnComplete(const THandleRef& req) {
            Complete_.push_back(req);
            Del(req);
        }

    private:
        typedef THashSet<THandleRef, TOps, TOps> TReqs;
        typedef TList<THandleRef> TComplete;
        TIntrusivePtr<TWaitQueue> WaitQueue_ = MakeIntrusive<TWaitQueue>();
        TReqs Reqs_;
        TComplete Complete_;
    };

    inline IProtocol* ProtocolForMessage(const TMessage& msg) {
        return ProtocolFactory()->Protocol(TStringBuf(msg.Addr).Before(':'));
    }
}

NNeh::TMessage NNeh::TMessage::FromString(const TStringBuf req) {
    TStringBuf addr;
    TStringBuf data;

    req.Split('?', addr, data);
    return TMessage(ToString(addr), ToString(data));
}

namespace {
    const TString svcFail = "service status: failed";
}

THandleRef NNeh::Request(const TMessage& msg, IOnRecv* fallback) {
    TServiceStatRef ss;

    if (TServiceStat::Disabled()) {
        return ProtocolForMessage(msg)->ScheduleRequest(msg, fallback, ss);
    }

    ss = GetServiceStat(msg.Addr);
    TServiceStat::EStatus es = ss->GetStatus();

    if (es == TServiceStat::Ok) {
        return ProtocolForMessage(msg)->ScheduleRequest(msg, fallback, ss);
    }

    if (es == TServiceStat::ReTry) {
        //send empty data request for validating service (update TServiceStat info)
        TMessage validator;

        validator.Addr = msg.Addr;

        ProtocolForMessage(msg)->ScheduleRequest(validator, nullptr, ss);
    }

    TNotifyHandleRef h(new TNotifyHandle(fallback, msg));
    h->NotifyError(new TError(svcFail));
    return h.Get();
}

THandleRef NNeh::Request(const TString& req, IOnRecv* fallback) {
    return Request(TMessage::FromString(req), fallback);
}

IMultiRequesterRef NNeh::CreateRequester() {
    return new TMultiRequester();
}

bool NNeh::SetProtocolOption(TStringBuf protoOption, TStringBuf value) {
    return ProtocolFactory()->Protocol(protoOption.Before('/'))->SetOption(protoOption.After('/'), value);
}
