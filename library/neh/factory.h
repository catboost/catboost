#pragma once

#include "neh.h"
#include "rpc.h"

namespace NNeh {
    struct TParsedLocation;

    class IProtocol {
    public:
        virtual ~IProtocol() {
        }
        virtual IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) = 0;
        virtual THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef&) = 0;
        virtual TStringBuf Scheme() const noexcept = 0;
        virtual bool SetOption(TStringBuf name, TStringBuf value) {
            Y_UNUSED(name);
            Y_UNUSED(value);
            return false;
        }
    };

    class IProtocolFactory {
    public:
        virtual IProtocol* Protocol(const TStringBuf& scheme) = 0;
        virtual void Register(IProtocol* proto) = 0;
        virtual ~IProtocolFactory() {
        }
    };

    void SetGlobalFactory(IProtocolFactory* factory);
    IProtocolFactory* ProtocolFactory();
}
