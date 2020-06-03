#pragma once

#include "par_remote.h"

namespace NPar {
    class TContextReplica;
    struct IGroupRequest;
    struct TJobDescription;
    struct IUserContext;
    struct IMRCommandCompleteNotify;

    class TMRCmdsProcessor: public ICmdProcessor {
        TIntrusivePtr<TContextReplica> Context;

        void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override;

    public:
        TMRCmdsProcessor(TRemoteQueryProcessor* p, TContextReplica* context)
            : Context(context)
        {
            p->RegisterCmdType("mr", this);
            p->RegisterCmdType("mr_low", this);
        }
    };

    // actually may block when there is no host with data for some hostId
    void AsyncStartGroupRequest(TJobDescription* descr,
                                TRemoteQueryProcessor* queryProc, IUserContext* userContext,
                                IMRCommandCompleteNotify* mrNotify);
}
