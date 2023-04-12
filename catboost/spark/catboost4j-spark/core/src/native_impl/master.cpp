#include "master.h"

#include <catboost/private/libs/app_helpers/mode_fit_helpers.h>

#include <library/cpp/par/par_log.h>
#include <library/cpp/par/par_network.h>
#include <library/cpp/par/par_remote.h>
#include <library/cpp/threading/atomic/bool.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/datetime/base.h>
#include <util/generic/cast.h>
#include <util/generic/guid.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/network/init.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/event.h>
#include <util/system/info.h>
#include <util/system/spin_wait.h>


int ModeFitImpl(const TVector<TString>& args) {
    const char* argv0 = "spark.native_impl.MasterApp";

    TVector<const char*> argv;
    argv.push_back(argv0);
    for (const auto& arg : args) {
        argv.push_back(arg.data());
    }

    return NCB::ModeFitImpl(SafeIntegerCast<int>(argv.size()), argv.data());
}


static NPar::TNetworkAddress CreateAddress(const TString& server) {
    NPar::TNetworkAddress::TPortNum port = 0;
    TString addr;
    if (server.Contains('[')) { // handle ipv6 address
        CB_ENSURE(server.Contains(']'), "invalid v6 address" << server);
        auto pos = server.rfind(']');
        addr = server.substr(0, pos + 1);
        if (pos != server.size() - 1) { // we have port
            CB_ENSURE(server[pos + 1] == ':' && server.size() > pos + 2, "invalid v6 address" << server);
            port = FromString<NPar::TNetworkAddress::TPortNum>(server.substr(pos + 2));
        }
    } else {
        if (!server.Contains(':')) {
            addr = server;
        } else {
            TString portStr;
            Split(server, ':', addr, portStr);
            port = FromString<NPar::TNetworkAddress::TPortNum>(portStr);
        }
    }
    return NPar::TNetworkAddress(addr, port);
}


class TStopMetaRequester {
    TAtomic ResultCount = 0;
    TSystemEvent Ready;

    TVector<NPar::TNetworkAddress> WorkerAddresses;

    NAtomic::TBool RequesterIsSet = false;
    TIntrusivePtr<NPar::IRequester> Requester;

public:
    explicit TStopMetaRequester(const TString& hostsFileName, i32 timeoutInSeconds) {
        {
            TFileInput queryFile(hostsFileName);
            TString host;
            while (queryFile.ReadLine(host)) {
                if (host.empty())
                    continue;
                WorkerAddresses.push_back(CreateAddress(host));
            }
        }

        SetRequester(NPar::CreateRequester(
            /*masterListenPort*/ 0,
            [this](const TGUID& canceledReq) { QueryCancelCallback(canceledReq); },
            [this](TAutoPtr<NPar::TNetworkRequest>& nlReq) { IncomingQueryCallback(nlReq); },
            [this](TAutoPtr<NPar::TNetworkResponse> response) { ReplyCallback(response); }));

        Ready.Reset();

        for (const auto& workerAddress : WorkerAddresses) {
            TGUID reqId;
            CreateGuid(&reqId);
            Requester->SendRequest(reqId, workerAddress, "stop", /*data*/ nullptr);
        }

        if (WorkerAddresses.size() != SafeIntegerCast<size_t>(AtomicGet(ResultCount))) {
            CB_ENSURE(
                Ready.WaitT(TDuration::Seconds(timeoutInSeconds)),
                "Shutdown workers: timeout of " << timeoutInSeconds << " s expired"
            );
        }
    }

    void SetRequester(TIntrusivePtr<NPar::IRequester> requester) noexcept {
        Requester = std::move(requester);
        RequesterIsSet = true;
    }

    void WaitUntilRequesterIsSet() noexcept {
        if (!RequesterIsSet) {
            TSpinWait sw;

            while (!RequesterIsSet) {
                sw.Sleep();
            }
        }
    }

    void QueryCancelCallback(const TGUID& canceledReq) {
        WaitUntilRequesterIsSet();
        QueryCancelCallbackImpl(canceledReq);
    }

    void QueryCancelCallbackImpl(const TGUID& canceledReq) {
        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Request " << GetGuidAsString(canceledReq)
            << " has been canceled";
    }

    void IncomingQueryCallback(TAutoPtr<NPar::TNetworkRequest>& nlReq) {
        WaitUntilRequesterIsSet();
        IncomingQueryCallbackImpl(nlReq);
    }

    void IncomingQueryCallbackImpl(TAutoPtr<NPar::TNetworkRequest>& nlReq) {
        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Got request " << nlReq->Url << " "
            << GetGuidAsString(nlReq->ReqId) << Endl;
    }

    void ReplyCallback(TAutoPtr<NPar::TNetworkResponse> response) {
        WaitUntilRequesterIsSet();
        ReplyCallbackImpl(response);
    }

    void ReplyCallbackImpl(TAutoPtr<NPar::TNetworkResponse> response) {
        PAR_DEBUG_LOG << "At " << Requester->GetHostAndPort() << " Got reply for redId "
            << GetGuidAsString(response->ReqId) << Endl;
        if (SafeIntegerCast<size_t>(AtomicIncrement(ResultCount)) == WorkerAddresses.size())
            Ready.Signal();
    }
};



void ShutdownWorkers(const TString& hostsFile, i32 timeoutInSeconds) {
    InitNetworkSubSystem();
    NPar::LocalExecutor().RunAdditionalThreads(NSystemInfo::CachedNumberOfCpus());

    TStopMetaRequester(hostsFile, timeoutInSeconds);
}
