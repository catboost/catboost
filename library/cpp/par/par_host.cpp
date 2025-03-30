#include "par_host.h"
#include "par_exec.h"
#include "par_mr.h"
#include <util/string/split.h>
#include <util/network/init.h>
#include <util/stream/file.h>
#include <util/generic/yexception.h>

namespace NPar {
    static TNetworkAddress CreateAddress(const TString& server, int defaultPort) {
        TNetworkAddress::TPortNum port = defaultPort;
        TString addr;
        if (server.Contains('[')) { // handle ipv6 address
            Y_ENSURE(server.Contains(']'), "invalid v6 address" << server);
            auto pos = server.rfind(']');
            addr = server.substr(0, pos + 1);
            if (pos != server.size() - 1) { // we have port
                Y_ENSURE(server[pos + 1] == ':' && server.size() > pos + 2, "invalid v6 address" << server);
                port = FromString<TNetworkAddress::TPortNum>(server.substr(pos + 2));
            }
        } else {
            if (!server.Contains(':')) {
                addr = server;
            } else {
                TString portStr;
                Split(server, ':', addr, portStr);
                port = FromString<TNetworkAddress::TPortNum>(portStr);
            }
        }
        return TNetworkAddress(addr, port);
    }

    //////////////////////////////////////////////////////////////////////////
    TRootEnvironment::TRootEnvironment(const char* hostsFileName, int defaultSlavePort, int masterPort, int debugPort) {
        Y_UNUSED(debugPort);
        TVector<TNetworkAddress> baseSearcherAddrs;
        {
            TFileInput queryFile(hostsFileName);
            TString host;
            while (queryFile.ReadLine(host)) {
                if (host.empty())
                    continue;
                baseSearcherAddrs.push_back(CreateAddress(host, defaultSlavePort));
            }
        }

        QueryProc = new TRemoteQueryProcessor();
        WriteBuffer = new TWriteBufferHandler(QueryProc.Get());
        QueryProc->RunMaster(baseSearcherAddrs, masterPort);

        ContextMaster = new TContextDistributor(QueryProc.Get(), WriteBuffer->GetWriteBuffer());
        Master = new TMaster(QueryProc.Get(), ContextMaster.Get());
    }

    TRootEnvironment::TRootEnvironment(TLocal) {
        WriteBuffer = new TWriteBufferHandler(nullptr);
        ContextMaster = new TContextDistributor(nullptr, WriteBuffer->GetWriteBuffer());
        Master = new TMaster(nullptr, ContextMaster.Get());
    }

    TRootEnvironment::~TRootEnvironment() {
        ContextMaster = nullptr;
    }

    void TRootEnvironment::Stop() {
        ContextMaster->WaitAllDistributionActivity();
        if (QueryProc.Get()) {
            QueryProc->StopSlaves();
        }
        Y_ASSERT(LocalExecutor().GetQueueSize() == 0);
        LocalExecutor().ClearLPQueue();
        ContextMaster = nullptr;
        Master = nullptr;
        if (QueryProc.Get()) {
            QueryProc->Stop();
            QueryProc = nullptr;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void RunSlave(int workerThreadCount, int port) {
        InitNetworkSubSystem();
        LocalExecutor().RunAdditionalThreads(workerThreadCount);

        TIntrusivePtr<TRemoteQueryProcessor> qp(new TRemoteQueryProcessor);
        TIntrusivePtr<TWriteBufferHandler> wb(new TWriteBufferHandler(qp.Get()));
        TIntrusivePtr<TContextReplica> context(new TContextReplica(qp.Get(), wb->GetWriteBuffer()));
        TIntrusivePtr<TMRCmdsProcessor> mr(new TMRCmdsProcessor(qp.Get(), context.Get()));
        qp->RunSlave(port);
        Y_ASSERT(LocalExecutor().GetQueueSize() == 0);
        LocalExecutor().ClearLPQueue();
        qp->Stop();
    }

    IRootEnvironment* RunLocalMaster(int workerThreadCount) {
        LocalExecutor().RunAdditionalThreads(workerThreadCount);

        return new TRootEnvironment(TRootEnvironment::TLocal());
    }

    IRootEnvironment* RunMaster(int masterPort, int workerThreadCount, const char* hostsFileName, int defaultSlavePort, int debugPort) {
        InitNetworkSubSystem();
        LocalExecutor().RunAdditionalThreads(workerThreadCount);

        return new TRootEnvironment(hostsFileName, defaultSlavePort, masterPort, debugPort);
    }
}
