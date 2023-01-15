#pragma once

#include "par_master.h"
#include "par_remote.h"
#include "par.h"
#include "par_context.h"

#include <util/generic/vector.h>

namespace NPar {
    class TRootEnvironment: public IRootEnvironment {
        OBJECT_NOCOPY_METHODS(TRootEnvironment);
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TContextDistributor> ContextMaster;
        TIntrusivePtr<TWriteBufferHandler> WriteBuffer;
        TIntrusivePtr<TMaster> Master;

        IEnvironment* CreateEnvironment(int envId, const TVector<int>& hostIds) override {
            return Master->CreateEnvironment(envId, hostIds);
        }
        IEnvironment* GetEverywhere() override {
            return Master->GetEverywhere();
        }
        IEnvironment* GetAnywhere() override {
            return Master->GetAnywhere();
        }
        TVector<int> MakeHostIdMapping(int groupCount) override {
            return Master->MakeHostIdMapping(groupCount);
        }
        int GetSlaveCount() override {
            return Master->GetSlaveCount();
        }
        void WaitDistribution() override {
            ContextMaster->WaitDistribution();
        }
        void Stop() override;

    public:
        struct TLocal {};

        TRootEnvironment() {
        }
        TRootEnvironment(const char* hostsFileName, int defaultSlavePort, int masterPort, int debugPort);
        TRootEnvironment(TLocal);
        ~TRootEnvironment() override;
    };
}
