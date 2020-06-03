#pragma once

#include <util/generic/ptr.h>
#include <util/ysafeptr.h>

namespace NPar {
    class TRemoteQueryProcessor;
    class TContextDistributor;
    struct IEnvironment;

    class TMaster: public TThrRefBase {
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TIntrusivePtr<TContextDistributor> ContextMaster;
        TObj<IEnvironment> Everywhere, Anywhere;

    public:
        TMaster(TRemoteQueryProcessor* queryProc, TContextDistributor* contextMaster);
        IEnvironment* CreateEnvironment(int envId, const TVector<int>& hostIds);
        IEnvironment* GetEverywhere() {
            return Everywhere;
        }
        IEnvironment* GetAnywhere() {
            return Anywhere;
        }
        TVector<int> MakeHostIdMapping(int groupCount);
        int GetSlaveCount();
    };
}
