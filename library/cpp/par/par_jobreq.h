#pragma once

#include <util/generic/ptr.h>
#include "par.h"

namespace NPar {
    struct TJobRequest: public TThrRefBase {
        TJobDescription Descr;
        TVector<ui16> ExecPlan;
        TVector<TVector<int>> HostId2Computer;
        THashMap<int, int> EnvId2Version;
        bool IsLowPriority;

        SAVELOAD(Descr, ExecPlan, HostId2Computer, EnvId2Version, IsLowPriority);

        TJobRequest()
            : IsLowPriority(false)
        {
        }
    };

    struct TJobRequestReply {
        bool IsCanceled;
        TVector<TVector<char>> Result;

        SAVELOAD(IsCanceled, Result);
    };

    void ProjectJob(TJobDescription* res, int thisPartId,
                    TVector<int>* dstPlace, TVector<bool>* remoteHasData,
                    TVector<int>* partId,
                    const TJobDescription& allJob,
                    const TVector<ui16>& subTask);

    void ProjectJob(TJobDescription* res,
                    int startIdx, int blockLen,
                    TVector<int>* dstPlace, TVector<bool>* remoteHasData,
                    const TJobDescription& allJob);

    // use comps from subsetHostId2Computer, test checkHostId2Computer for correctness
    bool RescheduleJobRequest(TJobDescription* job,
                              const TVector<TVector<int>>& subsetHostId2Computer,
                              const TVector<TVector<int>>& checkHostId2Computer,
                              TVector<bool>* selectedComps);
}
