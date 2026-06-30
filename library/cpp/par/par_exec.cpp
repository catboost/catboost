#include "par_exec.h"

#include <util/random/random.h>
#include <util/random/shuffle.h>

namespace NPar {
    TAtomic TMRCommandExec::LocalMapWins = 0;
    TAtomic TMRCommandExec::RemoteMapWins = 0;

    static void CheckSchedule(const TJobRequest& src) {
        for (int i = 0; i < src.Descr.ExecList.ysize(); ++i) {
            int compId = src.Descr.ExecList[i].CompId;
            Y_ASSERT(IsIn(src.ExecPlan, compId));
            (void)compId;
        }
    }

    static bool ScheduleJobRequest(TJobRequest* src, TRemoteQueryProcessor* queryProc) {
        TVector<bool> selectedComps;
        if (!RescheduleJobRequest(&src->Descr, src->HostId2Computer, src->HostId2Computer, &selectedComps))
            return false;

        queryProc->GetExecPlan(&src->ExecPlan);
        ProjectExecPlan(&src->ExecPlan, selectedComps);

        CheckSchedule(*src);
        return true;
    }

    bool RescheduleJobRequest(TJobRequest* src, const TVector<ui16>& parentExecPlan, int localCompId, int ignoreCompId) {
        Y_ASSERT(!parentExecPlan.empty());
        Y_ASSERT(localCompId != ignoreCompId);
        int hostIdCount = src->HostId2Computer.ysize();

        TVector<bool> isInParentExecPlan;
        GetSelectedCompList(&isInParentExecPlan, parentExecPlan);

        TVector<bool> needHostId;
        needHostId.resize(hostIdCount, false);
        bool hasAnywhere = false;
        for (int i = 0; i < src->Descr.ExecList.ysize(); ++i) {
            const TJobParams& jp = src->Descr.ExecList[i];
            if (jp.HostId == TJobDescription::ANYWHERE_HOST_ID)
                hasAnywhere = true;
            else
                needHostId[jp.HostId] = true;
        }

        src->ExecPlan = parentExecPlan;

        // make hostId2Computer out of computers in parentExecPlan except localCompId
        // excluding localCompId should help local-remote balance
        TVector<TVector<int>> subsetHostId2Computer;
        subsetHostId2Computer.resize(hostIdCount);
        for (int hostId = 0; hostId < hostIdCount; ++hostId) {
            TVector<int>& subsetHosts = subsetHostId2Computer[hostId];
            TVector<int>& srcHosts = src->HostId2Computer[hostId];
            if (srcHosts.empty())
                continue;
            subsetHosts.resize(srcHosts.ysize());
            int dst = 0;
            int ignoredCompPlace = -1;
            for (int i = 0; i < srcHosts.ysize(); ++i) {
                int compId = srcHosts[i];
                if (compId != localCompId) {
                    if (compId == ignoreCompId)
                        ignoredCompPlace = i;
                    else if (compId < isInParentExecPlan.ysize() && isInParentExecPlan[compId])
                        subsetHosts[dst++] = compId;
                }
            }
            if (ignoredCompPlace != -1) {
                // this hostId has ignored comp
                // remove it from the JobRequest's map
                srcHosts.erase(srcHosts.begin() + ignoredCompPlace);
            }
            if (dst == 0) {
                // have no replacement computer for this hostId in the parentExecPlan
                // test if we need replacement and can do it
                if ((hasAnywhere || needHostId[hostId]) && srcHosts.empty())
                    return false;
                if (needHostId[hostId]) {
                    // select replacement from outside of the group
                    int compId = srcHosts[RandomNumber(srcHosts.size())];
                    subsetHosts[dst++] = compId;
                    AddCompToPlan(&src->ExecPlan, compId);
                }
            }
            subsetHosts.resize(dst);

            // shuffle hosts
            Shuffle(subsetHosts.begin(), subsetHosts.end());
        }

        TVector<bool> selectedComps;
        if (!RescheduleJobRequest(&src->Descr, subsetHostId2Computer, src->HostId2Computer, &selectedComps)) {
            Y_ASSERT(0 && "should not happen");
            return false;
        }

        ProjectExecPlan(&src->ExecPlan, selectedComps);

        CheckSchedule(*src);
        return true;
    }

    void LaunchLocalJobRequest(TJobRequest* jr, int localCompId, IUserContext* userContext, IMRCommandCompleteNotify* mrNotify) {
        int hostIdCount = userContext->GetHostIdCount();
        jr->HostId2Computer.resize(0);
        jr->HostId2Computer.resize(hostIdCount);

        int mapJobCount = jr->Descr.ExecList.ysize();
        for (int i = 0; i < mapJobCount; ++i) {
            TJobParams& params = jr->Descr.ExecList[i];
            params.CompId = localCompId;
        }
        jr->ExecPlan.resize(0); // empty execplan means execute locally

        TMRCommandExec::Launch(jr, nullptr, localCompId, userContext, mrNotify);
    }

    bool LaunchJobRequest(TJobRequest* jr, TRemoteQueryProcessor* queryProc, IUserContext* userContext, IMRCommandCompleteNotify* mrNotify) {
        Y_ASSERT(queryProc);
        Y_ASSERT(!jr->HostId2Computer.empty());
        int hostIdCount = jr->HostId2Computer.ysize();
        int localCompId = queryProc->GetCompId();

        THashMap<int, bool> allHostIdSet;
        bool hasAllHostIds = true;
        for (int i = 0; i < hostIdCount; ++i) {
            allHostIdSet[i];
            hasAllHostIds &= !jr->HostId2Computer[i].empty();
        }

        if (hasAllHostIds) {
            // has at least one remote comp for each hostId
            // try regular execution
            bool chk = ScheduleJobRequest(jr, queryProc);
            Y_ABORT_UNLESS(chk);

            TMRCommandExec::Launch(jr, queryProc, localCompId, userContext, mrNotify);
        } else {
            // not all hostIds are ready
            // lets check if we can execute locally
            if (!userContext->HasHostIds(allHostIdSet)) {
                // can not execute locally
                return false;
            }
            LaunchLocalJobRequest(jr, localCompId, userContext, mrNotify);
        }
        return true;
    }
}
