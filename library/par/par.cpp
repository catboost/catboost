#include "par.h"
#include "par_host.h"

#include <library/cpp/binsaver/mem_io.h>

namespace NPar {
    void TJobDescription::SetCurrentOperation(IDistrCmd* op) {
        CHROMIUM_TRACE_FUNCTION();
        TObj<IDistrCmd> opPtr = op;
        SerializeToMem(&Cmds.emplace_back(), opPtr);
    }

    void TJobDescription::SetCurrentOperation(const TVector<char>& op) {
        CHROMIUM_TRACE_FUNCTION();
        Y_ASSERT(!op.empty());
        Cmds.push_back(op);
    }

    void TJobDescription::AddJob(int hostId, int paramId, int reduceId) {
        Y_ASSERT(!Cmds.empty());
        ExecList.push_back(TJobParams(Cmds.ysize() - 1, paramId, reduceId, -1, static_cast<short>(hostId)));
    }

    int TJobDescription::AddParamData(TVector<char>* data) {
        if (data) {
            int res = ParamsPtr.ysize() - 1;
            ParamsData.insert(ParamsData.end(), data->begin(), data->end());
            ParamsPtr.push_back(ParamsData.ysize());
            return res;
        }
        return 0;
    }

    int TJobDescription::GetReduceId() {
        if (ExecList.empty())
            return 0;
        return ExecList.back().ReduceId + 1;
    }

    TJobDescription::TJobDescription() {
        ParamsPtr.push_back(0);
        ParamsPtr.push_back(0);
    }

    void TJobDescription::AddMapImpl(int paramId) {
        CHROMIUM_TRACE_FUNCTION();
        AddJob(MAP_HOST_ID, paramId, GetReduceId());
    }

    void TJobDescription::AddQueryImpl(TVector<int> hostIds, int paramId) {
        CHROMIUM_TRACE_FUNCTION();
        Y_ASSERT(!hostIds.empty());
        int reduceId = GetReduceId();
        for (int i = 0; i < hostIds.ysize(); ++i) {
            Y_ASSERT(hostIds[i] >= 0);
            AddJob(hostIds[i], paramId, reduceId);
        }
    }

    void TJobDescription::AddQueryImpl(int hostId, int paramId) {
        CHROMIUM_TRACE_FUNCTION();
        Y_ASSERT(hostId >= 0 || hostId == ANYWHERE_HOST_ID || hostId == MAP_HOST_ID);
        AddJob(hostId, paramId, GetReduceId());
    }

    void TJobDescription::MergeResults() {
        CHROMIUM_TRACE_FUNCTION();
        for (int i = 0; i < ExecList.ysize(); ++i) {
            ExecList[i].ReduceId = 0;
        }
    }

    void TJobDescription::SeparateResults(int hostIdCount) {
        CHROMIUM_TRACE_FUNCTION();
        TVector<TJobParams> tmpList;
        for (int i = 0; i < ExecList.ysize(); ++i) {
            TJobParams jp = ExecList[i];
            if (jp.HostId == MAP_HOST_ID) {
                for (int hostId = 0; hostId < hostIdCount; ++hostId) {
                    jp.HostId = hostId;
                    tmpList.push_back(jp);
                }
            } else
                tmpList.push_back(jp);
        }
        for (int i = 0; i < tmpList.ysize(); ++i)
            tmpList[i].ReduceId = i;
        ExecList.swap(tmpList);
    }

    void TJobDescription::Swap(TJobDescription* res) {
        Cmds.swap(res->Cmds);
        ParamsData.swap(res->ParamsData);
        ParamsPtr.swap(res->ParamsPtr);
        ExecList.swap(res->ExecList);
    }
}
