#include "par_jobreq.h"
#include "distr_tree.h"

#include <util/random/random.h>
#include <util/random/shuffle.h>

namespace NPar {
    template <class T>
    struct TRemapper {
        TVector<int> NewId;
        const TVector<T>& SrcArray;
        TVector<T>* ResArray;

        TRemapper(TVector<T>* resArray, const TVector<T>& srcArray)
            : SrcArray(srcArray)
            , ResArray(resArray)
        {
            NewId.resize(srcArray.ysize(), -1);
            ResArray->reserve(srcArray.size());
            ResArray->resize(0);
        }
        int GetNewId(int id) {
            int res = NewId[id];
            if (res == -1) {
                res = ResArray->ysize();
                NewId[id] = res;
                ResArray->push_back(SrcArray[id]);
            }
            return res;
        }
    };

    struct TParamsRemapper {
        TVector<int> NewId;
        const TVector<char>& SrcDataArray;
        const TVector<int>& SrcPtrArray;
        TVector<char>* ResDataArray;
        TVector<int>* ResPtrArray;

        TParamsRemapper(TVector<char>* resDataArray, TVector<int>* resPtrArray,
                        const TVector<char>& srcDataArray, const TVector<int>& srcPtrArray)
            : SrcDataArray(srcDataArray)
            , SrcPtrArray(srcPtrArray)
            , ResDataArray(resDataArray)
            , ResPtrArray(resPtrArray)
        {
            NewId.resize(srcPtrArray.ysize(), -1);
            ResDataArray->reserve(srcDataArray.size());
            ResPtrArray->reserve(srcPtrArray.size());
            ResDataArray->resize(0);
            ResPtrArray->resize(1);
            (*ResPtrArray)[0] = 0;
        }
        int GetNewId(int id) {
            int res = NewId[id];
            if (res == -1) {
                res = ResPtrArray->ysize() - 1;
                NewId[id] = res;
                int sz = SrcPtrArray[id + 1] - SrcPtrArray[id];
                if (sz > 0) {
                    const char* data = &SrcDataArray[SrcPtrArray[id]];
                    ResDataArray->insert(ResDataArray->end(), data, data + sz);
                }
                ResPtrArray->push_back(ResDataArray->ysize());
            }
            return res;
        }
    };

    // get commands executed on subTask computers
    // dstPlace - places in original results vector to copy results of res
    // remoteHasData - whether result is not empty (can be empty due to reduce ops)
    // partId - vector of partIds for original cmd array
    void ProjectJob(TJobDescription* res, int thisPartId,
                    TVector<int>* dstPlace, TVector<bool>* remoteHasData,
                    TVector<int>* partId,
                    const TJobDescription& allJob,
                    const TVector<ui16>& subTask) {
        CHROMIUM_TRACE_FUNCTION();
        TVector<bool> selectedComps;
        GetSelectedCompList(&selectedComps, subTask);

        res->ExecList.resize(0);
        res->ExecList.reserve(allJob.ExecList.ysize());
        dstPlace->resize(0);

        TRemapper<TVector<char>> cmdRemap(&res->Cmds, allJob.Cmds);
        TParamsRemapper paramsRemap(&res->ParamsData, &res->ParamsPtr, allJob.ParamsData, allJob.ParamsPtr);
        int prevReduceId = -1;
        for (int i = 0; i < allJob.ExecList.ysize(); ++i) {
            TJobParams params = allJob.ExecList[i];
            if (params.CompId >= selectedComps.ysize() || selectedComps[params.CompId] == false)
                continue;

            bool hasData = params.ReduceId != prevReduceId;
            (*remoteHasData)[i] = hasData;
            if (hasData)
                dstPlace->push_back(i);

            params.CmdId = cmdRemap.GetNewId(params.CmdId);
            params.ParamId = paramsRemap.GetNewId(params.ParamId);
            res->ExecList.push_back(params);
            (*partId)[i] = thisPartId;
            prevReduceId = params.ReduceId;
        }
        Y_ASSERT(!res->ExecList.empty());
    }

    // get jobs subset by job index
    void ProjectJob(TJobDescription* res,
                    int startIdx, int blockLen,
                    TVector<int>* dstPlace, TVector<bool>* remoteHasData,
                    const TJobDescription& allJob) {
        CHROMIUM_TRACE_FUNCTION();
        res->ExecList.resize(blockLen);
        dstPlace->resize(0);

        TRemapper<TVector<char>> cmdRemap(&res->Cmds, allJob.Cmds);
        TParamsRemapper paramsRemap(&res->ParamsData, &res->ParamsPtr, allJob.ParamsData, allJob.ParamsPtr);
        int prevReduceId = -1;
        for (int i = 0; i < blockLen; ++i) {
            TJobParams params = allJob.ExecList[startIdx + i];

            bool hasData = params.ReduceId != prevReduceId;
            (*remoteHasData)[startIdx + i] = hasData;
            if (hasData)
                dstPlace->push_back(startIdx + i);

            params.CmdId = cmdRemap.GetNewId(params.CmdId);
            params.ParamId = paramsRemap.GetNewId(params.ParamId);
            res->ExecList[i] = params;
            prevReduceId = params.ReduceId;
        }
        Y_ASSERT(!res->ExecList.empty());
    }

    bool RescheduleJobRequest(TJobDescription* job,
                              const TVector<TVector<int>>& subsetHostId2Computer,
                              const TVector<TVector<int>>& checkHostId2Computer,
                              TVector<bool>* selectedComps) {
        CHROMIUM_TRACE_FUNCTION();

        Y_ASSERT(subsetHostId2Computer.ysize() == subsetHostId2Computer.ysize());
        int hostIdCount = subsetHostId2Computer.ysize();

        TVector<unsigned int> perHostIdCounter;
        perHostIdCounter.resize(hostIdCount);
        for (int i = 0; i < hostIdCount; ++i)
            perHostIdCounter[i] = RandomNumber<unsigned>();

        TVector<int> anywhereList;
        int anywherePtr = 0;

        selectedComps->resize(0);
        for (int i = 0; i < job->ExecList.ysize(); ++i) {
            TJobParams& params = job->ExecList[i];
            int hostId1 = params.HostId;
            int compId = 0x7fffffff;
            if (hostId1 == TJobDescription::ANYWHERE_HOST_ID) {
                // execute anywhere, but require all hostIds available
                if (anywhereList.empty()) {
                    for (int hostId2 = 0; hostId2 < hostIdCount; ++hostId2) {
                        const TVector<int>& hh = subsetHostId2Computer[hostId2];
                        if (checkHostId2Computer[hostId2].empty())
                            return false;
                        if (!hh.empty()) {
                            for (int j = 0; j < hh.ysize(); ++j)
                                anywhereList.push_back(hh[j]);
                        }
                    }
                    Shuffle(anywhereList.begin(), anywhereList.end());
                }
                compId = anywhereList[anywherePtr++ % anywhereList.size()];
            } else {
                const TVector<int>& hh = subsetHostId2Computer[hostId1];
                if (hh.empty())
                    return false;
                compId = hh[++perHostIdCounter[hostId1] % hh.size()]; // use comps in sequence to distribute load evenly
            }
            params.CompId = compId;
            if (compId >= selectedComps->ysize())
                selectedComps->resize(compId + 1, false);
            (*selectedComps)[compId] = true;
        }
        return true;
    }
}
