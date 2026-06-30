#include "par_wb.h"
#include "par.h"

namespace NPar {
    struct TWBCopyCompInfo {
        TVector<i64> Data;
        TVector<int> OriginalPlace;
    };

    struct TWBCopyCmd {
        TVector<i64> Data;

        SAVELOAD(Data);
    };

    class TDataCollector: public IRemoteQueryResponseNotify {
        THashMap<int, TWBCopyCompInfo>& RequestHash;
        TVector<TVector<char>> Result;
        TSystemEvent Ready;
        TAtomic ReqCount;

    public:
        TDataCollector(THashMap<int, TWBCopyCompInfo>* req)
            : RequestHash(*req)
            , ReqCount(0)
        {
            for (THashMap<int, TWBCopyCompInfo>::iterator i = req->begin(); i != req->end(); ++i) {
                TWBCopyCompInfo& info = i->second;
                for (int z = 0; z < info.OriginalPlace.ysize(); ++z) {
                    int place = info.OriginalPlace[z];
                    if (place >= Result.ysize())
                        Result.resize(place + 1);
                }
            }
        }
        void GotResponse(int queryId, TVector<char>* buf) override {
            CHROMIUM_TRACE_FUNCTION();

            TWBCopyCompInfo& info = RequestHash[queryId];
            TVector<TVector<char>> xx;
            SerializeFromMem(buf, xx);
            Y_ASSERT(xx.ysize() == info.OriginalPlace.ysize());
            for (int i = 0; i < xx.ysize(); ++i)
                Result[info.OriginalPlace[i]].swap(xx[i]);
            if (AtomicAdd(ReqCount, -1) == 0)
                Ready.Signal();
        }
        void Run(TLocalDataBuffer* writeBuffer, TRemoteQueryProcessor* queryProc,
                 TVector<TVector<char>>* result) {
            CHROMIUM_TRACE_FUNCTION();
            Ready.Reset();

            int localCompId = queryProc ? queryProc->GetCompId() : -1;
            AtomicAdd(ReqCount, 1);
            for (THashMap<int, TWBCopyCompInfo>::iterator i = RequestHash.begin(); i != RequestHash.end(); ++i) {
                TWBCopyCompInfo& info = i->second;
                int compId = i->first;
                if (compId == localCompId) {
                    for (int j = 0; j < info.Data.ysize(); ++j) {
                        writeBuffer->GetData(info.Data[j], &Result[info.OriginalPlace[j]], TLocalDataBuffer::DO_COPY);
                    }
                } else {
                    TVector<char> buf;
                    SerializeToMem(&buf, info.Data);
                    AtomicAdd(ReqCount, 1);
                    queryProc->SendQuery(compId, "wb_copy", &buf, this, compId);
                }
            }
            if (AtomicAdd(ReqCount, -1) == 0)
                Ready.Signal();
            Ready.Wait();

            result->swap(Result);
        }
    };

    void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res,
                     TLocalDataBuffer* writeBuffer, TRemoteQueryProcessor* queryProc) {
        CHROMIUM_TRACE_FUNCTION();

        THashMap<int, TWBCopyCompInfo> comp2req;
        for (int i = 0; i < data.ysize(); ++i) {
            const TDataLocation& dl = data[i];
            TWBCopyCompInfo& info = comp2req[dl.CompId];
            info.OriginalPlace.push_back(i);
            info.Data.push_back(dl.DataId);
        }
        TIntrusivePtr<TDataCollector> dc = new TDataCollector(&comp2req);
        dc->Run(writeBuffer, queryProc, res);
    }
}
