#pragma once

#include "par_remote.h"
#include "par_log.h"

#include <library/cpp/binsaver/mem_io.h>
#include <library/cpp/chromium_trace/interface.h>

#include <util/thread/lfstack.h>
#include <util/generic/vector.h>

namespace NPar {
    template <class T>
    inline ssize_t TotalSize(const TVector<TVector<T>>& data) {
        ssize_t res = 0;
        for (int i = 0; i < data.ysize(); ++i) {
            res += data[i].ysize();
        }
        return res;
    }

    class TLocalDataBuffer: public TThrRefBase {
        TMutex Lock;
        class TDataHolder { // Handles ptr to an object or it's serialized copy
            TVector<TVector<char>> BinData;
            TObj<IObjectBase> LocalObject;

        public:
            TDataHolder() {
            }
            TDataHolder(TVector<TVector<char>>& binData) {
                BinData.swap(binData);
            }
            TDataHolder(const IObjectBase* localObject) {
                LocalObject = const_cast<IObjectBase*>(localObject);
            }
            void SetData(TVector<TVector<char>>& binData) {
                TVector<TVector<char>>().swap(BinData);
                BinData.swap(binData);
                if (LocalObject) {
                    LocalObject = nullptr;
                }
            }
            void SetObject(const IObjectBase* localObject) {
                LocalObject = const_cast<IObjectBase*>(localObject);
                TVector<TVector<char>>().swap(BinData);
            }
            TVector<TVector<char>>& GetData() {
                if (LocalObject) {
                    if (!BinData.size()) {
                        SerializeToMem(&BinData, LocalObject);
                    }
                }
                return BinData;
            }
            TObj<IObjectBase> GetObject() {
                if (!LocalObject) {
                    TObj<IObjectBase> obj;
                    SerializeFromMem(&BinData, obj);
                    return obj;
                }
                return LocalObject;
            }
        };
        // can not use smarter pointers due to non thread safe ref counters in IObjectBase
        typedef THashMap<i64, TDataHolder> TDataHash;
        TDataHash Data;

        struct TTableInfo {
            ui64 Version;
            TVector<i64> Blocks;

            TTableInfo()
                : Version(0)
            {
            }
        };
        THashMap<int, TTableInfo> Tables;

        struct TSetDataOp {
            i64 Id;
            int TblId;
            ui64 VersionId;
            TVector<TVector<char>> Data;
            TObj<IObjectBase> Object;

            TSetDataOp()
                : Id(-1)
                , TblId(-1)
                , VersionId(0)
                , Object(nullptr)
            {
            }

            TSetDataOp(i64 id, int tblId, ui64 versionId, TVector<char>* p)
                : Id(id)
                , TblId(tblId)
                , VersionId(versionId)
                , Object(nullptr)
            {
                Data.resize(1);
                p->swap(Data[0]);
            }

            TSetDataOp(i64 id, int tblId, ui64 versionId, TVector<TVector<char>>* p)
                : Id(id)
                , TblId(tblId)
                , VersionId(versionId)
                , Object(nullptr)
            {
                p->swap(Data);
            }
            TSetDataOp(i64 id, int tblId, ui64 versionId, TObj<IObjectBase> object)
                : Id(id)
                , TblId(tblId)
                , VersionId(versionId)
                , Object(object)
            {
            }
        };
        TAtomic QueuedDataSize, LowId, HighId;
        TLockFreeStack<TSetDataOp*> SetQueue;

        enum {
            MAX_QUEUED_SIZE = 16 * 1024 * 1024
        };

        ~TLocalDataBuffer() override {
            SetDataFromQueue();
        }
        i64 GenId() {
            // poor man's "64bit" counter, we hope that threads will not stall for 2*10^9 ids
            TAtomic low = AtomicAdd(LowId, 1);
            int highIncrement = (low & 0x7fffffff) == 0;
            TAtomic high = AtomicAdd(HighId, highIncrement);
            return (((i64)high) << 32) + low;
        }

        void RemoveAllTableBlocks(int tblId) {
            TTableInfo& tbl = Tables[tblId];
            for (int i = 0; i < tbl.Blocks.ysize(); ++i) {
                TDataHash::iterator z = Data.find(tbl.Blocks[i]);
                if (z != Data.end()) {
                    Data.erase(z);
                }
            }
            tbl.Blocks.clear();
        }

        void SetDataFromQueueLocked() {
            for (TSetDataOp* op; SetQueue.Dequeue(&op); delete op) {
                AtomicAdd(QueuedDataSize, -TotalSize(op->Data));
                TTableInfo& tbl = Tables[op->TblId];

                if (tbl.Version > op->VersionId) {
                    continue; // ignore stale updates
                }
                if (tbl.Version < op->VersionId) {
                    tbl.Version = op->VersionId;
                    RemoveAllTableBlocks(op->TblId);
                }
                tbl.Blocks.push_back(op->Id);
                if (op->Object) {
                    Data[op->Id].SetObject(op->Object);
                } else {
                    Data[op->Id].SetData(op->Data);
                }
            }
        }
        void SetDataFromQueue() {
            TGuard<TMutex> gg(Lock);
            SetDataFromQueueLocked();
        }
        void EnqueSetDataOp(TSetDataOp* op, ssize_t dataSize) {
            AtomicAdd(QueuedDataSize, dataSize);
            SetQueue.Enqueue(op);
            if (QueuedDataSize > MAX_QUEUED_SIZE) {
                SetDataFromQueue();
            }
        }

    public:
        TLocalDataBuffer()
            : QueuedDataSize(0)
            , LowId(0)
            , HighId(0)
        {
        }
        // non blocking most of the time
        i64 SetData(int tblId, ui64 versionId, TVector<char>* p) {
            i64 id = GenId();
            const auto dataSize = p->ysize();
            EnqueSetDataOp(new TSetDataOp(id, tblId, versionId, p), dataSize);
            return id;
        }
        i64 SetData(int tblId, ui64 versionId, TVector<TVector<char>>* p) {
            // It's not static_assert because this code can be compiled on 32-bit platform
            Y_ABORT_UNLESS(sizeof(TAtomic) >= 8);

            i64 id = GenId();
            const auto dataSize = TotalSize(*p);
            EnqueSetDataOp(new TSetDataOp(id, tblId, versionId, p), dataSize);
            return id;
        }
        i64 SetObject(int tblId, ui64 versionId, const IObjectBase* obj) {
            i64 id = GenId();
            EnqueSetDataOp(new TSetDataOp(id, tblId, versionId, const_cast<IObjectBase*>(obj)), sizeof(obj));
            return id;
        }
        enum EGetOp {
            DO_COPY,
            DO_EXTRACT
        };
        bool GetData(i64 key, TVector<char>* buf, EGetOp op) {
            buf->resize(0);
            TGuard<TMutex> gg(Lock);
            SetDataFromQueueLocked();
            TDataHash::iterator z = Data.find(key);
            if (z == Data.end()) {
                Y_ASSERT(0);
                return false;
            }

            if (z->second.GetData().size() == 1) {
                if (op == DO_COPY)
                    *buf = z->second.GetData().at(0);
                else { // DO_EXTRACT
                    buf->swap(z->second.GetData().at(0));
                    Data.erase(z);
                }
            } else {
                buf->resize(0);
                if (op == DO_COPY) {
                    for (size_t i = 0; i < z->second.GetData().size(); ++i) {
                        buf->insert(buf->end(), z->second.GetData().at(i).begin(), z->second.GetData().at(i).end());
                    }
                } else { // DO_EXTRACT
                    for (size_t i = 0; i < z->second.GetData().size(); ++i) {
                        buf->insert(buf->end(), z->second.GetData().at(i).begin(), z->second.GetData().at(i).end());
                        TVector<char>().swap(z->second.GetData().at(i));
                    }
                    Data.erase(z);
                }
            }
            return true;
        }

        bool GetData(i64 key, TVector<TVector<char>>* buf, EGetOp op) {
            buf->resize(0);
            TGuard<TMutex> gg(Lock);
            SetDataFromQueueLocked();
            TDataHash::iterator z = Data.find(key);
            if (z == Data.end()) {
                Y_ASSERT(0);
                return false;
            }

            if (op == DO_COPY)
                *buf = z->second.GetData();
            else { // DO_EXTRACT
                buf->swap(z->second.GetData());
                Data.erase(z);
            }
            return true;
        }
        TObj<IObjectBase> GetObject(i64 key, EGetOp op) {
            TGuard<TMutex> gg(Lock);
            SetDataFromQueueLocked();
            TDataHash::iterator z = Data.find(key);
            if (z == Data.end()) {
                Y_ASSERT(0);
                return nullptr;
            }
            if (op == DO_COPY)
                return z->second.GetObject();
            else {
                TObj<IObjectBase> obj = z->second.GetObject();
                Data.erase(z);
                return obj;
            }
        }
    };

    class TWriteBufferHandler: public ICmdProcessor {
        TIntrusivePtr<TLocalDataBuffer> WriteBuffer;

        void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override {
            CHROMIUM_TRACE_FUNCTION();
            if (req->Url == "wb_copy") {
                Y_ASSERT(req->Url == "wb_copy");
                TVector<i64> data;
                SerializeFromMem(&req->Data, data);

                TVector<TVector<char>> res;
                res.resize(data.size());
                for (int i = 0; i < data.ysize(); ++i) {
                    WriteBuffer->GetData(data[i], &res[i], TLocalDataBuffer::DO_COPY);
                }

                TVector<char> tmp;
                SerializeToMem(&tmp, res);
                int dataSize = tmp.ysize();
                p->SendReply(req->ReqId, &tmp);
                PAR_DEBUG_LOG << "Sending " << dataSize << " bytes from write buffer data" << Endl;

            } else {
                Y_ASSERT(0);
            }
        }

    public:
        TWriteBufferHandler(TRemoteQueryProcessor* queryProc) {
            WriteBuffer = new TLocalDataBuffer;
            if (queryProc) {
                queryProc->RegisterCmdType("wb_copy", this);
            }
        }
        TLocalDataBuffer* GetWriteBuffer() const {
            return WriteBuffer.Get();
        }
    };

    struct TDataLocation;

    void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res,
                     TLocalDataBuffer* writeBuffer, TRemoteQueryProcessor* queryProc);
}
