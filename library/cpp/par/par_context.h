#pragma once

#include "par.h"
#include "par_remote.h"
#include "par_log.h"
#include "par_wb.h"

#include <library/cpp/binsaver/mem_io.h>

#include <library/cpp/chromium_trace/interface.h>
#include <library/cpp/binsaver/bin_saver.h>

#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/thread.h>

namespace NPar {
    class TThreadIdChecker {
        TThread::TId MainThreadId;

    public:
        TThreadIdChecker()
            : MainThreadId(0)
        {
        }
        void CheckThreadId() {
#ifndef NDEBUG
            if (MainThreadId == 0)
                MainThreadId = TThread::CurrentThreadId();
            else
                Y_ASSERT(MainThreadId == TThread::CurrentThreadId());
#else
            Y_UNUSED(MainThreadId);
#endif
        }
    };

    struct TContextDataHolder: public TThrRefBase {
        TVector<int> Computer2HostId;
        TObj<IObjectBase> Data;

        SAVELOAD(Computer2HostId, Data);

        ~TContextDataHolder() override {
            IObjectBase::SetThreadCheckMode(false);
            Data = nullptr;
            IObjectBase::SetThreadCheckMode(true);
        }
    };

    struct TContextDataPartCmd {
        int EnvId, Version;
        int Part, PartCount;
        bool KeepRawData;
        TVector<char> DataPart;

        SAVELOAD(EnvId, Version, Part, PartCount, KeepRawData, DataPart);
    };

    struct TContextForwardCmd {
        int EnvId, DstCompId, Part;
        bool KeepRawData;
        TContextForwardCmd(int EnvId, int DstCompId, int Part, bool KeepRawData = true)
            : EnvId(EnvId)
            , DstCompId(DstCompId)
            , Part(Part)
            , KeepRawData(KeepRawData)
        {
        }
        TContextForwardCmd() {
        }

        SAVELOAD(EnvId, DstCompId, Part, KeepRawData);
    };

    struct TContextFreeCmd {
        int EnvId;

        SAVELOAD(EnvId);
    };

    struct TContextSetData {
        int EnvId, Version, DstCompId;
        TVector<int> Computer2HostId;
        i64 DataId;
        bool KeepRawData;

        SAVELOAD(EnvId, Version, DstCompId, Computer2HostId, DataId, KeepRawData);
    };

    inline void MakeDataPartCmd(int envId, int version, int part, int partCount, bool keepRawData, const TVector<char>& dataPart,
                                TVector<char>* res) {
        CHROMIUM_TRACE_FUNCTION();
        Y_ASSERT(!dataPart.empty());
        TContextDataPartCmd cmd;
        cmd.EnvId = envId;
        cmd.Version = version;
        cmd.Part = part;
        cmd.PartCount = partCount;
        cmd.DataPart = dataPart;
        cmd.KeepRawData = keepRawData;
        SerializeToMem(res, cmd);
    }

    class TReplyForwarder: public IRemoteQueryResponseNotify {
        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        TGUID ReqId;
        TAtomic RespCount;

        void GotResponse(int, TVector<char>* response) override {
            CHROMIUM_TRACE_FUNCTION();
            if (AtomicDecrement(RespCount) == 0) {
                QueryProc->SendReply(ReqId, response);
            }
        }

    public:
        TReplyForwarder(TRemoteQueryProcessor* queryProc, const TGUID& reqId, ssize_t RespCount = 1)
            : QueryProc(queryProc)
            , ReqId(reqId)
            , RespCount(RespCount)
        {
        }
    };

    class TContextReplica: public ICmdProcessor {
        struct TCtxInfo {
            TIntrusivePtr<TContextDataHolder> Info;
            TVector<TVector<char>> BinData;
            int Version;
            bool KeepRawData = true;

            TCtxInfo()
                : Version(-1)
            {
            }
            int GetPartCount() const {
                return BinData.ysize();
            }
            bool IsReady() const {
                for (int i = 0; i < BinData.ysize(); ++i) {
                    if (BinData[i].empty()) {
                        return false;
                    }
                }
                return true;
            }
        };

        THashMap<int, TCtxInfo> EnvId2EnvData;
        TThreadIdChecker MainThreadId;
        TIntrusivePtr<TLocalDataBuffer> WriteBuffer;

        void NewRequest(TRemoteQueryProcessor* p, TNetworkRequest* req) override {
            CHROMIUM_TRACE_FUNCTION();
            MainThreadId.CheckThreadId();
            if (req->Url == "ctx") {
                TContextDataPartCmd cmd;
                SerializeFromMem(&req->Data, cmd);

                PAR_DEBUG_LOG << Sprintf("Got update envId %d, version %d, part %d/%d\n", cmd.EnvId, cmd.Version, cmd.Part, cmd.PartCount);
                TCtxInfo& dst = EnvId2EnvData[cmd.EnvId];
                if (cmd.Version >= dst.Version) {
                    // need to check version since we can manage to get stale update
                    if (cmd.Version > dst.Version) {
                        dst.Version = cmd.Version;
                        dst.BinData.resize(0);
                        dst.BinData.resize(cmd.PartCount);
                        dst.KeepRawData = cmd.KeepRawData;
                    }
                    swap(dst.BinData[cmd.Part], cmd.DataPart);
                    if (dst.KeepRawData && dst.IsReady()) {
                        // Deserialize only if KeepRawData, otherwise see ctx_free
                        dst.Info = new TContextDataHolder;
                        SerializeFromMem(&dst.BinData, *dst.Info);
                    }
                }
                p->SendReply(req->ReqId, nullptr);

            } else if (req->Url == "ctx_fwd") {
                TContextForwardCmd fwdCmd;
                SerializeFromMem(&req->Data, fwdCmd);

                Y_ASSERT(EnvId2EnvData.find(fwdCmd.EnvId) != EnvId2EnvData.end());
                TCtxInfo& ctx = EnvId2EnvData[fwdCmd.EnvId];
                TVector<char> data;
                MakeDataPartCmd(fwdCmd.EnvId, ctx.Version, fwdCmd.Part, ctx.GetPartCount(), fwdCmd.KeepRawData, ctx.BinData[fwdCmd.Part], &data);
                PAR_DEBUG_LOG << Sprintf("Forwarding envId %d version %d part %d to comp %d (%d bytes)\n", fwdCmd.EnvId, ctx.Version, fwdCmd.Part, fwdCmd.DstCompId, data.ysize());
                TReplyForwarder* rf = new TReplyForwarder(p, req->ReqId);
                p->SendQuery(fwdCmd.DstCompId, "ctx", &data, rf, 0);

            } else if (req->Url == "ctx_free") {
                TContextFreeCmd freeCmd;
                SerializeFromMem(&req->Data, freeCmd);
                Y_ASSERT(EnvId2EnvData.find(freeCmd.EnvId) != EnvId2EnvData.end());
                TCtxInfo& ctx = EnvId2EnvData[freeCmd.EnvId];
                if (!ctx.KeepRawData) {
                    // if !KeepRawData, we expect ctx_free after raw data distribution, and defer deserialation till now to minimize memory usage.
                    ctx.Info = new TContextDataHolder;
                    SerializeFromMemShrinkInput(&ctx.BinData, *ctx.Info);
                }
                ctx.BinData.clear();
                p->SendReply(req->ReqId, nullptr);
                PAR_DEBUG_LOG << "Freeing envId " << freeCmd.EnvId << Endl;

            } else if (req->Url == "ctx_wb") {
                TContextSetData wb;
                SerializeFromMem(&req->Data, wb);

                TIntrusivePtr<TContextDataHolder> info = new TContextDataHolder;
                info->Computer2HostId = wb.Computer2HostId;
                info->Data = WriteBuffer->GetObject(wb.DataId, TLocalDataBuffer::DO_EXTRACT);
                if (wb.DstCompId == p->GetCompId()) {
                    // local
                    PAR_DEBUG_LOG << "Copy local data to envId " << wb.EnvId << Endl;
                    TCtxInfo& dst = EnvId2EnvData[wb.EnvId];
                    if (wb.Version > dst.Version) {
                        // need to check version since we can manage to get stale update
                        dst.Version = wb.Version;
                        dst.KeepRawData = wb.KeepRawData;
                        dst.Info = info;
                        if (dst.KeepRawData)
                            SerializeToMem(&dst.BinData, *info);
                    }
                    p->SendReply(req->ReqId, nullptr);
                } else {
                    // remote
                    PAR_DEBUG_LOG << "Send local data to comp " << wb.DstCompId << " to set in envId " << wb.EnvId << Endl;

                    TVector<TVector<char>> allData;
                    SerializeToMem(&allData, *info);

                    int partCount = allData.ysize();
                    TReplyForwarder* rf = new TReplyForwarder(p, req->ReqId, partCount);

                    for (int part = 0; part < partCount; ++part) {
                        TVector<char> data;
                        MakeDataPartCmd(wb.EnvId, wb.Version, part, partCount, wb.KeepRawData, allData[part], &data);
                        p->SendQuery(wb.DstCompId, "ctx", &data, rf, 0);
                    }
                }

            } else {
                Y_ASSERT(0);
            }
        }

    public:
        TContextReplica(TRemoteQueryProcessor* queryProc, TLocalDataBuffer* writeBuffer)
            : WriteBuffer(writeBuffer)
        {
            queryProc->RegisterCmdType("ctx", this);
            queryProc->RegisterCmdType("ctx_fwd", this);
            queryProc->RegisterCmdType("ctx_free", this);
            queryProc->RegisterCmdType("ctx_wb", this);
        }
        bool GetContextState(const THashMap<int, int>& envId2version, THashMap<int, TIntrusivePtr<TContextDataHolder>>* res) {
            CHROMIUM_TRACE_FUNCTION();
            MainThreadId.CheckThreadId();
            res->clear();
            for (THashMap<int, int>::const_iterator i = envId2version.begin(); i != envId2version.end(); ++i) {
                int envId = i->first;
                int version = i->second;
                THashMap<int, TCtxInfo>::const_iterator z = EnvId2EnvData.find(envId);
                if (z == EnvId2EnvData.end()) {
                    PAR_DEBUG_LOG << "envId " << envId << " not found" << Endl;
                    return false;
                }
                const TCtxInfo& ctx = z->second;
                if (version != ctx.Version) {
                    PAR_DEBUG_LOG << "Request envId " << envId << "version " << version << " failed, current version " << ctx.Version << Endl;
                    return false;
                }
                (*res)[envId] = ctx.Info;
            }
            return true;
        }
        TLocalDataBuffer* GetWriteBuffer() const {
            return WriteBuffer.Get();
        }
    };

    class TContextDistributor: public IRemoteQueryResponseNotify {
        struct TCtxDataPart {
            TIntrusivePtr<TContextDataHolder> Info;
            TVector<TVector<char>> BinData;
            int Version;
            bool KeepRawData;

            TCtxDataPart()
                : Version(0)
                , KeepRawData(true)
            {
            }
            int GetPartCount() const {
                return BinData.ysize();
            }
        };
        struct TFullCtxInfo {
            TVector<int> Computer2HostId;
            TVector<TVector<int>> HostId2Computer;
            TVector<TVector<bool>> ReadyMask, CopyInitiated;
            TVector<TCtxDataPart> Data;
            TVector<bool> IsFullyDistributed;
            int MaxVersion; // max version of all groups
            int ParentEnvId;

            TFullCtxInfo()
                : MaxVersion(0)
                , ParentEnvId(0)
            {
            }

            template <class T>
            void ClearPodArray(TVector<T>* res, ssize_t size) {
                res->yresize(size);
                if (res->empty())
                    return;
                memset(&(*res)[0], 0, sizeof(T) * res->size());
            }

            void ResetHostIdReady(int hostId, int partCount) {
                IsFullyDistributed[hostId] = false;
                const TVector<int>& compList = HostId2Computer[hostId];
                for (int k = 0; k < compList.ysize(); ++k) {
                    int compId = compList[k];
                    ClearPodArray(&ReadyMask[compId], partCount);
                    ClearPodArray(&CopyInitiated[compId], partCount);
                }
            }
        };

        TIntrusivePtr<TRemoteQueryProcessor> QueryProc;
        THashMap<int, TFullCtxInfo> EnvId2Info;
        TMutex Sync;
        TVector<int> ComputerSendCount;
        TAtomic DistributionIsComplete;

        struct TTransferInfo {
            int EnvId, HostId;
            int Part;
            int SenderComp, DstComp;
            int Version;

            TTransferInfo() {
                Zero(*this);
            }
            TTransferInfo(int envId, int hostId, int part, int senderComp, int dstComp, int version)
                : EnvId(envId)
                , HostId(hostId)
                , Part(part)
                , SenderComp(senderComp)
                , DstComp(dstComp)
                , Version(version)
            {
            }
        };
        int QueryId;
        THashMap<int, TTransferInfo> TransferInfos;
        TAtomic ActiveReqCount;
        int SlaveCount;
        TIntrusivePtr<TLocalDataBuffer> WriteBuffer;

        void DoSend();
        void GotResponse(int id, TVector<char>* response) override;
        static void AssignData(TCtxDataPart* part, TFullCtxInfo& dst, const IObjectBase* data);
        void PerformSend(int srcComp, int dstComp,
                         int queryComp, const char* cmd,
                         TFullCtxInfo& fullCtx, int envId, int hostId, int part, int dataVersion,
                         TVector<char>* buf);

    public:
        TContextDistributor(TRemoteQueryProcessor* queryProc, TLocalDataBuffer* writeBuffer);
        ~TContextDistributor() override;
        void CreateNewContext(int envId, int parentEnvId, const TVector<int>& computer2HostId);
        void SetContextData(int envId, int hostId, const IObjectBase* data, EKeepDataFlags keepContextRawData = KEEP_CONTEXT_RAW_DATA);
        void SetContextData(int envId, const TVector<int>& compIds, const TVector<i64>& dataIds, EKeepDataFlags keepContextRawData = KEEP_CONTEXT_RAW_DATA);
        void DeleteContextRawData(int envId, int hostId, bool keepContextOnMaster = false);
        void GetVersions(int envId, int* hostIdCount, THashMap<int, int>* envId2version);
        bool GetReadyMask(const THashMap<int, int>& envId2Version,
                          TVector<TVector<int>>* hostId2Computer,
                          bool* hasNotReadyHost);
        bool GetContextState(int hostId, const THashMap<int, int>& envId2Version, THashMap<int, TIntrusivePtr<TContextDataHolder>>* res);
        const TVector<int>& GetComputer2HostId(int envId);
        int GetHostIdCount(int envId);
        void WaitDistribution();
        void WaitAllDistributionActivity();
        int GetSlaveCount() const {
            return SlaveCount;
        }
        TLocalDataBuffer* GetWriteBuffer() const {
            return WriteBuffer.Get();
        }
    };
}
