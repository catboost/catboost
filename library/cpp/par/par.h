#pragma once

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/chromium_trace/interface.h>

#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/hash.h>
#include <util/ysafeptr.h>

struct TGUID;
namespace NPar {
    struct IUserContext;
    struct IDCResultNotify : virtual public TThrRefBase {
        virtual void DistrCmdComplete(int reqId, TVector<char>* res) = 0;
        virtual bool IsDistrCmdNeeded() = 0;
        virtual TGUID GetDistrCmdMasterQueryId() = 0; // return query id which we are processing here or 0 if no such query exist
    };

    struct IDistrCmd: public IObjectBase {
        virtual void Exec(IUserContext* ctx, int hostId, TVector<char>* res) const {
            (void)ctx;
            (void)hostId;
            (void)res;
        }
        virtual void Merge(TVector<TVector<char>>* src, TVector<char>* res) const {
            (void)src;
            (void)res;
        }

        virtual void ExecAsync(IUserContext* ctx, int hostId, TVector<char>* params, IDCResultNotify* dcNotify, int reqId) const {
            Exec(ctx, hostId, params);
            dcNotify->DistrCmdComplete(reqId, params);
        }
        virtual void MergeAsync(TVector<TVector<char>>* src, IDCResultNotify* dcNotify, int reqId) const {
            TVector<char> buf;
            Merge(src, &buf);
            dcNotify->DistrCmdComplete(reqId, &buf);
        }
    };

    struct TJobParams {
        int CmdId, ParamId, ReduceId;
        short CompId, HostId;

        TJobParams() {
        }
        TJobParams(int cmdId, int paramId, int reduceId, short compId, short hostId)
            : CmdId(cmdId)
            , ParamId(paramId)
            , ReduceId(reduceId)
            , CompId(compId)
            , HostId(hostId)
        {
        }
    };

    struct TJobDescription {
        enum EMapId {
            ANYWHERE_HOST_ID = -1,
            MAP_HOST_ID = -2
        };
        TVector<TVector<char>> Cmds;
        TVector<char> ParamsData;
        TVector<int> ParamsPtr;
        TVector<TJobParams> ExecList;

        SAVELOAD(Cmds, ParamsData, ParamsPtr, ExecList);

    private:
        void AddJob(int hostId, int paramId, int reduceId);
        int AddParamData(TVector<char>* data);
        int GetReduceId();

    public:
        // helper functions
        template <class T>
        inline int AddParam(T* param) {
            CHROMIUM_TRACE_FUNCTION();
            if (IBinSaver::HasNonTrivialSerializer<T>(0u)) {
                TVector<char> tmp;
                SerializeToMem(&tmp, *param);
                return AddParamData(&tmp);
            } else {
                int res = ParamsPtr.ysize() - 1;
                int dstPtr = ParamsData.ysize();
                ParamsData.yresize(dstPtr + (int)sizeof(T));
                *reinterpret_cast<T*>(&ParamsData[dstPtr]) = *param;
                ParamsPtr.push_back(dstPtr + (int)sizeof(T));
                return res;
            }
        }
        void GetParam(int paramId, TVector<char>* res) {
            int sz = ParamsPtr[paramId + 1] - ParamsPtr[paramId];
            res->yresize(sz);
            if (sz > 0)
                memcpy(&(*res)[0], &ParamsData[ParamsPtr[paramId]], sz);
        }
        void AddMapImpl(int paramId);
        void AddQueryImpl(TVector<int> hostIds, int paramId);
        void AddQueryImpl(int hostId, int paramId);

    public:
        // interface
        TJobDescription();
        void SetCurrentOperation(IDistrCmd* op);
        void SetCurrentOperation(const TVector<char>& op);
        void AddMap(TVector<char>* data) {
            AddMapImpl(AddParamData(data));
        }
        void AddQuery(TVector<int> hostIds, TVector<char>* data) {
            AddQueryImpl(hostIds, AddParamData(data));
        }
        void AddQuery(int hostId, TVector<char>* data) {
            AddQueryImpl(hostId, AddParamData(data));
        }
        template <class T>
        void AddMap(T& data) {
            AddMapImpl(AddParam(&data));
        }
        template <class T>
        void AddQuery(TVector<int> hostIds, T& data) {
            AddQueryImpl(hostIds, AddParam(&data));
        }
        template <class T>
        void AddQuery(int hostId, T& data) {
            AddQueryImpl(hostId, AddParam(&data));
        }
        void MergeResults();                   // return single merged result of all ops
        void SeparateResults(int hostIdCount); // return all results separately, results for AddMap() will be returned separetely for each hostId
        void Swap(TJobDescription* res);
    };

    struct IMRCommandCompleteNotify : virtual public TThrRefBase {
        virtual void MRCommandComplete(bool isCanceled, TVector<TVector<char>>* res) = 0;
        virtual TGUID GetMasterQueryId() = 0; // return query id which we are processing here or 0 if no such query exist
        // for classes where MRNeedCheckCancel() returns true MRIsCmdNeeded() is regularly called and command is canceled if MRIsCmdNeeded() returns false
        virtual bool MRNeedCheckCancel() const {
            return false;
        }
        virtual bool MRIsCmdNeeded() const {
            return true;
        }
    };

    struct TDataLocation {
        i64 DataId;
        int CompId;

        SAVELOAD(DataId, CompId);

        TDataLocation()
            : DataId(0)
            , CompId(0)
        {
        }
        TDataLocation(i64 id, int compId)
            : DataId(id)
            , CompId(compId)
        {
        }
    };

    struct IUserContext: public TThrRefBase {
        enum EDataDistrState {
            DATA_COMPLETE,
            DATA_COPYING,
            DATA_UNAVAILABLE,
        };
        virtual const THashMap<int, int>& GetEnvId2Version() const = 0;
        virtual const IObjectBase* GetContextData(int envId, int hostId) = 0;
        virtual bool HasHostIds(const THashMap<int, bool>& hostIdSet) = 0;
        virtual int GetHostIdCount() = 0;
        virtual EDataDistrState UpdateDataDistrState(TVector<TVector<int>>* hostId2Computer) = 0;
        virtual IUserContext* CreateLocalOnlyContext() = 0;
        // tblId & versionID are used for memory management
        // when registering data for same table with newer version data for old version will be freed
        virtual TDataLocation RegisterData(int tblId, ui64 versionId, TVector<char>* p) = 0;
        virtual TDataLocation RegisterData(int tblId, ui64 versionId, const IObjectBase* obj) = 0;
        // For blocks larger than 2 Gb. It doesn't have CollectData() counterpart, because it's hard to
        // implement and doesn't seem worth doing. The main purpose is to allow SetContextData() with
        // TDataLocation for large (>2 Gb) objects.
        virtual TDataLocation RegisterData(int tblId, ui64 versionId, TVector<TVector<char>>* p) = 0;
        virtual void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res) = 0;

        virtual void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) = 0;
    };

    enum EKeepDataFlags {
        DELETE_RAW_DATA = 0,
        KEEP_CONTEXT_ON_MASTER = 1,
        KEEP_CONTEXT_RAW_DATA = 3,
    };

    struct IEnvironment: public IObjectBase {
        virtual int GetEnvId() = 0;
        virtual int GetHostIdCount() = 0;
        virtual void SetContextData(int hostId, const IObjectBase* data, EKeepDataFlags keepContextRawData = KEEP_CONTEXT_RAW_DATA) = 0;
        // number of locations should be equal to hostIdCount, data will be taken from specified data location
        // we assume that TObj<IObjectBase> is serialized in the data specified
        // used data will be freed, so you can not use these data locations after this func
        virtual void SetContextData(const TVector<TDataLocation>& data, EKeepDataFlags keepContextRawData = KEEP_CONTEXT_RAW_DATA) = 0;
        // Delete context data from master and binary copies from each comp.
        // Reduces memory consumption but makes computation on master impossible.
        virtual void DeleteContextRawData(int hostId, bool keepContextOnMaster = false) = 0;
        virtual void Run(TJobDescription* descr, IMRCommandCompleteNotify* completeNotify) = 0;
        virtual IEnvironment* CreateChildEnvironment(int envId) = 0;
        virtual void CollectData(const TVector<TDataLocation>& data, TVector<TVector<char>>* res) = 0;
    };

    struct IRootEnvironment: public IObjectBase {
        virtual IEnvironment* CreateEnvironment(int envId, const TVector<int>& hostIds) = 0;
        virtual IEnvironment* GetEverywhere() = 0;
        virtual IEnvironment* GetAnywhere() = 0;
        virtual TVector<int> MakeHostIdMapping(int groupCount) = 0;
        virtual int GetSlaveCount() = 0;
        virtual void WaitDistribution() = 0; // waits until all data is distributed
        virtual void Stop() = 0;
    };

    void RunSlave(int workerThreadCount, int port);
    IRootEnvironment* RunLocalMaster(int workerThreadCount);
    IRootEnvironment* RunMaster(int masterPort, int workerThreadCount, const char* hostsFileName, int defaultSlavePort, int debugPort);

    inline IRootEnvironment* RunMaster(int workerThreadCount, const char* hostsFileName, int defaultSlavePort) {
        const int DEBUG_PORT = 13013;
        const int AUTO_MASTER_PORT = 0;
        return RunMaster(AUTO_MASTER_PORT, workerThreadCount, hostsFileName, defaultSlavePort, DEBUG_PORT);
    }
    inline IRootEnvironment* RunMaster(int workerThreadCount, const char* hostsFileName, int defaultSlavePort, int debugPort) {
        const int AUTO_MASTER_PORT = 0;
        return RunMaster(AUTO_MASTER_PORT, workerThreadCount, hostsFileName, defaultSlavePort, debugPort);
    }
}
