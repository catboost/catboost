#pragma once

#include "par.h"

#include <library/cpp/binsaver/mem_io.h>

#include <util/system/type_name.h>
#include <util/generic/vector.h>
#include <util/system/compiler.h>
#include <util/system/event.h>
#include <util/system/yassert.h>

namespace NPar {
    template <class T>
    class TCtxPtr {
        const T* Ptr;

    public:
        TCtxPtr(IUserContext* ctx, int envId, int hostId)
            : Ptr(nullptr)
        {
            const IObjectBase* obj = ctx->GetContextData(envId, hostId);
            if (obj) {
                if (Y_UNLIKELY(typeid(*obj) != typeid(T))) {
                    const auto objTypeName = TypeName(*obj);
                    const auto expectedTypeName = TypeName<T>();
                    Y_ABORT("type mismatch: %s != %s", objTypeName.c_str(), expectedTypeName.c_str());
                }
                Ptr = CastToUserObject(const_cast<IObjectBase*>(obj), (T*)nullptr);
            }
        }
        const T* operator->() const {
            return Ptr;
        }
        operator const T*() const {
            return Ptr;
        }
        const T* GetPtr() const {
            return Ptr;
        }
    };

    template <class TInputArg, class TOutputArg>
    class TMapReduceCmd: public IDistrCmd {
        void ExecAsync(IUserContext* ctx, int hostId, TVector<char>* params, IDCResultNotify* dcNotify, int reqId) const override {
            CHROMIUM_TRACE_FUNCTION();
            TInputArg inp;
            SerializeFromMem(params, inp);
            TOutputArg outp;
            DoMapEx(ctx, hostId, &inp, &outp, dcNotify);
            TVector<char> buf;
            SerializeToMem(&buf, outp);
            dcNotify->DistrCmdComplete(reqId, &buf);
        }
        void MergeAsync(TVector<TVector<char>>* src, IDCResultNotify* dcNotify, int reqId) const override {
            CHROMIUM_TRACE_FUNCTION();
            TVector<TOutputArg> srcData;
            int count = src->ysize();
            srcData.resize(count);
            for (int i = 0; i < count; ++i)
                SerializeFromMem(&(*src)[i], srcData[i]);
            TOutputArg outp;
            DoReduce(&srcData, &outp);
            TVector<char> buf;
            SerializeToMem(&buf, outp);
            dcNotify->DistrCmdComplete(reqId, &buf);
        }

    public:
        using TInput = TInputArg;
        using TOutput = TOutputArg;

    protected:
        virtual void DoMapEx(IUserContext* ctx, int hostId, TInput* src, TOutput* dst, IDCResultNotify* dcNotify) const {
            CHROMIUM_TRACE_FUNCTION();
            (void)dcNotify;
            DoMap(ctx, hostId, src, dst);
        }
        virtual void DoMap(IUserContext* ctx, int hostId, TInput* src, TOutput* dst) const {
            CHROMIUM_TRACE_FUNCTION();

            (void)ctx;
            (void)hostId;
            (void)src;
            (void)dst;
            Y_ABORT("missing map implementation");
        }
        virtual void DoReduce(TVector<TOutput>* src, TOutput* dst) const {
            CHROMIUM_TRACE_FUNCTION();
            (void)src;
            (void)dst;
            Y_ABORT("missing reduce implementation");
        }
    };

    void RemoteMap(TJobDescription* job, IDistrCmd* finalMap);
    void RemoteMapReduce(TJobDescription* job, IDistrCmd* finalMap);

    template <class TInput, class TOutput>
    inline void Map(TJobDescription* job, TMapReduceCmd<TInput, TOutput>* cmd, TVector<TInput>* src) {
        job->SetCurrentOperation(cmd);
        TVector<char> buf;
        for (int i = 0; i < src->ysize(); ++i)
            job->AddMap((*src)[i]);
    }

    class TJobExecutor {
        class TCallback: public IMRCommandCompleteNotify {
        public:
            bool IsReadyFlag;
            TSystemEvent Ready;
            TVector<TVector<char>> Results;

            void MRCommandComplete(bool isCanceled, TVector<TVector<char>>* res) override {
                // easy way to get isCanceled here is to forget to call SetCotextData() for all hostIds
                Y_ABORT_UNLESS(!isCanceled);
                Y_ASSERT(!IsReadyFlag);
                Results.swap(*res);
                IsReadyFlag = true;
                Ready.Signal();
            }
            TGUID GetMasterQueryId() override;
            TCallback()
                : IsReadyFlag(false)
            {
            }
        };

        TIntrusivePtr<TCallback> Callback;

    public:
        TJobExecutor(TJobDescription* descr, IEnvironment* env) {
            Callback = new TCallback();
            env->Run(descr, Callback.Get());
        }
        TJobExecutor(TJobDescription* descr, IUserContext* ctx) {
            Callback = new TCallback();
            ctx->Run(descr, Callback.Get());
        }
        ~TJobExecutor() {
            Y_ASSERT(Callback->IsReadyFlag);
        }
        bool IsReady() const {
            return Callback->IsReadyFlag;
        }
        void GetRawResult(TVector<TVector<char>>* res) {
            Callback->Ready.Wait();
            Y_ASSERT(Callback->IsReadyFlag);
            if (res)
                res->swap(Callback->Results);
        }
        template <class T>
        void GetResult(T* res) {
            TVector<TVector<char>> buf;
            GetRawResult(&buf);
            Y_ABORT_UNLESS(buf.ysize() == 1, "buf.ysize()=%d", buf.ysize());
            SerializeFromMem(&buf[0], *res);
        }
        template <class T>
        void GetResultVec(TVector<T>* res) {
            CHROMIUM_TRACE_FUNCTION();
            TVector<TVector<char>> buf;
            GetRawResult(&buf);
            int resultCount = buf.ysize();
            res->resize(resultCount);
            for (int i = 0; i < resultCount; ++i)
                SerializeFromMem(&buf[i], (*res)[i]);
        }
        template <class T>
        void GetRemoteMapResults(TVector<T>* res) {
            CHROMIUM_TRACE_FUNCTION();
            TVector<TVector<char>> groups;
            GetRawResult(&groups);
            for (int g = 0; g < groups.ysize(); ++g) {
                TVector<TVector<char>> buf;
                SerializeFromMem(&groups[g], buf);
                int startIdx = res->ysize();
                res->resize(startIdx + buf.ysize());
                for (int i = 0; i < buf.ysize(); ++i)
                    SerializeFromMem(&buf[i], (*res)[startIdx + i]);
            }
        }
    };

    template <class TInput, class TOutput>
    inline void RunMap(IEnvironment* env, TMapReduceCmd<TInput, TOutput>* cmd, TVector<TInput>* src, TVector<TOutput>* dst) {
        TJobDescription jop;
        Map(&jop, cmd, src);
        TJobExecutor exec(&jop, env);
        exec.GetResultVec(dst);
    }

    template <class TInput, class TOutput>
    inline void RunMapReduce(IEnvironment* env, TMapReduceCmd<TInput, TOutput>* cmd, TVector<TInput>* src, TOutput* dst) {
        TJobDescription jop;
        Map(&jop, cmd, src);
        jop.MergeResults();
        TJobExecutor exec(&jop, env);
        exec.GetResult(dst);
    }

    void MakeRunOnRangeImpl(TJobDescription* job, int start, int finish, IDistrCmd* finalMap);
    template <class TOutput>
    void RunMapReduceOnRange(IEnvironment* env, TMapReduceCmd<int, TOutput>* cmd, int start, int finish, TOutput* dst) {
        TJobDescription jop;
        MakeRunOnRangeImpl(&jop, start, finish, cmd);
        TJobExecutor exec(&jop, env);
        exec.GetResult(dst);
    }

    // TDataCalcer inherits from TMapReduceCmd<TInput,TData>
    template <typename TDataCalcer, typename TData>
    void CalcDistributed(
        int partsCount,
        IEnvironment* env,
        const TDataCalcer& dataCalcer,
        TVector<TData>* data) {
        TJobDescription job;
        job.SetCurrentOperation(dataCalcer);
        for (int part = 0; part < partsCount; ++part) {
            job.AddQuery(part, part);
        }
        TJobExecutor exec(&job, env);
        exec.GetResultVec(data);
    }

    template <typename TDataUpdater>
    void UpdateDistributed(
        int partsCount,
        IEnvironment* sourceEnvironment,
        const TDataUpdater& dataUpdater,
        IEnvironment* targetEnvironment) {
        CHROMIUM_TRACE_FUNCTION();
        TVector<TDataLocation> targetLocations;
        CalcDistributed(partsCount, sourceEnvironment, dataUpdater, &targetLocations);
        targetEnvironment->SetContextData(targetLocations);
    }
}
