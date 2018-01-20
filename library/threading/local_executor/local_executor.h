#pragma once

#include <library/threading/future/future.h>

#include <util/generic/ptr.h>
#include <util/thread/lfqueue.h>
#include <util/system/event.h>
#include <util/generic/singleton.h>
#include <util/system/tls.h>
#include <functional>

class TThread;
namespace NPar {
    struct ILocallyExecutable : virtual public TThrRefBase {
        virtual void LocalExec(int id) = 0;
    };

    using TLocallyExecutableFunction = std::function<void(int)>;

#ifdef _freebsd_
    class TFastFreeBsdEvent: public TNonCopyable {
        enum {
            MAX_THREAD_COUNT = 100
        };
        long ThreadId[MAX_THREAD_COUNT];
        void* volatile ThreadState[MAX_THREAD_COUNT]; // 0 = run, -1 = wake, -2 = sleep
        TAtomic ThreadCount;
        void* volatile Signaled;

        int GetCurrentThreadIdx();

    public:
        TFastFreeBsdEvent();
        void Signal();
        void Reset();
        void Wait();
    };
#endif

    class TLocalRangeExecutor;
    class TLocalExecutor: public TNonCopyable {
        struct TSingleJob {
            TIntrusivePtr<ILocallyExecutable> Exec;
            int Id;

            TSingleJob()
                : Exec(nullptr)
                , Id(0)
            {
            }
            TSingleJob(ILocallyExecutable* exec, int id)
                : Exec(exec)
                , Id(id)
            {
            }
        };

        TLockFreeQueue<TSingleJob> JobQueue, MedJobQueue, LowJobQueue;
#ifdef _freebsd_
        TFastFreeBsdEvent HasJob;
#else
        Event HasJob;
#endif

        TAtomic ThreadCount, QueueSize, MPQueueSize, LPQueueSize;
        TAtomic ThreadId;

        Y_THREAD(int)
        CurrentTaskPriority;
        Y_THREAD(int)
        WorkerThreadId;

        static void* HostWorkerThread(void* p);
        bool GetJob(TSingleJob* job);
        void RunNewThread();
        void LaunchRange(TLocalRangeExecutor* execRange, int queueSizeLimit,
                         TAtomic* queueSize, TLockFreeQueue<TSingleJob>* jobQueue);

    public:
        enum EFlags {
            HIGH_PRIORITY = 0,
            MED_PRIORITY = 1,
            LOW_PRIORITY = 2,
            PRIORITY_MASK = 3,
            WAIT_COMPLETE = 4
        };

        class TExecRangeParams {
        public:
            TExecRangeParams(int firstId, int lastId) // [firstId..lastId)
                : FirstId(firstId)
                , LastId(lastId)
            {
                Y_ASSERT(lastId >= firstId);
                SetBlockSize(1);
            }
            TExecRangeParams& SetBlockCount(int blockCount) {
                BlockSize = CeilDiv(LastId - FirstId, blockCount);
                BlockCount = CeilDiv(LastId - FirstId, BlockSize);
                BlockEqualToThreads = false;
                return *this;
            }
            TExecRangeParams& SetBlockSize(int blockSize) {
                BlockSize = blockSize;
                BlockCount = CeilDiv(LastId - FirstId, blockSize);
                BlockEqualToThreads = false;
                return *this;
            }
            TExecRangeParams& SetBlockCountToThreadCount() {
                BlockEqualToThreads = true;
                return *this;
            }
            int GetBlockCount() const {
                Y_ASSERT(!BlockEqualToThreads);
                return BlockCount;
            }
            int GetBlockSize() const {
                Y_ASSERT(!BlockEqualToThreads);
                return BlockSize;
            }

            const int FirstId = 0;
            const int LastId = 0;

        private:
            friend TLocalExecutor;

            static inline int CeilDiv(int x, int y) {
                return (x + y - 1) / y;
            }

            int BlockSize;
            int BlockCount;
            bool BlockEqualToThreads;
        };

        template <typename TBody>
        inline static auto BlockedLoopBody(const TLocalExecutor::TExecRangeParams& params, const TBody& body) {
            return [=](int blockId) {
                const int blockFirstId = params.FirstId + blockId * params.BlockSize;
                const int blockLastId = Min(params.LastId, blockFirstId + params.BlockSize);
                for (int i = blockFirstId; i < blockLastId; ++i) {
                    body(i);
                }
            };
        }

        TLocalExecutor()
            : ThreadCount(0)
            , QueueSize(0)
            , MPQueueSize(0)
            , LPQueueSize(0)
            , ThreadId(0)
        {
        }
        ~TLocalExecutor();
        void RunAdditionalThreads(int threadCount);
        void Exec(ILocallyExecutable* exec, int id, int flags);
        void ExecRange(ILocallyExecutable* exec, int firstId, int lastId, int flags); // [firstId..lastId)
        void Exec(TLocallyExecutableFunction exec, int id, int flags);
        void ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags); // [firstId..lastId)
        void ExecRangeWithThrow(TLocallyExecutableFunction exec, int firstId, int lastId, int flags);
        TVector<NThreading::TFuture<void>> ExecRangeWithFutures(TLocallyExecutableFunction exec, int firstId, int lastId, int flags);
        template <typename TBody>
        inline void ExecRange(TBody&& body, TExecRangeParams params, int flags) {
            if (params.LastId == params.FirstId) {
                return;
            }
            if (params.BlockEqualToThreads) {
                params.SetBlockCount(ThreadCount + ((flags & WAIT_COMPLETE) != 0)); // ThreadCount or ThreadCount+1 depending on WaitFlag
            }
            ExecRange(BlockedLoopBody(params, body), 0, params.BlockCount, flags);
        }
        int GetQueueSize() const {
            return QueueSize;
        }
        int GetMPQueueSize() const {
            return MPQueueSize;
        }
        int GetLPQueueSize() const {
            return LPQueueSize;
        }
        void ClearLPQueue();
        int GetWorkerThreadId() {
            return WorkerThreadId;
        } // 0-based TLocalExecutor worker thread identification
        int GetThreadCount() const {
            return ThreadCount;
        }

        friend class TLocalEvent;
    };

    inline static TLocalExecutor& LocalExecutor() {
        return *Singleton<TLocalExecutor>();
    }

    template <typename TBody>
    inline void ParallelFor(TLocalExecutor& executor,
                            ui32 from, ui32 to, TBody&& body) {
        TLocalExecutor::TExecRangeParams params(from, to);
        params.SetBlockCountToThreadCount();
        executor.ExecRange(std::forward<TBody>(body), params, TLocalExecutor::WAIT_COMPLETE);
    }

    template <typename TBody>
    inline void ParallelFor(ui32 from, ui32 to, TBody&& body) {
        ParallelFor(LocalExecutor(), from, to, std::forward<TBody>(body));
    }

    template <typename TBody>
    inline void AsyncParallelFor(ui32 from, ui32 to, TBody&& body) {
        TLocalExecutor::TExecRangeParams params(from, to);
        params.SetBlockCountToThreadCount();
        LocalExecutor().ExecRange(std::forward<TBody>(body), params, 0);
    }
}
