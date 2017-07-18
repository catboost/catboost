#pragma once

// TODO(annaveronika): readme

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

        class TBlockParams {
        public:
            TBlockParams(int firstId, int lastId) // [firstId..lastId)
                : FirstId(firstId)
                , LastId(lastId)
            {
                Y_ASSERT(lastId >= firstId);
            }
            TBlockParams& WaitCompletion() {
                Flags |= TLocalExecutor::WAIT_COMPLETE;
                return *this;
            }
            TBlockParams& HighPriority() {
                Flags = (Flags & ~TLocalExecutor::PRIORITY_MASK) | TLocalExecutor::HIGH_PRIORITY;
                return *this;
            }
            TBlockParams& MediumPriority() {
                Flags = (Flags & ~TLocalExecutor::PRIORITY_MASK) | TLocalExecutor::MED_PRIORITY;
                return *this;
            }
            TBlockParams& LowPriority() {
                Flags = (Flags & ~TLocalExecutor::PRIORITY_MASK) | TLocalExecutor::LOW_PRIORITY;
                return *this;
            }
            TBlockParams& SetBlockCount(int blockCount) {
                BlockSize = CeilDiv(LastId - FirstId, blockCount);
                BlockCount = CeilDiv(LastId - FirstId, BlockSize);
                return *this;
            }
            TBlockParams& SetBlockSize(int blockSize) {
                BlockSize = blockSize;
                BlockCount = CeilDiv(LastId - FirstId, blockSize);
                return *this;
            }
            int GetBlockCount() const {
                return BlockCount;
            }
            int GetBlockSize() const {
                return BlockSize;
            }

            const int FirstId = 0;
            const int LastId = 0;

        private:
            friend TLocalExecutor;

            static inline int CeilDiv(int x, int y) {
                return (x + y - 1) / y;
            }

            // if BlockSize and BlockCount not set, SetBlockCount(ThreadCount) (wait flag unset) or SetBlockCount(ThreadCount+1) (wait flag set)
            int BlockSize = 0;
            int BlockCount = 0;
            int Flags = 0;
        };

        template<typename TBody>
        inline static auto BlockedLoopBody(const TLocalExecutor::TBlockParams& params, const TBody& body) {
            return [=] (int blockId) {
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
        template <typename TBody>
        inline void ExecRange(TBody&& body, TBlockParams params) {
            if (params.BlockCount == 0 || params.BlockSize == 0) { // blocking not specified
                params.SetBlockCount(ThreadCount + ((params.Flags & WAIT_COMPLETE) != 0)); // ThreadCount or ThreadCount+1 depending on WaitFlag
            }
            ExecRange(BlockedLoopBody(params, body), 0, params.BlockCount, params.Flags);
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
    inline void ParallelFor(ui32 from, ui32 to, TBody&& body) {
        TLocalExecutor::TBlockParams params(from, to);
        params.WaitCompletion();
        LocalExecutor().ExecRange(std::move(body), params);
    }
}
