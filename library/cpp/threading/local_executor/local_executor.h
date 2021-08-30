#pragma once

#include <library/cpp/threading/future/future.h>

#include <util/generic/cast.h>
#include <util/generic/fwd.h>
#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/singleton.h>
#include <util/generic/ymath.h>

#include <functional>

namespace NPar {
    struct ILocallyExecutable : virtual public TThrRefBase {
        // Must be implemented by the end user to define job that will be processed by one of
        // executor threads.
        //
        // @param id        Job parameter, typically an index pointing somewhere in array, or just
        //                  some dummy value, e.g. `0`.
        virtual void LocalExec(int id) = 0;
    };

    // Alternative and simpler way of describing a job for executor. Function argument has the
    // same meaning as `id` in `ILocallyExecutable::LocalExec`.
    //
    using TLocallyExecutableFunction = std::function<void(int)>;

    class ILocalExecutor: public TNonCopyable {
    public:
        ILocalExecutor() = default;
        virtual ~ILocalExecutor() = default;

        enum EFlags : int {
            HIGH_PRIORITY = 0,
            MED_PRIORITY = 1,
            LOW_PRIORITY = 2,
            PRIORITY_MASK = 3,
            WAIT_COMPLETE = 4
        };

        // Add task for further execution.
        //
        // @param exec          Task description.
        // @param id            Task argument.
        // @param flags         Bitmask composed by `HIGH_PRIORITY`, `MED_PRIORITY`, `LOW_PRIORITY`
        //                      and `WAIT_COMPLETE`.
        virtual void Exec(TIntrusivePtr<ILocallyExecutable> exec, int id, int flags) = 0;

        // Add tasks range for further execution.
        //
        // @param exec                      Task description.
        // @param firstId, lastId           Task arguments [firstId, lastId)
        // @param flags                     Same as for `Exec`.
        virtual void ExecRange(TIntrusivePtr<ILocallyExecutable> exec, int firstId, int lastId, int flags) = 0;

        // 0-based ILocalExecutor worker thread identification
        virtual int GetWorkerThreadId() const noexcept = 0;
        virtual int GetThreadCount() const noexcept = 0;

        // Describes a range of tasks with parameters from integer range [FirstId, LastId).
        //
        class TExecRangeParams {
        public:
            template <typename TFirst, typename TLast>
            TExecRangeParams(TFirst firstId, TLast lastId)
                : FirstId(SafeIntegerCast<int>(firstId))
                , LastId(SafeIntegerCast<int>(lastId))
            {
                Y_ASSERT(LastId >= FirstId);
                SetBlockSize(1);
            }
            // Partition tasks into `blockCount` blocks of approximately equal size, each of which
            // will be executed as a separate bigger task.
            //
            template <typename TBlockCount>
            TExecRangeParams& SetBlockCount(TBlockCount blockCount) {
                Y_ASSERT(SafeIntegerCast<int>(blockCount) > 0 || FirstId == LastId);
                BlockSize = FirstId == LastId ? 0 : CeilDiv(LastId - FirstId, SafeIntegerCast<int>(blockCount));
                BlockCount = BlockSize == 0 ? 0 : CeilDiv(LastId - FirstId, BlockSize);
                BlockEqualToThreads = false;
                return *this;
            }
            // Partition tasks into blocks of approximately `blockSize` size, each of which will
            // be executed as a separate bigger task.
            //
            template <typename TBlockSize>
            TExecRangeParams& SetBlockSize(TBlockSize blockSize) {
                Y_ASSERT(SafeIntegerCast<int>(blockSize) > 0 || FirstId == LastId);
                BlockSize = SafeIntegerCast<int>(blockSize);
                BlockCount = BlockSize == 0 ? 0 : CeilDiv(LastId - FirstId, BlockSize);
                BlockEqualToThreads = false;
                return *this;
            }
            // Partition tasks into thread count blocks of approximately equal size, each of which
            // will be executed as a separate bigger task.
            //
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
            bool GetBlockEqualToThreads() {
                return BlockEqualToThreads;
            }

            const int FirstId = 0;
            const int LastId = 0;

        private:
            int BlockSize;
            int BlockCount;
            bool BlockEqualToThreads;
        };

        // `Exec` and `ExecRange` versions that accept functions.
        //
        void Exec(TLocallyExecutableFunction exec, int id, int flags);
        void ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags);

        // Version of `ExecRange` that throws exception from task with minimal id if at least one of
        // task threw an exception.
        //
        void ExecRangeWithThrow(TLocallyExecutableFunction exec, int firstId, int lastId, int flags);

        // Version of `ExecRange` that returns vector of futures, thus allowing to retry any task if
        // it fails.
        //
        TVector<NThreading::TFuture<void>> ExecRangeWithFutures(TLocallyExecutableFunction exec, int firstId, int lastId, int flags);

        template <typename TBody>
        static inline auto BlockedLoopBody(const TExecRangeParams& params, const TBody& body) {
            return [=](int blockId) {
                const int blockFirstId = params.FirstId + blockId * params.GetBlockSize();
                const int blockLastId = Min(params.LastId, blockFirstId + params.GetBlockSize());
                for (int i = blockFirstId; i < blockLastId; ++i) {
                    body(i);
                }
            };
        }

        template <typename TBody>
        inline void ExecRange(TBody&& body, TExecRangeParams params, int flags) {
            if (TryExecRangeSequentially(body, params.FirstId, params.LastId, flags)) {
                return;
            }
            if (params.GetBlockEqualToThreads()) {
                params.SetBlockCount(GetThreadCount() + ((flags & WAIT_COMPLETE) != 0)); // ThreadCount or ThreadCount+1 depending on WaitFlag
            }
            ExecRange(BlockedLoopBody(params, body), 0, params.GetBlockCount(), flags);
        }

        template <typename TBody>
        inline void ExecRangeBlockedWithThrow(TBody&& body, int firstId, int lastId, int batchSizeOrZeroForAutoBatchSize, int flags) {
            if (firstId >= lastId) {
                return;
            }
            const int threadCount = Max(GetThreadCount(), 1);
            const int batchSize = batchSizeOrZeroForAutoBatchSize
                ? batchSizeOrZeroForAutoBatchSize
                : (lastId - firstId + threadCount - 1) / threadCount;
            const int batchCount = (lastId - firstId + batchSize - 1) / batchSize;
            const int batchCountPerThread = (batchCount + threadCount - 1) / threadCount;
            auto states = ExecRangeWithFutures(
                [=](int threadId) {
                    for (int batchIdPerThread = 0; batchIdPerThread < batchCountPerThread; ++batchIdPerThread) {
                        int batchId = batchIdPerThread * threadCount + threadId;
                        int begin = firstId + batchId * batchSize;
                        int end = Min(begin + batchSize, lastId);
                        for (int i = begin; i < end; ++i) {
                            body(i);
                        }
                    }
                },
                0, threadCount, flags);
            for (auto& state: states) {
                state.GetValueSync(); // Re-throw exception if any.
            }
        }

        template <typename TBody>
        static inline bool TryExecRangeSequentially(TBody&& body, int firstId, int lastId, int flags) {
            if (lastId == firstId) {
                return true;
            }
            if ((flags & WAIT_COMPLETE) && lastId - firstId == 1) {
                body(firstId);
                return true;
            }
            return false;
        }
    };

    // `TLocalExecutor` provides facilities for easy parallelization of existing code and cycles.
    //
    // Examples:
    // Execute one task with medium priority and wait for it completion.
    // ```
    // LocalExecutor().Run(4);
    // TEvent event;
    // LocalExecutor().Exec([](int) {
    //     SomeFunc();
    //     event.Signal();
    // }, 0, TLocalExecutor::MED_PRIORITY);
    //
    // SomeOtherCode();
    // event.WaitI();
    // ```
    //
    // Execute range of tasks with medium priority.
    // ```
    // LocalExecutor().Run(4);
    // LocalExecutor().ExecRange([](int id) {
    //     SomeFunc(id);
    // }, TExecRangeParams(0, 10), TLocalExecutor::WAIT_COMPLETE | TLocalExecutor::MED_PRIORITY);
    // ```
    //
    class TLocalExecutor final: public ILocalExecutor {
    public:
        using EFlags = ILocalExecutor::EFlags;

        // Creates executor without threads. You'll need to explicitly call `RunAdditionalThreads`
        // to add threads to underlying thread pool.
        //
        TLocalExecutor();
        ~TLocalExecutor();

        int GetQueueSize() const noexcept;
        int GetMPQueueSize() const noexcept;
        int GetLPQueueSize() const noexcept;
        void ClearLPQueue();

        // 0-based TLocalExecutor worker thread identification
        int GetWorkerThreadId() const noexcept override;
        int GetThreadCount() const noexcept override;

        // **Add** threads to underlying thread pool.
        //
        // @param threadCount       Number of threads to add.
        void RunAdditionalThreads(int threadCount);

        // Add task for further execution.
        //
        // @param exec          Task description.
        // @param id            Task argument.
        // @param flags         Bitmask composed by `HIGH_PRIORITY`, `MED_PRIORITY`, `LOW_PRIORITY`
        //                      and `WAIT_COMPLETE`.
        void Exec(TIntrusivePtr<ILocallyExecutable> exec, int id, int flags) override;

        // Add tasks range for further execution.
        //
        // @param exec                      Task description.
        // @param firstId, lastId           Task arguments [firstId, lastId)
        // @param flags                     Same as for `Exec`.
        void ExecRange(TIntrusivePtr<ILocallyExecutable> exec, int firstId, int lastId, int flags) override;

        using ILocalExecutor::Exec;
        using ILocalExecutor::ExecRange;

    private:
        class TImpl;
        THolder<TImpl> Impl_;
    };

    static inline TLocalExecutor& LocalExecutor() {
        return *Singleton<TLocalExecutor>();
    }

    template <typename TBody>
    inline void ParallelFor(ILocalExecutor& executor, ui32 from, ui32 to, TBody&& body) {
        ILocalExecutor::TExecRangeParams params(from, to);
        params.SetBlockCountToThreadCount();
        executor.ExecRange(std::forward<TBody>(body), params, TLocalExecutor::WAIT_COMPLETE);
    }

    template <typename TBody>
    inline void ParallelFor(ui32 from, ui32 to, TBody&& body) {
        ParallelFor(LocalExecutor(), from, to, std::forward<TBody>(body));
    }

    template <typename TBody>
    inline void AsyncParallelFor(ui32 from, ui32 to, TBody&& body) {
        ILocalExecutor::TExecRangeParams params(from, to);
        params.SetBlockCountToThreadCount();
        LocalExecutor().ExecRange(std::forward<TBody>(body), params, 0);
    }
}
