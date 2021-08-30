#pragma once

#include "local_executor.h"
#define __TBB_TASK_ISOLATION 1
#define __TBB_NO_IMPLICIT_LINKAGE 1

#include <contrib/libs/tbb/include/tbb/blocked_range.h>
#include <contrib/libs/tbb/include/tbb/parallel_for.h>
#include <contrib/libs/tbb/include/tbb/task_arena.h>
#include <contrib/libs/tbb/include/tbb/task_group.h>

namespace NPar {
    template <bool RespectTls = false>
    class TTbbLocalExecutor final: public ILocalExecutor  {
    public:
        TTbbLocalExecutor(int nThreads)
            : ILocalExecutor()
            , TbbArena(nThreads)
            , NumberOfTbbThreads(nThreads) {}
        ~TTbbLocalExecutor() noexcept override {}

        // 0-based ILocalExecutor worker thread identification
        virtual int GetWorkerThreadId() const noexcept override;
        virtual int GetThreadCount() const noexcept override;

        // Add task for further execution.
        //
        // @param exec          Task description.
        // @param id            Task argument.
        // @param flags         Bitmask composed by `HIGH_PRIORITY`, `MED_PRIORITY`, `LOW_PRIORITY`
        //                      and `WAIT_COMPLETE`.
        virtual void Exec(TIntrusivePtr<ILocallyExecutable> exec, int id, int flags) override;

        // Add tasks range for further execution.
        //
        // @param exec                      Task description.
        // @param firstId, lastId           Task arguments [firstId, lastId)
        // @param flags                     Same as for `Exec`.
        virtual void ExecRange(TIntrusivePtr<ILocallyExecutable> exec, int firstId, int lastId, int flags) override;

        // Submit tasks for async run
        void SubmitAsyncTasks(TLocallyExecutableFunction exec, int firstId, int lastId);

    private:
        mutable tbb::task_arena TbbArena;
        tbb::task_group Group;
        int NumberOfTbbThreads;
    };
}
