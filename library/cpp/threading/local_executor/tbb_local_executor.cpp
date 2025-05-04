#include "tbb_local_executor.h"

template <bool RespectTls>
void NPar::TTbbLocalExecutor<RespectTls>::SubmitAsyncTasks(TLocallyExecutableFunction exec, int firstId, int lastId) {
    for (int i = firstId; i < lastId; ++i) {
        Group.run([=] { exec(i); });
    }
}

template <bool RespectTls>
int NPar::TTbbLocalExecutor<RespectTls>::GetThreadCount() const noexcept {
    return NumberOfTbbThreads - 1;
}

template <bool RespectTls>
int NPar::TTbbLocalExecutor<RespectTls>::GetWorkerThreadId() const noexcept {
    static thread_local int WorkerThreadId = -1;
    if (WorkerThreadId == -1) {
        // Can't rely on return value except checking that it is 'not_initialized' because of
        //  "Since a thread may exit the arena at any time if it does not execute a task, the index of
        //   a thread may change between any two tasks"
        //  (https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/task_scheduler/task_arena/this_task_arena_ns#_CPPv4N3tbb15this_task_arena20current_thread_indexEv)
        const auto tbbThreadIndex = tbb::this_task_arena::current_thread_index();
        if (tbbThreadIndex == tbb::task_arena::not_initialized) {
            // This thread does not belong to TBB worker threads
            WorkerThreadId = 0;
        } else {
            WorkerThreadId = ++RegisteredThreadCounter;
        }
    }
    return WorkerThreadId;
}

template <bool RespectTls>
void NPar::TTbbLocalExecutor<RespectTls>::Exec(TIntrusivePtr<ILocallyExecutable> exec, int id, int flags) {
    if (flags & WAIT_COMPLETE) {
        exec->LocalExec(id);
    } else {
        TbbArena.execute([this, exec, id] {
            SubmitAsyncTasks([=] (int id) { exec->LocalExec(id); }, id, id + 1);
        });
    }
}

template <bool RespectTls>
void NPar::TTbbLocalExecutor<RespectTls>::ExecRange(TIntrusivePtr<ILocallyExecutable> exec, int firstId, int lastId, int flags) {
    if (flags & WAIT_COMPLETE) {
        TbbArena.execute([=] {
            if (RespectTls) {
                tbb::this_task_arena::isolate([=]{
                    tbb::parallel_for(firstId, lastId, [=] (int id) { exec->LocalExec(id); });
                });
            } else {
                tbb::parallel_for(firstId, lastId, [=] (int id) { exec->LocalExec(id); });
            }
        });
    } else {
        TbbArena.execute([this, exec, firstId, lastId] {
            SubmitAsyncTasks([exec] (int id) { exec->LocalExec(id); }, firstId, lastId);
        });
    }
}

template class NPar::TTbbLocalExecutor<true>;
template class NPar::TTbbLocalExecutor<false>;
