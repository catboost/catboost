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
    return TbbArena.execute([] {
        return tbb::this_task_arena::current_thread_index();
    });
}

template <bool RespectTls>
void NPar::TTbbLocalExecutor<RespectTls>::Exec(TIntrusivePtr<ILocallyExecutable> exec, int id, int flags) {
    if (flags & WAIT_COMPLETE) {
        exec->LocalExec(id);
    } else {
        TbbArena.execute([=] {
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
        TbbArena.execute([=] {
            SubmitAsyncTasks([=] (int id) { exec->LocalExec(id); }, firstId, lastId);
        });
    }
}

template class NPar::TTbbLocalExecutor<true>;
template class NPar::TTbbLocalExecutor<false>;
