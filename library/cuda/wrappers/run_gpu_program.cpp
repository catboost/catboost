#include "run_gpu_program.h"

#include <library/cpp/threading/future/async.h>
#include <util/thread/pool.h>

void RunGpuProgram(std::function<void()> func) {
    TThreadPool queue;
    queue.Start(1);
    NThreading::TFuture<void> future = NThreading::Async(
        func,
        queue
    );
    future.GetValueSync();
}
