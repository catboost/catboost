#include "run_gpu_program.h"

#include <library/threading/future/async.h>
#include <util/thread/pool.h>

void RunGpuProgram(std::function<void()> func) {
    auto queue = CreateThreadPool(1);
    NThreading::TFuture<void> future = NThreading::Async(
        func,
        *queue
    );
    future.Wait();
}
