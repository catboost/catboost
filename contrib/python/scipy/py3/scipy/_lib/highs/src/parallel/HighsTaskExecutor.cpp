#include "parallel/HighsTaskExecutor.h"

using namespace highs;

#ifdef _WIN32
static thread_local HighsSplitDeque* threadLocalWorkerDequePtr{nullptr};
HighsSplitDeque*& HighsTaskExecutor::threadLocalWorkerDeque() {
  return threadLocalWorkerDequePtr;
}

static thread_local HighsTaskExecutor::ExecutorHandle globalExecutorHandle{};

HighsTaskExecutor::ExecutorHandle&
HighsTaskExecutor::threadLocalExecutorHandle() {
  return globalExecutorHandle;
}
#else
thread_local HighsSplitDeque* HighsTaskExecutor::threadLocalWorkerDequePtr{
    nullptr};
thread_local HighsTaskExecutor::ExecutorHandle
    HighsTaskExecutor::globalExecutorHandle{};
#endif

HighsTaskExecutor::ExecutorHandle::~ExecutorHandle() {
  if (ptr && this == ptr->mainWorkerHandle.load(std::memory_order_relaxed))
    HighsTaskExecutor::shutdown();
}
