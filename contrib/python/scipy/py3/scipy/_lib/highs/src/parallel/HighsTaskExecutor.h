/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Qi Huangfu, Leona Gottwald    */
/*    and Michael Feldmeier                                              */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_TASKEXECUTOR_H_
#define HIGHS_TASKEXECUTOR_H_

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <thread>
#include <vector>

#include "parallel/HighsCacheAlign.h"
#include "parallel/HighsSplitDeque.h"
#include "parallel/HighsSchedulerConstants.h"
#include "util/HighsInt.h"
#include "util/HighsRandom.h"

class HighsTaskExecutor {
 public:
  using cache_aligned = highs::cache_aligned;
  struct ExecutorHandle {
    cache_aligned::shared_ptr<HighsTaskExecutor> ptr{nullptr};

    ~ExecutorHandle();
  };

 private:
#ifdef _WIN32
  static HighsSplitDeque*& threadLocalWorkerDeque();
  static ExecutorHandle& threadLocalExecutorHandle();
#else
  static thread_local HighsSplitDeque* threadLocalWorkerDequePtr;
  static thread_local ExecutorHandle globalExecutorHandle;

  static HighsSplitDeque*& threadLocalWorkerDeque() {
    return threadLocalWorkerDequePtr;
  }

  static ExecutorHandle& threadLocalExecutorHandle() {
    return globalExecutorHandle;
  }
#endif

  std::vector<cache_aligned::unique_ptr<HighsSplitDeque>> workerDeques;
  cache_aligned::shared_ptr<HighsSplitDeque::WorkerBunk> workerBunk;
  std::atomic<ExecutorHandle*> mainWorkerHandle;

  HighsTask* random_steal_loop(HighsSplitDeque* localDeque) {
    const int numWorkers = workerDeques.size();

    int numTries = 16 * (numWorkers - 1);

    auto tStart = std::chrono::high_resolution_clock::now();

    while (true) {
      for (int s = 0; s < numTries; ++s) {
        HighsTask* task = localDeque->randomSteal();
        if (task) return task;
      }

      if (!workerBunk->haveJobs.load(std::memory_order_relaxed)) break;

      auto numMicroSecs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - tStart)
              .count();

      if (numMicroSecs < HighsSchedulerConstants::kMicroSecsBeforeGlobalSync)
        numTries *= 2;
      else
        break;
    }

    return nullptr;
  }

  void run_worker(int workerId) {
    // spin until the global executor pointer is set up
    ExecutorHandle* executor;
    while (!(executor = mainWorkerHandle.load(std::memory_order_acquire)))
      HighsSpinMutex::yieldProcessor();
    // now acquire a reference count of the global executor
    threadLocalExecutorHandle() = *executor;
    HighsSplitDeque* localDeque = workerDeques[workerId].get();
    threadLocalWorkerDeque() = localDeque;
    HighsTask* currentTask = workerBunk->waitForNewTask(localDeque);
    while (currentTask != nullptr) {
      localDeque->runStolenTask(currentTask);

      currentTask = random_steal_loop(localDeque);
      if (currentTask != nullptr) continue;

      currentTask = workerBunk->waitForNewTask(localDeque);
    }
  }

 public:
  HighsTaskExecutor(int numThreads) {
    assert(numThreads > 0);
    mainWorkerHandle.store(nullptr, std::memory_order_relaxed);
    workerDeques.resize(numThreads);
    workerBunk = cache_aligned::make_shared<HighsSplitDeque::WorkerBunk>();
    for (int i = 0; i < numThreads; ++i)
      workerDeques[i] = cache_aligned::make_unique<HighsSplitDeque>(
          workerBunk, workerDeques.data(), i, numThreads);

    threadLocalWorkerDeque() = workerDeques[0].get();
    for (int i = 1; i < numThreads; ++i)
      std::thread([&](int id) { run_worker(id); }, i).detach();
  }

  static HighsSplitDeque* getThisWorkerDeque() {
    return threadLocalWorkerDeque();
  }

  static int getNumWorkerThreads() {
    return threadLocalWorkerDeque()->getNumWorkers();
  }

  static void initialize(int numThreads) {
    auto& executorHandle = threadLocalExecutorHandle();
    if (!executorHandle.ptr) {
      executorHandle.ptr =
          cache_aligned::make_shared<HighsTaskExecutor>(numThreads);
      executorHandle.ptr->mainWorkerHandle.store(&executorHandle,
                                                 std::memory_order_release);
    }
  }

  static void shutdown(bool blocking = false) {
    auto& executorHandle = threadLocalExecutorHandle();
    if (executorHandle.ptr) {
      // first spin until every worker has acquired its executor reference
      while (executorHandle.ptr.use_count() !=
             executorHandle.ptr->workerDeques.size())
        HighsSpinMutex::yieldProcessor();
      // set the active flag to false first with release ordering
      executorHandle.ptr->mainWorkerHandle.store(nullptr,
                                                 std::memory_order_release);
      // now inject the null task as termination signal to every worker
      for (auto& workerDeque : executorHandle.ptr->workerDeques)
        workerDeque->injectTaskAndNotify(nullptr);
      // finally release the global executor reference
      if (blocking) {
        while (executorHandle.ptr.use_count() != 1)
          HighsSpinMutex::yieldProcessor();
      }

      executorHandle.ptr.reset();
    }
  }

  static void sync_stolen_task(HighsSplitDeque* localDeque,
                               HighsTask* stolenTask) {
    HighsSplitDeque* stealer;
    if (!localDeque->leapfrogStolenTask(stolenTask, stealer)) {
      const int numWorkers = localDeque->getNumWorkers();
      int numTries = HighsSchedulerConstants::kNumTryFac * (numWorkers - 1);

      auto tStart = std::chrono::high_resolution_clock::now();

      while (true) {
        for (int s = 0; s < numTries; ++s) {
          if (stolenTask->isFinished()) {
            localDeque->popStolen();
            return;
          }
          localDeque->yield();
        }

        auto numMicroSecs =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - tStart)
                .count();

        if (numMicroSecs < HighsSchedulerConstants::kMicroSecsBeforeSleep)
          numTries *= 2;
        else
          break;
      }

      if (!stolenTask->isFinished())
        localDeque->waitForTaskToFinish(stolenTask, stealer);
    }

    localDeque->popStolen();
  }
};

#endif
