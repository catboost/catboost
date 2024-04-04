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

#ifndef HIGHS_BINARY_SEMAPHORE_H_
#define HIGHS_BINARY_SEMAPHORE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "parallel/HighsCacheAlign.h"
#include "parallel/HighsSchedulerConstants.h"
#include "parallel/HighsSpinMutex.h"

class HighsBinarySemaphore {
  struct Data {
    std::atomic<int> count;
    alignas(64) std::mutex mutex;
    std::condition_variable condvar;

    Data(int init) : count(init) {}
  };

  highs::cache_aligned::unique_ptr<Data> data_;

 public:
  HighsBinarySemaphore(bool init = false)
      : data_(highs::cache_aligned::make_unique<Data>(init)) {}

  void release() {
    int prev = data_->count.exchange(1, std::memory_order_release);
    if (prev < 0) {
      std::unique_lock<std::mutex> lg{data_->mutex};
      data_->condvar.notify_one();
    }
  }

  bool try_acquire() {
    int expected = 1;
    return data_->count.compare_exchange_weak(
        expected, 0, std::memory_order_acquire, std::memory_order_relaxed);
  }

  void acquire() {
    if (try_acquire()) return;

    auto tStart = std::chrono::high_resolution_clock::now();
    int spinIters = 10;
    while (true) {
      for (int i = 0; i < spinIters; ++i) {
        if (data_->count.load(std::memory_order_relaxed) == 1) {
          if (try_acquire()) return;
        }
        HighsSpinMutex::yieldProcessor();
      }

      auto numMicroSecs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - tStart)
              .count();

      if (numMicroSecs < HighsSchedulerConstants::kMicroSecsBeforeSleep)
        spinIters *= 2;
      else
        break;
    }

    std::unique_lock<std::mutex> lg{data_->mutex};
    int prev = data_->count.exchange(-1, std::memory_order_relaxed);
    if (prev == 1) {
      data_->count.store(0, std::memory_order_relaxed);
      return;
    }

    do {
      data_->condvar.wait(lg);
    } while (data_->count.load(std::memory_order_relaxed) != 1);

    data_->count.store(0, std::memory_order_relaxed);
  }

  std::unique_lock<std::mutex> lockMutexForAcquire() {
    return std::unique_lock<std::mutex>{data_->mutex};
  }

  void acquire(std::unique_lock<std::mutex> lockGuard) {
    int prev = data_->count.exchange(-1, std::memory_order_relaxed);
    if (prev == 1) {
      data_->count.store(0, std::memory_order_relaxed);
      return;
    }

    do {
      data_->condvar.wait(lockGuard);
    } while (data_->count.load(std::memory_order_relaxed) != 1);

    data_->count.store(0, std::memory_order_relaxed);
  }
};

#endif