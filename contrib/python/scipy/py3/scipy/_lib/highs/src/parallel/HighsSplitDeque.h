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
#ifndef HIGHS_SPLIT_DEQUE_H_
#define HIGHS_SPLIT_DEQUE_H_

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

#include "parallel/HighsBinarySemaphore.h"
#include "parallel/HighsCacheAlign.h"
#include "parallel/HighsSpinMutex.h"
#include "parallel/HighsTask.h"
#include "util/HighsInt.h"
#include "util/HighsRandom.h"

class HighsSplitDeque {
  using cache_aligned = highs::cache_aligned;

 public:
  enum Constants {
    kTaskArraySize = 8192,
  };
  struct WorkerBunk;

 private:
  struct OwnerData {
    cache_aligned::shared_ptr<WorkerBunk> workerBunk = nullptr;
    cache_aligned::unique_ptr<HighsSplitDeque>* workers = nullptr;
    HighsRandom randgen;
    uint32_t head = 0;
    uint32_t splitCopy = 0;
    int numWorkers = 0;
    int ownerId = -1;
    HighsTask* rootTask = nullptr;
    bool allStolenCopy = true;
  };

  struct StealerData {
    HighsBinarySemaphore semaphore{0};
    HighsTask* injectedTask{nullptr};
    std::atomic<uint64_t> ts{0};
    std::atomic<bool> allStolen{true};
  };

  struct TaskMetadata {
    std::atomic<HighsSplitDeque*> stealer;
  };

  static constexpr uint64_t makeTailSplit(uint32_t tail, uint32_t split) {
    return (uint64_t(tail) << 32) | split;
  }

  static constexpr uint32_t tail(uint64_t tailSplit) { return tailSplit >> 32; }

  static constexpr uint32_t split(uint64_t tailSplit) {
    return static_cast<uint32_t>(tailSplit);
  }
  struct WorkerBunkData {
    std::atomic<HighsSplitDeque*> nextSleeper{nullptr};
    int ownerId;
  };

 public:
  struct WorkerBunk {
    static constexpr uint64_t kAbaTagShift = 20;
    static constexpr uint64_t kIndexMask = (uint64_t{1} << kAbaTagShift) - 1;
    alignas(64) std::atomic<int> haveJobs;
    alignas(64) std::atomic<uint64_t> sleeperStack;

    WorkerBunk() : haveJobs{0}, sleeperStack(0) {}

    void pushSleeper(HighsSplitDeque* deque) {
      uint64_t stackState = sleeperStack.load(std::memory_order_relaxed);
      uint64_t newStackState;
      HighsSplitDeque* head;

      do {
        head =
            stackState & kIndexMask
                ? deque->ownerData.workers[(stackState & kIndexMask) - 1].get()
                : nullptr;
        deque->workerBunkData.nextSleeper.store(head,
                                                std::memory_order_relaxed);

        newStackState = (stackState >> kAbaTagShift) + 1;
        newStackState = (newStackState << kAbaTagShift) |
                        uint64_t(deque->workerBunkData.ownerId + 1);
      } while (!sleeperStack.compare_exchange_weak(stackState, newStackState,
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed));
    }

    HighsSplitDeque* popSleeper(HighsSplitDeque* localDeque) {
      uint64_t stackState = sleeperStack.load(std::memory_order_relaxed);
      HighsSplitDeque* head;
      HighsSplitDeque* newHead;
      uint64_t newStackState;

      do {
        if ((stackState & kIndexMask) == 0) return nullptr;
        head =
            localDeque->ownerData.workers[(stackState & kIndexMask) - 1].get();
        newHead =
            head->workerBunkData.nextSleeper.load(std::memory_order_relaxed);
        int newHeadId =
            newHead != nullptr ? newHead->workerBunkData.ownerId + 1 : 0;
        newStackState = (stackState >> kAbaTagShift) + 1;
        newStackState = (newStackState << kAbaTagShift) | uint64_t(newHeadId);
      } while (!sleeperStack.compare_exchange_weak(stackState, newStackState,
                                                   std::memory_order_acquire,
                                                   std::memory_order_relaxed));

      head->workerBunkData.nextSleeper.store(nullptr,
                                             std::memory_order_relaxed);

      return head;
    }

    void publishWork(HighsSplitDeque* localDeque) {
      HighsSplitDeque* sleeper = popSleeper(localDeque);
      while (sleeper) {
        uint32_t t = localDeque->selfStealAndGetTail();
        if (t == localDeque->ownerData.splitCopy) {
          if (localDeque->ownerData.head == localDeque->ownerData.splitCopy) {
            localDeque->ownerData.allStolenCopy = true;
            localDeque->stealerData.allStolen.store(true,
                                                    std::memory_order_relaxed);
            haveJobs.fetch_add(-1, std::memory_order_release);
          }
          pushSleeper(sleeper);
          return;
        } else {
          sleeper->injectTaskAndNotify(&localDeque->taskArray[t]);
        }

        if (t == localDeque->ownerData.splitCopy - 1) {
          if (localDeque->ownerData.head == localDeque->ownerData.splitCopy) {
            localDeque->ownerData.allStolenCopy = true;
            localDeque->stealerData.allStolen.store(true,
                                                    std::memory_order_relaxed);
            haveJobs.fetch_add(-1, std::memory_order_release);
          }
          return;
        }

        sleeper = popSleeper(localDeque);
      }
    }

    HighsTask* waitForNewTask(HighsSplitDeque* localDeque) {
      pushSleeper(localDeque);
      localDeque->stealerData.semaphore.acquire();
      return localDeque->stealerData.injectedTask;
    }
  };

 private:
  static_assert(sizeof(OwnerData) <= 64,
                "sizeof(OwnerData) exceeds cache line size");
  static_assert(sizeof(StealerData) <= 64,
                "sizeof(StealerData) exceeds cache line size");
  static_assert(sizeof(WorkerBunkData) <= 64,
                "sizeof(GlobalQueueData) exceeds cache line size");

  alignas(64) OwnerData ownerData;
  alignas(64) std::atomic<bool> splitRequest;
  alignas(64) StealerData stealerData;
  alignas(64) WorkerBunkData workerBunkData;
  alignas(64) std::array<HighsTask, kTaskArraySize> taskArray;

  void growShared() {
    int haveJobs =
        ownerData.workerBunk->haveJobs.load(std::memory_order_relaxed);
    bool splitRq = false;
    uint32_t newSplit;
    if (haveJobs == ownerData.numWorkers) {
      splitRq = splitRequest.load(std::memory_order_relaxed);
      if (!splitRq) return;
    }

    newSplit = std::min(uint32_t{kTaskArraySize}, ownerData.head);

    assert(newSplit > ownerData.splitCopy);

    // we want to replace the old split point with the new splitPoint
    // but not alter the upper 32 bits of tail.
    // Hence with xor we can xor or the copy of the current split point
    // to set the lower bits to zero and then xor the bits of the new split
    // point to the lower bits that are then zero. First doing the xor of the
    // old and new split point and then doing the xor with the stealerData
    // will not alter the result. Also the upper 32 bits of the xor mask are
    // zero and will therefore not alter the value of tail.
    uint64_t xorMask = ownerData.splitCopy ^ newSplit;
    // since we publish the task data here, we need to use
    // std::memory_order_release which ensures all writes to set up the task
    // are done
    assert((xorMask >> 32) == 0);

    stealerData.ts.fetch_xor(xorMask, std::memory_order_release);
    ownerData.splitCopy = newSplit;
    if (splitRq)
      splitRequest.store(false, std::memory_order_relaxed);
    else
      ownerData.workerBunk->publishWork(this);
  }

  bool shrinkShared() {
    uint32_t t = tail(stealerData.ts.load(std::memory_order_relaxed));
    uint32_t s = ownerData.splitCopy;

    if (t != s) {
      ownerData.splitCopy = (t + s) / 2;
      t = tail(stealerData.ts.fetch_add(uint64_t{ownerData.splitCopy} - s,
                                        std::memory_order_acq_rel));
      if (t != s) {
        if (t > ownerData.splitCopy) {
          ownerData.splitCopy = (t + s) / 2;
          stealerData.ts.store(makeTailSplit(t, ownerData.splitCopy),
                               std::memory_order_relaxed);
        }

        return false;
      }
    }

    stealerData.allStolen.store(true, std::memory_order_relaxed);
    ownerData.allStolenCopy = true;
    ownerData.workerBunk->haveJobs.fetch_add(-1, std::memory_order_relaxed);
    return true;
  }

 public:
  HighsSplitDeque(const cache_aligned::shared_ptr<WorkerBunk>& workerBunk,
                  cache_aligned::unique_ptr<HighsSplitDeque>* workers,
                  int ownerId, int numWorkers) {
    ownerData.ownerId = ownerId;
    ownerData.workers = workers;
    ownerData.numWorkers = numWorkers;
    workerBunkData.ownerId = ownerId;
    ownerData.randgen.initialise(ownerId);
    ownerData.workerBunk = workerBunk;
    splitRequest.store(false, std::memory_order_relaxed);

    assert((reinterpret_cast<uintptr_t>(this) & 63u) == 0);
    static_assert(offsetof(HighsSplitDeque, splitRequest) == 64,
                  "alignas failed to guarantee 64 byte alignment");
    static_assert(offsetof(HighsSplitDeque, stealerData) == 128,
                  "alignas failed to guarantee 64 byte alignment");
    static_assert(offsetof(HighsSplitDeque, workerBunkData) == 192,
                  "alignas failed to guarantee 64 byte alignment");
    static_assert(offsetof(HighsSplitDeque, taskArray) == 256,
                  "alignas failed to guarantee 64 byte alignment");
  }

  void checkInterrupt() {
    if (ownerData.rootTask && ownerData.rootTask->isCancelled())
      throw HighsTask::Interrupt();
  }

  void cancelTask(HighsInt taskIndex) {
    assert(taskIndex < ownerData.head);
    assert(taskIndex >= 0);
    taskArray[taskIndex].cancel();
  }

  template <typename F>
  void push(F&& f) {
    if (ownerData.head >= kTaskArraySize) {
      // task queue is full, execute task directly
      if (ownerData.splitCopy < kTaskArraySize && !ownerData.allStolenCopy)
        growShared();

      ownerData.head += 1;
      f();
      return;
    }

    taskArray[ownerData.head++].setTaskData(std::forward<F>(f));
    if (ownerData.allStolenCopy) {
      assert(ownerData.head > 0);
      stealerData.ts.store(makeTailSplit(ownerData.head - 1, ownerData.head),
                           std::memory_order_release);
      stealerData.allStolen.store(false, std::memory_order_relaxed);
      ownerData.splitCopy = ownerData.head;
      ownerData.allStolenCopy = false;
      if (splitRequest.load(std::memory_order_relaxed))
        splitRequest.store(false, std::memory_order_relaxed);

      int haveJobs = ownerData.workerBunk->haveJobs.fetch_add(
          1, std::memory_order_release);
      if (haveJobs < ownerData.numWorkers - 1)
        ownerData.workerBunk->publishWork(this);
    } else
      growShared();
  }

  enum class Status {
    kEmpty,
    kStolen,
    kWork,
    kOverflown,
  };

  std::pair<Status, HighsTask*> pop() {
    if (ownerData.head == 0) return std::make_pair(Status::kEmpty, nullptr);

    if (ownerData.head > kTaskArraySize) {
      // task queue was full and the overflown tasks have
      // been directly executed
      ownerData.head -= 1;
      return std::make_pair(Status::kOverflown, nullptr);
    }

    if (ownerData.allStolenCopy)
      return std::make_pair(Status::kStolen, &taskArray[ownerData.head - 1]);

    if (ownerData.splitCopy == ownerData.head) {
      if (shrinkShared())
        return std::make_pair(Status::kStolen, &taskArray[ownerData.head - 1]);
    }

    ownerData.head -= 1;

    if (ownerData.head == 0) {
      if (!ownerData.allStolenCopy) {
        ownerData.allStolenCopy = true;
        stealerData.allStolen.store(true, std::memory_order_relaxed);
        ownerData.workerBunk->haveJobs.fetch_add(-1, std::memory_order_release);
      }
    } else if (ownerData.head != ownerData.splitCopy)
      growShared();

    return std::make_pair(Status::kWork, &taskArray[ownerData.head]);
  }

  void popStolen() {
    ownerData.head -= 1;
    if (!ownerData.allStolenCopy) {
      ownerData.allStolenCopy = true;
      stealerData.allStolen.store(true, std::memory_order_relaxed);
      ownerData.workerBunk->haveJobs.fetch_add(-1, std::memory_order_release);
    }
  }

  HighsTask* steal() {
    if (stealerData.allStolen.load(std::memory_order_relaxed)) return nullptr;

    uint64_t ts = stealerData.ts.load(std::memory_order_relaxed);
    uint32_t t = tail(ts);
    uint32_t s = split(ts);
    if (t < s) {
      if (stealerData.ts.compare_exchange_weak(ts, makeTailSplit(t + 1, s),
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed))
        return &taskArray[t];

      t = tail(ts);
      s = split(ts);
      if (t < s) {
        return nullptr;
      }
    }

    if (t < kTaskArraySize && !splitRequest.load(std::memory_order_relaxed))
      splitRequest.store(true, std::memory_order_relaxed);

    return nullptr;
  }

  HighsTask* stealWithRetryLoop() {
    if (stealerData.allStolen.load(std::memory_order_relaxed)) return nullptr;

    uint64_t ts = stealerData.ts.load(std::memory_order_relaxed);
    uint32_t t = tail(ts);
    uint32_t s = split(ts);

    while (t < s) {
      if (stealerData.ts.compare_exchange_weak(ts, makeTailSplit(t + 1, s),
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed))
        return &taskArray[t];

      t = tail(ts);
      s = split(ts);
    }

    if (t < kTaskArraySize && !splitRequest.load(std::memory_order_relaxed))
      splitRequest.store(true, std::memory_order_relaxed);

    return nullptr;
  }

  uint32_t selfStealAndGetTail() {
    if (ownerData.allStolenCopy) return ownerData.splitCopy;

    // when we steal from ourself we can simply do a fetch_add predictively
    // instead of a cas loop. If the tail we read like this ends up to be
    // above already equal to the splitPoint then we correct it with a simple
    // store. When tail > split instead of tail == split no wrong result can
    // occur as long as we know that the task at taskArray[split] is not
    // actually considered to be stolen and tail is corrected before the owner
    // enters shrinkShared.

    uint32_t t = tail(stealerData.ts.fetch_add(makeTailSplit(1, 0),
                                               std::memory_order_relaxed));

    if (t == ownerData.splitCopy)
      stealerData.ts.store(makeTailSplit(t, ownerData.splitCopy),
                           std::memory_order_relaxed);

    return t;
  }

  HighsTask* randomSteal() {
    HighsInt next = ownerData.randgen.integer(ownerData.numWorkers - 1);
    next += next >= ownerData.ownerId;
    assert(next != ownerData.ownerId);
    assert(next >= 0);
    assert(next < ownerData.numWorkers);

    return ownerData.workers[next]->steal();
  }

  void injectTaskAndNotify(HighsTask* t) {
    stealerData.injectedTask = t;
    stealerData.semaphore.release();
  }

  void notify() { stealerData.semaphore.release(); }

  /// wait until notified
  void wait() { stealerData.semaphore.acquire(); }

  void runStolenTask(HighsTask* task) {
    HighsTask* prevRootTask = ownerData.rootTask;
    ownerData.rootTask = task;
    uint32_t currentHead = ownerData.head;
    try {
      HighsSplitDeque* owner = task->run(this);
      if (owner) owner->notify();
    } catch (const HighsTask::Interrupt&) {
      // in case the task was interrupted we unwind and cancel all subtasks of
      // the stolen task

      // first cancel all tasks
      for (uint32_t i = currentHead; i < ownerData.head; ++i)
        taskArray[i].cancel();

      // now remove them from our deque so that we arrive at the original state
      // before the stolen task was executed
      while (ownerData.head != currentHead) {
        std::pair<Status, HighsTask*> popResult = pop();
        assert(popResult.first != Status::kEmpty);
        // if the task was not stolen it would be up to this worker to execute
        // it now and we simply skip its execution as it is cancelled
        if (popResult.first != Status::kStolen) continue;

        // The task was stolen. Check if the stealer is already finished with
        // its execution in which case we just remove it from the deque.
        HighsSplitDeque* stealer = popResult.second->getStealerIfUnfinished();
        if (stealer == nullptr) {
          popStolen();
          continue;
        }

        // The task was stolen and the stealer is still executing the task.
        // We now wait in a spin loop until the task is marked as finished for
        // some time. We do not proceed with stealing other tasks as when there
        // is a cancelled task the likelihood of that task being cancelled too
        // might be high and our priority is to finish unwinding the chain of
        // cancelled tasks as fast as possible.
        // When the spinning proceeds for too long we request a notification
        // from the stealer when it is finished and yield control to the
        // operating system until then.
        int numTries = HighsSchedulerConstants::kNumTryFac;
        auto tStart = std::chrono::high_resolution_clock::now();

        bool isFinished = popResult.second->isFinished();

        while (!isFinished) {
          for (int i = 0; i < numTries; ++i) {
            HighsSpinMutex::yieldProcessor();
            isFinished = popResult.second->isFinished();
            if (isFinished) break;
          }

          if (!isFinished) {
            auto numMicroSecs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - tStart)
                    .count();

            if (numMicroSecs < HighsSchedulerConstants::kMicroSecsBeforeSleep)
              numTries *= 2;
            else {
              waitForTaskToFinish(popResult.second, stealer);
              break;
            }
          }
        }

        // the task is finished and we can safely proceed to unwind to the next
        // task
        popStolen();
      }

      // unwinding is finished for all subtasks and we can mark the task as
      // finished and then notify the owner if it is waiting for a signal
      HighsSplitDeque* owner = task->markAsFinished(this);
      if (owner) owner->notify();
    }

    ownerData.rootTask = prevRootTask;
    checkInterrupt();
  }

  // steal from the stealer until this task is finished or no more work can be
  // stolen from the stealer. If the task is finished then true is returned,
  // otherwise false is returned
  bool leapfrogStolenTask(HighsTask* task, HighsSplitDeque*& stealer) {
    bool cancelled;
    stealer = task->getStealerIfUnfinished(&cancelled);

    if (stealer == nullptr) return true;

    if (!cancelled) {
      do {
        HighsTask* t = stealer->stealWithRetryLoop();
        if (t == nullptr) break;
        runStolenTask(t);
      } while (!task->isFinished());
    }

    return task->isFinished();
  }

  void waitForTaskToFinish(HighsTask* t, HighsSplitDeque* stealer) {
    std::unique_lock<std::mutex> lg =
        stealerData.semaphore.lockMutexForAcquire();
    // exchange the value stored in stealer with the pointer to owner
    // so that the stealer will see this pointer instead of nullptr
    // when it stores whether the task is finished. In that case the
    // stealer will additionally acquire the wait mutex and signal the owner
    // thread that the task is finished

    if (!t->requestNotifyWhenFinished(this, stealer)) return;

    stealerData.semaphore.acquire(std::move(lg));
  }

  void yield() {
    HighsTask* t = randomSteal();
    if (t) runStolenTask(t);
  }

  int getOwnerId() const { return ownerData.ownerId; }

  int getNumWorkers() const { return ownerData.numWorkers; }

  int getCurrentHead() const { return ownerData.head; }

  HighsSplitDeque* getWorkerById(int id) const {
    return ownerData.workers[id].get();
  }
};

#endif
