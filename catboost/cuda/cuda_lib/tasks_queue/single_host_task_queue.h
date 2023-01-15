#pragma once

#include <catboost/cuda/cuda_lib/task.h>
#include <library/cpp/threading/chunk_queue/queue.h>
#include <util/datetime/base.h>
#include <util/system/event.h>
#include <util/system/yield.h>

namespace NCudaLib {
    class TSingleHostTaskQueue {
    public:
        using TGpuTaskPtr = THolder<ICommand>;
        using TQueue = typename NThreading::TOneOneQueue<TGpuTaskPtr>;

        bool IsEmpty() {
            return InputTaskQueue.IsEmpty();
        }

        TGpuTaskPtr Dequeue() {
            THolder<ICommand> task;
            bool done = InputTaskQueue.Dequeue(task);
            CB_ENSURE(done, "Error: dequeue failed");
            return task;
        }

        void Wait(TDuration time) {
            JobsEvent.Reset();

            auto fastWaitInterval = TDuration::Seconds(1);
            auto startTime = TInstant::Now();

            ui64 fastIters = 10000;
            ui64 i = 0;
            while ((TInstant::Now() - startTime) < fastWaitInterval) {
                ++i;
                if (!InputTaskQueue.IsEmpty()) {
                    return;
                }
                if (i % fastIters == 0) {
                    SchedYield();
                }
            }
            if (InputTaskQueue.IsEmpty()) {
                JobsEvent.WaitT(time);
            }
        }

        template <class TTask>
        void AddTask(THolder<TTask>&& task) {
            InputTaskQueue.Enqueue(std::move(task));
            JobsEvent.Signal();
        }

        template <class TTask,
                  class... Args>
        void EmplaceTask(Args&&... args) {
            AddTask(MakeHolder<TTask>(std::forward<Args>(args)...));
        }

    private:
        TManualEvent JobsEvent;
        TQueue InputTaskQueue;
    };
}
