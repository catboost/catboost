#pragma once

#include <catboost/cuda/cuda_lib/task.h>
#include <library/threading/chunk_queue/queue.h>
#include <util/datetime/base.h>
#include <util/system/event.h>
#include <util/system/yield.h>

namespace NCudaLib {
    class TSingleHostTaskQueue {
    public:
        using TGpuTaskPtr = THolder<ICommand>;
        using TQueue = typename NThreading::TManyOneQueue<TGpuTaskPtr>;

        bool IsEmpty() {
            return InputTaskQueue.IsEmpty();
        }

        TGpuTaskPtr Dequeue() {
            THolder<ICommand> task;
            bool done = InputTaskQueue.Dequeue(task);
            CB_ENSURE(done, "Error: dequeue failed");
            return task;
        }

        void Wait(TInstant time) {
            JobsEvent.Reset();

            const ui32 waitIters = 100000;
            for (ui32 iter = 0; iter < waitIters; ++iter) {
                SchedYield();
                if (!InputTaskQueue.IsEmpty()) {
                    return;
                }
            }
            if (InputTaskQueue.IsEmpty()) {
                JobsEvent.WaitD(time);
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
