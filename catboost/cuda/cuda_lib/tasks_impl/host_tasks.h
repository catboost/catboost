#pragma once

#include "remote_device_future.h"
#include <catboost/cuda/cuda_lib/task.h>

namespace NCudaLib {
    template <class TTask, bool IsSystem = false>
    class THostTask: public IHostTask {
    public:
        using TOutput = typename TTask::TOutput;

        TDeviceFuture<TOutput> GetResult() {
            Promise = NThreading::NewPromise<TOutput>();
            return TDeviceFuture<TOutput>(Promise.GetFuture());
        }

        virtual void Exec() override {
            Promise.SetValue(std::move<TOutput>(Task()));
        }

        virtual bool IsSystemTask() override {
            return IsSystem;
        }

        THostTask(TTask&& task)
            : Task(std::move(task))
        {
        }

        template <class... TArgs>
        THostTask(TArgs... args)
            : Task(std::forward<TArgs>(args)...)
        {
        }

    private:
        TTask Task;
        NThreading::TPromise<TOutput> Promise;
    };

    struct TBlockingSyncDevice {
        using TOutput = ui64;

        ui64 operator()() {
            return 0;
        }
    };

    using TBlockingDeviceSynchronize = THostTask<TBlockingSyncDevice>;

    class TWaitSubmitCommand: public IGpuCommand {
    public:
        TWaitSubmitCommand()
            : IGpuCommand(EGpuHostCommandType::WaitSubmit)
        {
        }
    };

    struct TRequestHandlesTask {
        using TOutput = TVector<ui64>;
        ui32 Count;

        explicit TRequestHandlesTask(ui32 count)
            : Count(count)
        {
        }

        TOutput operator()() {
            return GetHandleStorage().GetHandle(Count);
        }
    };

    using TRequestHandles = THostTask<TRequestHandlesTask, true>;

}
