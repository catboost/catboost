#pragma once

#include <catboost/cuda/cuda_lib/task.h>

namespace NCudaLib {
    class TWaitSubmitCommand: public ICommand {
    public:
        TWaitSubmitCommand()
            : ICommand(EComandType::WaitSubmit)
        {
        }

        Y_STATELESS_TASK();
    };

    struct TNonBlockingFunc {
        static constexpr ECpuFuncType FuncType() {
            return ECpuFuncType::DeviceNonblocking;
        }
    };

    struct TBlockingFunc {
        static constexpr ECpuFuncType FuncType() {
            return ECpuFuncType::DeviceBlocking;
        }
    };

    struct TBlockingSyncDevice: public TBlockingFunc {
        using TOutput = ui64;

        ui64 operator()() {
            return 0;
        }

        Y_SAVELOAD_EMPTY();
    };

    struct TRequestHandlesTask: public TNonBlockingFunc {
        using TOutput = TVector<ui64>;
        ui32 Count = 0;

        explicit TRequestHandlesTask(ui32 count)
            : Count(count)
        {
        }

        TRequestHandlesTask() = default;

        TOutput operator()() {
            return GetHandleStorage().GetHandle(Count);
        }

        Y_SAVELOAD_DEFINE(Count);
    };

    struct TFreeHandlesTask: public TBlockingFunc {
        using TOutput = int;
        TVector<ui64> Handles;

        explicit TFreeHandlesTask(TVector<ui64>&& handles)
            : Handles(std::move(handles))
        {
        }

        TFreeHandlesTask() = default;

        TOutput operator()() {
            auto& storage = GetHandleStorage();

            for (const auto& handle : Handles) {
                storage.FreeHandle(handle);
            }
            return 0;
        }

        Y_SAVELOAD_DEFINE(Handles);
    };

}
