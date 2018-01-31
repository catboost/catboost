#pragma once

#include <catboost/cuda/cuda_lib/task.h>

namespace NCudaLib {
    template <EPtrType PtrType>
    class TCudaMallocTask: public IAllocateMemoryTask {
    private:
        ui64 Handle;
        ui64 Size;

    public:
        TCudaMallocTask(ui64 handle,
                        ui64 size)
            : Handle(handle)
            , Size(size)
        {
        }

        TCudaMallocTask() {
        }

        ui64 GetHandle() const final {
            return Handle;
        }

        ui64 GetSize() const final {
            return Size;
        }

        EPtrType GetPtrType() const final {
            return PtrType;
        }

        Y_SAVELOAD_TASK(Handle, Size);
    };

    template <class T,
              bool FreeHandle = false>
    class TResetPointerCommand: public IFreeMemoryTask {
    private:
        ui64 Handle;

    public:
        explicit TResetPointerCommand(ui64 handle)
            : Handle(handle)
        {
        }

        TResetPointerCommand() {
        }

        void Exec() final {
            THandleBasedPointer<T>(Handle).Reset();
            if (FreeHandle) {
                GetHandleStorage().FreeHandle(Handle);
            }
        }

        Y_SAVELOAD_TASK(Handle);
    };

    template <class T>
    struct TCreateObjectCommandTrait {
        class TWithoutConstructCommand: public IHostTask {
        private:
            ui64 Handle;

        public:
            explicit TWithoutConstructCommand(ui64 handle)
                : Handle(handle)
            {
            }

            void Exec(const IWorkerStateProvider&) final {
                THandleBasedPointer<T> ptr(Handle);
                ptr.Reset(new T);
            }

            virtual ECpuFuncType GetHostTaskType() {
                return ECpuFuncType::DeviceNonblocking;
            }

            Y_SAVELOAD_TASK(Handle);
        };

        static THolder<TWithoutConstructCommand> Create(ui64 handle) {
            return MakeHolder<TWithoutConstructCommand>(handle);
        }
    };

}
