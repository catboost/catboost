#pragma once

#include "cuda_base.h"
#include "remote_objects.h"
#include "kernel.h"
#include "worker_state.h"

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <util/generic/buffer.h>

namespace NCudaLib {
    enum class EComandType {
        StreamKernel,       //async tasks, will be launch in stream
        HostTask,           //sync task, ensure every task in stream was completed
        MemoryAllocation,   // usually async, but could sync or memory defragmentation
        MemoryDeallocation, // sync everything and free memory
        RequestStream,
        FreeStream,
        WaitSubmit,
        Reset,
        StopWorker,
        SerializedCommand
    };

    class ICommand {
    private:
        EComandType Type;

    public:
        explicit ICommand(EComandType type)
            : Type(type)
        {
        }

        virtual ~ICommand() {
        }

        EComandType GetCommandType() const {
            return Type;
        }

        virtual void Load(IInputStream*) = 0;
        virtual void Save(IOutputStream*) const = 0;
    };

#define Y_STATELESS_TASK()                  \
    void Save(IOutputStream*) const final { \
    }                                       \
                                            \
    void Load(IInputStream*) final {        \
    }

#define Y_SAVELOAD_EMPTY()                   \
    inline void Save(IOutputStream*) const { \
    }                                        \
                                             \
    inline void Load(IInputStream*) {        \
    }

//for local shared memory tasks
#define Y_NON_SERIALIZABLE_TASK()                          \
    void Save(IOutputStream*) const final {                \
        CB_ENSURE(false, "Error: can't save this command"); \
    }                                                      \
                                                           \
    void Load(IInputStream*) override {                    \
        CB_ENSURE(false, "Error: can't load this command"); \
    }

#define Y_SAVELOAD_TASK(...)                  \
    void Save(IOutputStream* s) const final { \
        ::SaveMany(s, __VA_ARGS__);           \
    }                                         \
                                              \
    void Load(IInputStream* s) final {        \
        ::LoadMany(s, __VA_ARGS__);           \
    }

#define Y_SAVELOAD_IMPL(...)                \
    void SaveImpl(IOutputStream* s) const { \
        ::SaveMany(s, __VA_ARGS__);         \
    }                                       \
                                            \
    void LoadImpl(IInputStream* s) {        \
        ::LoadMany(s, __VA_ARGS__);         \
    }

    struct TResetCommand: public ICommand {
    public:
        TResetCommand(double gpuMemoryPart,
                      ui64 pinnedMemorySize)
            : ICommand(EComandType::Reset)
            , GpuMemoryPart(gpuMemoryPart)
            , PinnedMemorySize(pinnedMemorySize)
        {
        }

        TResetCommand()
            : ICommand(EComandType::Reset)
        {
        }

        double GpuMemoryPart = 0;
        ui64 PinnedMemorySize = 0;

        Y_SAVELOAD_TASK(GpuMemoryPart, PinnedMemorySize)
    };

    class IAllocateMemoryTask: public ICommand {
    public:
        IAllocateMemoryTask()
            : ICommand(EComandType::MemoryAllocation)
        {
        }

        virtual ui64 GetHandle() const = 0;

        virtual ui64 GetSize() const = 0;

        virtual EPtrType GetPtrType() const = 0;
    };

    class IFreeMemoryTask: public ICommand {
    public:
        IFreeMemoryTask()
            : ICommand(EComandType::MemoryDeallocation)
        {
        }

        virtual void Exec() = 0;
    };

    enum class ECpuFuncType {
        DeviceBlocking,    //wait all previous tasks done
        DeviceNonblocking, //system async
    };

    inline bool constexpr IsBlockingHostTask(ECpuFuncType type) {
        return type == ECpuFuncType::DeviceBlocking;
    }

    class IHostTask: public ICommand {
    public:
        IHostTask()
            : ICommand(EComandType::HostTask)
        {
        }

        //system tasks skip stream semantic
        virtual ECpuFuncType GetHostTaskType() = 0;

        virtual void Exec(const IWorkerStateProvider& workerState) = 0;
    };

    class TStopWorkerCommand: public ICommand {
    public:
        explicit TStopWorkerCommand()
            : ICommand(EComandType::StopWorker)
        {
        }

        Y_STATELESS_TASK();
    };

    class TSerializedCommand: public ICommand {
    public:
        TSerializedCommand(TBuffer&& data);

        THolder<ICommand> Deserialize();

        Y_NON_SERIALIZABLE_TASK();

    private:
        TBuffer Data;
    };

}
