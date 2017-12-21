#pragma once

#include "cuda_base.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "cuda_events_provider.h"
#include "kernel.h"

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <library/threading/future/future.h>
#include <util/ysafeptr.h>
#include <util/system/event.h>

namespace NCudaLib {
    enum class EGpuHostCommandType {
        StreamKernel,       //async tasks, will be launch in stream
        HostTask,           //sync task, ensure every task in stream was completed
        MemoryAllocation,   // usually async, but could sync or memory defragmentation
        MemoryDeallocation, // sync everything and free memory
        MemoryState,
        RequestStream,
        FreeStream,
        WaitSubmit,
        StopWorker,
    };

    class IGpuCommand {
    private:
        EGpuHostCommandType Type;

    public:
        explicit IGpuCommand(EGpuHostCommandType type)
            : Type(type)
        {
        }

        virtual ~IGpuCommand() {
        }

        EGpuHostCommandType GetCommandType() const {
            return Type;
        }
    };

    class IAllocateMemoryTask: public IGpuCommand {
    public:
        IAllocateMemoryTask()
            : IGpuCommand(EGpuHostCommandType::MemoryAllocation)
        {
        }

        virtual ui64 GetHandle() const = 0;

        virtual ui64 GetSize() const = 0;

        virtual EPtrType GetPtrType() const = 0;
    };

    class IFreeMemoryTask: public IGpuCommand {
    public:
        IFreeMemoryTask()
            : IGpuCommand(EGpuHostCommandType::MemoryDeallocation)
        {
        }

        virtual void Exec() = 0;
    };

    class IHostTask: public IGpuCommand {
    public:
        IHostTask()
            : IGpuCommand(EGpuHostCommandType::HostTask)
        {
        }

        //system tasks skip stream semantic
        virtual bool IsSystemTask() {
            return false;
        }

        virtual void Exec() = 0;
    };

    class TStopCommand: public IGpuCommand {
    public:
        explicit TStopCommand()
            : IGpuCommand(EGpuHostCommandType::StopWorker)
        {
        }
    };

}
