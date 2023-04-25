#include "gpu_single_worker.h"

#include <catboost/cuda/cuda_lib/tasks_impl/memory_allocation.h>

#if defined(WITH_HWLOC)
#include "hwloc_wrapper.h"
#endif

namespace NCudaLib {
    void TGpuOneDeviceWorker::AllocateTempMemory(ui64 handle, EPtrType ptrType, ui64 size) {
        switch (ptrType) {
            case EPtrType::CudaHost: {
                using TRawPtr = typename TMemoryProviderImplTrait<EPtrType::CudaHost>::TRawFreeMemory;
                using TCmd = TResetPointerCommand<TRawPtr, true>;
                Y_ASSERT(HostMemoryProvider);
                TCudaMallocTask<EPtrType::CudaHost> task(handle, size);
                auto cmd = MakeHolder<TCmd>(handle);
                TempMemoryAllocatedObjects.push_back(std::move(cmd));
                AllocateMemory(task);
                return;
            }
            case EPtrType::CudaDevice: {
                using TRawPtr = typename TMemoryProviderImplTrait<EPtrType::CudaDevice>::TRawFreeMemory;
                using TCmd = TResetPointerCommand<TRawPtr, true>;

                Y_ASSERT(DeviceMemoryProvider);
                TCudaMallocTask<EPtrType::CudaDevice> task(handle, size);
                auto cmd = MakeHolder<TCmd>(handle);
                TempMemoryAllocatedObjects.push_back(std::move(cmd));
                AllocateMemory(task);
                return;
            }
            case EPtrType::Host: {
                using TRawPtr = typename TMemoryProviderImplTrait<EPtrType::Host>::TRawFreeMemory;
                using TCmd = TResetPointerCommand<TRawPtr, true>;

                TCudaMallocTask<EPtrType::Host> task(handle, size);
                auto cmd = MakeHolder<TCmd>(handle);
                TempMemoryAllocatedObjects.push_back(std::move(cmd));
                AllocateMemory(task);
                return;
            }
            default: {
                ythrow TCatBoostException() << "Unsupported operation type";
            }
        }
    }

    bool TGpuOneDeviceWorker::RunIteration() {
        bool shouldStop = false;
        try {
            const bool hasRunning = CheckRunningTasks();
            const bool isEmpty = InputTaskQueue.IsEmpty();

            while (TempMemoryAllocatedObjects.size()) {
                ObjectsToFree.push_back(std::move(TempMemoryAllocatedObjects.back()));
                TempMemoryAllocatedObjects.pop_back();
            }

            if (!hasRunning && isEmpty) {
                InputTaskQueue.Wait(TDuration::Max());
            } else if (!isEmpty) {
                THolder<ICommand> task = InputTaskQueue.Dequeue();

                if (task->GetCommandType() == EComandType::SerializedCommand) {
                    task = reinterpret_cast<TSerializedCommand*>(task.Get())->Deserialize();
                }

                switch (task->GetCommandType()) {
                    case EComandType::Reset: {
                        TResetCommand* init = dynamic_cast<TResetCommand*>(task.Get());
                        WaitSubmitAndSync();
                        Reset(*init);
                        break;
                    }
                    //could be run async
                    case EComandType::StreamKernel: {
                        THolder<IGpuKernelTask> kernelTask(reinterpret_cast<IGpuKernelTask*>(task.Release()));
                        const ui32 streamId = kernelTask->GetStreamId();
                        if (streamId == 0) {
                            WaitAllTaskToSubmit();
                            SyncActiveStreams(true);
                            if (ObjectsToFree.size()) {
                                DeleteObjects();
                                SyncStream(0);
                            }
                        }
                        auto& stream = *Streams[streamId];
                        THolder<NKernel::IKernelContext> data;
                        data = kernelTask->PrepareExec(TempMemoryManager);

                        stream.AddTask(std::move(kernelTask), std::move(data));
                        break;
                    }
                    //synchronized on memory defragmentation
                    case EComandType::MemoryAllocation: {
                        IAllocateMemoryTask* memoryTask = reinterpret_cast<IAllocateMemoryTask*>(task.Get());
                        AllocateMemory(*memoryTask);
                        break;
                    }
                    case EComandType::MemoryDeallocation: {
                        THolder<IFreeMemoryTask> freeMemoryTask(reinterpret_cast<IFreeMemoryTask*>(task.Release()));
                        ObjectsToFree.push_back(std::move(freeMemoryTask));
                        WaitSubmitAndSync();
                        break;
                    }
                    case EComandType::WaitSubmit: {
                        WaitAllTaskToSubmit();
                        break;
                    }
                    case EComandType::HostTask: {
                        auto taskPtr = reinterpret_cast<IHostTask*>(task.Get());
                        auto type = taskPtr->GetHostTaskType();
                        if (IsBlockingHostTask(type)) {
                            WaitSubmitAndSync();
                        }
                        taskPtr->Exec(*this);
                        break;
                    }
                    case EComandType::RequestStream: {
                        const ui32 streamId = RequestStreamImpl();
                        reinterpret_cast<IRequestStreamCommand*>(task.Get())->SetStreamId(streamId);
                        break;
                    }
                    case EComandType::FreeStream: {
                        const auto& streams = reinterpret_cast<TFreeStreamCommand*>(task.Get())->GetStreams();
                        WaitAllTaskToSubmit();
                        for (const auto& stream : streams) {
                            SyncStream(stream);
                            FreeStreams.push_back(stream);
                        }
                        break;
                    }
                    case EComandType::StopWorker: {
                        WaitSubmitAndSync();
                        shouldStop = true;
                        break;
                    }
                    case EComandType::SerializedCommand: {
                        Y_UNREACHABLE();
                    }
                    default: {
                        ythrow TCatBoostException() << "Unknown command type";
                    }
                }
            }
            CheckLastError();
        } catch (...) {
            RunErrorCallbacks(CurrentExceptionMessage());
        }
        return shouldStop;
    }

    void TGpuOneDeviceWorker::Run() {
        AtomicSet(Stopped, 0);
        SetDevice(LocalDeviceId);
#if defined(WITH_HWLOC)
        auto& localityHelper = HardwareLocalityHelper();
        localityHelper.BindThreadForDevice(LocalDeviceId);
#endif

        CreateNewComputationStream();
        SetDefaultStream(Streams[0]->GetStream());

        while (true) {
            const bool shouldStop = RunIteration();
            if (shouldStop) {
                break;
            }
        }
        CB_ENSURE(InputTaskQueue.IsEmpty(), "Error: found tasks after stop command");

        CB_ENSURE((1 + FreeStreams.size()) == Streams.size());
        CB_ENSURE(ObjectsToFree.size() == 0);
        Streams.clear();
        FreeStreams.clear();
        ObjectsToFree.clear();

        AtomicSet(Stopped, 1);
    }

}
