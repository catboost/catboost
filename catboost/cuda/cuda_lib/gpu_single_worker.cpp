#include "gpu_single_worker.h"

namespace NCudaLib {
    bool TGpuOneDeviceWorker::RunIteration() {
        bool shouldStop = false;
        try {
            const bool hasRunning = CheckRunningTasks();
            const bool isEmpty = InputTaskQueue.IsEmpty();
            if (!hasRunning && isEmpty) {
                InputTaskQueue.Wait(TInstant::Seconds(1));
            } else if (!isEmpty) {
                THolder<IGpuCommand> task = InputTaskQueue.Dequeue();

                switch (task->GetCommandType()) {
                    //could be run async
                    case EGpuHostCommandType::StreamKernel: {
                        IGpuKernelTask* kernelTask = dynamic_cast<IGpuKernelTask*>(task.Get());
                        const ui32 streamId = kernelTask->GetStreamId();
                        if (streamId == 0) {
                            WaitAllTaskToSubmit();
                            SyncActiveStreams(true);
                        }
                        auto& stream = *Streams[streamId];
                        auto data = kernelTask->PrepareExec(TempMemoryManager);
                        task.Release();
                        stream.AddTask(THolder<IGpuKernelTask>(kernelTask), std::move(data));
                        break;
                    }
                        //synchronized on memory defragmentation
                    case EGpuHostCommandType::MemoryAllocation: {
                        IAllocateMemoryTask* memoryTask = dynamic_cast<IAllocateMemoryTask*>(task.Get());
                        AllocateMemory(*memoryTask);
                        break;
                    }
                    case EGpuHostCommandType::MemoryDeallocation: {
                        WaitSubmitAndSync();
                        IFreeMemoryTask* freeMemoryTask = dynamic_cast<IFreeMemoryTask*>(task.Get());
                        task.Release();
                        ObjectsToFree.push_back(THolder<IFreeMemoryTask>(freeMemoryTask));
                        DeleteObjects();
                        break;
                    }
                    case EGpuHostCommandType::MemoryState: {
                        WaitAllTaskToSubmit();
                        TMemoryStateTask* memoryStateTask = dynamic_cast<TMemoryStateTask*>(task.Get());
                        task.Release();
                        memoryStateTask->Set(GetMemoryState());
                        break;
                    }
                    case EGpuHostCommandType::WaitSubmit: {
                        WaitAllTaskToSubmit();
                        break;
                    }
                        //synchronized always
                    case EGpuHostCommandType::HostTask: {
                        auto taskPtr = dynamic_cast<IHostTask*>(task.Get());
                        if (!taskPtr->IsSystemTask()) {
                            WaitSubmitAndSync();
                            DeleteObjects();
                        }
                        taskPtr->Exec();
                        break;
                    }
                    case EGpuHostCommandType::RequestStream: {
                        const ui32 streamId = RequestStreamImpl();
                        dynamic_cast<TRequestStreamCommand*>(task.Get())->SetStreamId(streamId);
                        break;
                    }
                    case EGpuHostCommandType::FreeStream: {
                        auto stream = dynamic_cast<TFreeStreamCommand*>(task.Get())->GetStream();
                        WaitAllTaskToSubmit();
                        SyncStream(stream);
                        FreeStreams.push_back(stream);
                        break;
                    }
                    case EGpuHostCommandType::StopWorker: {
                        WaitSubmitAndSync();
                        shouldStop = true;
                        break;
                    }
                    default: {
                        ythrow yexception() << "Unknown command type";
                    }
                }
            }
            CheckLastError();
        } catch (...) {
            RunErrorCallbacks(CurrentExceptionMessage());
        }
        return shouldStop;
    }

    void TGpuOneDeviceWorker::Run(ui64 gpuMemoryLimit, ui64 pinnedMemoryLimit) {
        Stopped = false;
        SetDevice(LocalDeviceId);
        DeviceMemoryProvider = MakeHolder<TDeviceMemoryProvider>(gpuMemoryLimit);
        HostMemoryProvider = MakeHolder<THostMemoryProvider>(pinnedMemoryLimit);
        CreateNewComputationStream();
        SetDefaultStream(Streams[0]->GetStream());

        ui64 Iteration = 0;
        while (true) {
            ++Iteration;
            const bool shouldStop = RunIteration();
            if (shouldStop) {
                break;
            }
        }
        CB_ENSURE(InputTaskQueue.IsEmpty(), "Error: found tasks after stop command");
        Stopped = true;
    }

}
