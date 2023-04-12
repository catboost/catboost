#pragma once

#include "single_host_task_queue.h"
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/cuda/cuda_lib/device_id.h>
#include <catboost/cuda/cuda_lib/task.h>
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>
#include <util/generic/buffer.h>
#include <util/stream/buffer.h>
#include <util/stream/file.h>
#include <util/digest/city.h>

namespace NCudaLib {
#if defined(USE_MPI)

    class TMpiTaskSlaveForwarder {
    public:
        using TGpuTaskPtr = THolder<ICommand>;

        explicit TMpiTaskSlaveForwarder(TVector<TSingleHostTaskQueue*>&& taskQueues)
            : Manager(GetMpiManager())
            , TaskQueues(std::move(taskQueues))
        {
            const ui64 maxTaskSize = 1024 * 1024;
            CB_ENSURE(TaskQueues.size() == static_cast<size_t>(NCudaLib::NCudaHelpers::GetDeviceCount()));
            TaskBuffer.resize(TaskQueues.size());
            for (auto& buffer : TaskBuffer) {
                buffer.Resize(maxTaskSize);
            }

            DeviceIds.resize(TaskQueues.size());
            TaskCount.resize(TaskQueues.size());
            Requests.resize(TaskQueues.size());
            const int hostId = Manager.GetHostId();
            for (ui32 i = 0; i < DeviceIds.size(); ++i) {
                DeviceIds[i].HostId = hostId;
                DeviceIds[i].DeviceId = i;
            }
        }

        template <class TNeedStop>
        void Run(TNeedStop&& needStop) {
            for (ui32 i = 0; i < TaskQueues.size(); ++i) {
                ReceiveNextTaskAsync(i);
            }

            ui32 iter = 0;
            const ui32 checkStopIters = 10000;

            while (true) {
                for (ui32 i = 0; i < TaskQueues.size(); ++i) {
                    if (Requests[i]->IsComplete()) {
                        const TBuffer& taskData = TaskBuffer[i];
                        const ui32 dataSize = Requests[i]->ReceivedBytes();

                        Y_ASSERT(dataSize > 0);
                        TBuffer task;
                        task.Append(taskData.Begin(), dataSize);
                        TaskQueues[i]->EmplaceTask<TSerializedCommand>(std::move(task));
                        ReceiveNextTaskAsync(i);
                    }
                }

                const bool timeToCheckForStop = (((++iter) % checkStopIters) == 0);
                if (timeToCheckForStop && needStop()) {
                    break;
                }
            }

            for (TMpiRequestPtr& request : Requests) {
                if (request) {
                    CB_ENSURE(!request->IsComplete(), "Request completed unexpectedly");
                    request->Abort();
                }
            }
        }

    private:
        void ReceiveNextTaskAsync(ui32 i) {
            TSerializedTask& buffer = TaskBuffer[i];
            Y_ASSERT(Requests[i] == nullptr || (Requests[i]) && Requests[i]->IsComplete());
            Requests[i] = Manager.ReadAsync(buffer.Data(),
                                            static_cast<int>(buffer.Size()),
                                            Manager.GetMasterId(),
                                            Manager.GetTaskTag(DeviceIds[i]));
        }

    private:
        TMpiManager& Manager;
        TVector<TSerializedTask> TaskBuffer;
        TVector<TSingleHostTaskQueue*> TaskQueues;
        TVector<TMpiRequestPtr> Requests;
        TVector<TDeviceId> DeviceIds;
        TVector<ui64> TaskCount;
    };

    class TRemoteHostTasksForwarder {
    public:
        TRemoteHostTasksForwarder(TDeviceId deviceId)
            : Manager(GetMpiManager())
            , DeviceId(deviceId)
        {
        }

        template <class TTask>
        void AddTask(THolder<TTask>&& task) {
            ForwardTask(*task);
        }

        template <class TTask,
                  class... Args>
        void EmplaceTask(Args&&... args) {
            TTask task(std::forward<Args>(args)...);
            ForwardTask(task);
        }

    private:
        TMpiManager& Manager;
        TDeviceId DeviceId;

    private:
        template <class TTask>
        void ForwardTask(const TTask& task) {
            TSerializedTask serializedTask;
            TBufferOutput out(serializedTask);
            TTaskSerializer::SaveCommand(task, &out);
            Manager.SendTask(DeviceId,
                             std::move(serializedTask));
        }
    };

#endif
}
