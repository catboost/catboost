#include "mpi_manager.h"

#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <util/system/env.h>
#include <util/string/cast.h>
#include <catboost/cuda/cuda_lib/tasks_queue/mpi_task_queue.h>
#include <library/cpp/blockcodecs/codecs.h>

namespace NCudaLib {
    void TMpiManager::Start(int* argc, char*** argv) {
        int providedLevel;
        int threadLevel = MPI_THREAD_SERIALIZED;

        MPI_SAFE_CALL(MPI_Init_thread(argc, argv, threadLevel, &providedLevel));
        CB_ENSURE(providedLevel >= threadLevel, "Error: MPI implementation doesn't support thread serialized level");
        Communicator = MPI_COMM_WORLD;

        MPI_SAFE_CALL(MPI_Comm_size(Communicator, &HostCount));
        MPI_SAFE_CALL(MPI_Comm_rank(Communicator, &HostId));
        CATBOOST_DEBUG_LOG << "Host count: " << HostCount << " Host id: " << HostId << Endl;
        CommandsBuffer.resize(BufferSize);
        MPI_SAFE_CALL(MPI_Buffer_attach(CommandsBuffer.data(), CommandsBuffer.size()));

        //env config
        TString codecName = GetEnv("CB_COMPRESS_CODEC", "lz4fast");
        CompressCodec = NBlockCodecs::Codec(codecName);
        MinCompressSize = FromString(GetEnv("CB_MIN_COMPRESS_SIZE", "10000"));

        TString taskSendType = GetEnv("CB_BSEND_TASKS", "false");
        if (taskSendType == "true") {
            UseBSendForTasks = true;
        }

        int deviceCount = NCudaHelpers::GetDeviceCount();
        int deviceCountTypeBytes = sizeof(decltype(deviceCount));

        MpiProxyThread = MakeHolder<std::thread>([this]() {
            this->ProceedRequests();
        });

        if (IsMaster()) {
            CATBOOST_DEBUG_LOG << "Starting master" << Endl;
            TVector<int> devicesOnHost(HostCount);
            devicesOnHost[0] = deviceCount;

            TVector<TVector<TCudaDeviceProperties>> deviceProps;
            deviceProps.resize(HostCount);
            deviceProps[0] = NCudaHelpers::GetDevicesProps();

            for (int i = 1; i < HostCount; ++i) {
                Read(reinterpret_cast<char*>(&devicesOnHost[i]), deviceCountTypeBytes, i, 0);
                for (int dev = 0; dev < devicesOnHost[i]; ++dev) {
                    TCudaDeviceProperties props;
                    ReceivePodAsync(i, 0, &props)->WaitComplete();
                    deviceProps[i].push_back(props);
                }
            }

            for (int host = 0; host < HostCount; ++host) {
                for (int dev = 0; dev < devicesOnHost[host]; ++dev) {
                    Devices.push_back(TDeviceId(host, dev));
                    DeviceProps.push_back(deviceProps[host][dev]);
                }
            }
        } else {
            CATBOOST_DEBUG_LOG << "Starting slave" << Endl;
            TVector<TCudaDeviceProperties> props = NCudaHelpers::GetDevicesProps();
            Write(reinterpret_cast<const char*>(&deviceCount), deviceCountTypeBytes, 0, 0);
            for (const auto& prop : props) {
                SendPod(prop, 0, 0);
            }
        }

        CB_ENSURE(HostCount >= 1, "Error: need at least one worker");
    }

    void TMpiManager::Stop() {
        if (IsMaster()) {
            NCudaLib::GetDevicesProvider().FreeDevices();
        }

        AtomicSet(StopFlag, true);
        HasWorkEvent.Signal();
        MpiProxyThread->join();

        MPI_SAFE_CALL(MPI_Finalize());
    }

    void TMpiManager::SendTask(const TDeviceId& deviceId,
                               TSerializedTask&& task) {
        Y_ASSERT(IsMaster());
        TSendTaskRequest request;
        request.DeviceId = deviceId;
        request.Task = std::move(task);
        SendCommands.Enqueue(std::move(request));
        HasWorkEvent.Signal();
    }

    TMpiRequestPtr TMpiManager::ReadAsync(char* data, int dataSize, int sourceRank, int tag) {
        TMpiRequestPtr request = new TMpiRequest();
        TMemcpyReceiveRequest readRequest;
        readRequest.Request = request;
        readRequest.DataSize = dataSize;
        readRequest.Data = data;
        readRequest.SourceRank = sourceRank;
        readRequest.Tag = tag;
        ReceiveRequests.Enqueue(std::move(readRequest));
        HasWorkEvent.Signal();
        return request;
    }

    TMpiRequestPtr TMpiManager::WriteAsync(const char* data, int dataSize, int destRank, int tag) {
        TMpiRequestPtr request = new TMpiRequest();
        TMemcpySendRequest sendRequest;
        sendRequest.Request = request;
        sendRequest.DataSize = dataSize;
        sendRequest.Data = data;
        sendRequest.DestRank = destRank;
        sendRequest.Tag = tag;
        SendRequests.Enqueue(std::move(sendRequest));
        HasWorkEvent.Signal();
        return request;
    }

    TMpiManager::TMpiRequest::EState TMpiManager::InvokeRunningRequest(TMpiRequest* request) {
        if (request->CancelFlag == 1) {
            MPI_SAFE_CALL(MPI_Cancel(&(request->Request)));
            request->SetState(TMpiRequest::EState::Canceled);
            request->WaitEvent.Signal();
            return TMpiRequest::EState::Canceled;
        } else {
            Y_ASSERT(request->GetState() == TMpiRequest::EState::Running);
            Y_ASSERT(request->Request != MPI_REQUEST_NULL);

            int flag = 0;
            MPI_SAFE_CALL(MPI_Test(&request->Request, &flag, &request->Status));

            if (flag) {
                int length = 0;
                //TODO(noxoomo): check performance impact for this method, should be negilable
                MPI_SAFE_CALL(MPI_Get_count(&request->Status, MPI_CHAR, &length));

                AtomicSet(request->ReceivedBytesCount, length);
                request->SetState(TMpiRequest::EState::Completed);
                request->WaitEvent.Signal();
                return TMpiRequest::EState::Completed;
            }
            return TMpiRequest::EState::Running;
        }
    }

    void TMpiManager::ProceedRequests() {
        bool isMaster = IsMaster();
        while (true) {
            HasWorkEvent.Reset();

            bool hasWorkToDo = false;
            if (isMaster) {
                const int maxSendTasksTries = 16 * Max<int>(Devices.size(), 1);

                for (int k = 0; k < maxSendTasksTries; ++k) {
                    if (!SendCommands.IsEmpty()) {
                        hasWorkToDo = true;
                        TSendTaskRequest request;
                        CB_ENSURE(SendCommands.Dequeue(request), "Dequeue of send command failed");
                        const auto& deviceId = request.DeviceId;

                        const int size = static_cast<const int>(request.Task.Size());
                        Y_ASSERT(size < (int)BufferSize);
                        Y_ASSERT(size);

                        if (UseBSendForTasks) {
                            MPI_SAFE_CALL(MPI_Bsend(request.Task.Data(), size, MPI_CHAR,
                                                    deviceId.HostId, GetTaskTag(deviceId),
                                                    Communicator));
                        } else {
                            MPI_SAFE_CALL(MPI_Send(request.Task.Data(), size, MPI_CHAR,
                                                   deviceId.HostId, GetTaskTag(deviceId),
                                                   Communicator));
                        }

                    } else {
                        break;
                    }
                }
            }

            const int memcpyRequestTries = 32;

            for (int k = 0; k < memcpyRequestTries; ++k) {
                if (!ReceiveRequests.IsEmpty()) {
                    hasWorkToDo = true;
                    TMemcpyReceiveRequest readRequest;
                    const auto rc = ReceiveRequests.Dequeue(readRequest);
                    CB_ENSURE(rc, "Dequeue from receive requests failed");
                    CB_ENSURE(readRequest.Request != nullptr, "Dequeued read request is nullptr");
                    Y_ASSERT(readRequest.Request->GetState() == TMpiRequest::EState::Created);

                    MPI_SAFE_CALL(MPI_Irecv(readRequest.Data, readRequest.DataSize,
                                            MPI_CHAR, readRequest.SourceRank, readRequest.Tag,
                                            Communicator,
                                            &readRequest.Request->Request));

                    readRequest.Request->SetState(TMpiRequest::EState::Running);
                    if (InvokeRunningRequest(readRequest.Request.Get()) == TMpiRequest::EState::Running) {
                        RunningRequests.push_back(std::move(readRequest.Request));
                    }
                }

                if (!SendRequests.IsEmpty()) {
                    hasWorkToDo = true;
                    TMemcpySendRequest writeRequest;
                    const auto rc = SendRequests.Dequeue(writeRequest);
                    CB_ENSURE(rc, "Dequeue from send requests failed");
                    CB_ENSURE(writeRequest.Request != nullptr, "Dequeued write request is nullptr");
                    Y_ASSERT(writeRequest.Request->GetState() == TMpiRequest::EState::Created);

                    MPI_SAFE_CALL(MPI_Issend(writeRequest.Data, writeRequest.DataSize,
                                             MPI_CHAR, writeRequest.DestRank,
                                             writeRequest.Tag, Communicator,
                                             &writeRequest.Request->Request));

                    writeRequest.Request->SetState(TMpiRequest::EState::Running);
                    if (InvokeRunningRequest(writeRequest.Request.Get()) == TMpiRequest::EState::Running) {
                        RunningRequests.push_back(std::move(writeRequest.Request));
                    }
                }
            }

            {
                TVector<TMpiRequestPtr> stillRunning;

                while (RunningRequests.size()) {
                    TMpiRequestPtr request = RunningRequests.back();
                    RunningRequests.pop_back();

                    TMpiRequest::EState state = request->GetState();
                    if (state == TMpiRequest::EState::Created) {
                        if (request->CancelFlag) {
                            request->SetState(TMpiRequest::EState::Canceled);
                            request->WaitEvent.Signal();
                        }
                    } else if (state == TMpiRequest::EState::Running) {
                        if (InvokeRunningRequest(request.Get()) == TMpiRequest::EState::Running) {
                            stillRunning.push_back(request);
                        }
                    }
                }
                RunningRequests.swap(stillRunning);
                hasWorkToDo |= !RunningRequests.empty();
            }

            if (!hasWorkToDo) {
                if (StopFlag) {
                    return;
                } else {
                    HasWorkEvent.WaitT(TDuration::Max());
                }
            }
        }
    }
}

#endif
