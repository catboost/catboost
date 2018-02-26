#include "mpi_manager.h"
#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <util/system/env.h>
#include <catboost/cuda/cuda_lib/tasks_queue/mpi_task_queue.h>
#include <library/blockcodecs/codecs.h>

namespace NCudaLib {
#if defined(WRITE_MPI_MESSAGE_LOG)
    TOperationsLogger::TOperationsLogger() {
        Out = new TOFStream("mpi_messages.log");
    }
#endif
    void TMpiManager::Start(int* argc, char*** argv) {
        int providedLevel;
        int threadLevel = MPI_THREAD_MULTIPLE;

        MPI_SAFE_CALL(MPI_Init_thread(argc, argv, threadLevel, &providedLevel));
        CB_ENSURE(providedLevel >= threadLevel, "Error: MPI implementation doesn't support thread multiple level");
        Communicator = MPI_COMM_WORLD;

        MPI_SAFE_CALL(MPI_Comm_size(Communicator, &HostCount));
        MPI_SAFE_CALL(MPI_Comm_rank(Communicator, &HostId));

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

        if (IsMaster()) {
            TVector<int> devicesOnHost(HostCount);
            devicesOnHost[0] = deviceCount;

            TVector<TVector<TCudaDeviceProperties>> deviceProps;
            deviceProps.resize(HostCount);
            deviceProps[0] = NCudaHelpers::GetDevicesProps();

            for (int i = 1; i < HostCount; ++i) {
                Read(reinterpret_cast<char*>(&devicesOnHost[i]), deviceCountTypeBytes, i, 0);
                deviceProps[i] = Receive<TVector<TCudaDeviceProperties>>(i, 0);
            }

            for (int host = 0; host < HostCount; ++host) {
                for (int dev = 0; dev < devicesOnHost[host]; ++dev) {
                    Devices.push_back(TDeviceId(host, dev));
                    DeviceProps.push_back(deviceProps[host][dev]);
                }
            }
        } else {
            TVector<TCudaDeviceProperties> props = NCudaHelpers::GetDevicesProps();
            Write(reinterpret_cast<const char*>(&deviceCount), deviceCountTypeBytes, 0, 0);
            Send(props, 0, 0);
        }

        CB_ENSURE(HostCount >= 1, "Error: need at least one worker");
    }

    void TMpiManager::Stop() {
        if (IsMaster()) {
            NCudaLib::GetDevicesProvider().FreeDevices();
        }
        MPI_SAFE_CALL(MPI_Finalize());
    }


    TMpiLock& GetMpiLock() {
        return GetMpiManager().GetLock();
    }
}

#endif
