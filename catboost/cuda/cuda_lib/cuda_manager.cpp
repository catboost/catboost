#include "cuda_manager.h"
#include "cuda_profiler.h"
#include "mapping.h"
#include "cuda_buffer.h"
#include <catboost/cuda/cuda_lib/tasks_impl/memory_copy_tasks.h>
#include <catboost/cuda/cuda_lib/tasks_queue/mpi_task_queue.h>
#include <catboost/cuda/cuda_lib/tasks_impl/enable_peers.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

using namespace NCudaLib;

void TCudaManager::CreateProfiler() {
    Profiler = new TCudaProfiler(EProfileMode::LabelAsync, 0, false);
}

TCudaManager::~TCudaManager() {
    Y_VERIFY(Profiler == nullptr, "Reset profile before stopping cuda manager");
    CB_ENSURE(FreeStreams.size() == 0, "Error: CudaManager was not stopped");
    CB_ENSURE(Streams.size() == 0, "Error: CudaManager was not stopped");
}

void TCudaManager::ResetProfiler(bool printInfo) {
    if (Profiler) {
        if (printInfo) {
            Profiler->PrintInfo();
        }
        delete Profiler;
        Profiler = nullptr;
    }
}

double TCudaManager::FreeMemoryMb(ui32 devId, bool waitComplete) const {
    if (waitComplete) {
        GetCudaManager().WaitComplete();
    }
    auto devPtr = GetState().Devices[devId];
    return devPtr->GetMemoryState().FreeGpuRam * 1.0 / 1024 / 1024;
}

double TCudaManager::TotalMemoryMb(ui32 devId) const {
    auto devPtr = GetState().Devices[devId];
    return devPtr->GetMemoryState().RequestedGpuRam * 1.0 / 1024 / 1024;
}

void TCudaManager::StopChild() {
    CB_ENSURE(IsChildManager);
    CB_ENSURE(ParentProfiler != nullptr);
    //add stats from child to parent
    {
        TGuard<TAdaptiveLock> guard(GetState().Lock);
        ParentProfiler->Add(*Profiler);
    }
    ResetProfiler(false);

    FreeComputationStreams();
    WaitComplete();
    State = nullptr;
    OnStopChildEvent.Signal();
}

void TCudaManager::StartChild(TCudaManager& parent,
                              const TDevicesList& devices,
                              TManualEvent& stopEvent) {
    CB_ENSURE(!State, "Error: can't start, state already exists");
    State = parent.State;
    IsChildManager = true;
    DevicesList = devices;
    OnStopChildEvent = stopEvent;

    IsActiveDevice.clear();
    IsActiveDevice.resize(GetDeviceCount(), false);

    for (auto& dev : DevicesList) {
        IsActiveDevice[dev] = true;
    }
    InitDefaultStream();
    CreateProfiler();
    GetProfiler().SetDefaultProfileMode(parent.GetProfiler().GetDefaultProfileMode());
    ParentProfiler = &parent.GetProfiler();
}

template <class TPeersKernel>
inline void TogglePeersKernel(TCudaManager& manager) {
    const ui64 deviceCount = manager.GetDeviceCount();
    for (ui64 dev = 0; dev < deviceCount; ++dev) {
        NCudaLib::TDeviceId myDevice = manager.GetDeviceId(dev);
        TPeersKernel peersKernel;
        for (ui32 peerDev = 0; peerDev < deviceCount; ++peerDev) {
            if (dev != peerDev) {
                NCudaLib::TDeviceId peerDevice = manager.GetDeviceId(peerDev);
                if (myDevice.HostId == peerDevice.HostId) {
                    peersKernel.Devices.push_back(peerDevice);
                }
            }
            manager.LaunchKernel(std::move(peersKernel), dev, 0);
        }
    }
}

void TCudaManager::EnablePeers() {
    TogglePeersKernel<NKernelHost::TEnablePeersKernel>(*this);
}

void TCudaManager::DisablePeers() {
    TogglePeersKernel<NKernelHost::TDisablePeersKernel>(*this);
}

void RunSlave() {
#if defined(USE_MPI)
    THostDevices hostWorkers(GetMpiManager().GetHostId());
    TVector<TSingleHostTaskQueue*> workerQueues;
    for (ui32 i = 0; i < hostWorkers.GetDeviceCount(); ++i) {
        workerQueues.push_back(&hostWorkers.GetWorkerQueue(i));
    }
    TMpiTaskSlaveForwarder taskForwarder(std::move(workerQueues));
    auto areWorkersStopped = [&]() -> bool {
        return !hostWorkers.IsRunning();
    };
    taskForwarder.Run(areWorkersStopped);
    hostWorkers.Join();
    GetMpiManager().Stop();
#endif
}

inline void InitMemPerformanceTables(TCudaManager& manager) {
    manager.WaitComplete();
    auto singleMapping = TSingleMapping(0, 42);
    auto mirrorMapping = TMirrorMapping(42);

    auto bufferSingle = TSingleBuffer<float>::Create(singleMapping);
    auto bufferMirror = TMirrorBuffer<float>::Create(mirrorMapping);
    Reshard(bufferSingle, bufferMirror);
    manager.WaitComplete();
}

TFinallyGuard<TStopCudaManagerCallback> StartCudaManager(const NCudaLib::TDeviceRequestConfig& requestConfig,
                                                         const ELoggingLevel loggingLevel) {
    SetLogingLevel(loggingLevel);

#if defined(USE_MPI)
    CB_ENSURE(GetMpiManager().IsMaster(), "Error: can't run cudaManager on slave");
#endif

    auto& manager = NCudaLib::GetCudaManager();
    manager.Start(requestConfig);
    manager.WaitComplete();

    ui32 devCount = manager.GetDeviceCount();
    for (ui32 dev = 0; dev < devCount; ++dev) {
        MATRIXNET_INFO_LOG << "Free memory on device #" << dev << " " << manager.FreeMemoryMb(dev) << "MB" << Endl;
    }
    InitMemPerformanceTables(manager);

    return TFinallyGuard<TStopCudaManagerCallback>(TStopCudaManagerCallback());
}

TFinallyGuard<TStopCudaManagerCallback> StartCudaManager(const ELoggingLevel loggingLevel) {
    return StartCudaManager(NCudaLib::GetDefaultDeviceRequestConfig(), loggingLevel);
}
