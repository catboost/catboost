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

TCudaManager::~TCudaManager() noexcept(false) {
    CB_ENSURE(Profiler == nullptr, "Reset profile before stopping cuda manager");
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

void TCudaManager::FreeStream(const ui32 streamId) {
    TDistributedObject<ui32>& stream = Streams[streamId];

    for (ui64 dev = 0; dev < State->Devices.size(); ++dev) {
        const auto devStreamId = stream.At(dev);
        if (devStreamId != State->Devices[dev]->DefaultStream()) {
            State->Devices[dev]->FreeStream(devStreamId);
        } else {
            CB_ENSURE(!IsActiveDevice[dev]);
        }
    }
}

void TCudaManager::InitDefaultStream() {
    CB_ENSURE(Streams.size() == 0);

    ui32 defaultStream = 0;
    for (auto& dev : DevicesList) {
        CB_ENSURE(defaultStream == State->Devices[dev]->DefaultStream());
    }
    {
        TDistributedObject<ui32> stream(GetDeviceCount());
        stream.Fill(defaultStream);
        Streams.push_back(std::move(stream));
    }
}

void TCudaManager::SetDevices(TVector<TCudaSingleDevice*>&& devices) {
    CB_ENSURE(!HasDevices(), "Error: CudaManager already has devices");
    GetState().Devices = std::move(devices);
    CB_ENSURE(Streams.size() == 0);
    CB_ENSURE(FreeStreams.size() == 0);
    const auto deviceCount = GetState().Devices.size();
    DevicesList = TDevicesListBuilder::Range(0, deviceCount);
    IsActiveDevice.clear();
    IsActiveDevice.resize(GetDeviceCount(), true);
    State->BuildDevPtrToDevId();
    InitDefaultStream();
}

void TCudaManager::FreeDevices() {
    auto& provider = GetDevicesProvider();
    for (auto dev : GetState().Devices) {
        provider.Free(dev);
    }
    GetState().Devices.resize(0);
    GetState().DevPtrToDevId.clear();
}

void TCudaManager::FreeComputationStreams() {
    CB_ENSURE((1 + FreeStreams.size()) == Streams.size(), "Error: not all streams are free");
    for (int i = Streams.size() - 1; i > 0; --i) {
        FreeStream(i);
    }
    Streams.clear();
    FreeStreams.resize(0);
}

TVector<ui32> TCudaManager::GetDevices(bool onlyLocalIfHasAny) const {
    TVector<ui32> devices;
    for (auto& dev : DevicesList) {
        if (onlyLocalIfHasAny && GetState().Devices[dev]->IsRemoteDevice()) {
            continue;
        }
        devices.push_back(dev);
    }
    if (devices.size() == 0) {
        for (auto& dev : DevicesList) {
            devices.push_back(dev);
        }
    }
    return devices;
}

void TCudaManager::WaitComplete(TDevicesList&& devices) {
    using TEventPtr = THolder<IDeviceFuture<ui64>>;
    TVector<TEventPtr> waitComplete;

    for (auto dev : devices) {
        CB_ENSURE(dev < GetState().Devices.size());
        CB_ENSURE(IsActiveDevice[dev], "Device should be active");
        waitComplete.push_back(GetState().Devices[dev]->WaitComplete());
    }

    for (auto& event : waitComplete) {
        event->Wait();
        CB_ENSURE(event->Has(), "Wait completed without value");
    }
}

void TCudaManager::Start(const NCudaLib::TDeviceRequestConfig& config) {
    CB_ENSURE(State == nullptr);
    State.Reset(new TCudaManagerState());
    CB_ENSURE(!HasDevices());
    SetDevices(GetDevicesProvider().RequestDevices(config));
    if (config.EnablePeers) {
        EnablePeers();
        State->PeersSupportEnabled = true;
    }
    CreateProfiler();
}

void TCudaManager::Stop() {
    CB_ENSURE(!IsChildManager);
    CB_ENSURE(State);

    if (State->PeersSupportEnabled) {
        DisablePeers();
    }

    FreeComputationStreams();
    WaitComplete();
    FreeDevices();
    ResetProfiler(true);
    State = nullptr;
}

TComputationStream TCudaManager::RequestStream() {
    if (FreeStreams.size() == 0) {
        TDistributedObject<ui32> stream = CreateDistributedObject<ui32>();
        for (ui64 dev = 0; dev < stream.DeviceCount(); ++dev) {
            if (IsActiveDevice[dev]) {
                stream.Set(dev, GetState().Devices[dev]->RequestStream());
            } else {
                stream.Set(dev, 0);
            }
        }
        FreeStreams.push_back(Streams.size());
        Streams.push_back(stream);
    }

    ui32 id = FreeStreams.back();
    FreeStreams.pop_back();
    return TComputationStream(id, this);
}

bool TCudaManager::HasRemoteDevices() const {
    for (auto dev : State->Devices) {
        if (dev->IsRemoteDevice()) {
            return true;
        }
    }
    return false;
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

THolder<TStopCudaManagerCallback> StartCudaManager(const NCudaLib::TDeviceRequestConfig& requestConfig,
                                                   const ELoggingLevel loggingLevel) {
    TSetLogging inThisScope(loggingLevel);

#if defined(USE_MPI)
    CB_ENSURE(GetMpiManager().IsMaster(), "Error: can't run cudaManager on slave");
#endif

    auto& manager = NCudaLib::GetCudaManager();
    manager.Start(requestConfig);
    manager.WaitComplete();

    ui32 devCount = manager.GetDeviceCount();
    for (ui32 dev = 0; dev < devCount; ++dev) {
        CATBOOST_INFO_LOG << "Free memory on device #" << dev << " " << manager.FreeMemoryMb(dev) << "MB" << Endl;
    }
    InitMemPerformanceTables(manager);

    return MakeHolder<TStopCudaManagerCallback>();
}

THolder<TStopCudaManagerCallback> StartCudaManager(const ELoggingLevel loggingLevel) {
    return StartCudaManager(NCudaLib::GetDefaultDeviceRequestConfig(), loggingLevel);
}
