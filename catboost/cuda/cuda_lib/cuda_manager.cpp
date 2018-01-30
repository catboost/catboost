#include "cuda_manager.h"
#include "cuda_profiler.h"
#include <catboost/cuda/cuda_lib/tasks_impl/single_host_memory_copy_tasks.h>

using namespace NCudaLib;

void TCudaManager::CreateProfiler() {
    Profiler = new TCudaProfiler;
}

TCudaManager::~TCudaManager() {
    Y_VERIFY(Profiler == nullptr, "Reset profile before stopping cuda manager");
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

void TCudaManager::SyncStream(ui32 stream) {
    TSingleHostStreamSync streamSync(stream);
    for (auto dev : DevicesList) {
        streamSync.AddDevice(GetState().Devices[dev]);
    }
    streamSync();
}

void TCudaManager::DumpFreeMemory(TString message) const {
    GetCudaManager().WaitComplete();
    MATRIXNET_INFO_LOG << message << Endl;
    for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
        auto devPtr = GetState().Devices[dev];
        const double totalMb = devPtr->GetGpuRamSize() * 1.0 / 1024 / 1024;
        const double freeMb = devPtr->GetFreeMemorySize() * 1.0 / 1024 / 1024;
        MATRIXNET_INFO_LOG << "    Device memory #" << dev << " " << freeMb << " / " << totalMb << Endl;
    }
}

double TCudaManager::TotalMemoryMb(ui32 devId) const {
    auto devPtr = GetState().Devices[devId];
    return devPtr->GetGpuRamSize() * 1.0 / 1024 / 1024;
}

double TCudaManager::MinFreeMemoryFraction() const {
    GetCudaManager().WaitComplete();
    double min = 1.0;
    for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
        auto devPtr = GetState().Devices[dev];
        const double totalMb = devPtr->GetGpuRamSize() * 1.0 / 1024 / 1024;
        const double freeMb = devPtr->GetFreeMemorySize() * 1.0 / 1024 / 1024;
        min = Min<double>(min, freeMb / totalMb);
    }
    return min;
}

double TCudaManager::FreeMemoryMb(ui32 deviceId,
                                  bool waitComplete) const {
    if (waitComplete) {
        GetCudaManager().WaitComplete();
    }
    auto devPtr = GetState().Devices[deviceId];
    return devPtr->GetFreeMemorySize() * 1.0 / 1024 / 1024;
}

void TCudaManager::StopChild() {
    CB_ENSURE(IsChildManager);
    CB_ENSURE(ParentProfiler != nullptr);
    WaitComplete();
    //add stats from child to parent
    {
        TGuard<TAdaptiveLock> guard(GetState().Lock);
        ParentProfiler->Add(*Profiler);
    }
    ResetProfiler(false);
    State = nullptr;
    OnStopChildEvent.Signal();
}

void TCudaManager::StartChild(TCudaManager& parent,
                              const TDevicesList& devices,
                              TAutoEvent& stopEvent) {
    CB_ENSURE(!State, "Error: can't start, state already exists");
    State = parent.State;
    IsChildManager = true;
    DevicesList = devices;
    OnStopChildEvent = stopEvent;

    IsActiveDevice.clear();
    IsActiveDevice.resize(GetDeviceCount(), false);
    for (auto& dev : devices) {
        IsActiveDevice[dev] = true;
    }
    CreateProfiler();
    GetProfiler().SetDefaultProfileMode(parent.GetProfiler().GetDefaultProfileMode());
    ParentProfiler = &parent.GetProfiler();
}
