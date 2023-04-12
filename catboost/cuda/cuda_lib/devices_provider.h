#pragma once

#include "cuda_base.h"
#include "single_device.h"
#include "helpers.h"

#include <catboost/private/libs/options/catboost_options.h>

#include <util/generic/vector.h>

namespace NCudaLib {
    class TTerminateOnErrorCallback: public IExceptionCallback {
    public:
        void Call(const TString& message) override {
            CATBOOST_ERROR_LOG << "Application terminated with error: " << message << Endl;
            std::terminate();
        }
    };

    class TSetDeviceExceptionCallback: public IExceptionCallback {
    public:
        void Call(const TString& message) override {
            CATBOOST_ERROR_LOG << "Exception on device #" << Device->GetDeviceId() << " Error: " << message << Endl;
            AtomicIncrement(Device->ExceptionsCount);
        }

    private:
        TCudaSingleDevice* Device;
    };

    inline TSet<ui32> GetEnabledDevices(const TString& deviceConfig, ui32 devCount) {
        TSet<ui32> enabledDevices;
        if (deviceConfig == "-1") {
            for (ui32 i = 0; i < devCount; ++i) {
                enabledDevices.insert(i);
            }
        } else {
            enabledDevices = ::NHelpers::ParseRangeString(deviceConfig, devCount);
        }
        return enabledDevices;
    }

    struct TDeviceRequestConfig {
        constexpr static ui64 MB = 1024 * 1024;

        ui64 PinnedMemorySize = 1024 * MB;
        double GpuMemoryPartByWorker = 0.95;
        bool EnablePeers = false;
        TString DeviceConfig = "-1";

        TDeviceRequestConfig(const TDeviceRequestConfig& other) = default;
        TDeviceRequestConfig() = default;

        ui64 GetGpuMemoryToUse(ui64 freeMemoryToUse) const {
            return (ui64)(freeMemoryToUse * GpuMemoryPartByWorker);
        }

        ui64 GetPinnedMemoryToUse(ui64 deviceMemorySize) const {
            Y_UNUSED(deviceMemorySize);
            return PinnedMemorySize;
        }
    };

    TDeviceRequestConfig CreateDeviceRequestConfig(const NCatboostOptions::TCatBoostOptions& options);

    //for mpi test purpose
    inline NCudaLib::TDeviceRequestConfig& GetDefaultDeviceRequestConfig() {
        return *Singleton<NCudaLib::TDeviceRequestConfig>();
    }

    class THostDevices: public TNonCopyable {
    public:
        explicit THostDevices(int hostId)
            : HostId(hostId)
        {
            Workers.resize(NCudaHelpers::GetDeviceCount());
            for (int device = 0; device < static_cast<int>(Workers.size()); ++device) {
                Workers[device] = MakeHolder<TGpuOneDeviceWorker>(device, new TTerminateOnErrorCallback);
            }
            DeviceProps = NCudaHelpers::GetDevicesProps();
        }

        ~THostDevices() {
            Join();
        }

        void Join() {
            for (auto& worker : Workers) {
                if (worker) {
                    worker->Join();
                }
            }
        }

        ui32 GetDeviceCount() const {
            return Workers.size();
        }

        TDeviceId GetDeviceId(ui32 devId) const {
            return TDeviceId(HostId, devId);
        }

        const TCudaDeviceProperties& GetDeviceProps(ui32 devId) const {
            return DeviceProps.at(devId);
        }

        TSingleHostTaskQueue& GetWorkerQueue(ui32 devId) {
            return Workers[devId]->GetTaskQueue();
        }

        bool IsRunning() const {
            for (const auto& worker : Workers) {
                if (worker && worker->IsRunning()) {
                    return true;
                }
            }
            return false;
        }

    private:
        int HostId = 0;
        TVector<THolder<TGpuOneDeviceWorker>> Workers;
        TVector<TCudaDeviceProperties> DeviceProps;
    };

    class TDevicesProvider {
    private:
        THolder<THostDevices> MasterWorkers;
#if defined(USE_MPI)
        TVector<THolder<TRemoteHostTasksForwarder>> SlaveForwarders;
#endif

        TVector<THolder<TCudaSingleDevice>> Devices;
        bool IsInitialized = false;
        TSpinLock Lock;

        inline void InitLocalDevices() {
            CB_ENSURE(MasterWorkers == nullptr, "Can't init more than once");

            MasterWorkers.Reset(new THostDevices(0));

            for (ui32 dev = 0; dev < MasterWorkers->GetDeviceCount(); ++dev) {
                TDeviceId deviceId = MasterWorkers->GetDeviceId(dev);
                TCudaDeviceProperties props = MasterWorkers->GetDeviceProps(dev);
                void* queue = &MasterWorkers->GetWorkerQueue(dev);
                Devices.push_back(MakeHolder<TCudaSingleDevice>(queue, deviceId, props));
            }
        }

#if defined(USE_MPI)
        inline void InitSlaveForwarders() {
            CB_ENSURE(MasterWorkers, "Create local workers first");
            CB_ENSURE(SlaveForwarders.size() == 0, "Can't init more than once");

            const auto& devices = GetMpiManager().GetDevices();
            const auto& deviceProps = GetMpiManager().GetDeviceProperties();

            const ui32 masterDevice = MasterWorkers->GetDeviceCount();

            for (ui32 dev = masterDevice; dev < devices.size(); ++dev) {
                TDeviceId deviceId = devices[dev];
                CB_ENSURE(deviceId.HostId != 0, "Error: host should be remote");
                SlaveForwarders.push_back(MakeHolder<TRemoteHostTasksForwarder>(deviceId));
                const TCudaDeviceProperties& props = deviceProps[dev];
                void* queue = SlaveForwarders.back().Get();
                Devices.push_back(MakeHolder<TCudaSingleDevice>(queue, deviceId, props));
            }
        }

#endif

        void Initilize() {
            CB_ENSURE(!IsInitialized, "Error: Initialization could be done only once");
            CB_ENSURE(GetHostId() == 0, "Error: could use devices provider only on master host");

            InitLocalDevices();
#if defined(USE_MPI)
            InitSlaveForwarders();
#endif

            IsInitialized = true;
        }

        void FreeDevices(ui32 dev) {
            CB_ENSURE(!Devices[dev]->IsStopped(), "Error: device already stopped");
            Devices[dev]->Stop();
        }

        TCudaSingleDevice* RequestDevice(ui32 dev, double gpuRamPart, double pinnedMemorySize) {
            CB_ENSURE(Devices[dev]->IsStopped(), "Error: device already requested " << dev);
            Devices[dev]->Start(gpuRamPart, pinnedMemorySize);
            return Devices[dev].Get();
        }

    private:
        void FreeDevices() {
            for (const auto& device : Devices) {
                CB_ENSURE(device->IsStopped());
            }
            Devices.clear();
        }

        friend class TMpiManager;

    public:
        ~TDevicesProvider() noexcept(false) {
#if defined(USE_MPI)
            if (Devices.size() > 0) {
                CATBOOST_ERROR_LOG << "CatBoost did not free some GPU devices. "
                    << " This may result in memory leakage/fragmentation, "
                    << " abandoned threads on CPU and GPU on local or remoted hosts" << Endl;
                if (!std::uncaught_exceptions()) {
                    CB_ENSURE(Devices.size() == 0);
                }
            }
#else
            Devices.resize(0);
#endif
        }

        void Free(TCudaSingleDevice* device) {
            TGuard<TSpinLock> guard(Lock);

            for (ui64 i = 0; i < Devices.size(); ++i) {
                if (Devices[i] == device) {
                    FreeDevices(i);
                    break;
                }
                CB_ENSURE(i != (Devices.size() - 1), "Error: unknown worker");
            }
        }

        ui32 GetDeviceCount() {
            TGuard<TSpinLock> guard(Lock);
            if (!IsInitialized) {
                Initilize();
            }
            return Devices.size();
        }

        TVector<TCudaSingleDevice*> RequestDevices(const TDeviceRequestConfig& config) {
            TGuard<TSpinLock> guard(Lock);
            if (!IsInitialized) {
                Initilize();
            }
            const ui32 devCount = Devices.size();

            TSet<ui32> requestedDevices = GetEnabledDevices(config.DeviceConfig,
                                                            Devices.size());

            TVector<TCudaSingleDevice*> devices;

            for (auto dev : requestedDevices) {
                devices.push_back(RequestDevice(dev, config.GpuMemoryPartByWorker, config.PinnedMemorySize));
            }

            CB_ENSURE(requestedDevices.size(), "Error: no devices found");
            CATBOOST_INFO_LOG << "Requested devices:" << Endl;
            for (ui32 i = 0; i < devCount; i++) {
                if (requestedDevices.count(i)) {
                    const auto& props = Devices[i]->GetDeviceProperties();
                    TDeviceId id = Devices[i]->GetDevice();

                    CATBOOST_INFO_LOG << "  " << i << ". " << props.GetName() << " (compute capability "
                                      << props.GetMajor() << "." << props.GetMinor();
                    if (id.HostId != 0) {
                        CATBOOST_INFO_LOG << ", host " << id.HostId << ")" << Endl;
                    } else {
                        CATBOOST_INFO_LOG << ")" << Endl;
                    }
                }
            }
            return devices;
        }
    };

    inline TDevicesProvider& GetDevicesProvider() {
        return *Singleton<TDevicesProvider>();
    }

}
