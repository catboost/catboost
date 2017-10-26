#pragma once

#include "cuda_base.h"
#include "single_device.h"
#include "helpers.h"
#include <thread>
#include <util/generic/vector.h>

namespace NCudaLib {

    inline yset<ui32> GetEnabledDevices(const TString& deviceConfig)
    {
        const ui32 devCount = (ui32)NCudaHelpers::GetDeviceCount();
        yset<ui32> enabledDevices;
        if (deviceConfig == "-1") {
            for (ui32 i = 0; i < devCount; ++i) {
                enabledDevices.insert(i);
            }
        } else {
            enabledDevices = ::NHelpers::ParseRangeString(deviceConfig);
        }
        return enabledDevices;
    }

    struct TCudaApplicationConfig {
        constexpr static ui64 MB = 1024 * 1024;

        enum class EDevicesUsageType {
            Single, //one thread use one device. Thread-local cuda-managers
            All,    //one thread uses all devices
        } UsageType = EDevicesUsageType::All;

        ui32 WorkersPerDevice = 1;
        ui64 PinnedMemorySize = 1024 * MB;
        double GpuMemoryPartByWorker = 0.95;

        TString DeviceConfig = "-1";

        TCudaApplicationConfig(const TCudaApplicationConfig& other) = default;
        TCudaApplicationConfig() = default;

        ui64 GetGpuMemoryToUse(ui64 freeMemoryToUse) const {
            return (ui64)(freeMemoryToUse * GpuMemoryPartByWorker / WorkersPerDevice);
        }

        ui64 GetPinnedMemoryToUse(ui64 deviceMemorySize) const {
            Y_UNUSED(deviceMemorySize);
            return PinnedMemorySize;
        }

        ui32 GetDeviceCount() const {
            return GetEnabledDevices(DeviceConfig).size() * WorkersPerDevice;
        }
    };


    class TSingleHostDevicesProvider {
    private:
        yvector<THolder<TGpuOneDeviceWorker>> Workers;
        yvector<ui32> Devices;
        yvector<THolder<TCudaSingleDevice>> SingleDevices;
        bool IsInitialized = false;
        TCudaApplicationConfig Config;
        TSpinLock Lock;

        friend void SetApplicationConfig(const TCudaApplicationConfig& config);
        yvector<yset<ui32>> PeerDevices;

    private:
        void EnablePeerAccess() {
            if (Config.UsageType == TCudaApplicationConfig::EDevicesUsageType::All) {
                ui32 devCount = (ui32)NCudaHelpers::GetDeviceCount();
                PeerDevices.resize(devCount);
                for (ui32 i = 0; i < devCount; ++i) {
                    for (ui32 j = i + 1; j < devCount; ++j) {
                        int can = 0;
                        SetDevice(i);
                        cudaDeviceCanAccessPeer(&can, i, j);
                        if (can) {
                            cudaDeviceEnablePeerAccess(j, 0);
                            SetDevice(j);
                            cudaDeviceEnablePeerAccess(i, 0);
                            PeerDevices[i].insert(j);
                            PeerDevices[j].insert(i);
                        }
                    }
                }
            }
        }

        THolder<TGpuOneDeviceWorker> CreateWorker(int dev) {
            auto props = NCudaHelpers::GetDeviceProps(dev);
            ui64 free = 0;
            ui64 total = 0;
            NCudaLib::SetDevice(dev);
            CUDA_SAFE_CALL(cudaMemGetInfo(&free, &total));
            if (free * 1.0 / props.GetDeviceMemory() < 0.75) {
                MATRIXNET_WARNING_LOG << "Warning: less than 75% gpu memory available for training. Free: " << free * 1.0 / 1024 / 1024 << " Total: "<< free * 1.0 / 1024 / 1024 << Endl;
            }
            ui64 gpuMemoryToUse = Config.GetGpuMemoryToUse(free);

            ui64 pinnedMemoryToUse = Config.GetPinnedMemoryToUse(props.GetDeviceMemory());
            return MakeHolder<TGpuOneDeviceWorker>(dev, gpuMemoryToUse, pinnedMemoryToUse);
        }

        void Initilize() {
            CB_ENSURE(!IsInitialized, "Error: Initialization could be done only once");
            const ui32 devCount = (ui32)NCudaHelpers::GetDeviceCount();
            yset<ui32> enabledDevices = GetEnabledDevices(Config.DeviceConfig);
            const ui32 workersPerDevice = Config.WorkersPerDevice;
            const ui32 freeCount = enabledDevices.size() * workersPerDevice;

            Workers.resize(enabledDevices.size() * workersPerDevice);
            SingleDevices.resize(enabledDevices.size() * workersPerDevice);
            Devices.resize(freeCount);

            {
                ui32 offset = 0;
                for (ui32 i = 0; i < workersPerDevice; ++i) {
                    for (ui32 dev = 0; dev < devCount; ++dev) {
                        if (enabledDevices.count(dev)) {
                            Devices[offset++] = dev;
                        }
                    }
                }
            }
            MATRIXNET_INFO_LOG << "Available devices:" << Endl;

            EnablePeerAccess();

            for (ui32 i = 0; i < devCount; i++) {
                if (enabledDevices.count(i)) {
                    const auto props = NCudaHelpers::GetDeviceProps(i);
                    MATRIXNET_INFO_LOG << "  " << i << ". " << props.GetName() << " (compute capability "
                                       << props.GetMajor() << "." << props.GetMinor() << ")" << Endl;
                }
            }
            IsInitialized = true;
        }

        void ResetDevice(ui32 dev) {
            Workers[dev] = CreateWorker(Devices[dev]);
            SingleDevices[dev].Reset(new TCudaSingleDevice(*Workers[dev]));
        }

    public:
        void Free(TCudaSingleDevice* device) {
            TGuard<TSpinLock> guard(Lock);

            for (ui64 i = 0; i < Workers.size(); ++i) {
                if (SingleDevices[i] == device) {
                    SingleDevices[i].Reset(nullptr);
                    Workers[i].Reset(nullptr);
                    break;
                }
                CB_ENSURE(i != (Workers.size() - 1), "Error: unknown worker");
            }
        }

        yvector<TCudaSingleDevice*> GetDevices() {
            TGuard<TSpinLock> guard(Lock);
            if (!IsInitialized) {
                Initilize();
            }
            yvector<TCudaSingleDevice*> devices;

            switch (Config.UsageType) {
                case TCudaApplicationConfig::EDevicesUsageType::Single: {
                    for (ui32 i = 0; i < Workers.size(); ++i) {
                        if (Workers[i] == nullptr) {
                            ResetDevice(i);
                            devices.push_back(SingleDevices[i].Get());
                            break;
                        }
                        CB_ENSURE((i + 1) < Workers.size(), "Error: requested too many devices");
                    }
                    break;
                }
                case TCudaApplicationConfig::EDevicesUsageType::All: {
                    for (ui32 i = 0; i < Workers.size(); ++i) {
                        CB_ENSURE(Workers[i] == nullptr, "Error: device already used, can't return device");
                        ResetDevice(i);
                        devices.push_back(SingleDevices[i].Get());
                    }
                    break;
                }
            }
            for (ui32 i = 0; i < devices.size(); ++i) {
                devices[i]->PeerDevices.clear();

                for (ui32 j = 0; j < devices.size(); ++j) {
                    if (i == j) {
                        continue;
                    }
                    const ui32 leftDev = devices[i]->CudaDeviceId();
                    const ui32 rightDev = devices[i]->CudaDeviceId();
                    if (PeerDevices[leftDev].count(rightDev)) {
                        devices[i]->PeerDevices.insert(devices[j]);
                    }
                }
            }
            return devices;
        }

        void Reset() {
            IsInitialized = false;
            Workers.clear();
            SingleDevices.clear();
            Devices.clear();
            PeerDevices.clear();
        }

        ui64 TotalWorkerCount() {
            return Workers.size();
        }
    };

    inline TSingleHostDevicesProvider& GetDevicesProvider() {
        return *Singleton<TSingleHostDevicesProvider>();
    }

    inline void SetApplicationConfig(const TCudaApplicationConfig& config) {
        TSingleHostDevicesProvider& provider = GetDevicesProvider();
        CB_ENSURE(!provider.IsInitialized, "Error: can't update config after initialization");
        provider.Config = config;
    }
}
