#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/cuda/cuda_lib/task.h>
#include <catboost/cuda/cuda_lib/device_id.h>
#include <catboost/cuda/cuda_lib/peer_devices.h>

namespace NKernelHost {
    using TDeviceId = NCudaLib::TDeviceId;

    struct TEnablePeersKernel: public TStatelessKernel {
        TVector<TDeviceId> Devices;

        TEnablePeersKernel(TVector<TDeviceId>&& devices)
            : Devices(devices)
        {
        }

        TEnablePeersKernel() {
        }

        void Run(const TCudaStream&) const {
            int myHostId = NCudaLib::GetHostId();
            int myDevice = NCudaLib::GetDevice();
            auto& peerHelper = NCudaLib::GetPeerDevicesHelper();

            for (auto& deviceId : Devices) {
                if (deviceId.HostId == myHostId) {
                    peerHelper.EnablePeerAccess(myDevice, deviceId.DeviceId);
                }
            }
        }

        Y_SAVELOAD_DEFINE(Devices);
    };

    struct TDisablePeersKernel: public TStatelessKernel {
        TVector<TDeviceId> Devices;

        TDisablePeersKernel(TVector<TDeviceId>&& devices)
            : Devices(devices)
        {
        }

        TDisablePeersKernel() {
        }

        void Run(const TCudaStream&) const {
            int myHostId = NCudaLib::GetHostId();
            int myDevice = NCudaLib::GetDevice();
            auto& peerHelper = NCudaLib::GetPeerDevicesHelper();

            for (auto& deviceId : Devices) {
                if (deviceId.HostId == myHostId) {
                    peerHelper.DisablePeerAccess(myDevice, deviceId.DeviceId);
                }
            }
        }

        Y_SAVELOAD_DEFINE(Devices);
    };
}
