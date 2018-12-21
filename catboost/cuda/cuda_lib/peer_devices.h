#pragma once

#include "cuda_base.h"
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>

namespace NCudaLib {
    class TPeerDevicesHelper {
    public:
        bool HasPeerAccess(int from, int to) const {
            if (PeerDevices.size() && PeerDevices.size() > static_cast<ui64>(from)) {
                return PeerDevices[from].contains(to);
            }
            return false;
        }

        void EnablePeerAccess(int currentDevice, int targetDevice) {
            if (PeerDevices[currentDevice].contains(targetDevice)) {
                return;
            }
            Y_ASSERT(GetDevice() == currentDevice);
            int can = 0;
            CUDA_SAFE_CALL(cudaDeviceCanAccessPeer(&can, currentDevice, targetDevice));
            if (can) {
                CUDA_SAFE_CALL(cudaDeviceEnablePeerAccess(targetDevice, 0));
                PeerDevices[currentDevice].insert(targetDevice);
            }
        }

        void DisablePeerAccess(int currentDevice, int targetDevice) {
            if (!PeerDevices[currentDevice].contains(targetDevice)) {
                return;
            }
            CUDA_SAFE_CALL(cudaDeviceDisablePeerAccess(targetDevice));
            TSet<ui32>& currentDevicePeersSet = PeerDevices[currentDevice];
            currentDevicePeersSet.erase(targetDevice);
        }

        TPeerDevicesHelper() {
            PeerDevices.resize(NCudaHelpers::GetDeviceCount());
        }

    private:
        TVector<TSet<ui32>> PeerDevices;
    };

    inline TPeerDevicesHelper& GetPeerDevicesHelper() {
        return *Singleton<TPeerDevicesHelper>();
    }
}
