#pragma once

#include "cuda_base.h"
#include <util/system/types.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>

namespace NCudaLib {
    class TPeerDevicesHelper {
    public:
        bool HasPeerAccess(ui32 from, ui32 to) const {
            if (PeerDevices.size() && PeerDevices.size() > from) {
                return PeerDevices[from].has(to);
            }
            return false;
        }

        void EnablePeerAccess() {
            if (PeerDevices.size() == 0) {
                ui32 devCount = (ui32)NCudaHelpers::GetDeviceCount();
                PeerDevices.resize(devCount);
                int devId = GetDevice();
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
                SetDevice(devId);
            } else {
                CB_ENSURE((int)PeerDevices.size() == NCudaHelpers::GetDeviceCount());
            }
        }

    private:
        TVector<TSet<ui32>> PeerDevices;
    };

    inline TPeerDevicesHelper& GetPeerDevicesHelper() {
        return *Singleton<TPeerDevicesHelper>();
    }
}
