#pragma once
#include <catboost/libs/helpers/exception.h>

#include <util/system/yassert.h>
#include <util/ysaveload.h>
#include <tuple>

namespace NCudaLib {
    struct TDeviceId {
        int HostId = -1;
        int DeviceId = -1;

        TDeviceId() = default;

        TDeviceId(int hostId,
                  int deviceId)
            : HostId(hostId)
            , DeviceId(deviceId)
        {
#ifndef USE_MPI
            CB_ENSURE(hostId == 0, "Remote device support is not enabled");
#endif
        }

        bool operator==(const TDeviceId& rhs) const {
            return std::tie(HostId, DeviceId) == std::tie(rhs.HostId, rhs.DeviceId);
        }

        bool operator!=(const TDeviceId& rhs) const {
            return !(rhs == *this);
        }

        inline bool operator<(const NCudaLib::TDeviceId& right) const {
            const auto& left = *this;
            return left.HostId < right.HostId || (left.HostId == right.HostId && left.DeviceId < right.DeviceId);
        }

        inline bool operator<=(const NCudaLib::TDeviceId& right) const {
            const auto& left = *this;
            return left.HostId < right.HostId || (left.HostId == right.HostId && left.DeviceId <= right.DeviceId);
        }

        inline bool operator>(const NCudaLib::TDeviceId& right) const {
            const auto& left = *this;
            return !(left <= right);
        }

        inline bool operator>=(const NCudaLib::TDeviceId& right) const {
            const auto& left = *this;
            return !(left < right);
        }

        Y_SAVELOAD_DEFINE(HostId, DeviceId);
    };

}

Y_DECLARE_PODTYPE(NCudaLib::TDeviceId);

template <>
struct THash<NCudaLib::TDeviceId> {
    inline size_t operator()(const NCudaLib::TDeviceId& deviceId) const {
        return (static_cast<ui64>(deviceId.HostId) << 32) | deviceId.DeviceId;
    }
};
