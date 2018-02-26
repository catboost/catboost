#pragma once

#if defined(WITH_HWLOC)

#include <hwloc.h>
#include <util/generic/yexception.h>
#include <catboost/libs/logging/logging.h>
#include <util/system/spinlock.h>
#include <util/generic/map.h>

namespace NCudaLib {
    class THardwareLocalityHelper {
    public:
        THardwareLocalityHelper();

        void BindThreadForDevice(int deviceId);

        ~THardwareLocalityHelper() {
            hwloc_topology_destroy(Context);
            HasContext = false;
        }

    private:
        hwloc_topology_t Context;
        bool HasContext = false;
    };

    inline THardwareLocalityHelper& HardwareLocalityHelper() {
        return *Singleton<THardwareLocalityHelper>();
    }
}

#endif
