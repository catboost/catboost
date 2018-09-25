#include "hwloc_wrapper.h"

#if defined(WITH_HWLOC)
#include "cuda_base.h"
#include <hwloc/cudart.h>
#include <util/system/info.h>
#include <util/generic/set.h>

namespace NCudaLib {
    struct THwlocSet {
        THwlocSet() {
            Set = hwloc_bitmap_alloc();
        }

        ~THwlocSet() {
            hwloc_bitmap_free(Set);
        }

        void UseOnly(ui32 id) {
            hwloc_bitmap_only(Set, id);
        }

        hwloc_bitmap_t Set;
    };

    THardwareLocalityHelper::THardwareLocalityHelper() {
        int errCode = hwloc_topology_init(&Context);
        if (errCode != -1) {
            errCode = hwloc_topology_load(Context);
        }
        if (errCode == -1) {
            CATBOOST_ERROR_LOG << "Error: can't init hwloc topology" << Endl;
            HasContext = false;
        } else {
            HasContext = true;
        }
    }

    void THardwareLocalityHelper::BindThreadForDevice(int deviceId) {
        if (!HasContext) {
            return;
        }
        THwlocSet deviceCpu;
        THwlocSet numaNode;
        int errCode = hwloc_cudart_get_device_cpuset(Context, deviceId, deviceCpu.Set);
        hwloc_cpuset_to_nodeset(Context, deviceCpu.Set, numaNode.Set);

        errCode = hwloc_set_cpubind(Context, deviceCpu.Set, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
        if (errCode == -1) {
            CATBOOST_ERROR_LOG << "Can't bind thread for " << deviceId << " with err " << errno << Endl;
        }

        errCode = hwloc_set_membind_nodeset(Context, numaNode.Set, HWLOC_MEMBIND_BIND, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
        if (errCode == -1) {
            CATBOOST_ERROR_LOG << "Can't bind memory for " << deviceId << " with err " << errno << Endl;
        }
    }
}
#endif
