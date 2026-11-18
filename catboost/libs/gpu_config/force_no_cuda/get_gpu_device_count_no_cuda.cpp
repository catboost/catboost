#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>
#if defined(CATBOOST_HAVE_METAL)
#include <catboost/metal/native/metal_trainer.h>
#endif

namespace NCB {

    bool IsMetalBackend() {
#if defined(CATBOOST_HAVE_METAL)
        return true;
#else
        return false;
#endif
    }

    int GetGpuDeviceCount() {
#if defined(CATBOOST_HAVE_METAL)
        char name[256] = {};
        char error[2048] = {};
        return cbm_device_info(name, sizeof(name), error, sizeof(error)) == 0 ? 1 : 0;
#else
        return 0;
#endif
    }

}
