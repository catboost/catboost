#pragma once

namespace NCB {

    /*
     * will always return 0 if CUDA support in not enabled in the build config (-DHAVE_CUDA=no) or
     * source code is linked with catboost/libs/gpu_config/force_no_cuda library
     *  (see catboost/libs/gpu_config/README.md)
     */
    int GetGpuDeviceCount();

}
