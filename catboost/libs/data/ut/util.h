#pragma once

#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/vector.h>


namespace NCB {
    namespace NDataNewUT {

    inline TVector<ETaskType> GetTaskTypes() {
        TVector<ETaskType> result = {ETaskType::CPU};
        if (GetGpuDeviceCount() > 0) {
            result.push_back(ETaskType::GPU);
        }
        return result;
    }

    }
}
