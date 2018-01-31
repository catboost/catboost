#pragma once

#include "cpu_func.h"
#include "host_tasks.h"
#include <catboost/cuda/cuda_lib/worker_state.h>

namespace NCudaLib {
    struct TMemoryStateFunc: public TBlockingFunc {
        TMemoryState operator()(const IWorkerStateProvider& stateProvider) const {
            return stateProvider.GetMemoryState();
        }

        Y_SAVELOAD_EMPTY();
    };

    template <>
    struct THostFuncTrait<TMemoryStateFunc> {
        static constexpr bool NeedWorkerState() {
            return true;
        }
    };

}
