#pragma once

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {
    struct TCholeskySolverContext : public IKernelContext {

        inline TCholeskySolverContext(ui32 /*rowSize*/) {
        }

        static bool UseCuSolver(ui32 /*rowSize*/, ui32 /*matCount*/) {
            return false;
        }

        void AllocateBuffers(NKernelHost::IMemoryManager& /*manager*/) {
        }

        inline TCholeskySolverContext() = default;
    };

    void RunCholeskyCuSolver(
        float* matrices,
        float* solutions,
        int rowSize,
        int matCount,
        bool removeLast,
        TCholeskySolverContext& context,
        TCudaStream stream
    );
}
