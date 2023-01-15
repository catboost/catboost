#pragma once

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

#include <cusolverDn.h>

namespace NKernel {
    struct TCholeskySolverContext : public IKernelContext {

        inline TCholeskySolverContext(ui32 rowSize)
        : RowSize(rowSize)
        , BatchSize(CuSolverBufferSize / (rowSize * rowSize))
        {
            CB_ENSURE(BatchSize > 0);
            CB_ENSURE(CUSOLVER_STATUS_SUCCESS == cusolverDnCreate(&CuSolverHandle));
        }

        static constexpr ui32 CuSolverBufferSize = 1u << 22;

        static bool UseCuSolver(ui32 rowSize, ui32 matCount) {
            return rowSize >= 32 && matCount >= 10000;
        }

        void AllocateBuffers(NKernelHost::IMemoryManager& manager);

        ui32 RowSize = 0;
        ui32 BatchSize = 0;

        TDevicePointer<float> CuSolverBuffer;
        TDevicePointer<int> CuSolverInfo;
        TDevicePointer<float*> CuSolverLhs;
        TDevicePointer<float*> CuSolverRhs;

        cusolverDnHandle_t CuSolverHandle = NULL;

        inline TCholeskySolverContext() = default;
        inline ~TCholeskySolverContext() {
            if (CuSolverHandle) {
                cusolverDnDestroy(CuSolverHandle);
            }
        }
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
