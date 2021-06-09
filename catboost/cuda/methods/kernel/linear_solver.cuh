#pragma once

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

        static bool IsCuSolverFast(ui32 rowSize, ui32 matCount) {
            return rowSize >= 32 && matCount >= 10000;
        }

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

    void ExtractMatricesAndTargets(const float* linearSystem, int matCount, int rowSize, float* matrices, float* targets, float* matrixDiag, TCudaStream stream);

    void ZeroMean(float* solutions, int rowSize, int matCount, TCudaStream stream);

    void CholeskySolver(float* matrices, float* solutions, int rowSize, int matCount, bool removeLast, TCholeskySolverContext& context, TCudaStream stream);

    void CalcScores(const float* linearSystem, const float* solutions, float* scores, int rowSize, int matCount, TCudaStream stream);

    void SolverForward(float* matrices, float* solutions, int rowSize, int matCount, TCudaStream stream);

    void SolverBackward(float* matrices, float* solutions, int rowSize, int matCount, TCudaStream stream);

    void Regularize(float* matrices, int rowSize, int matCount, double lambdaNonDiag, double lambdaDiag, TCudaStream stream);
}
