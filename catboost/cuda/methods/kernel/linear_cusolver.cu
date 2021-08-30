#include "linear_cusolver.cuh"

namespace NKernel {
    void TCholeskySolverContext::AllocateBuffers(NKernelHost::IMemoryManager& manager) {
        CuSolverBuffer = manager.Allocate<float>(CuSolverBufferSize);
        CuSolverInfo = manager.Allocate<int>(BatchSize);
        CuSolverLhs = manager.Allocate<float*>(BatchSize);
        CuSolverRhs = manager.Allocate<float*>(BatchSize);
    }

    __global__ void UnpackTriangularImpl(float* lower, float* solutions, int rowSize, int matCount, float* full, float** rhs, float** lhs) {
        int matIdx = blockIdx.x;
        int row = threadIdx.x;
        if (matIdx >= matCount || row >= rowSize) {
            return;
        }
        rhs[matIdx] = solutions + (size_t)matIdx * rowSize; // safe write-write conflict
        lhs[matIdx] = full + (size_t)matIdx * rowSize * rowSize;
        float* lowerRow = lower + (size_t)matIdx * rowSize * (rowSize + 1) / 2 + row * (row + 1) / 2;
        float* fullRow = full + (size_t)matIdx * rowSize * rowSize + row * rowSize;
        #pragma unroll 32
        for (int col = 0; col <= row; ++col) {
            fullRow[col] = lowerRow[col];
        }
        row = rowSize - 1 - row;
        lowerRow = lower + (size_t)matIdx * rowSize * (rowSize + 1) / 2 + row * (row + 1) / 2;
        fullRow = full + (size_t)matIdx * rowSize * rowSize + row * rowSize;
        #pragma unroll 32
        for (int col = 0; col <= row; ++col) {
            fullRow[col] = lowerRow[col];
        }
    }

    void RunCholeskyCuSolver(
        float* matrices, float* solutions,
        int rowSize, int matCount, bool removeLast,
        TCholeskySolverContext& context,
        TCudaStream stream
    ) {
        CB_ENSURE(CUSOLVER_STATUS_SUCCESS == cusolverDnSetStream(context.CuSolverHandle, stream));
        const auto uplo = CUBLAS_FILL_MODE_UPPER;
        const int nrhs = 1;
        const auto cuSolverBuffer = context.CuSolverBuffer.Get();
        const auto cuSolverLhs = context.CuSolverLhs.Get();
        const auto cuSolverRhs = context.CuSolverRhs.Get();
        const auto cuSolverInfo = context.CuSolverInfo.Get();
        const auto batchSize = context.BatchSize;
        const auto lowerSize = rowSize * (rowSize + 1) / 2;
        for (int matIdx = 0; matIdx < matCount; matIdx += batchSize) {
            auto count = Min<ui32>(batchSize, matCount - matIdx);
            UnpackTriangularImpl<< <count, (rowSize + 1) / 2, 0, stream>> >(
                matrices + matIdx * lowerSize,
                solutions + matIdx * rowSize,
                rowSize,
                count,
                cuSolverBuffer,
                cuSolverRhs,
                cuSolverLhs);
            CB_ENSURE(
                CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrfBatched(
                    context.CuSolverHandle,
                    uplo,
                    rowSize - removeLast,
                    cuSolverLhs,
                    rowSize,
                    cuSolverInfo,
                    count));
            CB_ENSURE(
                CUSOLVER_STATUS_SUCCESS == cusolverDnSpotrsBatched(
                    context.CuSolverHandle,
                    uplo,
                    rowSize - removeLast,
                    nrhs, /*only support rhs = 1*/
                    cuSolverLhs,
                    rowSize,
                    cuSolverRhs,
                    rowSize,
                    cuSolverInfo,
                    count));
        }
    }
}
