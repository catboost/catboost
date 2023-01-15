#include "linear_cusolver_stub.cuh"

namespace NKernel {
    void RunCholeskyCuSolver(
        float* matrices, float* solutions,
        int rowSize, int matCount, bool removeLast,
        TCholeskySolverContext& context,
        TCudaStream stream
    ) {
        Y_UNUSED(matrices);
        Y_UNUSED(solutions);
        Y_UNUSED(rowSize);
        Y_UNUSED(matCount);
        Y_UNUSED(removeLast);
        Y_UNUSED(context);
        Y_UNUSED(stream);
    }
}
