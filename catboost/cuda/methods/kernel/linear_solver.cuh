#pragma once

#ifdef USE_CUSOLVER
#include "linear_cusolver.cuh"
#else
#include "linear_cusolver_stub.cuh"
#endif

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {
    void ExtractMatricesAndTargets(const float* linearSystem, int matCount, int rowSize, float* matrices, float* targets, float* matrixDiag, TCudaStream stream);

    void ZeroMean(float* solutions, int rowSize, int matCount, TCudaStream stream);

    void CholeskySolver(float* matrices, float* solutions, int rowSize, int matCount, bool removeLast, TCholeskySolverContext& context, TCudaStream stream);

    void CalcScores(const float* linearSystem, const float* solutions, float* scores, int rowSize, int matCount, TCudaStream stream);

    void SolverForward(float* matrices, float* solutions, int rowSize, int matCount, TCudaStream stream);

    void SolverBackward(float* matrices, float* solutions, int rowSize, int matCount, TCudaStream stream);

    void Regularize(float* matrices, int rowSize, int matCount, double lambdaNonDiag, double lambdaDiag, TCudaStream stream);
}
