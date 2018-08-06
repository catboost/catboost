#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/targets/kernel/multilogit.cuh>

namespace NKernelHost {

    class TMultiLogitValueAndDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> TargetWeights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const ui32> LoadPredictionIndices;
        int NumClasses;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
    public:
        TMultiLogitValueAndDerKernel() = default;

        TMultiLogitValueAndDerKernel(TCudaBufferPtr<const float> targetClasses,
                                     TCudaBufferPtr<const float> targetWeights,
                                     TCudaBufferPtr<const float> predictions,
                                     TCudaBufferPtr<const ui32> loadPredictionIndices,
                                     int numClasses,
                                     TCudaBufferPtr<float> functionValue,
                                     TCudaBufferPtr<float> der)
                : TargetClasses(targetClasses)
                  , TargetWeights(targetWeights)
                  , Predictions(predictions)
                  , LoadPredictionIndices(loadPredictionIndices)
                  , NumClasses(numClasses)
                  , FunctionValue(functionValue)
                  , Der(der)
        {
        }

        Y_SAVELOAD_DEFINE(TargetClasses, NumClasses, TargetWeights, Predictions, FunctionValue, Der, LoadPredictionIndices);

        void Run(const TCudaStream& stream) const {
            const int approxDim = static_cast<const int>(Predictions.GetColumnCount());
            CB_ENSURE(approxDim == NumClasses - 1);
            if (Der.Get()) {
                CB_ENSURE((int)Der.GetColumnCount() == NumClasses - 1);
            }
            NKernel::MultiLogitValueAndDer(TargetClasses.Get(), NumClasses, TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), Predictions.AlignedColumnSize(), LoadPredictionIndices.Get(), FunctionValue.Get(), Der.Get(), Der.AlignedColumnSize(), stream.GetStream());
        }
    };

    class TMultiLogitSecondDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> TargetWeights;
        TCudaBufferPtr<const float> Predictions;
        int NumClasses;
        int RowIdx;
        TCudaBufferPtr<float> Der2;

    public:
        TMultiLogitSecondDerKernel() = default;

        TMultiLogitSecondDerKernel(TCudaBufferPtr<const float> targetClasses,
                                   TCudaBufferPtr<const float> targetWeights,
                                   TCudaBufferPtr<const float> predictions,
                                   int numClasses,
                                   int rowIdx,
                                   TCudaBufferPtr<float> der2)
                : TargetClasses(targetClasses)
                  , TargetWeights(targetWeights)
                  , Predictions(predictions)
                  , NumClasses(numClasses)
                  , RowIdx(rowIdx)
                  , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(TargetClasses, NumClasses, TargetWeights, Predictions,  Der2, RowIdx);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE((ui32)RowIdx <= Der2.GetColumnCount());
            NKernel::MultiLogitSecondDer(TargetClasses.Get(), NumClasses, TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), Predictions.AlignedColumnSize(),  Der2.Get(), RowIdx, Der2.AlignedColumnSize(), stream.GetStream());
        }
    };

}

template <class TMapping, class TFloat>
inline void MultiLogitValueAndDer(const TCudaBuffer<TFloat, TMapping>& target,
                                  const TCudaBuffer<TFloat, TMapping>& weights,
                                  const TCudaBuffer<TFloat, TMapping>& approx,
                                  const TCudaBuffer<ui32, TMapping>* loadPredictionsIndices,
                                  int numClasses,
                                  TCudaBuffer<float, TMapping>* score,
                                  TCudaBuffer<float, TMapping>* weightedDer,
                                  ui32 stream = 0) {
    using TKernel = NKernelHost::TMultiLogitValueAndDerKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(),
                           stream,
                           target,
                           weights,
                           approx,
                           loadPredictionsIndices,
                           numClasses,
                           score,
                           weightedDer);
}



template <class TMapping, class TFloat>
inline void MultiLogitSecondDerRow(const TCudaBuffer<TFloat, TMapping>& target,
                                   const TCudaBuffer<TFloat, TMapping>& weights,
                                   const TCudaBuffer<TFloat, TMapping>& approx,
                                   int numClasses,
                                   int rowIdx,
                                   TCudaBuffer<float, TMapping>* weightedDer2Row,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TMultiLogitSecondDerKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(),
                           stream,
                           target,
                           weights,
                           approx,
                           numClasses,
                           rowIdx,
                           weightedDer2Row);
}
