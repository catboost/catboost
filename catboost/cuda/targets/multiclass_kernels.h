#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/targets/kernel/multilogit.cuh>

#include <util/stream/labeled.h>

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
            CB_ENSURE(approxDim == NumClasses - 1, LabeledOutput(approxDim, NumClasses - 1));
            if (Der.Get()) {
                CB_ENSURE((int)Der.GetColumnCount() == NumClasses - 1, LabeledOutput(Der.GetColumnCount(), NumClasses - 1));
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

        Y_SAVELOAD_DEFINE(TargetClasses, NumClasses, TargetWeights, Predictions, Der2, RowIdx);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE((ui32)RowIdx <= Der2.GetColumnCount(), LabeledOutput(RowIdx, Der2.GetColumnCount()));
            NKernel::MultiLogitSecondDer(TargetClasses.Get(), NumClasses, TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), Predictions.AlignedColumnSize(), Der2.Get(), RowIdx, Der2.AlignedColumnSize(), stream.GetStream());
        }
    };

    class TRMSEWithUncertaintyValueAndDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const ui32> LoadPredictionIndices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;

    public:
        TRMSEWithUncertaintyValueAndDerKernel() = default;

        TRMSEWithUncertaintyValueAndDerKernel(TCudaBufferPtr<const float> target,
                                     TCudaBufferPtr<const float> weights,
                                     TCudaBufferPtr<const float> predictions,
                                     TCudaBufferPtr<const ui32> loadPredictionIndices,
                                     TCudaBufferPtr<float> functionValue,
                                     TCudaBufferPtr<float> der)
            : Target(target)
            , Weights(weights)
            , Predictions(predictions)
            , LoadPredictionIndices(loadPredictionIndices)
            , FunctionValue(functionValue)
            , Der(der)
        {
        }

        Y_SAVELOAD_DEFINE(Target, Weights, Predictions, FunctionValue, Der, LoadPredictionIndices);

        void Run(const TCudaStream& stream) const {
            const int approxDim = static_cast<const int>(Predictions.GetColumnCount());
            CB_ENSURE(approxDim == 2, LabeledOutput(approxDim, 2));
            if (Der.Get()) {
                CB_ENSURE((int)Der.GetColumnCount() == 2, LabeledOutput(Der.GetColumnCount(), 2));
            }
            NKernel::RMSEWithUncertaintyValueAndDer(Target.Get(), Weights.Get(), Target.Size(), Predictions.Get(), Predictions.AlignedColumnSize(), LoadPredictionIndices.Get(), FunctionValue.Get(), Der.Get(), Der.AlignedColumnSize(), stream.GetStream());
        }
    };

    class TRMSEWithUncertaintySecondDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        int RowIdx;
        TCudaBufferPtr<float> Der2;

    public:
        TRMSEWithUncertaintySecondDerKernel() = default;

        TRMSEWithUncertaintySecondDerKernel(TCudaBufferPtr<const float> target,
                                   TCudaBufferPtr<const float> weights,
                                   TCudaBufferPtr<const float> predictions,
                                   int rowIdx,
                                   TCudaBufferPtr<float> der2)
            : Target(target)
            , Weights(weights)
            , Predictions(predictions)
            , RowIdx(rowIdx)
            , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(Target, Weights, Predictions, Der2, RowIdx);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE((ui32)RowIdx <= Der2.GetColumnCount(), LabeledOutput(RowIdx, Der2.GetColumnCount()));
            NKernel::RMSEWithUncertaintySecondDer(Target.Get(), Weights.Get(), Target.Size(), Predictions.Get(), Predictions.AlignedColumnSize(), Der2.Get(), RowIdx, Der2.AlignedColumnSize(), stream.GetStream());
        }
    };

    class TMultiCrossEntropyValueAndDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const ui32> LoadPredictionIndices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;

    public:
        TMultiCrossEntropyValueAndDerKernel() = default;

        TMultiCrossEntropyValueAndDerKernel(
            TCudaBufferPtr<const float> target,
            TCudaBufferPtr<const float> weights,
            TCudaBufferPtr<const float> predictions,
            TCudaBufferPtr<const ui32> loadPredictionIndices,
            TCudaBufferPtr<float> functionValue,
            TCudaBufferPtr<float> der
        )
            : Target(target)
            , Weights(weights)
            , Predictions(predictions)
            , LoadPredictionIndices(loadPredictionIndices)
            , FunctionValue(functionValue)
            , Der(der)
        {
        }

        Y_SAVELOAD_DEFINE(Target, Weights, Predictions, LoadPredictionIndices, FunctionValue, Der);

        void Run(const TCudaStream& stream) const {
            const auto approxDim = Predictions.GetColumnCount();
            const auto targetDim = Target.GetColumnCount();
            CB_ENSURE(approxDim == targetDim, LabeledOutput(approxDim, targetDim));
            if (Der.Get()) {
                CB_ENSURE(Der.GetColumnCount() == approxDim, LabeledOutput(Der.GetColumnCount(), approxDim));
            }
            NKernel::MultiCrossEntropyValueAndDer(
                targetDim,
                Target.Size(),
                Target.Get(), Target.AlignedColumnSize(),
                Weights.Get(),
                Predictions.Get(), Predictions.AlignedColumnSize(),
                LoadPredictionIndices.Get(),
                FunctionValue.Get(),
                Der.Get(), Der.AlignedColumnSize(),
                stream.GetStream());
        }
    };

    class TMultiCrossEntropySecondDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        ui32 RowIdx;
        TCudaBufferPtr<float> Der2;

    public:
        TMultiCrossEntropySecondDerKernel() = default;

        TMultiCrossEntropySecondDerKernel(
            TCudaBufferPtr<const float> target,
            TCudaBufferPtr<const float> weights,
            TCudaBufferPtr<const float> predictions,
            ui32 rowIdx,
            TCudaBufferPtr<float> der2
        )
            : Target(target)
            , Weights(weights)
            , Predictions(predictions)
            , RowIdx(rowIdx)
            , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(Target, Weights, Predictions, RowIdx, Der2);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(
                Target.GetColumnCount() == Predictions.GetColumnCount(),
                LabeledOutput(Target.GetColumnCount(), Predictions.GetColumnCount()));
            CB_ENSURE(RowIdx <= Der2.GetColumnCount(), LabeledOutput(RowIdx, Der2.GetColumnCount()));
            NKernel::MultiCrossEntropySecondDer(
                Target.GetColumnCount(),
                Target.Size(),
                Target.Get(), Target.AlignedColumnSize(),
                Weights.Get(),
                Predictions.Get(), Predictions.AlignedColumnSize(),
                Der2.Get(),
                RowIdx, Der2.AlignedColumnSize(),
                stream.GetStream());
        }
    };

    class TMultiRMSEValueAndDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const ui32> LoadPredictionIndices;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;

    public:
        TMultiRMSEValueAndDerKernel() = default;

        TMultiRMSEValueAndDerKernel(
            TCudaBufferPtr<const float> target,
            TCudaBufferPtr<const float> weights,
            TCudaBufferPtr<const float> predictions,
            TCudaBufferPtr<const ui32> loadPredictionIndices,
            TCudaBufferPtr<float> functionValue,
            TCudaBufferPtr<float> der
        )
            : Target(target)
            , Weights(weights)
            , Predictions(predictions)
            , LoadPredictionIndices(loadPredictionIndices)
            , FunctionValue(functionValue)
            , Der(der)
        {
        }

        Y_SAVELOAD_DEFINE(Target, Weights, Predictions, LoadPredictionIndices, FunctionValue, Der);

        void Run(const TCudaStream& stream) const {
            const auto approxDim = Predictions.GetColumnCount();
            const auto targetDim = Target.GetColumnCount();
            CB_ENSURE(approxDim == targetDim, LabeledOutput(approxDim, targetDim));
            if (Der.Get()) {
                CB_ENSURE(Der.GetColumnCount() == approxDim, LabeledOutput(Der.GetColumnCount(), approxDim));
            }
            NKernel::MultiRMSEValueAndDer(
                targetDim,
                Target.Size(),
                Target.Get(), Target.AlignedColumnSize(),
                Weights.Get(),
                Predictions.Get(), Predictions.AlignedColumnSize(),
                LoadPredictionIndices.Get(),
                FunctionValue.Get(),
                Der.Get(), Der.AlignedColumnSize(),
                stream.GetStream());
        }
    };

    class TMultiRMSESecondDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Weights;
        ui32 RowIdx;
        TCudaBufferPtr<float> Der2;

    public:
        TMultiRMSESecondDerKernel() = default;

        TMultiRMSESecondDerKernel(
            TCudaBufferPtr<const float> weights,
            ui32 rowIdx,
            TCudaBufferPtr<float> der2
        )
            : Weights(weights)
            , RowIdx(rowIdx)
            , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(Weights, RowIdx, Der2);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(RowIdx <= Der2.GetColumnCount(), LabeledOutput(RowIdx, Der2.GetColumnCount()));
            NKernel::MultiRMSESecondDer(
                Weights.Size(),
                Weights.Get(),
                Der2.Get(),
                RowIdx, Der2.AlignedColumnSize(),
                stream.GetStream());
        }
    };

    class TMultiClassOneVsAllValueAndDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> TargetWeights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const ui32> LoadPredictionIndices;
        int NumClasses;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;

    public:
        TMultiClassOneVsAllValueAndDerKernel() = default;

        TMultiClassOneVsAllValueAndDerKernel(TCudaBufferPtr<const float> targetClasses,
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
            if (Der.Get()) {
                CB_ENSURE((int)Der.GetColumnCount() == NumClasses, LabeledOutput(Der.GetColumnCount(), NumClasses));
            }
            CB_ENSURE((int)Predictions.GetColumnCount() == NumClasses, LabeledOutput(Predictions.GetColumnCount(), NumClasses));

            NKernel::MultiClassOneVsAllValueAndDer(TargetClasses.Get(), NumClasses, TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), Predictions.AlignedColumnSize(), LoadPredictionIndices.Get(), FunctionValue.Get(), Der.Get(), Der.AlignedColumnSize(), stream.GetStream());
        }
    };

    class TMultiClassOneVsAllSecondDerKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> TargetWeights;
        TCudaBufferPtr<const float> Predictions;
        int NumClasses;
        TCudaBufferPtr<float> Der2;

    public:
        TMultiClassOneVsAllSecondDerKernel() = default;

        TMultiClassOneVsAllSecondDerKernel(TCudaBufferPtr<const float> targetClasses,
                                           TCudaBufferPtr<const float> targetWeights,
                                           TCudaBufferPtr<const float> predictions,
                                           int numClasses,
                                           TCudaBufferPtr<float> der2)
            : TargetClasses(targetClasses)
            , TargetWeights(targetWeights)
            , Predictions(predictions)
            , NumClasses(numClasses)
            , Der2(der2)
        {
        }

        Y_SAVELOAD_DEFINE(TargetClasses, NumClasses, TargetWeights, Predictions, Der2);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Der2.GetColumnCount() == Predictions.GetColumnCount(), LabeledOutput(Der2.GetColumnCount(), Predictions.GetColumnCount()));
            NKernel::MultiClassOneVsAllSecondDer(TargetClasses.Get(), NumClasses, TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), Predictions.AlignedColumnSize(), Der2.Get(), Der2.AlignedColumnSize(), stream.GetStream());
        }
    };

    class TBuildConfusionMatrixKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> Predictions;
        int NumClasses;
        bool IsBinClass;
        float BinTargetProbabilityThreshold;
        TCudaBufferPtr<ui32> Bins;

    public:
        TBuildConfusionMatrixKernel() = default;

        TBuildConfusionMatrixKernel(TCudaBufferPtr<const float> targetClasses,
                                    TCudaBufferPtr<const float> predictions,
                                    int numClasses,
                                    bool isBinClass,
                                    float binTargetProbabilityThreshold,
                                    TCudaBufferPtr<ui32> bins)
            : TargetClasses(targetClasses)
            , Predictions(predictions)
            , NumClasses(numClasses)
            , IsBinClass(isBinClass)
            , BinTargetProbabilityThreshold(binTargetProbabilityThreshold)
            , Bins(bins)
        {
        }

        Y_SAVELOAD_DEFINE(TargetClasses, NumClasses, Predictions, IsBinClass, BinTargetProbabilityThreshold, Bins);

        void Run(const TCudaStream& stream) const {
            NKernel::BuildConfusionMatrixBins(TargetClasses.Get(), NumClasses, TargetClasses.Size(), Predictions.Get(), Predictions.GetColumnCount(), Predictions.AlignedColumnSize(), IsBinClass, BinTargetProbabilityThreshold, Bins.Get(), stream.GetStream());
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

template <class TMapping, class TFloat>
inline void RMSEWithUncertaintyValueAndDer(const TCudaBuffer<TFloat, TMapping>& target,
                                  const TCudaBuffer<TFloat, TMapping>& weights,
                                  const TCudaBuffer<TFloat, TMapping>& approx,
                                  const TCudaBuffer<ui32, TMapping>* loadPredictionsIndices,
                                  TCudaBuffer<float, TMapping>* score,
                                  TCudaBuffer<float, TMapping>* weightedDer,
                                  ui32 stream = 0) {
    using TKernel = NKernelHost::TRMSEWithUncertaintyValueAndDerKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(),
                           stream,
                           target,
                           weights,
                           approx,
                           loadPredictionsIndices,
                           score,
                           weightedDer);
}

template <class TMapping, class TFloat>
inline void RMSEWithUncertaintySecondDerRow(const TCudaBuffer<TFloat, TMapping>& target,
                                   const TCudaBuffer<TFloat, TMapping>& weights,
                                   const TCudaBuffer<TFloat, TMapping>& approx,
                                   int rowIdx,
                                   TCudaBuffer<float, TMapping>* weightedDer2Row,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TRMSEWithUncertaintySecondDerKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(),
                           stream,
                           target,
                           weights,
                           approx,
                           rowIdx,
                           weightedDer2Row);
}

template <class TMapping, class TFloat>
inline void MultiCrossEntropyValueAndDer(
    const TCudaBuffer<TFloat, TMapping>& target,
    const TCudaBuffer<TFloat, TMapping>& weights,
    const TCudaBuffer<TFloat, TMapping>& approx,
    const TCudaBuffer<ui32, TMapping>* loadPredictionsIndices,
    TCudaBuffer<float, TMapping>* score,
    TCudaBuffer<float, TMapping>* weightedDer,
    ui32 stream = 0
) {
    using TKernel = NKernelHost::TMultiCrossEntropyValueAndDerKernel;
    LaunchKernels<TKernel>(
        target.NonEmptyDevices(),
        stream,
        target,
        weights,
        approx,
        loadPredictionsIndices,
        score,
        weightedDer);
}

template <class TMapping, class TFloat>
inline void MultiCrossEntropySecondDerRow(
    const TCudaBuffer<TFloat, TMapping>& target,
    const TCudaBuffer<TFloat, TMapping>& weights,
    const TCudaBuffer<TFloat, TMapping>& approx,
    ui32 rowIdx,
    TCudaBuffer<float, TMapping>* weightedDer2Row,
    ui32 stream = 0
) {
    using TKernel = NKernelHost::TMultiCrossEntropySecondDerKernel;
    LaunchKernels<TKernel>(
        target.NonEmptyDevices(),
        stream,
        target,
        weights,
        approx,
        rowIdx,
        weightedDer2Row);
}

template <class TMapping, class TFloat>
inline void MultiRMSEValueAndDer(
    const TCudaBuffer<TFloat, TMapping>& target,
    const TCudaBuffer<TFloat, TMapping>& weights,
    const TCudaBuffer<TFloat, TMapping>& approx,
    const TCudaBuffer<ui32, TMapping>* loadPredictionsIndices,
    TCudaBuffer<float, TMapping>* score,
    TCudaBuffer<float, TMapping>* weightedDer,
    ui32 stream = 0
) {
    using TKernel = NKernelHost::TMultiRMSEValueAndDerKernel;
    LaunchKernels<TKernel>(
        target.NonEmptyDevices(),
        stream,
        target,
        weights,
        approx,
        loadPredictionsIndices,
        score,
        weightedDer);
}

template <class TMapping, class TFloat>
inline void MultiRMSESecondDerRow(
    const TCudaBuffer<TFloat, TMapping>& target,
    const TCudaBuffer<TFloat, TMapping>& weights,
    ui32 rowIdx,
    TCudaBuffer<float, TMapping>* weightedDer2Row,
    ui32 stream = 0
) {
    using TKernel = NKernelHost::TMultiRMSESecondDerKernel;
    LaunchKernels<TKernel>(
        target.NonEmptyDevices(),
        stream,
        weights,
        rowIdx,
        weightedDer2Row);
}

template <class TMapping, class TFloat>
inline void MultiClassOneVsAllValueAndDer(const TCudaBuffer<TFloat, TMapping>& target,
                                          const TCudaBuffer<TFloat, TMapping>& weights,
                                          const TCudaBuffer<TFloat, TMapping>& approx,
                                          const TCudaBuffer<ui32, TMapping>* loadPredictionsIndices,
                                          int numClasses,
                                          TCudaBuffer<float, TMapping>* score,
                                          TCudaBuffer<float, TMapping>* weightedDer,
                                          ui32 stream = 0) {
    using TKernel = NKernelHost::TMultiClassOneVsAllValueAndDerKernel;
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
inline void MultiClassOneVsAllSecondDer(const TCudaBuffer<TFloat, TMapping>& target,
                                        const TCudaBuffer<TFloat, TMapping>& weights,
                                        const TCudaBuffer<TFloat, TMapping>& approx,
                                        int numClasses,
                                        TCudaBuffer<float, TMapping>* weightedDer2Row,
                                        ui32 stream = 0) {
    using TKernel = NKernelHost::TMultiClassOneVsAllSecondDerKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(),
                           stream,
                           target,
                           weights,
                           approx,
                           numClasses,
                           weightedDer2Row);
}

template <class TMapping, class TFloat>
inline void BuildConfusionMatrix(const TCudaBuffer<TFloat, TMapping>& target,
                                 const TCudaBuffer<TFloat, TMapping>& approx,
                                 int numClasses,
                                 bool isBinClass,
                                 float binTargetProbabilityThreshold,
                                 TCudaBuffer<ui32, TMapping>* bins,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TBuildConfusionMatrixKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(),
                           stream,
                           target,
                           approx,
                           numClasses,
                           isBinClass,
                           binTargetProbabilityThreshold,
                           bins);
}
