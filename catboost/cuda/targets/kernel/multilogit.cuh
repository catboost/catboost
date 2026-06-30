#pragma once


#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/private/libs/options/enums.h>

namespace NKernel {

    void MultiLogitValueAndDer(const float* targetClasses, int numClasses,
                               const float* targetWeights,
                               ui32 size,
                               const float* predictions, ui32 predictionsAlignSize,
                               const ui32* loadPredictionsIndices,
                               float* functionValue,
                               float* der, ui32 derAlignSize,
                               TCudaStream stream);

    void MultiLogitSecondDer(const float* targetClasses, int numClasses,
                             const float* targetWeights,
                             ui32 size,
                             const float* predictions, ui32 predictionsAlignSize,
                             float* der2,
                             int der2Row, ui32 der2AlignSize,
                             TCudaStream stream);


    void RMSEWithUncertaintyValueAndDer(const float* target,
                               const float* weights,
                               ui32 size,
                               const float* predictions, ui32 predictionsAlignSize,
                               const ui32* loadPredictionsIndices,
                               float* functionValue,
                               float* der, ui32 derAlignSize,
                               TCudaStream stream);

    void RMSEWithUncertaintySecondDer(const float* target,
                             const float* weights,
                             ui32 size,
                             const float* predictions, ui32 predictionsAlignSize,
                             float* der2,
                             int der2Row, ui32 der2AlignSize,
                             TCudaStream stream);

    void MultiClassOneVsAllValueAndDer(const float* targetClasses, int numClasses,
                                       const float* targetWeights,
                                       ui32 size,
                                       const float* predictions, ui32 predictionsAlignSize,
                                       const ui32* loadPredictionsIndices,
                                       float* functionValue,
                                       float* der, ui32 derAlignSize,
                                       TCudaStream stream);

    void MultiClassOneVsAllSecondDer(const float* targetClasses, int numClasses,
                                     const float* targetWeights,
                                     ui32 size,
                                     const float* predictions, ui32 predictionsAlignSize,
                                     float* der2,
                                     ui32 der2AlignSize,
                                     TCudaStream stream);

    void MultiCrossEntropyValueAndDer(
        ui32 targetCount,
        ui32 size,
        const float* target, ui32 targetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize,
        TCudaStream stream);

    void MultiCrossEntropySecondDer(
        ui32 targetCount,
        ui32 size,
        const float* target, ui32 targetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        float* der2,
        ui32 der2Row, ui32 der2AlignSize,
        TCudaStream stream);

    void MultiRMSEValueAndDer(
        ui32 targetCount,
        ui32 size,
        const float* target, ui32 targetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize,
        TCudaStream stream);

    void MultiRMSESecondDer(
        ui32 size,
        const float* weights,
        float* der2,
        ui32 der2Row, ui32 der2AlignSize,
        TCudaStream stream);



    void BuildConfusionMatrixBins(const float* targetClasses, int numClasses, ui32 size,
                                  const float* predictions, int predictionsDim,
                                  ui32 predictionsAlignSize,
                                  bool isBinClass,
                                  float binTargetProbabilityThreshold,
                                  ui32* bins,
                                  TCudaStream stream);

}
