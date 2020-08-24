#pragma once

#include <library/cpp/grid_creator/binarization.h>


void BestSplit(
    // can be modified inside, should be TVector&& but SWIG does not support that
    TVector<float>* values,
    bool valuesSorted,
    // SWIG is unable to parse TMaybe definition, so pointer here is a poor substitution
    const TDefaultValue<float>* defaultValue,
    bool featureValuesMayContainNans,
    int maxBordersCount,
    EBorderSelectionType type,
    // SWIG is unable to parse TMaybe definition, so pointer here is a poor substitution
    const float* quantizedDefaultBinFraction,
    // SWIG is unable to parse TMaybe definition, so pointer here is a poor substitution
    const TVector<float>* initialBorders, // can be nullptr is initialBorders is not defined
    TVector<float>* outBorders,
    bool* outHasDefaultQuantizedBin,
    NSplitSelection::TDefaultQuantizedBin* outDefaultQuantizedBin
) throw (yexception);
