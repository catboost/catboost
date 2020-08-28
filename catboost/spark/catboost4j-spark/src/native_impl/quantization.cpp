#include "quantization.h"


void BestSplit(
    TVector<float>* values,
    bool valuesSorted,
    const TDefaultValue<float>* defaultValue,
    bool featureValuesMayContainNans,
    int maxBordersCount,
    EBorderSelectionType type,
    const float* quantizedDefaultBinFraction,
    const TVector<float>* initialBorders,
    TVector<float>* outBorders,
    bool* outHasDefaultQuantizedBin,
    NSplitSelection::TDefaultQuantizedBin* outDefaultQuantizedBin
)  throw (yexception) {
    NSplitSelection::TFeatureValues featureValues(
        std::move(*values),
        valuesSorted,
        defaultValue ? TMaybe<TDefaultValue<float>>(*defaultValue) : Nothing()
    );

    NSplitSelection::TQuantization quantization = NSplitSelection::BestSplit(
        std::move(featureValues),
        featureValuesMayContainNans,
        maxBordersCount,
        type,
        quantizedDefaultBinFraction ? TMaybe<float>(*quantizedDefaultBinFraction) : Nothing(),
        initialBorders ? TMaybe<TVector<float>>(*initialBorders) : Nothing()
    );

    *outBorders = std::move(quantization.Borders);

    if (quantization.DefaultQuantizedBin) {
        *outHasDefaultQuantizedBin = true;
        *outDefaultQuantizedBin = std::move(*quantization.DefaultQuantizedBin);
    } else {
        *outHasDefaultQuantizedBin = false;
    }
}
