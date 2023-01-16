%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/quantization.h>
%}

%include "catboost_enums.i"


%catches(yexception) PrepareQuantizationParameters(
    const NCB::TFeaturesLayout& featuresLayout,
    const TString& plainJsonParamsAsString
);

%catches(yexception) TNanModeAndBordersBuilder::TNanModeAndBordersBuilder(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
);

%catches(std::exception) TNanModeAndBordersBuilder::SetSampleSize(i32 sampleSize);
%catches(std::exception) TNanModeAndBordersBuilder::AddSample(TConstArrayRef<double> objectData);

%catches(yexception, std::exception) TNanModeAndBordersBuilder::CalcBordersWithoutNans(i32 threadCount);

%catches(yexception, std::exception) TNanModeAndBordersBuilder::Finish(TConstArrayRef<i8> hasNans);

%catches(yexception) Quantize(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TRawObjectsDataProviderPtr* rawObjectsDataProvider, // moved into
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) GetActiveFeaturesIndices(
    NCB::TFeaturesLayoutPtr featuresLayout,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TVector<i32>* ui8FlatIndices,
    TVector<i32>* ui16FlatIndices,
    TVector<i32>* ui32FlatIndices
);

%include <catboost/spark/catboost4j-spark/core/src/native_impl/quantization.h>
