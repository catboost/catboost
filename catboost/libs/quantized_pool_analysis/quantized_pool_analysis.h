#pragma once

#include <catboost/libs/algo/index_calcer.h>
#include <catboost/libs/data_new/quantization.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <util/generic/vector.h>

struct TBinarizedFloatFeatureStatistics {
    TVector<float> Borders;
    TVector<ui8> BinarizedFeature;
    TVector<float> MeanTarget;
    TVector<float> MeanPrediction;
    TVector<size_t> ObjectsPerBin;
};

namespace NCB {

    TBinarizedFloatFeatureStatistics GetBinarizedStatistics(
        const TFullModel& model,
        const TDataProvider& dataset,
        const TVector<float>& target,
        const TVector<float>& prediction,
        const size_t featureNum) {
        TVector<float> borders = model.ObliviousTrees.FloatFeatures[featureNum].Borders;

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE(objectsPtr);
        TRawObjectsDataProviderPtr rawObjectsDataProvider(objectsPtr);

        TVector<ui32> ignoredFeatureNums;

        for (auto& feature : model.ObliviousTrees.FloatFeatures) {
            if (feature.FeatureIndex != -1 && static_cast<ui32>(feature.FeatureIndex) != featureNum) {
                ignoredFeatureNums.push_back(feature.FeatureIndex);
            }
        }
        for (auto& feature : model.ObliviousTrees.CatFeatures) {
            if (feature.FeatureIndex != -1 && static_cast<ui32>(feature.FeatureIndex) != featureNum) {
                ignoredFeatureNums.push_back(feature.FeatureIndex);
            }
        }

        TConstArrayRef<ui32> ignoredFeatures = MakeConstArrayRef(ignoredFeatureNums);
        TFeaturesLayoutPtr featuresLayout = dataset.MetaInfo.FeaturesLayout;
        NCatboostOptions::TBinarizationOptions commonFloatFeaturesBinarization;
        auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *(featuresLayout.Get()),
            ignoredFeatures,
            commonFloatFeaturesBinarization);
        quantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(featureNum), std::move(borders));
        quantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(featureNum), ENanMode::Forbidden);

        TQuantizationOptions options;
        TRestorableFastRng64 rand(0);
        NPar::TLocalExecutor executor;

        TQuantizedObjectsDataProviderPtr ptr = Quantize(
            options,
            rawObjectsDataProvider,
            quantizedFeaturesInfo,
            &rand,
            &executor);

        TMaybeData<const IQuantizedFloatValuesHolder*> feature = ptr->GetFloatFeature(featureNum);
        CB_ENSURE(feature);
        const TQuantizedFloatValuesHolder* values = dynamic_cast<const TQuantizedFloatValuesHolder*>(feature.GetRef());
        CB_ENSURE(values);
        TArrayRef extractedValues = *(values->ExtractValues(&executor));
        TVector<ui8> binNums(extractedValues.begin(), extractedValues.end());

        size_t numBins = quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)).size() + 1;
        TVector<float> meanTarget(numBins, 0.);
        TVector<float> meanPrediction(numBins, 0.);
        TVector<size_t> countObjectsPerBin(numBins, 0);
        for (size_t numObj = 0; numObj < binNums.size(); ++numObj) {
            ui8 binNum = binNums[numObj];
            meanTarget[binNum] += target[numObj];
            meanPrediction[binNum] += prediction[numObj];
            countObjectsPerBin[binNum] += 1;
        }
        for (size_t binNum = 0; binNum < borders.size() + 1; ++binNum) {
            size_t numObjs = countObjectsPerBin[binNum];
            if (numObjs == 0) {
                continue;
            }
            meanTarget[binNum] /= static_cast<float>(numObjs);
            meanPrediction[binNum] /= static_cast<float>(numObjs);
        }

        return TBinarizedFloatFeatureStatistics{
            quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)),
            binNums,
            meanTarget,
            meanPrediction,
            countObjectsPerBin
        };
    }
}
