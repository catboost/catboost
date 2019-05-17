#pragma once

#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/index_calcer.h>
#include <catboost/libs/data_new/quantization.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/data_provider_builders.cpp>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/target/data_providers.h>

#include <util/generic/vector.h>


namespace NCB {

    struct TBinarizedFloatFeatureStatistics {
        TVector<float> Borders;
        TVector<ui8> BinarizedFeature;
        TVector<float> MeanTarget;
        TVector<float> MeanPrediction;
        TVector<size_t> ObjectsPerBin;
        TVector<float> Target;          // for debugging,
        TVector<double> Prediction;     // will remove in further versions
        TVector<double> PredictionsOnVaryingFeature;
    };

    void GetPredictionsOnVaryingFeature(
        const TFullModel& model,
        const size_t featureNum,
        const TVector<float>& featureValues,
        const EPredictionType predictionType,
        TDataProvider& dataProvider,
        TVector<double>* predictions,
        NPar::TLocalExecutor* executor) {

        TRawDataProvider rawDataProvider(
            std::move(dataProvider.MetaInfo),
            std::move(dynamic_cast<TRawObjectsDataProvider*>(dataProvider.ObjectsData.Get())),
            dataProvider.ObjectsGrouping,
            std::move(dataProvider.RawTargetData));
        size_t objectCount = rawDataProvider.GetObjectCount();
        TIntrusivePtr<TRawDataProvider> rawDataProviderPtr(&rawDataProvider);
        TRawBuilderData data = TRawBuilderDataHelper::Extract(std::move(*(rawDataProviderPtr.Get())));
        auto initialHolder = std::move(data.ObjectsData.FloatFeatures[featureNum]);

        for (size_t numVal = 0; numVal < featureValues.size(); ++numVal) {
            THolder<TFloatValuesHolder> holder = MakeHolder<TFloatValuesHolder>(
                featureNum,
                TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>(objectCount, featureValues[numVal])),
                data.CommonObjectsData.SubsetIndexing.Get());
            data.ObjectsData.FloatFeatures[featureNum] = std::move(holder);

            rawDataProviderPtr = MakeDataProvider<TRawObjectsDataProvider>(
                Nothing(),
                std::move(data),
                false,
                executor);

            auto pred = ApplyModel(model, *(rawDataProviderPtr->ObjectsData), false, predictionType);
            (*predictions)[numVal] = std::accumulate(pred.begin(), pred.end(), 0.) / static_cast<double>(pred.size());
            data = TRawBuilderDataHelper::Extract(std::move(*(rawDataProviderPtr.Get())));
        }

        data.ObjectsData.FloatFeatures[featureNum] = std::move(initialHolder);
        rawDataProviderPtr = MakeDataProvider<TRawObjectsDataProvider>(
            Nothing(),
            std::move(data),
            false,
            executor);
        rawDataProvider = *(rawDataProviderPtr.Get());

        dataProvider = TDataProvider(
            std::move(rawDataProvider.MetaInfo),
            std::move(rawDataProvider.ObjectsData),
            rawDataProvider.ObjectsGrouping,
            std::move(rawDataProvider.RawTargetData));
    }


    TBinarizedFloatFeatureStatistics GetBinarizedStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const EPredictionType predictionType) {

        NPar::TLocalExecutor executor;
        TRestorableFastRng64 rand(0);

        TVector<float> borders = model.ObliviousTrees.FloatFeatures[featureNum].Borders;
        size_t bordersSize = borders.size();
        TVector<double> prediction = ApplyModel(model, dataset, false, predictionType, 0, 0, 1);

        TMaybeData<TConstArrayRef<float>> targetData = CreateModelCompatibleProcessedDataProvider(
                dataset, {}, model, &rand, &executor).TargetData->GetTarget();
        CB_ENSURE_INTERNAL(targetData, "No target found in pool");
        TVector<float> target(targetData.GetRef().begin(), targetData.GetRef().end());

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        TRawObjectsDataProviderPtr rawObjectsDataProviderPtr(objectsPtr);

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
        quantizedFeaturesInfo->SetNanMode(TFloatFeatureIdx(featureNum),
            model.ObliviousTrees.FloatFeatures[featureNum].HasNans ? ENanMode::Min : ENanMode::Forbidden);

        TQuantizationOptions options;

        TQuantizedObjectsDataProviderPtr ptr = Quantize(
            options,
            rawObjectsDataProviderPtr,
            quantizedFeaturesInfo,
            &rand,
            &executor);

        TMaybeData<const IQuantizedFloatValuesHolder*> feature = ptr->GetFloatFeature(featureNum);
        CB_ENSURE_INTERNAL(feature, "Float feature #" << featureNum << " not found");
        const TQuantizedFloatValuesHolder* values = dynamic_cast<const TQuantizedFloatValuesHolder*>(feature.GetRef());
        CB_ENSURE_INTERNAL(values, "Cannot access values of float feature #" << featureNum);
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
        for (size_t binNum = 0; binNum < bordersSize + 1; ++binNum) {
            size_t numObjs = countObjectsPerBin[binNum];
            if (numObjs == 0) {
                continue;
            }
            meanTarget[binNum] /= static_cast<float>(numObjs);
            meanPrediction[binNum] /= static_cast<float>(numObjs);
        }

        TVector<double> predictionsOnVarying(bordersSize, 0.);
        GetPredictionsOnVaryingFeature(
            model,
            featureNum,
            quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)),
            predictionType,
            dataset,
            &predictionsOnVarying,
            &executor);

        return TBinarizedFloatFeatureStatistics{
            quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)),
            binNums,
            meanTarget,
            meanPrediction,
            countObjectsPerBin,
            target,
            prediction,
            predictionsOnVarying
        };
    }

}
