#include "quantized_pool_analysis.h"

namespace NCB {

    template <class T, class THolderType, EFeatureType FeatureType>
    static void GetPredictionsOnVaryingFeature(
        const TFullModel& model,
        const size_t featureNum,
        const TVector<T>& featureValues,
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
        THolder<THolderType> initialHolder;
        if constexpr (FeatureType == EFeatureType::Float) {
            initialHolder = std::move(data.ObjectsData.FloatFeatures[featureNum]);
        }
        if constexpr (FeatureType == EFeatureType::Categorical) {
            initialHolder = std::move(data.ObjectsData.CatFeatures[featureNum]);
        }

        for (size_t numVal = 0; numVal < featureValues.size(); ++numVal) {
            if constexpr (FeatureType == EFeatureType::Categorical) {
                THolder<THashedCatValuesHolder> holder = MakeHolder<THashedCatValuesHolder>(
                    featureNum,
                    TMaybeOwningConstArrayHolder<ui32>::CreateOwning(TVector<ui32>(objectCount, featureValues[numVal])),
                    data.CommonObjectsData.SubsetIndexing.Get());
                data.ObjectsData.CatFeatures[featureNum] = std::move(holder);
            }
            if constexpr (FeatureType == EFeatureType::Float) {
                THolder<TFloatValuesHolder> holder = MakeHolder<TFloatValuesHolder>(
                    featureNum,
                    TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>(objectCount, featureValues[numVal])),
                    data.CommonObjectsData.SubsetIndexing.Get());
                data.ObjectsData.FloatFeatures[featureNum] = std::move(holder);
            }

            rawDataProviderPtr = MakeDataProvider<TRawObjectsDataProvider>(
                Nothing(),
                std::move(data),
                false,
                executor);

            auto pred = ApplyModel(model, *(rawDataProviderPtr->ObjectsData), false, predictionType);
            (*predictions)[numVal] = std::accumulate(pred.begin(), pred.end(), 0.) / static_cast<double>(pred.size());
            data = TRawBuilderDataHelper::Extract(std::move(*(rawDataProviderPtr.Get())));
        }

        if constexpr (FeatureType == EFeatureType::Categorical) {
            data.ObjectsData.CatFeatures[featureNum] = std::move(initialHolder);
        } else if (FeatureType == EFeatureType::Float) {
            data.ObjectsData.FloatFeatures[featureNum] = std::move(initialHolder);
        }

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

    TBinarizedFloatFeatureStatistics GetBinarizedFloatFeatureStatistics(
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
                ignoredFeatureNums.push_back(feature.FlatFeatureIndex);
            }
        }
        for (auto& feature : model.ObliviousTrees.CatFeatures) {
            ignoredFeatureNums.push_back(feature.FlatFeatureIndex);
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

        TQuantizedObjectsDataProviderPtr quantizedPtr = Quantize(
            options,
            rawObjectsDataProviderPtr,
            quantizedFeaturesInfo,
            &rand,
            &executor);

        TMaybeData<const IQuantizedFloatValuesHolder*> feature = quantizedPtr->GetFloatFeature(featureNum);
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
        GetPredictionsOnVaryingFeature<float, TFloatValuesHolder, EFeatureType::Float>(
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
            predictionsOnVarying
        };
    }

    TBinarizedOneHotFeatureStatistics GetBinarizedOneHotFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const EPredictionType predictionType) {

        NPar::TLocalExecutor executor;
        TRestorableFastRng64 rand(0);

        TVector<double> prediction = ApplyModel(model, dataset, false, predictionType, 0, 0, 1);

        TMaybeData<TConstArrayRef<float>> targetData = CreateModelCompatibleProcessedDataProvider(
            dataset, {}, model, &rand, &executor).TargetData->GetTarget();
        CB_ENSURE_INTERNAL(targetData, "No target found in pool");
        TVector<float> target(targetData.GetRef().begin(), targetData.GetRef().end());

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        TRawObjectsDataProviderPtr rawObjectsDataProviderPtr(objectsPtr);

        TVector<int> oneHotUniqueValues = model.ObliviousTrees.OneHotFeatures[featureNum].Values;
        TVector<TString> stringValues = model.ObliviousTrees.OneHotFeatures[featureNum].StringValues;
        stringValues.push_back("test_value");
        stringValues.push_back(model.ObliviousTrees.CatFeatures[featureNum].FeatureId);

        TMaybeData<const THashedCatValuesHolder*> catFeatureMaybe = \
            rawObjectsDataProviderPtr->GetCatFeature(featureNum);
        const THashedCatValuesHolder* catFeatureHolder = catFeatureMaybe.GetRef();
        TArrayRef<const ui32> featureValuesRef = *(*(catFeatureHolder->GetArrayData().GetSrc()));
        TVector<ui32> featureValues(featureValuesRef.begin(), featureValuesRef.end());

        TVector<int> binNums;
        binNums.reserve(featureValues.size());
        for (auto val : featureValues) {
            auto it = std::find(oneHotUniqueValues.begin(), oneHotUniqueValues.end(), val);
            binNums.push_back(it - oneHotUniqueValues.begin()); // Unknown values will be at bin #oneHotUniqueValues.size()
        }

        size_t numBins = oneHotUniqueValues.size() + 1;
        TVector<float> meanTarget(numBins, 0.);
        TVector<float> meanPrediction(numBins, 0.);
        TVector<size_t> countObjectsPerBin(numBins, 0);
        for (size_t numObj = 0; numObj < binNums.size(); ++numObj) {
            ui8 binNum = binNums[numObj];
            meanTarget[binNum] += target[numObj];
            meanPrediction[binNum] += prediction[numObj];
            countObjectsPerBin[binNum] += 1;
        }
        if (countObjectsPerBin[numBins - 1] == 0) {
            meanTarget.pop_back();
            meanPrediction.pop_back();
            countObjectsPerBin.pop_back();
        }
        for (size_t binNum = 0; binNum < numBins; ++binNum) {
            size_t numObjs = countObjectsPerBin[binNum];
            if (numObjs == 0) {
                continue;
            }
            meanTarget[binNum] /= static_cast<float>(numObjs);
            meanPrediction[binNum] /= static_cast<float>(numObjs);
        }

        TVector<double> predictionsOnVarying(numBins, 0.);
        GetPredictionsOnVaryingFeature<int, THashedCatValuesHolder, EFeatureType::Categorical>(
            model,
            featureNum,
            oneHotUniqueValues,
            predictionType,
            dataset,
            &predictionsOnVarying,
            &executor);

        return TBinarizedOneHotFeatureStatistics{
            binNums,
            meanTarget,
            meanPrediction,
            countObjectsPerBin,
            predictionsOnVarying
        };
    }

    ui32 GetCatFeaturePerfectHash(
            const TFullModel& model,
            const TStringBuf& value,
            const size_t featureNum) {

        int hash = static_cast<int>(CalcCatFeatureHash(value));
        const TVector<int>& oneHotUniqueValues = model.ObliviousTrees.OneHotFeatures[featureNum].Values;
        auto it = std::find(oneHotUniqueValues.begin(), oneHotUniqueValues.end(), hash);
        return it - oneHotUniqueValues.begin();
    }

    TFeatureTypeAndInternalIndex GetFeatureTypeAndInternalIndex(const TFullModel& model, const int flatFeatureIndex) {
        for (auto& feature: model.ObliviousTrees.FloatFeatures) {
            if (feature.FlatFeatureIndex == flatFeatureIndex) {
                return TFeatureTypeAndInternalIndex{EFeatureType::Float, feature.FeatureIndex};
            }
        }
        for (auto& feature: model.ObliviousTrees.CatFeatures) {
            if (feature.FlatFeatureIndex == flatFeatureIndex) {
                return TFeatureTypeAndInternalIndex{EFeatureType::Categorical, feature.FeatureIndex};
            }
        }
        CB_ENSURE(false, "Unsupported feature type");
    }

}
