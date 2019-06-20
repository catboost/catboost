#include "quantized_pool_analysis.h"

namespace NCB {

    template <class T, class THolderType, EFeatureType FeatureType>
    static void GetPredictionsOnVaryingFeature(
        const TFullModel& model,
        const size_t featureNum,
        const TVector<T>& featureValues,
        const EPredictionType predictionType,
        const int threadCount,
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
        TRawBuilderData data = TRawBuilderDataHelper::Extract(std::move(*(rawDataProviderPtr.Release())));

        THolder<THolderType> initialHolder;
        if constexpr (FeatureType == EFeatureType::Categorical) {
            initialHolder = std::move(data.ObjectsData.CatFeatures[featureNum]);
        } else {
            CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
            initialHolder = std::move(data.ObjectsData.FloatFeatures[featureNum]);
        }

        for (size_t numVal = 0; numVal < featureValues.size(); ++numVal) {
            if constexpr (FeatureType == EFeatureType::Categorical) {
                THolder<THashedCatValuesHolder> holder = MakeHolder<THashedCatValuesHolder>(
                    featureNum,
                    TMaybeOwningConstArrayHolder<ui32>::CreateOwning(TVector<ui32>(objectCount, featureValues[numVal])),
                    data.CommonObjectsData.SubsetIndexing.Get());
                data.ObjectsData.CatFeatures[featureNum] = std::move(holder);
            } else {
                CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
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

            auto pred = ApplyModel(model, *(rawDataProviderPtr->ObjectsData), false, predictionType, 0, 0, threadCount);
            (*predictions)[numVal] = std::accumulate(pred.begin(), pred.end(), 0.) / static_cast<double>(pred.size());
            data = TRawBuilderDataHelper::Extract(std::move(*(rawDataProviderPtr.Release())));
        }

        if constexpr (FeatureType == EFeatureType::Categorical) {
            data.ObjectsData.CatFeatures[featureNum] = std::move(initialHolder);
        } else {
            CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
            data.ObjectsData.FloatFeatures[featureNum] = std::move(initialHolder);
        }

        rawDataProviderPtr = MakeDataProvider<TRawObjectsDataProvider>(
            Nothing(),
            std::move(data),
            false,
            executor);
        rawDataProvider = *(rawDataProviderPtr.Release());

        dataProvider = TDataProvider(
            std::move(rawDataProvider.MetaInfo),
            std::move(rawDataProvider.ObjectsData),
            rawDataProvider.ObjectsGrouping,
            std::move(rawDataProvider.RawTargetData));
    }

    TBinarizedFeatureStatistics GetBinarizedFloatFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const TVector<double>& prediction,
        const EPredictionType predictionType,
        const int threadCount) {
        CB_ENSURE_INTERNAL(!model.ObliviousTrees.FloatFeatures[featureNum].HasNans,
            "Features with nan values not supported");

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(threadCount - 1);
        TRestorableFastRng64 rand(0);

        TVector<float> borders = model.ObliviousTrees.FloatFeatures[featureNum].Borders;
        size_t bordersSize = borders.size();

        TProcessedDataProvider processedDataProvider = CreateModelCompatibleProcessedDataProvider(
            dataset, {}, model, &rand, &executor);
        TVector<float> target(
            processedDataProvider.TargetData->GetTarget().GetRef().begin(),
            processedDataProvider.TargetData->GetTarget().GetRef().end());

        if (bordersSize == 0) {
            return {};
        }

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
        TMaybeOwningArrayHolder<ui8> extractedValues = values->ExtractValues(&executor);
        TVector<int> binNums((*extractedValues).begin(), (*extractedValues).end());

        size_t numBins = quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)).size() + 1;
        TVector<float> meanTarget(numBins, 0.);
        TVector<float> meanPrediction(numBins, 0.);
        TVector<size_t> countObjectsPerBin(numBins, 0);

        for (size_t numObj = 0; numObj < binNums.size(); ++numObj) {
            int binNum = binNums[numObj];
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
            threadCount,
            dataset,
            &predictionsOnVarying,
            &executor);

        return TBinarizedFeatureStatistics{
            quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)),
            binNums,
            meanTarget,
            meanPrediction,
            countObjectsPerBin,
            predictionsOnVarying
        };
    }

    int GetOneHotFeatureFlatNum(const TFullModel& model, const size_t featureNum) {
        int featureFlatNum = -1;
        for(auto & feature: model.ObliviousTrees.OneHotFeatures) {
            const int distToNearestCatFeature = featureNum - feature.CatFeatureIndex;
            ++featureFlatNum;
            if (distToNearestCatFeature < 0) {
                return -1;
            }
            else{
                if (distToNearestCatFeature == 0) {
                    return featureFlatNum;
                }
            }
        }
        return -1;
    }

    TBinarizedFeatureStatistics GetBinarizedOneHotFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const int featureFlatNum,
        const TVector<double>& prediction,
        const EPredictionType predictionType,
        const int threadCount) {

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(threadCount - 1);
        TRestorableFastRng64 rand(0);

        TProcessedDataProvider processedDataProvider = CreateModelCompatibleProcessedDataProvider(
            dataset, {}, model, &rand, &executor);
        TVector<float> target(
            processedDataProvider.TargetData->GetTarget().GetRef().begin(),
            processedDataProvider.TargetData->GetTarget().GetRef().end());

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        TRawObjectsDataProviderPtr rawObjectsDataProviderPtr(objectsPtr);

        if (model.ObliviousTrees.OneHotFeatures.empty()) {
            return {};
        }

        const TVector<int>& oneHotUniqueValues = model.ObliviousTrees.OneHotFeatures[featureFlatNum].Values;
        TMaybeData<const THashedCatValuesHolder*> catFeatureMaybe = \
            rawObjectsDataProviderPtr->GetCatFeature(featureNum);
        CB_ENSURE_INTERNAL(catFeatureMaybe, "Categorical feature #" << featureNum << " not found");
        const THashedCatValuesHolder* catFeatureHolder = catFeatureMaybe.GetRef();
        CB_ENSURE_INTERNAL(catFeatureHolder, "Cannot access values of categorical feature #" << featureNum);
        TArrayRef<const ui32> featureValuesRef = *(*(catFeatureHolder->GetArrayData().GetSrc()));
        TVector<ui32> featureValues(featureValuesRef.begin(), featureValuesRef.end());

        TVector<int> binNums;
        binNums.reserve(oneHotUniqueValues.size());
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
            --numBins;
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
            threadCount,
            dataset,
            &predictionsOnVarying,
            &executor);

        return TBinarizedFeatureStatistics{
            TVector<float>(),
            binNums,
            meanTarget,
            meanPrediction,
            countObjectsPerBin,
            predictionsOnVarying
        };
    }

    TBinarizedFeatureStatistics GetBinarizedCatFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const TVector<double>& prediction,
        const EPredictionType predictionType,
        const int threadCount) {
        const int featureFlatNum = GetOneHotFeatureFlatNum(model, featureNum);
        CB_ENSURE_INTERNAL(
            featureFlatNum != -1,
            "Binarized statistics supported only for one-hot encoded features. Use one_hot_max_size when training to manage that."
        );
        return GetBinarizedOneHotFeatureStatistics(
            model,
            dataset,
            featureNum,
            featureFlatNum,
            prediction,
            predictionType,
            threadCount
        );
    }

    TVector<TBinarizedFeatureStatistics> GetBinarizedStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const TVector<size_t>& catFeaturesNums,
        const TVector<size_t>& floatFeaturesNums,
        const EPredictionType predictionType,
        const int threadCount) {

        const TVector<double> prediction = ApplyModel(model, dataset, false, predictionType, 0, 0, threadCount);
        TVector<TBinarizedFeatureStatistics> statistics;

        for (const auto catFeatureNum: catFeaturesNums) {
            statistics.push_back(
                GetBinarizedCatFeatureStatistics(
                    model, dataset,
                    catFeatureNum,
                    prediction,
                    predictionType,
                    threadCount
                )
            );
        }

        for (const auto floatFeatureNum: floatFeaturesNums) {
            statistics.push_back(
                GetBinarizedFloatFeatureStatistics(
                    model, dataset,
                    floatFeatureNum,
                    prediction,
                    predictionType,
                    threadCount
                )
            );
        }

        return statistics;
    }

    ui32 GetCatFeaturePerfectHash(
            const TFullModel& model,
            const TStringBuf& value,
            const size_t featureNum) {

        int hash = static_cast<int>(CalcCatFeatureHash(value));
        if (model.ObliviousTrees.OneHotFeatures.empty()) {
            return 0;
        }
        const int featureFlatNum = GetOneHotFeatureFlatNum(model, featureNum);

        if (featureFlatNum == -1) {
            return 0;
        }
        const TVector<int>& oneHotUniqueValues = model.ObliviousTrees.OneHotFeatures[featureFlatNum].Values;
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

    TVector<TString> GetCatFeatureValues(const TDataProvider& dataset, const size_t flatFeatureIndex) {
        auto featureIdx = dataset.ObjectsData->GetFeaturesLayout()->GetInternalFeatureIdx(flatFeatureIndex);
        const auto& hashToString = dataset.ObjectsData->GetCatFeaturesHashToString(featureIdx);

        TVector<TString> stringValues;
        stringValues.reserve(hashToString.size());
        for (const auto& [value, stringValue] : hashToString) {
            stringValues.push_back(stringValue);
        }

        return stringValues;
    }

}
