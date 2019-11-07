#include "quantized_pool_analysis.h"

#include <catboost/libs/helpers/mem_usage.h>

namespace NCB {

    template <class T, class THolderType, EFeatureType FeatureType>
    static void GetPredictionsOnVaryingFeature(
        const TFullModel& model,
        const size_t featureNum,
        const TVector<size_t>& ignoredValues, // for cat features, ordered
        const TVector<T>& featureValues,
        const EPredictionType predictionType,
        const int threadCount,
        TDataProvider& dataProvider,
        TVector<double>* predictions,
        NPar::TLocalExecutor* executor) {

        size_t objectCount = dataProvider.GetObjectCount();

        TIntrusivePtr<TRawDataProvider> rawDataProviderPtr
            = dataProvider.CastMoveTo<TRawObjectsDataProvider>();
        CB_ENSURE(rawDataProviderPtr, "Only non-quantized datasets are supported");
        TRawBuilderData data = TRawBuilderDataHelper::Extract(std::move(*rawDataProviderPtr));

        THolder<THolderType> initialHolder;
        if constexpr (FeatureType == EFeatureType::Categorical) {
            initialHolder = std::move(data.ObjectsData.CatFeatures[featureNum]);
        } else {
            CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
            initialHolder = std::move(data.ObjectsData.FloatFeatures[featureNum]);
        }
        size_t ignoreIndex = 0;
        const size_t ignoreSize = ignoredValues.size();
        float prevBorder = featureValues[0];
        for (size_t numVal = 0; numVal <= featureValues.size(); ++numVal) {
            if constexpr (FeatureType == EFeatureType::Categorical) {
                if (ignoreIndex < ignoreSize && ignoredValues[ignoreIndex] == numVal) {
                    ++ignoreIndex;
                    continue;
                }
                if (numVal == featureValues.size()) {
                    break;
                }
                THolder<THashedCatValuesHolder> holder = MakeHolder<THashedCatArrayValuesHolder>(
                    featureNum,
                    TMaybeOwningConstArrayHolder<ui32>::CreateOwning(TVector<ui32>(objectCount, featureValues[numVal])),
                    data.CommonObjectsData.SubsetIndexing.Get());
                data.ObjectsData.CatFeatures[featureNum] = std::move(holder);
            } else {
                CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
                float newBorder;
                if (numVal == featureValues.size()) {
                    newBorder = prevBorder;
                } else {
                    newBorder = featureValues[numVal];
                }
                THolder<TFloatValuesHolder> holder = MakeHolder<TFloatArrayValuesHolder>(
                    featureNum,
                    TMaybeOwningConstArrayHolder<float>::CreateOwning(
                        TVector<float>(
                            objectCount,
                            (prevBorder + newBorder) / 2.0
                        )
                    ),
                    data.CommonObjectsData.SubsetIndexing.Get());
                data.ObjectsData.FloatFeatures[featureNum] = std::move(holder);
                prevBorder = newBorder;
            }

            rawDataProviderPtr = MakeDataProvider<TRawObjectsDataProvider>(
                Nothing(),
                std::move(data),
                true,
                executor
            );

            auto pred = ApplyModelMulti(model, *(rawDataProviderPtr->ObjectsData), false, predictionType, 0, 0, threadCount)[0];
            (*predictions)[numVal - ignoreIndex] = std::accumulate(pred.begin(), pred.end(), 0.) / static_cast<double>(pred.size());
            data = TRawBuilderDataHelper::Extract(std::move(*rawDataProviderPtr));
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

        dataProvider = std::move(*(rawDataProviderPtr->CastMoveTo<TObjectsDataProvider>()));
    }

    TBinarizedFeatureStatistics GetBinarizedFloatFeatureStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const size_t featureNum,
        const TVector<double>& prediction,
        const EPredictionType predictionType,
        const int threadCount) {

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(threadCount - 1);
        TRestorableFastRng64 rand(0);

        TVector<float> borders = model.ModelTrees->GetFloatFeatures()[featureNum].Borders;
        size_t bordersSize = borders.size();
        if (bordersSize == 0) {
            return {};
        }

        TProcessedDataProvider processedDataProvider = CreateModelCompatibleProcessedDataProvider(
            dataset, {}, model, GetMonopolisticFreeCpuRam(), &rand, &executor);
        TVector<float> target(
            processedDataProvider.TargetData->GetOneDimensionalTarget()->begin(),
            processedDataProvider.TargetData->GetOneDimensionalTarget()->end());

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        TRawObjectsDataProviderPtr rawObjectsDataProviderPtr(objectsPtr);

        TVector<ui32> ignoredFeatureNums;

        for (const auto& feature : model.ModelTrees->GetFloatFeatures()) {
            if (feature.Position.Index != -1 && static_cast<ui32>(feature.Position.Index) != featureNum) {
                ignoredFeatureNums.push_back(feature.Position.FlatIndex);
            }
        }
        for (const auto& feature : model.ModelTrees->GetCatFeatures()) {
            ignoredFeatureNums.push_back(feature.Position.FlatIndex);
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
            model.ModelTrees->GetFloatFeatures()[featureNum].HasNans ? ENanMode::Min : ENanMode::Forbidden);

        TQuantizationOptions options;

        TQuantizedObjectsDataProviderPtr quantizedPtr = Quantize(
            options,
            rawObjectsDataProviderPtr,
            quantizedFeaturesInfo,
            &rand,
            &executor);

        TMaybeData<const IQuantizedFloatValuesHolder*> feature = quantizedPtr->GetFloatFeature(featureNum);
        CB_ENSURE_INTERNAL(feature, "Float feature #" << featureNum << " not found");

        const IQuantizedFloatValuesHolder* values = dynamic_cast<const IQuantizedFloatValuesHolder*>(feature.GetRef());
        CB_ENSURE_INTERNAL(values, "Cannot access values of float feature #" << featureNum);
        TMaybeOwningArrayHolder<ui8> extractedValues = values->ExtractValues(&executor);
        TVector<int> binNums((*extractedValues).begin(), (*extractedValues).end());

        size_t numBins = quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureNum)).size() + 1;
        TVector<float> meanTarget(numBins, 0.);
        TVector<float> meanPrediction(numBins, 0.);
        TVector<size_t> countObjectsPerBin(numBins, 0);

        TVector<float> meanWeightedTarget;
        TVector<double> sumWeightsObjectsPerBin;
        auto maybeWeights = processedDataProvider.TargetData->GetWeights();
        bool useWeights = maybeWeights && !maybeWeights->IsTrivial();
        if (useWeights) {
            meanWeightedTarget.resize(numBins, 0);
            sumWeightsObjectsPerBin.resize(numBins, 0);
        }
        for (size_t numObj = 0; numObj < binNums.size(); ++numObj) {
            int binNum = binNums[numObj];

            meanTarget[binNum] += target[numObj];
            meanPrediction[binNum] += prediction[numObj];

            countObjectsPerBin[binNum] += 1;
            if (useWeights) {
                double weight = (*maybeWeights)[numObj];
                meanWeightedTarget[binNum] += target[numObj] * weight;
                sumWeightsObjectsPerBin[binNum] += weight;
            }
        }

        TVector<size_t> skippedValues;
        for (size_t binNum = 0; binNum < bordersSize + 1; ++binNum) {
            size_t numObjs = countObjectsPerBin[binNum];
            if (numObjs == 0) {
                continue;
            }
            countObjectsPerBin[binNum] = countObjectsPerBin[binNum];
            meanTarget[binNum] /= static_cast<float>(numObjs);
            meanPrediction[binNum] /= static_cast<float>(numObjs);

            if (useWeights) {
                double sumWeight = sumWeightsObjectsPerBin[binNum];
                meanWeightedTarget[binNum ] /= sumWeight;
            }
        }

        TVector<double> predictionsOnVarying(bordersSize + 1, 0.);
        GetPredictionsOnVaryingFeature<float, TFloatValuesHolder, EFeatureType::Float>(
            model,
            featureNum,
            TVector<size_t>(),
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
            useWeights ? meanWeightedTarget : TVector<float> {},
            meanPrediction,
            countObjectsPerBin,
            predictionsOnVarying
        };
    }

    int GetOneHotFeatureFlatNum(const TFullModel& model, const size_t featureNum) {
        int featureFlatNum = -1;
        for(const auto& feature: model.ModelTrees->GetOneHotFeatures()) {
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
            dataset, {}, model, GetMonopolisticFreeCpuRam(), &rand, &executor);
        TVector<float> target(
            processedDataProvider.TargetData->GetOneDimensionalTarget()->begin(),
            processedDataProvider.TargetData->GetOneDimensionalTarget()->end());

        auto objectsPtr = dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get());
        CB_ENSURE_INTERNAL(objectsPtr, "Zero pointer to raw objects");
        TRawObjectsDataProviderPtr rawObjectsDataProviderPtr(objectsPtr);

        if (model.ModelTrees->GetOneHotFeatures().empty()) {
            return {};
        }

        const TVector<int>& oneHotUniqueValues = model.ModelTrees->GetOneHotFeatures()[featureFlatNum].Values;
        TMaybeData<const THashedCatValuesHolder*> catFeatureMaybe = \
            rawObjectsDataProviderPtr->GetCatFeature(featureNum);
        CB_ENSURE_INTERNAL(catFeatureMaybe, "Categorical feature #" << featureNum << " not found");
        const THashedCatValuesHolder* catFeatureHolder = catFeatureMaybe.GetRef();
        CB_ENSURE_INTERNAL(catFeatureHolder, "Cannot access values of categorical feature #" << featureNum);
        TMaybeOwningArrayHolder<ui32> featureValues = catFeatureHolder->ExtractValues(&executor);

        TVector<int> binNums;
        binNums.reserve(oneHotUniqueValues.size());
        for (auto val : *featureValues) {
            auto it = std::find(oneHotUniqueValues.begin(), oneHotUniqueValues.end(), val);
            binNums.push_back(it - oneHotUniqueValues.begin()); // Unknown values will be at bin #oneHotUniqueValues.size()
        }

        size_t numBins = oneHotUniqueValues.size() + 1;
        TVector<float> meanTarget(numBins, 0.);
        TVector<float> meanPrediction(numBins, 0.);
        TVector<size_t> countObjectsPerBin(numBins, 0);

        TVector<float> meanWeightedTarget;
        TVector<double> sumWeightsObjectsPerBin;
        auto maybeWeights = processedDataProvider.TargetData->GetWeights();
        bool useWeights = maybeWeights && !maybeWeights->IsTrivial();
        if (useWeights) {
            meanWeightedTarget.resize(numBins, 0);
            sumWeightsObjectsPerBin.resize(numBins, 0);
        }
        for (size_t numObj = 0; numObj < binNums.size(); ++numObj) {
            ui8 binNum = binNums[numObj];

            meanTarget[binNum] += target[numObj];
            meanPrediction[binNum] += prediction[numObj];
            countObjectsPerBin[binNum] += 1;

            if (useWeights) {
                double weight = (*maybeWeights)[numObj];
                meanWeightedTarget[binNum] += target[numObj] * weight;
                sumWeightsObjectsPerBin[binNum] += weight;
            }
        }

        if (countObjectsPerBin[numBins - 1] == 0) {
            meanTarget.pop_back();
            meanPrediction.pop_back();
            countObjectsPerBin.pop_back();
            if (useWeights) {
                meanWeightedTarget.pop_back();
                sumWeightsObjectsPerBin.pop_back();
            }
            --numBins;
        }
        int emptyOffset = 0;
        TVector<size_t> skippedValues;
        for (size_t binNum = 0; binNum < numBins; ++binNum) {
            size_t numObjs = countObjectsPerBin[binNum];
            if (numObjs == 0) {
                ++emptyOffset;
                skippedValues.push_back(binNum);
                continue;
            }
            countObjectsPerBin[binNum - emptyOffset] = countObjectsPerBin[binNum];
            binNums[binNum - emptyOffset] = binNums[binNum];
            meanTarget[binNum - emptyOffset] = meanTarget[binNum] / static_cast<float>(numObjs);
            meanPrediction[binNum - emptyOffset] = meanPrediction[binNum] / static_cast<float>(numObjs);

            if (useWeights) {
                double sumWeight = sumWeightsObjectsPerBin[binNum];
                meanWeightedTarget[binNum - emptyOffset] = meanWeightedTarget[binNum] / sumWeight;
            }
        }
        countObjectsPerBin.resize(countObjectsPerBin.size() - emptyOffset);
        binNums.resize(binNums.size() - emptyOffset);
        meanTarget.resize(meanTarget.size() - emptyOffset);
        meanPrediction.resize(meanPrediction.size() - emptyOffset);

        TVector<double> predictionsOnVarying(numBins - emptyOffset, 0.);
        GetPredictionsOnVaryingFeature<int, THashedCatValuesHolder, EFeatureType::Categorical>(
            model,
            featureNum,
            skippedValues,
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
            useWeights ? meanWeightedTarget : TVector<float> {},
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

        const TVector<double> prediction = ApplyModelMulti(model, dataset, false, predictionType, 0, 0, threadCount)[0];
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
        if (model.ModelTrees->GetOneHotFeatures().empty()) {
            return 0;
        }
        const int featureFlatNum = GetOneHotFeatureFlatNum(model, featureNum);

        if (featureFlatNum == -1) {
            return 0;
        }
        const TVector<int>& oneHotUniqueValues = model.ModelTrees->GetOneHotFeatures()[featureFlatNum].Values;
        auto it = std::find(oneHotUniqueValues.begin(), oneHotUniqueValues.end(), hash);
        return it - oneHotUniqueValues.begin();
    }

    TFeatureTypeAndInternalIndex GetFeatureTypeAndInternalIndex(const TFullModel& model, const int flatFeatureIndex) {
        for (const auto& feature: model.ModelTrees->GetFloatFeatures()) {
            if (feature.Position.FlatIndex == flatFeatureIndex) {
                return TFeatureTypeAndInternalIndex{EFeatureType::Float, feature.Position.Index};
            }
        }
        for (const auto& feature: model.ModelTrees->GetCatFeatures()) {
            if (feature.Position.FlatIndex == flatFeatureIndex) {
                return TFeatureTypeAndInternalIndex{EFeatureType::Categorical, feature.Position.Index};
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
