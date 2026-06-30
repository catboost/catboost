#include "quantized_pool_analysis.h"

#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/private/libs/quantization/utils.h>

#include <util/generic/fwd.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>

#include <limits>

namespace NCB {

    namespace {

        class TFloatFeatureHolderGenerator {
        public:
            explicit TFloatFeatureHolderGenerator(size_t featureNum, TConstArrayRef<float> quantizationBorders)
                : FeatureNum(featureNum)
                , QuantizationBorders(quantizationBorders)
                , CurrentIndex(0) {
            }

            template <class TSubsetIndexing>
            THolder<TFloatValuesHolder> Next(size_t objectCount, const TSubsetIndexing* subsetIndexing) {
                if (CurrentIndex > QuantizationBorders.size()) {
                    return {};
                }

                float value;
                if (CurrentIndex == 0) {
                    value = std::numeric_limits<float>::lowest();
                } else if (CurrentIndex == QuantizationBorders.size()) {
                    value = std::numeric_limits<float>::max();
                } else {
                    value = (QuantizationBorders[CurrentIndex - 1] + QuantizationBorders[CurrentIndex]) / 2.f;
                }

                auto result = MakeHolder<TFloatArrayValuesHolder>(
                    FeatureNum,
                    TMaybeOwningConstArrayHolder<float>::CreateOwning(TVector<float>(objectCount, value)),
                    subsetIndexing);

                ++CurrentIndex;
                return result;
            }

        private:
            const size_t FeatureNum;
            const TConstArrayRef<float> QuantizationBorders;

            size_t CurrentIndex;
        };

        class TQuantizedFloatFeatureHolderGenerator {
        public:
            explicit TQuantizedFloatFeatureHolderGenerator(
                size_t featureNum,
                TConstArrayRef<ui32> binNums,
                size_t bitsPerValue)

                : FeatureNum(featureNum)
                , BinNums(binNums)
                , BitsPerValue(bitsPerValue)
                , CurrentIndex(0) {
            }

            template <class TSubsetIndexing>
            THolder<IQuantizedFloatValuesHolder>
            Next(size_t objectCount, const TSubsetIndexing* subsetIndexing) {
                if (CurrentIndex >= BinNums.size()) {
                    return {};
                }

                ui64 value = BinNums[CurrentIndex];
                ui64 compressedValues = 0;
                for (ui8 i = 0; i < 64; i += BitsPerValue) {
                    compressedValues |= (value << i);
                }
                ui64 compressedCount = CeilDiv(objectCount, size_t(64) / BitsPerValue);
                auto result = MakeHolder<TCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>>(
                    FeatureNum,
                    TCompressedArray(
                        objectCount,
                        BitsPerValue,
                        TMaybeOwningArrayHolder<ui64>::CreateOwning(
                            TVector<ui64>(compressedCount, compressedValues))),
                    subsetIndexing);

                ++CurrentIndex;
                return result;
            }

        private:
            const size_t FeatureNum;
            const TConstArrayRef<ui32> BinNums;
            const ui8 BitsPerValue;

            size_t CurrentIndex;
        };

        class TCatFeatureHolderGenerator {
        public:
            explicit TCatFeatureHolderGenerator(
                size_t featureNum,
                TConstArrayRef<int> featureValues,
                TConstArrayRef<size_t> ignoredValues)

                : FeatureNum(featureNum)
                , FeatureValues(featureValues)
                , IgnoredValues(ignoredValues)
                , IgnoredIndex(0)
                , CurrentIndex(0) {
            }

            template <class TSubsetIndexing>
            THolder<THashedCatValuesHolder> Next(size_t objectCount, const TSubsetIndexing* subsetIndexing) {
                while (IgnoredIndex < IgnoredValues.size() && IgnoredValues[IgnoredIndex] == CurrentIndex) {
                    ++IgnoredIndex;
                    ++CurrentIndex;
                }

                if (CurrentIndex == FeatureValues.size()) {
                    return {};
                }

                auto result = MakeHolder<THashedCatArrayValuesHolder>(
                    FeatureNum,
                    TMaybeOwningConstArrayHolder<ui32>::CreateOwning(
                        TVector<ui32>(objectCount, FeatureValues[CurrentIndex])),
                    subsetIndexing);

                ++CurrentIndex;
                return result;
            }

        private:
            const size_t FeatureNum;
            const TConstArrayRef<int> FeatureValues;
            const TConstArrayRef<size_t> IgnoredValues;

            size_t IgnoredIndex;
            size_t CurrentIndex;
        };

        template <class TTBuilderData>
        auto& GetObjectsDataWithFeatures(TTBuilderData& builderData) {
            return builderData.ObjectsData;
        }

        template <>
        auto& GetObjectsDataWithFeatures(TQuantizedBuilderData& builderData) {
            return builderData.ObjectsData;
        }

    }  // anonymous namespace

    template <class TTObjectsDataProvider, EFeatureType FeatureType, class TFeatureHolderGenerator>
    static void GetPredictionsOnVaryingFeature(
        const TFullModel& model,
        const size_t featureNum,
        TFeatureHolderGenerator featureHolderGenerator,
        const EPredictionType predictionType,
        const int threadCount,
        TDataProvider& dataProvider,
        TVector<double>* predictions,
        NPar::ILocalExecutor* executor) {

        size_t objectCount = dataProvider.GetObjectCount();

        TIntrusivePtr<TDataProviderTemplate<TTObjectsDataProvider>> dataProviderPtr =
            dataProvider.CastMoveTo<TTObjectsDataProvider>();
        CB_ENSURE_INTERNAL(dataProviderPtr, "data provider type is different from expected");
        auto data = TBuilderDataHelper<TTObjectsDataProvider>::Extract(std::move(*dataProviderPtr));

        auto initialHolder = [&] {
            if constexpr (FeatureType == EFeatureType::Categorical) {
                return std::move(GetObjectsDataWithFeatures(data).CatFeatures[featureNum]);
            } else {
                CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
                return std::move(GetObjectsDataWithFeatures(data).FloatFeatures[featureNum]);
            }
        }();
        size_t numVal = 0;
        while (auto holder = featureHolderGenerator.Next(objectCount, data.CommonObjectsData.SubsetIndexing.Get())) {
            if constexpr (FeatureType == EFeatureType::Categorical) {
                GetObjectsDataWithFeatures(data).CatFeatures[featureNum] = std::move(holder);
            } else {
                CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
                GetObjectsDataWithFeatures(data).FloatFeatures[featureNum] = std::move(holder);
            }

            dataProviderPtr = MakeDataProvider<TTObjectsDataProvider>(
                Nothing(),
                std::move(data),
                true,
                dataProvider.MetaInfo.ForceUnitAutoPairWeights,
                executor
            );

            auto pred = ApplyModelMulti(model, *(dataProviderPtr->ObjectsData), false, predictionType, 0, 0, threadCount, dataProviderPtr->RawTargetData.GetBaseline())[0];
            (*predictions)[numVal] = std::accumulate(pred.begin(), pred.end(), 0.) / static_cast<double>(pred.size());
            data = TBuilderDataHelper<TTObjectsDataProvider>::Extract(std::move(*dataProviderPtr));
            ++numVal;
        }

        if constexpr (FeatureType == EFeatureType::Categorical) {
            GetObjectsDataWithFeatures(data).CatFeatures[featureNum] = std::move(initialHolder);
        } else {
            CB_ENSURE_INTERNAL(FeatureType == EFeatureType::Float, "Unsupported FeatureType");
            GetObjectsDataWithFeatures(data).FloatFeatures[featureNum] = std::move(initialHolder);
        }

        dataProviderPtr = MakeDataProvider<TTObjectsDataProvider>(
            Nothing(),
            std::move(data),
            false,
            dataProvider.MetaInfo.ForceUnitAutoPairWeights,
            executor);

        dataProvider = std::move(*(dataProviderPtr->template CastMoveTo<TObjectsDataProvider>()));
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

        const TVector<float>& modelBorders = model.ModelTrees->GetFloatFeatures()[featureNum].Borders;
        size_t modelBordersSize = modelBorders.size();
        if (modelBordersSize == 0) {
            return {};
        }
        size_t numModelBins = modelBordersSize + 1;

        TProcessedDataProvider processedDataProvider = CreateModelCompatibleProcessedDataProvider(
            dataset, {}, model, GetMonopolisticFreeCpuRam(), &rand, &executor);
        TVector<float> target(
            processedDataProvider.TargetData->GetOneDimensionalTarget()->begin(),
            processedDataProvider.TargetData->GetOneDimensionalTarget()->end());

        TQuantizedObjectsDataProviderPtr quantizedPtr;
        bool isDatasetQuantized = false;
        if (auto* rawObjectsDataProvider =
            dynamic_cast<TRawObjectsDataProvider*>(dataset.ObjectsData.Get()))
        {
            TRawObjectsDataProviderPtr rawObjectsDataProviderPtr(rawObjectsDataProvider);

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
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
                *(featuresLayout.Get()),
                ignoredFeatures,
                commonFloatFeaturesBinarization);
            quantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(featureNum), TVector<float>(modelBorders));
            quantizedFeaturesInfo->SetNanMode(
                TFloatFeatureIdx(featureNum),
                model.ModelTrees->GetFloatFeatures()[featureNum].HasNans ? ENanMode::Min
                                                                         : ENanMode::Forbidden);

            TQuantizationOptions options;

            quantizedPtr = Quantize(
                options,
                rawObjectsDataProviderPtr,
                quantizedFeaturesInfo,
                &rand,
                &executor);
        } else if (auto* quantizedForCpuObjectsData =
                   dynamic_cast<TQuantizedObjectsDataProvider*>(dataset.ObjectsData.Get()))
        {
            quantizedPtr = quantizedForCpuObjectsData;
            isDatasetQuantized = true;
        } else {
            CB_ENSURE(false, "Unsupported objects data - neither raw nor quantized for CPU");
        }

        TMaybeData<const IQuantizedFloatValuesHolder*> feature = quantizedPtr->GetFloatFeature(featureNum);
        CB_ENSURE_INTERNAL(feature, "Float feature #" << featureNum << " not found");

        const IQuantizedFloatValuesHolder* values = dynamic_cast<const IQuantizedFloatValuesHolder*>(feature.GetRef());
        CB_ENSURE_INTERNAL(values, "Cannot access values of float feature #" << featureNum);
        TVector<ui32> binNums(values->GetSize());
        auto binNumsWriteIter = binNums.begin();

        // initialized only when dataset is quantized
        TVector<ui8> poolBinsRemap;

        if (isDatasetQuantized) {
            poolBinsRemap = GetFloatFeatureBordersRemap(
                model.ModelTrees->GetFloatFeatures()[featureNum],
                featureNum,
                *quantizedPtr->GetQuantizedFeaturesInfo().Get());
            CB_ENSURE_INTERNAL(!poolBinsRemap.empty(), "Float feature #" << featureNum << " cannot be remapped");
            auto blockFunc = [&binNumsWriteIter, &poolBinsRemap](size_t /*blockStartIdx*/, auto block) {
                Transform(
                    block.begin(),
                    block.end(),
                    binNumsWriteIter,
                    [&poolBinsRemap](const auto binNum) {return poolBinsRemap[binNum];});
                binNumsWriteIter += block.end() - block.begin();
            };
            values->ForEachBlock(std::move(blockFunc));
        } else {
            auto blockFunc = [&binNumsWriteIter](size_t /*blockStartIdx*/, auto block) {
                binNumsWriteIter = Copy(block.begin(), block.end(), binNumsWriteIter);
            };
            values->ForEachBlock(std::move(blockFunc));
        }

        TVector<float> meanTarget(numModelBins, 0.);
        TVector<float> meanPrediction(numModelBins, 0.);
        TVector<size_t> countObjectsPerBin(numModelBins, 0);

        TVector<float> meanWeightedTarget;
        TVector<double> sumWeightsObjectsPerBin;
        auto maybeWeights = processedDataProvider.TargetData->GetWeights();
        bool useWeights = maybeWeights && !maybeWeights->IsTrivial();
        if (useWeights) {
            meanWeightedTarget.resize(numModelBins, 0);
            sumWeightsObjectsPerBin.resize(numModelBins, 0);
        }
        for (size_t numObj = 0; numObj < binNums.size(); ++numObj) {
            ui32 binNum = binNums[numObj];

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
        for (size_t binNum = 0; binNum < numModelBins; ++binNum) {
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

        TVector<double> predictionsOnVarying(numModelBins, 0.);
        if (isDatasetQuantized) {
            TVector<ui32> modelBinsToPoolBins(numModelBins);
            for (const auto poolBin : xrange(poolBinsRemap.size())) {
                modelBinsToPoolBins[poolBinsRemap[poolBin]] = poolBin;
            }
            GetPredictionsOnVaryingFeature<TQuantizedObjectsDataProvider, EFeatureType::Float>(
                model,
                featureNum,
                TQuantizedFloatFeatureHolderGenerator(
                    featureNum,
                    modelBinsToPoolBins,
                    CalcHistogramWidthForBorders(poolBinsRemap.size() - 1)),
                predictionType,
                threadCount,
                dataset,
                &predictionsOnVarying,
                &executor);
        } else {
            GetPredictionsOnVaryingFeature<TRawObjectsDataProvider, EFeatureType::Float>(
                model,
                featureNum,
                TFloatFeatureHolderGenerator(featureNum, modelBorders),
                predictionType,
                threadCount,
                dataset,
                &predictionsOnVarying,
                &executor);
        }

        return TBinarizedFeatureStatistics{
            modelBorders,
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

        TVector<ui32> binNums;
        binNums.reserve(oneHotUniqueValues.size());
        auto blockIterator = catFeatureHolder->GetBlockIterator();
        while (auto block = blockIterator->Next(1024)) {
            for (auto val : block) {
                auto it = std::find(oneHotUniqueValues.begin(), oneHotUniqueValues.end(), val);
                binNums.push_back(it - oneHotUniqueValues.begin()); // Unknown values will be at bin #oneHotUniqueValues.size()
            }
        };

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
        GetPredictionsOnVaryingFeature<TRawObjectsDataProvider, EFeatureType::Categorical>(
            model,
            featureNum,
            TCatFeatureHolderGenerator(featureNum, oneHotUniqueValues, skippedValues),
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
