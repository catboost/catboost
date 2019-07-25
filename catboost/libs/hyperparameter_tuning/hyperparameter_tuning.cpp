#include "hyperparameter_tuning.h"

#include <catboost/libs/algo/data.h>
#include <catboost/libs/algo/approx_dimension.h>
#include <catboost/libs/data_new/objects_grouping.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <util/generic/algorithm.h>
#include <util/generic/deque.h>
#include <util/generic/set.h>
#include <util/generic/xrange.h>
#include <util/random/shuffle.h>

#include <numeric>

namespace {

    const TVector<TString> NanModeParamAliaces {"nan_mode"};
    const TVector<TString> BorderCountParamAliaces {"border_count", "max_bin"};
    const TVector<TString> BorderTypeParamAliaces {"feature_border_type"};

    // TEnumeratedSet - type of sets, TValue - type of values in sets
    // Set should have access to elements by index and size() method
    // Uniqueness of elements is not required: 'set' is just unformal term
    template <class TEnumeratedSet, class TValue>
    class TProductIteratorBase: public NCB::IDynamicIterator<TConstArrayRef<TValue>> {
    protected:
        bool IsStopIteration = false;
        size_t FirstVaryingDigit = 0;
        TVector<size_t> MultiIndex;
        TVector<TEnumeratedSet> Sets;
        TVector<TValue> State;

    protected:
        explicit TProductIteratorBase(const TVector<TEnumeratedSet>& sets)
            : Sets(sets) {
            InitClassFields(sets);
        }

        void InitClassFields(const TVector<TEnumeratedSet>& sets) {
             if (sets.size() == 0) {
                IsStopIteration = true;
                return;
            }
            MultiIndex.resize(sets.size(), 0);
            size_t idx = 0;
            for (const auto& set : sets) {
                State.push_back(set[0]);
                MultiIndex[idx] = set.size() - 1;
                ++idx;
            }
        }

        const TVector<TValue>& NextWithOffset(ui64 offset) {
            for (size_t setIdx = MultiIndex.size() - 1; setIdx > 0; --setIdx) {
                size_t oldDigit = MultiIndex[setIdx];
                MultiIndex[setIdx] = (MultiIndex[setIdx] + offset) % Sets[setIdx].size();
                State[setIdx] = Sets[setIdx][MultiIndex[setIdx]];

                if (MultiIndex[setIdx] == Sets[setIdx].size() - 1 && FirstVaryingDigit == setIdx) {
                    ++FirstVaryingDigit;
                }
                if (FirstVaryingDigit == MultiIndex.size()) {
                    IsStopIteration = true;
                    return State;
                }
                if (oldDigit + offset < Sets[setIdx].size()) {
                    return State;
                }
                offset = (offset - (Sets[setIdx].size() - oldDigit)) / Sets[setIdx].size() + 1;
            }
            MultiIndex[0] = (MultiIndex[0] + offset) % Sets[0].size();
            State[0] = Sets[0][MultiIndex[0]];
            if (MultiIndex[0] == Sets[0].size() - 1 && FirstVaryingDigit == 0)
                ++FirstVaryingDigit;
            if (FirstVaryingDigit == MultiIndex.size()) {
                IsStopIteration = true;
            }
            return State;
        }
    };

    template <class TEnumeratedSet, class TValue>
    class TCartesianProductIterator: public TProductIteratorBase<TEnumeratedSet, TValue> {
    public:
        explicit TCartesianProductIterator(const TVector<TEnumeratedSet>& sets)
            : TProductIteratorBase<TEnumeratedSet, TValue>(sets)
        {}

        virtual TMaybe<TConstArrayRef<TValue>> Next() override {
            if (this->IsStopIteration) {
                 return this->END_VALUE;
            }
            return this->NextWithOffset(1);
        }
    };

    template <class TEnumeratedSet, class TValue>
    class TRandomizedProductIterator: public TProductIteratorBase<TEnumeratedSet, TValue> {
    private:
        TVector<ui64> FlatOffsets;
        size_t OffsetIndex = 0;

    public:
        // pass count={any positive number} to iterate over random {count} elements
        TRandomizedProductIterator(const TVector<TEnumeratedSet>& sets, ui32 count, bool allowRepeat = false)
            : TProductIteratorBase<TEnumeratedSet, TValue>(sets) {

            CB_ENSURE(count > 0, "Error: param count for TRandomizedProductIterator should be a positive number");
            ui64 totalCount = 1;
            ui64 logTotalCount = 0;
            for (const auto& set : sets) {
                CB_ENSURE(set.size() > 0, "Error: set should be not empty");
                logTotalCount += log2(set.size());
                CB_ENSURE(logTotalCount < 64, "Error: The parameter grid is too large. Try to reduce it.");
                totalCount *= set.size();
            }
            if (count > totalCount && !allowRepeat) {
                count = totalCount;
            }

            TVector<ui64> indexes;
            if (static_cast<double>(count) / totalCount > 0.7 && !allowRepeat) {
                indexes.resize(totalCount);
                std::iota(indexes.begin(), indexes.end(), 1);
                Shuffle(indexes.begin(), indexes.end());
                indexes.resize(count);
            } else {
                TSet<ui64> choosenIndexes;
                TRandom random;
                while (indexes.size() != count) {
                    ui64 nextRandom = random.NextUniformL() % totalCount;
                    while (choosenIndexes.contains(nextRandom)) {
                        nextRandom = random.NextUniformL() % totalCount;
                    }
                    indexes.push_back(nextRandom);
                    if (!allowRepeat) {
                        choosenIndexes.insert(nextRandom);
                    }
                }
            }
            Sort(indexes);
            ui64 lastIndex = 0;
            for (const auto& index : indexes) {
                FlatOffsets.push_back(index - lastIndex);
                lastIndex = index;
            }
        }

        virtual TMaybe<TConstArrayRef<TValue>> Next() override {
            if (this->IsStopIteration) {
                 return this->END_VALUE;
            }
            ui64 offset = 1;
            offset = FlatOffsets[OffsetIndex];
            ++OffsetIndex;
            if (OffsetIndex == FlatOffsets.size()) {
                this->IsStopIteration = true;
            }

            return this->NextWithOffset(offset);
        }
    };

    struct TGeneralQuatizationParamsInfo {
        bool IsBordersCountInGrid = false;
        bool IsBorderTypeInGrid = false;
        bool IsNanModeInGrid = false;
        TString BordersCountParamName = BorderCountParamAliaces[0];
        TString BorderTypeParamName = BorderTypeParamAliaces[0];
        TString NanModeParamName = NanModeParamAliaces[0];
    };

    struct TQuantizationParamsInfo {
        int BinsCount = -1;
        EBorderSelectionType BorderType;
        ENanMode NanMode;
        TGeneralQuatizationParamsInfo GeneralInfo;
    };

    struct TGridParamsInfo {
        TQuantizationParamsInfo QuantizationParamsSet;
        NCB::TQuantizedFeaturesInfoPtr QuantizedFeatureInfo;
        NJson::TJsonValue OthersParamsSet;
        TVector<TString> GridParamNames;
    };

    bool CheckIfRandomDisribution(const TString& value) {
        return value.rfind("CustomRandomDistributionGenerator", 0) == 0;
    }

    NJson::TJsonValue GetRandomValueIfNeeded(
        const NJson::TJsonValue& value,
        const THashMap<TString, NCB::TCustomRandomDistributionGenerator>& randDistGen) {

        if (value.GetType() == NJson::EJsonValueType::JSON_STRING) {
            if (CheckIfRandomDisribution(value.GetString())) {
                CB_ENSURE(
                    randDistGen.find(value.GetString()) != randDistGen.end(),
                    "Error: Reference to unknown random distribution generator"
                );
                const auto& rnd = randDistGen.at(value.GetString());
                return NJson::TJsonValue(rnd.EvalFunc(rnd.CustomData));
            }
        }
        return value;
    }

    void AssignOptionsToJson(
        TConstArrayRef<TString> names,
        TConstArrayRef<NJson::TJsonValue> values,
        const THashMap<TString, NCB::TCustomRandomDistributionGenerator>& randDistGen,
        NJson::TJsonValue* jsonValues) {

        CB_ENSURE(names.size() == values.size(), "Error: names and values should have same size");
        for (size_t i : xrange(names.size())) {
            (*jsonValues)[names[i]] = GetRandomValueIfNeeded(values[i], randDistGen);
        }
    }

    template <class TDataProvidersTemplate>
    TDataProvidersTemplate PrepareTrainTestSplit(
        typename TDataProvidersTemplate::TDataPtr srcData,
        const TTrainTestSplitParams& trainTestSplitParams,
        NPar::TLocalExecutor* localExecutor) {

        CB_ENSURE(
            srcData->ObjectsData->GetOrder() != NCB::EObjectsOrder::Ordered,
            "Params search for ordered objects data is not yet implemented"
        );
        CB_ENSURE(!trainTestSplitParams.Stratified, "Stratified train-test split is not yet supported");
        NCB::TArraySubsetIndexing<ui32> trainIndices;
        NCB::TArraySubsetIndexing<ui32> testIndices;
        TrainTestSplit(
            *srcData->ObjectsGrouping,
            trainTestSplitParams.TrainPart,
            &trainIndices,
            &testIndices
        );
        return NCB::CreateTrainTestSubsets<TDataProvidersTemplate>(
            srcData,
            std::move(trainIndices),
            std::move(testIndices),
            localExecutor
        );
    }

    bool TryCheckParamType(
        const TString& paramName,
        const TSet<NJson::EJsonValueType>& allowedTypes,
        const NJson::TJsonValue& gridJsonValues) {

        if (!gridJsonValues.GetMap().contains(paramName)) {
            return false;
        }

        const auto& jsonValues = gridJsonValues.GetMap().at(paramName);
        for (const auto& value : jsonValues.GetArray()) {
            const auto type = value.GetType();
            if (allowedTypes.find(type) != allowedTypes.end()) {
                continue;
            }
            if (type == NJson::EJsonValueType::JSON_STRING && CheckIfRandomDisribution(value.GetString())) {
                continue;
            }
            ythrow TCatBoostException() << "Can't parse parameter \"" << paramName
                << "\" with value: " << value;
        }
        return true;
    }

    template <class T, typename Func>
    void FindAndExtractParam(
        const TVector<TString>& paramAliases,
        const NCatboostOptions::TOption<T>& option,
        const TSet<NJson::EJsonValueType>& allowedTypes,
        const Func& typeCaster,
        bool* isInGrid,
        TString* exactParamName,
        TDeque<NJson::TJsonValue>* values,
        NJson::TJsonValue* gridJsonValues,
        NJson::TJsonValue* modelJsonParams) {

        for (const auto& paramName : paramAliases) {
            *exactParamName = paramName;
            *isInGrid = TryCheckParamType(
                *exactParamName,
                allowedTypes,
                *gridJsonValues
            );
            if (*isInGrid) {
                break;
            }
        }

        if (*isInGrid) {
            *values = (*gridJsonValues)[*exactParamName].GetArray();
            gridJsonValues->EraseValue(*exactParamName);
            modelJsonParams->EraseValue(*exactParamName);
        } else {
            values->push_back(
                NJson::TJsonValue(
                    typeCaster(option.Get())
                )
            );
        }
    }

    void FindAndExtractGridQuantizationParams(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        TDeque<NJson::TJsonValue>* borderMaxCounts,
        bool* isBordersCountInGrid,
        TString* borderCountsParamName,
        TDeque<NJson::TJsonValue>* borderTypes,
        bool* isBorderTypeInGrid,
        TString* borderTypesParamName,
        TDeque<NJson::TJsonValue>* nanModes,
        bool* isNanModeInGrid,
        TString* nanModesParamName,
        NJson::TJsonValue* gridJsonValues,
        NJson::TJsonValue* modelJsonParams) {

        FindAndExtractParam(
            BorderCountParamAliaces,
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get().BorderCount,
            {
                NJson::EJsonValueType::JSON_INTEGER,
                NJson::EJsonValueType::JSON_UINTEGER,
                NJson::EJsonValueType::JSON_DOUBLE
            },
            [](ui32 value){ return value; },
            isBordersCountInGrid,
            borderCountsParamName,
            borderMaxCounts,
            gridJsonValues,
            modelJsonParams
        );

        FindAndExtractParam(
            BorderTypeParamAliaces,
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get().BorderSelectionType,
            {NJson::EJsonValueType::JSON_STRING},
            [](EBorderSelectionType value){ return ToString(value); },
            isBorderTypeInGrid,
            borderTypesParamName,
            borderTypes,
            gridJsonValues,
            modelJsonParams
        );

        FindAndExtractParam(
            NanModeParamAliaces,
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get().NanMode,
            {NJson::EJsonValueType::JSON_STRING},
            [](ENanMode value){ return ToString(value); },
            isNanModeInGrid,
            nanModesParamName,
            nanModes,
            gridJsonValues,
            modelJsonParams
        );
    }

    bool QuantizeDataIfNeeded(
        bool allowWriteFiles,
        NCB::TFeaturesLayoutPtr featuresLayout,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NCB::TDataProviderPtr data,
        const TQuantizationParamsInfo& oldQuantizedParamsInfo,
        const TQuantizationParamsInfo& newQuantizedParamsInfo,
        TLabelConverter* labelConverter,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        NCatboostOptions::TCatBoostOptions* catBoostOptions,
        NCB::TTrainingDataProviderPtr* result) {

        if (oldQuantizedParamsInfo.BinsCount != newQuantizedParamsInfo.BinsCount ||
            oldQuantizedParamsInfo.BorderType != newQuantizedParamsInfo.BorderType ||
            oldQuantizedParamsInfo.NanMode != newQuantizedParamsInfo.NanMode)
        {
            NCatboostOptions::TBinarizationOptions commonFloatFeaturesBinarization(
                newQuantizedParamsInfo.BorderType,
                newQuantizedParamsInfo.BinsCount,
                newQuantizedParamsInfo.NanMode
            );

            TVector<ui32> ignoredFeatureNums; // TODO(ilikepugs): MLTOOLS-3838
            TMaybe<float> targetBorder = catBoostOptions->DataProcessingOptions->TargetBorder;

            quantizedFeaturesInfo = MakeIntrusive<NCB::TQuantizedFeaturesInfo>(
                *(featuresLayout.Get()),
                MakeConstArrayRef(ignoredFeatureNums),
                commonFloatFeaturesBinarization
            );
            // Quantizing training data
            *result = GetTrainingData(
                data,
                /*isLearnData*/ true,
                /*datasetName*/ TStringBuf(),
                /*bordersFile*/ Nothing(),  // Already at quantizedFeaturesInfo
                /*unloadCatFeaturePerfectHashFromRamIfPossible*/ true,
                /*ensureConsecutiveLearnFeaturesDataForCpu*/ true,
                allowWriteFiles,
                quantizedFeaturesInfo,
                catBoostOptions,
                labelConverter,
                &targetBorder,
                localExecutor,
                rand
            );
            return true;
        }
        return false;
    }

    bool QuantizeAndSplitDataIfNeeded(
        bool allowWriteFiles,
        const TTrainTestSplitParams& trainTestSplitParams,
        NCB::TFeaturesLayoutPtr featuresLayout,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NCB::TDataProviderPtr data,
        const TQuantizationParamsInfo& oldQuantizedParamsInfo,
        const TQuantizationParamsInfo& newQuantizedParamsInfo,
        TLabelConverter* labelConverter,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        NCatboostOptions::TCatBoostOptions* catBoostOptions,
        NCB::TTrainingDataProviders* result) {

        NCB::TTrainingDataProviderPtr quantizedData;
        bool isNeedSplit = QuantizeDataIfNeeded(
            allowWriteFiles,
            featuresLayout,
            quantizedFeaturesInfo,
            data,
            oldQuantizedParamsInfo,
            newQuantizedParamsInfo,
            labelConverter,
            localExecutor,
            rand,
            catBoostOptions,
            &quantizedData
        );

        if (isNeedSplit) {
            // Train-test split
            *result = PrepareTrainTestSplit<NCB::TTrainingDataProviders>(
                quantizedData,
                trainTestSplitParams,
                localExecutor
            );
            return true;
        }
        return false;
    }

    void ParseGridParams(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        NJson::TJsonValue* jsonGrid,
        NJson::TJsonValue* modelJsonParams,
        TVector<TString>* paramNames,
        TVector<TDeque<NJson::TJsonValue>>* paramPossibleValues,
        TGeneralQuatizationParamsInfo* generalQuantizeParamsInfo) {

        paramPossibleValues->resize(3);
        FindAndExtractGridQuantizationParams(
            catBoostOptions,

            &(*paramPossibleValues)[0],
            &generalQuantizeParamsInfo->IsBordersCountInGrid,
            &generalQuantizeParamsInfo->BordersCountParamName,

            &(*paramPossibleValues)[1],
            &generalQuantizeParamsInfo->IsBorderTypeInGrid,
            &generalQuantizeParamsInfo->BorderTypeParamName,

            &(*paramPossibleValues)[2],
            &generalQuantizeParamsInfo->IsNanModeInGrid,
            &generalQuantizeParamsInfo->NanModeParamName,

            jsonGrid,
            modelJsonParams
        );

        for (const auto& set : jsonGrid->GetMap()) {
            paramNames->push_back(set.first);
            paramPossibleValues->resize(paramPossibleValues->size() + 1);
            CB_ENSURE(set.second.GetArray().size() > 0, "Error: an empty set of values for parameter " + paramNames->back());
            for (auto& value : set.second.GetArray()) {
                (*paramPossibleValues)[paramPossibleValues->size() - 1].push_back(value);
            }
        }
    }

    void SetGridParamsToBestOptionValues(
        const TGridParamsInfo & gridParams,
        NCB::TBestOptionValuesWithCvResult* namedOptionsCollection) {
        namedOptionsCollection->SetOptionsFromJson(gridParams.OthersParamsSet.GetMap(), gridParams.GridParamNames);
        // Adding quantization params if needed
        if (gridParams.QuantizationParamsSet.GeneralInfo.IsBordersCountInGrid) {
            const TString& paramName = gridParams.QuantizationParamsSet.GeneralInfo.BordersCountParamName;
            namedOptionsCollection->IntOptions[paramName] = gridParams.QuantizationParamsSet.BinsCount;
        }
        if (gridParams.QuantizationParamsSet.GeneralInfo.IsBorderTypeInGrid) {
            const TString& paramName = gridParams.QuantizationParamsSet.GeneralInfo.BorderTypeParamName;
            namedOptionsCollection->StringOptions[paramName] = ToString(gridParams.QuantizationParamsSet.BorderType);
        }
        if (gridParams.QuantizationParamsSet.GeneralInfo.IsNanModeInGrid) {
            const TString& paramName = gridParams.QuantizationParamsSet.GeneralInfo.NanModeParamName;
            namedOptionsCollection->StringOptions[paramName] = ToString(gridParams.QuantizationParamsSet.NanMode);
        }
    }

    bool SetBestParamsAndUpdateMetricValueIfNeeded(
        double metricValue,
        const TVector<THolder<IMetric>>& metrics,
        const TQuantizationParamsInfo& quantizationParamsSet,
        const NJson::TJsonValue& modelParamsToBeTried,
        const TVector<TString>& paramNames,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TGridParamsInfo* bestGridParams,
        double* bestParamsSetMetricValue) {

        EMetricBestValue metricValueType;
        metrics[0]->GetBestValue(&metricValueType, nullptr);  // Choosing best params only by first metric
        int metricSign;
        if (metricValueType == EMetricBestValue::Min) {
            metricSign = 1;
        } else if (metricValueType == EMetricBestValue::Max) {
            metricSign = -1;
        } else {
            CB_ENSURE(false, "Error: metric for grid search must be minimized or maximized");
        }
        if (metricSign * metricValue < *bestParamsSetMetricValue) {
            *bestParamsSetMetricValue = metricValue;
            bestGridParams->QuantizationParamsSet = quantizationParamsSet;
            bestGridParams->OthersParamsSet = modelParamsToBeTried;
            bestGridParams->QuantizedFeatureInfo = quantizedFeaturesInfo;
            bestGridParams->GridParamNames = paramNames;
            return true;
        }
        return false;
    }

    double TuneHyperparamsCV(
        const TVector<TString>& paramNames,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TCrossValidationParams& cvParams,
        NCB::TDataProviderPtr data,
        TProductIteratorBase<TDeque<NJson::TJsonValue>, NJson::TJsonValue>* gridIterator,
        NJson::TJsonValue* modelParamsToBeTried,
        TGridParamsInfo* bestGridParams,
        TVector<TCVResult>* bestCvResult,
        NPar::TLocalExecutor* localExecutor,
        const THashMap<TString, NCB::TCustomRandomDistributionGenerator>& randDistGenerators = {}) {
        TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

        if (cvParams.Shuffle) {
            auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
            data = data->GetSubset(objectsGroupingSubset, localExecutor);
        }

        double bestParamsSetMetricValue = Max<double>();
        // Other parameters
        NCB::TTrainingDataProviderPtr quantizedData;
        TQuantizationParamsInfo lastQuantizationParamsSet;
        while (auto paramsSet = gridIterator->Next()) {
            // paramsSet: {border_count, feature_border_type, nan_mode, [others]}
            TQuantizationParamsInfo quantizationParamsSet;
            quantizationParamsSet.BinsCount = GetRandomValueIfNeeded((*paramsSet)[0], randDistGenerators).GetInteger();
            quantizationParamsSet.BorderType = FromString<EBorderSelectionType>((*paramsSet)[1].GetString());
            quantizationParamsSet.NanMode = FromString<ENanMode>((*paramsSet)[2].GetString());

            AssignOptionsToJson(
                TConstArrayRef<TString>(paramNames),
                TConstArrayRef<NJson::TJsonValue>(
                    paramsSet->begin() + 3,
                    paramsSet->end()
                ), // Ignoring quantization params
                randDistGenerators,
                modelParamsToBeTried
            );

            NJson::TJsonValue jsonParams;
            NJson::TJsonValue outputJsonParams;
            NCatboostOptions::PlainJsonToOptions(*modelParamsToBeTried, &jsonParams, &outputJsonParams);
            NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
            NCatboostOptions::TOutputFilesOptions outputFileOptions;
            outputFileOptions.Load(outputJsonParams);

            TLabelConverter labelConverter;
            NCB::TFeaturesLayoutPtr featuresLayout = data->MetaInfo.FeaturesLayout;
            NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;

            QuantizeDataIfNeeded(
                outputFileOptions.AllowWriteFiles(),
                featuresLayout,
                quantizedFeaturesInfo,
                data,
                lastQuantizationParamsSet,
                quantizationParamsSet,
                &labelConverter,
                localExecutor,
                &rand,
                &catBoostOptions,
                &quantizedData
            );

            lastQuantizationParamsSet = quantizationParamsSet;
            TVector<TCVResult> cvResult;
            CrossValidate(
                *modelParamsToBeTried,
                objectiveDescriptor,
                evalMetricDescriptor,
                labelConverter,
                quantizedData,
                cvParams,
                localExecutor,
                &cvResult);

            ui32 approxDimension = NCB::GetApproxDimension(catBoostOptions, labelConverter);
            const TVector<THolder<IMetric>> metrics = CreateMetrics(
                catBoostOptions.MetricOptions,
                evalMetricDescriptor,
                approxDimension
            );

            bool isUpdateBest = SetBestParamsAndUpdateMetricValueIfNeeded(
                cvResult[0].AverageTest.back(),
                metrics,
                quantizationParamsSet,
                *modelParamsToBeTried,
                paramNames,
                quantizedFeaturesInfo,
                bestGridParams,
                &bestParamsSetMetricValue);
            if (isUpdateBest) {
                *bestCvResult = cvResult;
            }
        }
        return bestParamsSetMetricValue;
    }

    double TuneHyperparamsTrainTest(
        const TVector<TString>& paramNames,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TTrainTestSplitParams& trainTestSplitParams,
        NCB::TDataProviderPtr data,
        TProductIteratorBase<TDeque<NJson::TJsonValue>, NJson::TJsonValue>* gridIterator,
        NJson::TJsonValue* modelParamsToBeTried,
        TGridParamsInfo * bestGridParams,
        NPar::TLocalExecutor* localExecutor,
        const THashMap<TString, NCB::TCustomRandomDistributionGenerator>& randDistGenerators = {}) {
        TRestorableFastRng64 rand(trainTestSplitParams.PartitionRandSeed);

        if (trainTestSplitParams.Shuffle) {
            auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
            data = data->GetSubset(objectsGroupingSubset, localExecutor);
        }

        double bestParamsSetMetricValue = Max<double>();
        // Other parameters
        NCB::TTrainingDataProviders trainTestData;
        TQuantizationParamsInfo lastQuantizationParamsSet;
        while (auto paramsSet = gridIterator->Next()) {
            // paramsSet: {border_count, feature_border_type, nan_mode, [others]}
            TQuantizationParamsInfo quantizationParamsSet;
            quantizationParamsSet.BinsCount = GetRandomValueIfNeeded((*paramsSet)[0], randDistGenerators).GetInteger();
            quantizationParamsSet.BorderType = FromString<EBorderSelectionType>((*paramsSet)[1].GetString());
            quantizationParamsSet.NanMode = FromString<ENanMode>((*paramsSet)[2].GetString());

            AssignOptionsToJson(
                TConstArrayRef<TString>(paramNames),
                TConstArrayRef<NJson::TJsonValue>(
                    paramsSet->begin() + 3,
                    paramsSet->end()
                ), // Ignoring quantization params
                randDistGenerators,
                modelParamsToBeTried
            );

            NJson::TJsonValue jsonParams;
            NJson::TJsonValue outputJsonParams;
            NCatboostOptions::PlainJsonToOptions(*modelParamsToBeTried, &jsonParams, &outputJsonParams);
            NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
            NCatboostOptions::TOutputFilesOptions outputFileOptions;
            outputFileOptions.Load(outputJsonParams);

            TLabelConverter labelConverter;
            NCB::TFeaturesLayoutPtr featuresLayout = data->MetaInfo.FeaturesLayout;
            NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;

            QuantizeAndSplitDataIfNeeded(
                outputFileOptions.AllowWriteFiles(),
                trainTestSplitParams,
                featuresLayout,
                quantizedFeaturesInfo,
                data,
                lastQuantizationParamsSet,
                quantizationParamsSet,
                &labelConverter,
                localExecutor,
                &rand,
                &catBoostOptions,
                &trainTestData
            );
            lastQuantizationParamsSet = quantizationParamsSet;
            THolder<IModelTrainer> modelTrainerHolder = TTrainerFactory::Construct(catBoostOptions.GetTaskType());

            // Iteration callback
            // TODO(ilikepugs): MLTOOLS-3540
            const TOnEndIterationCallback onEndIterationCallback =
                [] (const TMetricsAndTimeLeftHistory& /*metricsAndTimeHistory*/) -> bool { return true; };

            TMetricsAndTimeLeftHistory metricsAndTimeHistory;
            TEvalResult evalRes;

            TTrainModelInternalOptions internalOptions;
            internalOptions.CalcMetricsOnly = true;
            internalOptions.ForceCalcEvalMetricOnEveryIteration = false;
            internalOptions.OffsetMetricPeriodByInitModelSize = true;

            // Training model
            modelTrainerHolder->TrainModel(
                internalOptions,
                catBoostOptions,
                outputFileOptions,
                objectiveDescriptor,
                evalMetricDescriptor,
                onEndIterationCallback,
                trainTestData,
                labelConverter,
                /*initModel*/ Nothing(),
                /*initLearnProgress*/ nullptr,
                /*initModelApplyCompatiblePools*/ NCB::TDataProviders(),
                localExecutor,
                &rand,
                /*dstModel*/ nullptr,
                /*evalResultPtrs*/ {&evalRes},
                &metricsAndTimeHistory,
                /*dstLearnProgress*/nullptr
            );

            ui32 approxDimension = NCB::GetApproxDimension(catBoostOptions, labelConverter);
            const TVector<THolder<IMetric>> metrics = CreateMetrics(
                catBoostOptions.MetricOptions,
                evalMetricDescriptor,
                approxDimension
            );

            const TString& lossDescription = metrics[0]->GetDescription();
            double bestMetricValue = metricsAndTimeHistory.TestBestError[0][lossDescription]; //[testId][lossDescription]

            SetBestParamsAndUpdateMetricValueIfNeeded(
                bestMetricValue,
                metrics,
                quantizationParamsSet,
                *modelParamsToBeTried,
                paramNames,
                quantizedFeaturesInfo,
                bestGridParams,
                &bestParamsSetMetricValue);
        }
        return bestParamsSetMetricValue;
    }
} // anonymous namespace

namespace NCB {
    void TBestOptionValuesWithCvResult::SetOptionsFromJson(
        const THashMap<TString, NJson::TJsonValue>& options,
        const TVector<TString>& optionsNames) {
        BoolOptions.clear();
        IntOptions.clear();
        UIntOptions.clear();
        DoubleOptions.clear();
        StringOptions.clear();
        for (const auto& optionName : optionsNames) {
            const auto& option = options.at(optionName);
            NJson::EJsonValueType type = option.GetType();
            switch(type) {
                case NJson::EJsonValueType::JSON_BOOLEAN: {
                    BoolOptions[optionName] = option.GetBoolean();
                    break;
                }
                case NJson::EJsonValueType::JSON_INTEGER: {
                    IntOptions[optionName] = option.GetInteger();
                    break;
                }
                case NJson::EJsonValueType::JSON_UINTEGER: {
                    UIntOptions[optionName] = option.GetUInteger();
                    break;
                }
                case NJson::EJsonValueType::JSON_DOUBLE: {
                    DoubleOptions[optionName] = option.GetDouble();
                    break;
                }
                case NJson::EJsonValueType::JSON_STRING: {
                    StringOptions[optionName] = option.GetString();
                    break;
                }
                default: {
                    CB_ENSURE(false, "Error: option value should be bool, int, ui32, double or string");
                }
            }
        }
    }

    void GridSearch(
        const NJson::TJsonValue& gridJsonValues,
        const NJson::TJsonValue& modelJsonParams,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TCrossValidationParams& cvParams,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TDataProviderPtr data,
        TBestOptionValuesWithCvResult* bestOptionValuesWithCvResult,
        bool isSearchUsingTrainTestSplit,
        bool returnCvStat) {

        // CatBoost options
        NJson::TJsonValue jsonParams;
        NJson::TJsonValue outputJsonParams;
        NCatboostOptions::PlainJsonToOptions(modelJsonParams, &jsonParams, &outputJsonParams);
        NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
        NCatboostOptions::TOutputFilesOptions outputFileOptions;
        outputFileOptions.Load(outputJsonParams);
        CB_ENSURE(!outputJsonParams["save_snapshot"].GetBoolean(), "Snapshots are not yet supported for GridSearchCV");

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(catBoostOptions.SystemOptions->NumThreads.Get() - 1);

        TGridParamsInfo bestGridParams;
        TDeque<NJson::TJsonValue> paramGrids;
        if (gridJsonValues.GetType() == NJson::EJsonValueType::JSON_MAP) {
            paramGrids.push_back(gridJsonValues);
        } else {
            paramGrids = gridJsonValues.GetArray();
        }

        double bestParamsSetMetricValue = Max<double>();
        TVector<TCVResult> bestCvResult;
        for (auto grid : paramGrids) {
            // Preparing parameters for cartesian product
            TVector<TDeque<NJson::TJsonValue>> paramPossibleValues; // {border_count, feature_border_type, nan_mode, ...}
            TGeneralQuatizationParamsInfo generalQuantizeParamsInfo;
            TQuantizationParamsInfo quantizationParamsSet;
            TVector<TString> paramNames;

            NJson::TJsonValue modelParamsToBeTried(modelJsonParams);
            TGridParamsInfo gridParams;
            ParseGridParams(
                catBoostOptions,
                &grid,
                &modelParamsToBeTried,
                &paramNames,
                &paramPossibleValues,
                &generalQuantizeParamsInfo
            );

            TCartesianProductIterator<TDeque<NJson::TJsonValue>, NJson::TJsonValue> gridIterator(paramPossibleValues);
            double metricValue;
            if (isSearchUsingTrainTestSplit) {
                metricValue = TuneHyperparamsTrainTest(
                    paramNames,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    trainTestSplitParams,
                    data,
                    &gridIterator,
                    &modelParamsToBeTried,
                    &gridParams,
                    &localExecutor
                );
            } else {
                metricValue = TuneHyperparamsCV(
                    paramNames,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    cvParams,
                    data,
                    &gridIterator,
                    &modelParamsToBeTried,
                    &gridParams,
                    &bestCvResult,
                    &localExecutor
                );
            }

            if (metricValue < bestParamsSetMetricValue) {
                bestGridParams = gridParams;
                bestGridParams.QuantizationParamsSet.GeneralInfo = generalQuantizeParamsInfo;
                SetGridParamsToBestOptionValues(bestGridParams, bestOptionValuesWithCvResult);
            }
        }
        if (returnCvStat || isSearchUsingTrainTestSplit) {
            if (isSearchUsingTrainTestSplit) {
                CrossValidate(
                    bestGridParams.OthersParamsSet,
                    bestGridParams.QuantizedFeatureInfo,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    data,
                    cvParams,
                    &(bestOptionValuesWithCvResult->CvResult)
                );
            } else {
                bestOptionValuesWithCvResult->CvResult = bestCvResult;
            }
        }
    }

    void RandomizedSearch(
        ui32 numberOfTries,
        const THashMap<TString, TCustomRandomDistributionGenerator>& randDistGenerators,
        const NJson::TJsonValue& gridJsonValues,
        const NJson::TJsonValue& modelJsonParams,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TCrossValidationParams& cvParams,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TDataProviderPtr data,
        TBestOptionValuesWithCvResult* bestOptionValuesWithCvResult,
        bool isSearchUsingTrainTestSplit,
        bool returnCvStat) {

        // CatBoost options
        NJson::TJsonValue jsonParams;
        NJson::TJsonValue outputJsonParams;
        NCatboostOptions::PlainJsonToOptions(modelJsonParams, &jsonParams, &outputJsonParams);
        NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
        NCatboostOptions::TOutputFilesOptions outputFileOptions;
        outputFileOptions.Load(outputJsonParams);
        CB_ENSURE(!outputJsonParams["save_snapshot"].GetBoolean(), "Snapshots are not yet supported for RandomizedSearchCV");

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(catBoostOptions.SystemOptions->NumThreads.Get() - 1);

        NJson::TJsonValue paramGrid;
        if (gridJsonValues.GetType() == NJson::EJsonValueType::JSON_MAP) {
            paramGrid = gridJsonValues;
        } else {
            paramGrid = gridJsonValues.GetArray()[0];
        }
        // Preparing parameters for cartesian product
        TVector<TDeque<NJson::TJsonValue>> paramPossibleValues; // {border_count, feature_border_type, nan_mode, ...}
        TGeneralQuatizationParamsInfo generalQuantizeParamsInfo;
        TQuantizationParamsInfo quantizationParamsSet;
        TVector<TString> paramNames;

        NJson::TJsonValue modelParamsToBeTried(modelJsonParams);

        ParseGridParams(
            catBoostOptions,
            &paramGrid,
            &modelParamsToBeTried,
            &paramNames,
            &paramPossibleValues,
            &generalQuantizeParamsInfo
        );

        TRandomizedProductIterator<TDeque<NJson::TJsonValue>, NJson::TJsonValue> gridIterator(
            paramPossibleValues,
            numberOfTries,
            randDistGenerators.size() > 0
        );

        TGridParamsInfo bestGridParams;
        TVector<TCVResult> cvResult;
        if (isSearchUsingTrainTestSplit) {
            TuneHyperparamsTrainTest(
                paramNames,
                objectiveDescriptor,
                evalMetricDescriptor,
                trainTestSplitParams,
                data,
                &gridIterator,
                &modelParamsToBeTried,
                &bestGridParams,
                &localExecutor,
                randDistGenerators
            );
        } else {
            TuneHyperparamsCV(
                paramNames,
                objectiveDescriptor,
                evalMetricDescriptor,
                cvParams,
                data,
                &gridIterator,
                &modelParamsToBeTried,
                &bestGridParams,
                &cvResult,
                &localExecutor,
                randDistGenerators
            );
        }
        bestGridParams.QuantizationParamsSet.GeneralInfo = generalQuantizeParamsInfo;
        SetGridParamsToBestOptionValues(bestGridParams, bestOptionValuesWithCvResult);
        if (returnCvStat || isSearchUsingTrainTestSplit) {
            if (isSearchUsingTrainTestSplit) {
                CrossValidate(
                    bestGridParams.OthersParamsSet,
                    bestGridParams.QuantizedFeatureInfo,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    data,
                    cvParams,
                    &(bestOptionValuesWithCvResult->CvResult)
                );
            } else {
                bestOptionValuesWithCvResult->CvResult = cvResult;
            }
        }
    }
}
