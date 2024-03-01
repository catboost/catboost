#include "hyperparameter_tuning.h"

#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/objects_grouping.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/train_lib/dir_helper.h>
#include <catboost/libs/train_lib/trainer_env.h>
#include <catboost/private/libs/options/defaults_helper.h>
#include <catboost/private/libs/options/plain_options_helper.h>

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
    constexpr ui32 IndexOfFirstTrainingParameter = 3;

    // TEnumeratedSet - type of sets, TValue - type of values in sets
    // Set should have access to elements by index and size() method
    // Uniqueness of elements is not required: 'set' is just unformal term
    template <class TEnumeratedSet, class TValue>
    class TProductIteratorBase: public NCB::IDynamicIterator<TConstArrayRef<TValue>> {
    protected:
        bool IsStopIteration = false;
        size_t FirstVaryingDigit = 0;
        ui64 PassedElementsCount = 0;
        ui64 TotalElementsCount;
        TVector<size_t> MultiIndex;
        TVector<TEnumeratedSet> Sets;
        TVector<TValue> State;

    protected:
        explicit TProductIteratorBase(const TVector<TEnumeratedSet>& sets)
            : Sets(sets) {
            InitClassFields(sets);
            ui64 totalCount = 1;
            ui64 logTotalCount = 0;
            for (const auto& set : sets) {
                CB_ENSURE(set.size() > 0, "Error: set should be not empty");
                logTotalCount += log2(set.size());
                CB_ENSURE(logTotalCount < 64, "Error: The parameter grid is too large. Try to reduce it.");
                totalCount *= set.size();
            }
            TotalElementsCount = totalCount;
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

                if (oldDigit + offset < Sets[setIdx].size()) {
                    return State;
                }
                offset = (offset - (Sets[setIdx].size() - oldDigit)) / Sets[setIdx].size() + 1;
            }
            MultiIndex[0] = (MultiIndex[0] + offset) % Sets[0].size();
            State[0] = Sets[0][MultiIndex[0]];
            return State;
        }

        bool IsIteratorReachedEnd() {
            return PassedElementsCount >= TotalElementsCount;
        }

    public:
        ui64 GetTotalElementsCount() {
            return TotalElementsCount;
        }
    };

    template <class TEnumeratedSet, class TValue>
    class TCartesianProductIterator: public TProductIteratorBase<TEnumeratedSet, TValue> {
    public:
        explicit TCartesianProductIterator(const TVector<TEnumeratedSet>& sets)
            : TProductIteratorBase<TEnumeratedSet, TValue>(sets)
        {}

        bool Next(TConstArrayRef<TValue>* value) override {
            if (this->IsIteratorReachedEnd()) {
                 return false;
            }
            this->PassedElementsCount++;
            *value = this->NextWithOffset(1);
            return true;
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
            ui64 totalCount = this->TotalElementsCount;
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
            this->TotalElementsCount = count;
        }

        bool Next(TConstArrayRef<TValue>* values) override {
            if (this->IsIteratorReachedEnd()) {
                 return false;
            }
            ui64 offset = 1;
            offset = FlatOffsets[OffsetIndex];
            ++OffsetIndex;
            this->PassedElementsCount++;
            *values = this->NextWithOffset(offset);
            return true;
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


    NCB::TTrainingDataProviders PrepareTrainTestSplit(
        NCB::TTrainingDataProviderPtr srcData,
        const TTrainTestSplitParams& trainTestSplitParams,
        ui64 cpuUsedRamLimit,
        NPar::ILocalExecutor* localExecutor) {

        CB_ENSURE(
            srcData->ObjectsData->GetOrder() != NCB::EObjectsOrder::Ordered,
            "Params search for ordered objects data is not yet implemented"
        );
        NCB::TArraySubsetIndexing<ui32> trainIndices;
        NCB::TArraySubsetIndexing<ui32> testIndices;

        if (trainTestSplitParams.Stratified) {
            NCB::TMaybeData<TConstArrayRef<float>> maybeTarget
                = srcData->TargetData->GetOneDimensionalTarget();
            CB_ENSURE(maybeTarget, "Cannot do stratified split: Target data is unavailable");

            NCB::StratifiedTrainTestSplit(
                *srcData->ObjectsGrouping,
                *maybeTarget,
                trainTestSplitParams.TrainPart,
                &trainIndices,
                &testIndices
            );
        } else {
            TrainTestSplit(
                *srcData->ObjectsGrouping,
                trainTestSplitParams.TrainPart,
                &trainIndices,
                &testIndices
            );
        }
        return NCB::CreateTrainTestSubsets<NCB::TTrainingDataProviders>(
            srcData,
            std::move(trainIndices),
            std::move(testIndices),
            cpuUsedRamLimit,
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
        const TString& tmpDir,
        NCB::TFeaturesLayoutPtr featuresLayout,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NCB::TDataProviderPtr data,
        const TQuantizationParamsInfo& oldQuantizedParamsInfo,
        const TQuantizationParamsInfo& newQuantizedParamsInfo,
        TLabelConverter* labelConverter,
        NPar::ILocalExecutor* localExecutor,
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
                commonFloatFeaturesBinarization,
                /*perFloatFeatureQuantization*/TMap<ui32, NCatboostOptions::TBinarizationOptions>(),
                /*floatFeaturesAllowNansInTestOnly*/true
            );
            // Quantizing training data
            *result = GetTrainingData(
                data,
                /*dataCanBeEmpty*/ false,
                /*isLearnData*/ true,
                /*datasetName*/ TStringBuf(),
                /*bordersFile*/ Nothing(),  // Already at quantizedFeaturesInfo
                /*unloadCatFeaturePerfectHashFromRam*/ allowWriteFiles,
                /*ensureConsecutiveLearnFeaturesDataForCpu*/ false, // data will be split afterwards anyway
                tmpDir,
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
        const TString& tmpDir,
        const TTrainTestSplitParams& trainTestSplitParams,
        ui64 cpuUsedRamLimit,
        NCB::TFeaturesLayoutPtr featuresLayout,
        NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NCB::TDataProviderPtr data,
        const TQuantizationParamsInfo& oldQuantizedParamsInfo,
        const TQuantizationParamsInfo& newQuantizedParamsInfo,
        TLabelConverter* labelConverter,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        NCatboostOptions::TCatBoostOptions* catBoostOptions,
        NCB::TTrainingDataProviders* result) {

        NCB::TTrainingDataProviderPtr quantizedData;
        bool isNeedSplit = QuantizeDataIfNeeded(
            allowWriteFiles,
            tmpDir,
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
            *result = PrepareTrainTestSplit(
                quantizedData,
                trainTestSplitParams,
                cpuUsedRamLimit,
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
            namedOptionsCollection->BestParams[paramName] = NJson::TJsonValue(gridParams.QuantizationParamsSet.BinsCount);
        }
        if (gridParams.QuantizationParamsSet.GeneralInfo.IsBorderTypeInGrid) {
            const TString& paramName = gridParams.QuantizationParamsSet.GeneralInfo.BorderTypeParamName;
            namedOptionsCollection->BestParams[paramName] = NJson::TJsonValue(ToString(gridParams.QuantizationParamsSet.BorderType));
        }
        if (gridParams.QuantizationParamsSet.GeneralInfo.IsNanModeInGrid) {
            const TString& paramName = gridParams.QuantizationParamsSet.GeneralInfo.NanModeParamName;
            namedOptionsCollection->BestParams[paramName] = NJson::TJsonValue(ToString(gridParams.QuantizationParamsSet.NanMode));
        }
    }

    int GetSignForMetricMinimization(const THolder<IMetric>& metric) {
        EMetricBestValue metricValueType;
        metric->GetBestValue(&metricValueType, nullptr);  // Choosing best params only by first metric
        int metricSign;
        if (metricValueType == EMetricBestValue::Min) {
            metricSign = 1;
        } else if (metricValueType == EMetricBestValue::Max) {
            metricSign = -1;
        } else {
            CB_ENSURE(false, "Error: metric for grid search must be minimized or maximized");
        }
        return metricSign;
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

        int metricSign = GetSignForMetricMinimization(metrics[0]);
        if (metricSign * metricValue < *bestParamsSetMetricValue * metricSign) {
            *bestParamsSetMetricValue = metricValue;
            bestGridParams->QuantizationParamsSet = quantizationParamsSet;
            bestGridParams->OthersParamsSet = modelParamsToBeTried;
            bestGridParams->QuantizedFeatureInfo = quantizedFeaturesInfo;
            bestGridParams->GridParamNames = paramNames;
            return true;
        }
        return false;
    }

    static TString GetNamesPrefix(ui32 foldIdx) {
        return "fold_" + ToString(foldIdx) + "_";
    }

    static void InitializeFilesLoggers(
        const TVector<THolder<IMetric>>& metrics,
        const TOutputFiles& outputFiles,
        const int iterationCount,
        const ELaunchMode launchMode,
        const int foldCountOrTestSize,
        const TString& parametersToken,
        TLogger* logger
    ) {
        TVector<TString> learnSetNames;
        TVector<TString> testSetNames;
        switch (launchMode) {
            case ELaunchMode::CV: {
                for (auto foldIdx : xrange(foldCountOrTestSize)) {
                    learnSetNames.push_back("fold_" + ToString(foldIdx) + "_learn");
                    testSetNames.push_back("fold_" + ToString(foldIdx) + "_test");
                }
                break;
            }
            case ELaunchMode::Train: {
                const auto& learnToken = GetTrainModelLearnToken();
                const auto& testTokens = GetTrainModelTestTokens(foldCountOrTestSize);
                learnSetNames = { outputFiles.NamesPrefix + learnToken };
                for (int testIdx = 0; testIdx < testTokens.ysize(); ++testIdx) {
                    testSetNames.push_back({ outputFiles.NamesPrefix + testTokens[testIdx] });
                }
                break;
            }
            default: CB_ENSURE(false, "unexpected launchMode" << launchMode);
        }

        AddFileLoggers(
            false,
            outputFiles.LearnErrorLogFile,
            outputFiles.TestErrorLogFile,
            outputFiles.TimeLeftLogFile,
            outputFiles.JsonLogFile,
            outputFiles.ProfileLogFile,
            outputFiles.TrainDir,
            GetJsonMeta(
                iterationCount,
                outputFiles.ExperimentName,
                GetConstPointers(metrics),
                learnSetNames,
                testSetNames,
                parametersToken,
                launchMode),
                /*metric period*/ 1,
                logger
        );
    }

    static void LogTrainTest(
        const TString& lossDescription,
        TOneInterationLogger& oneIterLogger,
        const TMaybe<double> bestLearnResult,
        const double bestTestResult,
        const TString& learnToken,
        const TString& testToken,
        bool isMainMetric) {
        if (bestLearnResult.Defined()) {
            oneIterLogger.OutputMetric(
                learnToken,
                TMetricEvalResult(
                    lossDescription,
                    *bestLearnResult,
                    isMainMetric
                )
            );
        }
        oneIterLogger.OutputMetric(
            testToken,
            TMetricEvalResult(
                lossDescription,
                bestTestResult,
                isMainMetric
            )
        );
    }

    static void LogParameters(
        const TVector<TString>& paramNames,
        TConstArrayRef<NJson::TJsonValue> paramsSet,
        const TString& parametersToken,
        const TGeneralQuatizationParamsInfo& generalQuantizeParamsInfo,
        TOneInterationLogger& oneIterLogger) {
        NJson::TJsonValue jsonParams;
        // paramsSet: {border_count, feature_border_type, nan_mode, [others]}
        if (generalQuantizeParamsInfo.IsBordersCountInGrid) {
            jsonParams.InsertValue(generalQuantizeParamsInfo.BordersCountParamName, paramsSet[0]);
        }
        if (generalQuantizeParamsInfo.IsBorderTypeInGrid) {
            jsonParams.InsertValue(generalQuantizeParamsInfo.BorderTypeParamName, paramsSet[1]);
        }
        if (generalQuantizeParamsInfo.IsNanModeInGrid) {
            jsonParams.InsertValue(generalQuantizeParamsInfo.NanModeParamName, paramsSet[2]);
        }
        for (size_t idx = IndexOfFirstTrainingParameter; idx < paramsSet.size(); ++idx) {
            const auto key = paramNames[idx - IndexOfFirstTrainingParameter];
            jsonParams.InsertValue(key, paramsSet[idx]);
        }
        oneIterLogger.OutputParameters(parametersToken, jsonParams);
    }

    bool ParseJsonParams(
        const NCB::TDataMetaInfo& metaInfo,
        const NJson::TJsonValue& modelParamsToBeTried,
        NCatboostOptions::TCatBoostOptions *catBoostOptions,
        NCatboostOptions::TOutputFilesOptions *outputFileOptions,
        TString* paramsErrorMessage) {
        try {
            NJson::TJsonValue jsonParams;
            NJson::TJsonValue outputJsonParams;
            NCatboostOptions::PlainJsonToOptions(modelParamsToBeTried, &jsonParams, &outputJsonParams);
            ConvertParamsToCanonicalFormat(metaInfo, &jsonParams);
            *catBoostOptions = NCatboostOptions::LoadOptions(jsonParams);
            outputFileOptions->Load(outputJsonParams);

            return true;
        } catch (const TCatBoostException& exception) {
            *paramsErrorMessage = ToString(exception.what());
            return false;
        }
    }

    double TuneHyperparamsCV(
        const TVector<TString>& paramNames,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TCrossValidationParams& cvParams,
        const TGeneralQuatizationParamsInfo& generalQuantizeParamsInfo,
        ui64 cpuUsedRamLimit,
        NCB::TDataProviderPtr data,
        TProductIteratorBase<TDeque<NJson::TJsonValue>, NJson::TJsonValue>* gridIterator,
        NJson::TJsonValue* modelParamsToBeTried,
        TGridParamsInfo* bestGridParams,
        TVector<TCVResult>* bestCvResult,
        NPar::ILocalExecutor* localExecutor,
        int verbose,
        const THashMap<TString, NCB::TCustomRandomDistributionGenerator>& randDistGenerators = {}) {
        TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

        if (cvParams.Shuffle) {
            auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
            data = data->GetSubset(objectsGroupingSubset, cpuUsedRamLimit, localExecutor);
        }

        TSetLogging inThisScope(ELoggingLevel::Debug);
        TLogger logger;
        const auto parametersToken = GetParametersToken();
        TString searchToken = "loss";
        AddConsoleLogger(
            searchToken,
            {},
            /*hasTrain=*/true,
            verbose,
            gridIterator->GetTotalElementsCount(),
            &logger
        );
        double bestParamsSetMetricValue = 0;
        // Other parameters
        TQuantizationParamsInfo lastQuantizationParamsSet;
        TLabelConverter labelConverter;
        int iterationIdx = 0;
        int bestIterationIdx = 0;

        TProfileInfo profile(gridIterator->GetTotalElementsCount());
        TConstArrayRef<NJson::TJsonValue> paramsSet;
        TString paramsErrorString;
        bool foundValidParams = false;
        while (gridIterator->Next(&paramsSet)) {
            profile.StartIterationBlock();
            // paramsSet: {border_count, feature_border_type, nan_mode, [others]}
            TQuantizationParamsInfo quantizationParamsSet;
            quantizationParamsSet.BinsCount = GetRandomValueIfNeeded(paramsSet[0], randDistGenerators).GetInteger();
            quantizationParamsSet.BorderType = FromString<EBorderSelectionType>(paramsSet[1].GetString());
            quantizationParamsSet.NanMode = FromString<ENanMode>(paramsSet[2].GetString());

            AssignOptionsToJson(
                TConstArrayRef<TString>(paramNames),
                TConstArrayRef<NJson::TJsonValue>(
                    paramsSet.begin() + IndexOfFirstTrainingParameter,
                    paramsSet.end()
                ), // Ignoring quantization params
                randDistGenerators,
                modelParamsToBeTried
            );

            NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
            NCatboostOptions::TOutputFilesOptions outputFileOptions;
            bool areParamsValid = ParseJsonParams(
                data.Get()->MetaInfo,
                *modelParamsToBeTried,
                &catBoostOptions,
                &outputFileOptions,
                &paramsErrorString
            );
            if (!areParamsValid) {
                continue;
            }
            foundValidParams = true;

            TString tmpDir;
            if (outputFileOptions.AllowWriteFiles()) {
                NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputFileOptions.GetTrainDir(), &tmpDir);
            }
            InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric, &catBoostOptions.MetricOptions->EvalMetric);

            UpdateMetricPeriodOption(catBoostOptions, &outputFileOptions);

            NCB::TFeaturesLayoutPtr featuresLayout = data->MetaInfo.FeaturesLayout;
            NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;

            TMetricsAndTimeLeftHistory metricsAndTimeHistory;
            TVector<TCVResult> cvResult;
            {
                TSetLogging inThisScope(catBoostOptions.LoggingLevel);
                lastQuantizationParamsSet = quantizationParamsSet;
                CrossValidate(
                    *modelParamsToBeTried,
                    quantizedFeaturesInfo,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    labelConverter,
                    data,
                    cvParams,
                    localExecutor,
                    &cvResult);
            }
            ui32 approxDimension = NCB::GetApproxDimension(catBoostOptions, labelConverter, data->RawTargetData.GetTargetDimension());
            const TVector<THolder<IMetric>> metrics = CreateMetrics(
                catBoostOptions.MetricOptions,
                evalMetricDescriptor,
                approxDimension,
                data->MetaInfo.HasWeights
            );
            double bestMetricValue = cvResult[0].AverageTest.back(); //[testId][lossDescription]
            if (iterationIdx == 0) {
                // We guarantee to update the parameters on the first iteration
                bestParamsSetMetricValue = cvResult[0].AverageTest.back() + GetSignForMetricMinimization(metrics[0]);
                if (outputFileOptions.AllowWriteFiles()) {
                    // Initialize Files Loggers
                    TString namesPrefix = "fold_0_";
                    TOutputFiles outputFiles(outputFileOptions, namesPrefix);
                    InitializeFilesLoggers(
                        metrics,
                        outputFiles,
                        gridIterator->GetTotalElementsCount(),
                        ELaunchMode::CV,
                        cvParams.FoldCount,
                        parametersToken,
                        &logger
                    );
                }
            }
            bool isUpdateBest = SetBestParamsAndUpdateMetricValueIfNeeded(
                bestMetricValue,
                metrics,
                quantizationParamsSet,
                *modelParamsToBeTried,
                paramNames,
                quantizedFeaturesInfo,
                bestGridParams,
                &bestParamsSetMetricValue);
            if (isUpdateBest) {
                bestIterationIdx = iterationIdx;
                *bestCvResult = cvResult;
            }
            const TString& lossDescription = metrics[0]->GetDescription();
            TOneInterationLogger oneIterLogger(logger);
            oneIterLogger.OutputMetric(
                searchToken,
                TMetricEvalResult(
                    lossDescription,
                    bestMetricValue,
                    bestParamsSetMetricValue,
                    bestIterationIdx,
                    true
                )
            );
            if (outputFileOptions.AllowWriteFiles()) {
                //log metrics
                const auto& skipMetricOnTrain = GetSkipMetricOnTrain(metrics);
                for (auto foldIdx : xrange((size_t)cvParams.FoldCount)) {
                    for (auto metricIdx : xrange(metrics.size())) {
                        LogTrainTest(
                            metrics[metricIdx]->GetDescription(),
                            oneIterLogger,
                            skipMetricOnTrain[metricIdx] ? Nothing() :
                                MakeMaybe<double>(cvResult[metricIdx].LastTrainEvalMetric[foldIdx]),
                            cvResult[metricIdx].LastTestEvalMetric[foldIdx],
                            GetNamesPrefix(foldIdx) + "learn",
                            GetNamesPrefix(foldIdx) + "test",
                            metricIdx == 0
                        );
                    }
                }
                //log parameters
                LogParameters(
                    paramNames,
                    paramsSet,
                    parametersToken,
                    generalQuantizeParamsInfo,
                    oneIterLogger
                );
            }
            profile.FinishIterationBlock(1);
            oneIterLogger.OutputProfile(profile.GetProfileResults());
            iterationIdx++;
        }
        if (!foundValidParams) {
            ythrow TCatBoostException() << "All params in grid were invalid, last error message: " << paramsErrorString;
        }
        return bestParamsSetMetricValue;
    }

    double TuneHyperparamsTrainTest(
        const TVector<TString>& paramNames,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TGeneralQuatizationParamsInfo& generalQuantizeParamsInfo,
        ui64 cpuUsedRamLimit,
        NCB::TDataProviderPtr data,
        TProductIteratorBase<TDeque<NJson::TJsonValue>, NJson::TJsonValue>* gridIterator,
        NJson::TJsonValue* modelParamsToBeTried,
        TGridParamsInfo * bestGridParams,
        TMetricsAndTimeLeftHistory* trainTestResult,
        NPar::ILocalExecutor* localExecutor,
        int verbose,
        const THashMap<TString, NCB::TCustomRandomDistributionGenerator>& randDistGenerators = {}) {
        TRestorableFastRng64 rand(trainTestSplitParams.PartitionRandSeed);

        if (trainTestSplitParams.Shuffle) {
            auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
            data = data->GetSubset(objectsGroupingSubset, cpuUsedRamLimit, localExecutor);
        }

        TSetLogging inThisScope(ELoggingLevel::Debug);
        TLogger logger;
        TString searchToken = "loss";
        const auto parametersToken = GetParametersToken();
        AddConsoleLogger(
            searchToken,
            {},
            /*hasTrain=*/true,
            verbose,
            gridIterator->GetTotalElementsCount(),
            &logger
        );
        double bestParamsSetMetricValue = 0;
        // Other parameters
        NCB::TTrainingDataProviders trainTestData;
        TQuantizationParamsInfo lastQuantizationParamsSet;
        TLabelConverter labelConverter;
        int iterationIdx = 0;
        int bestIterationIdx = 0;
        TProfileInfo profile(gridIterator->GetTotalElementsCount());
        TConstArrayRef<NJson::TJsonValue> paramsSet;
        bool foundValidParams = false;
        TString paramsErrorString;
        while (gridIterator->Next(&paramsSet)) {
            profile.StartIterationBlock();
            // paramsSet: {border_count, feature_border_type, nan_mode, [others]}
            TQuantizationParamsInfo quantizationParamsSet;
            quantizationParamsSet.BinsCount = GetRandomValueIfNeeded(paramsSet[0], randDistGenerators).GetInteger();
            quantizationParamsSet.BorderType = FromString<EBorderSelectionType>(paramsSet[1].GetString());
            quantizationParamsSet.NanMode = FromString<ENanMode>(paramsSet[2].GetString());

            AssignOptionsToJson(
                TConstArrayRef<TString>(paramNames),
                TConstArrayRef<NJson::TJsonValue>(
                    paramsSet.begin() + IndexOfFirstTrainingParameter,
                    paramsSet.end()
                ), // Ignoring quantization params
                randDistGenerators,
                modelParamsToBeTried
            );

            NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
            NCatboostOptions::TOutputFilesOptions outputFileOptions;
            bool areParamsValid = ParseJsonParams(
                data.Get()->MetaInfo,
                *modelParamsToBeTried,
                &catBoostOptions,
                &outputFileOptions,
                &paramsErrorString
            );
            if (!areParamsValid) {
                continue;
            }
            foundValidParams = true;

            static const bool allowWriteFiles = outputFileOptions.AllowWriteFiles();
            TString tmpDir;
            if (allowWriteFiles) {
                NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputFileOptions.GetTrainDir(), &tmpDir);
            }

            InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric, &catBoostOptions.MetricOptions->EvalMetric);

            UpdateMetricPeriodOption(catBoostOptions, &outputFileOptions);

            UpdateSampleRateOption(data->GetObjectCount(), &catBoostOptions);
            NCB::TFeaturesLayoutPtr featuresLayout = data->MetaInfo.FeaturesLayout;
            NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;

            TMetricsAndTimeLeftHistory metricsAndTimeHistory;
            {
                TSetLogging inThisScope(catBoostOptions.LoggingLevel);
                QuantizeAndSplitDataIfNeeded(
                    allowWriteFiles,
                    tmpDir,
                    trainTestSplitParams,
                    cpuUsedRamLimit,
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
                THolder<IModelTrainer> modelTrainerHolder = THolder<IModelTrainer>(TTrainerFactory::Construct(catBoostOptions.GetTaskType()));

                TEvalResult evalRes;

                TTrainModelInternalOptions internalOptions;
                internalOptions.CalcMetricsOnly = true;
                internalOptions.ForceCalcEvalMetricOnEveryIteration = false;
                internalOptions.OffsetMetricPeriodByInitModelSize = true;
                outputFileOptions.SetAllowWriteFiles(false);
                const auto defaultTrainingCallbacks = MakeHolder<ITrainingCallbacks>();
                const auto defaultCustomCallbacks = MakeHolder<TCustomCallbacks>(Nothing());
                // Training model
                modelTrainerHolder->TrainModel(
                    internalOptions,
                    catBoostOptions,
                    outputFileOptions,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    trainTestData,
                    /*precomputedSingleOnlineCtrDataForSingleFold*/ Nothing(),
                    labelConverter,
                    defaultTrainingCallbacks.Get(), // TODO(ilikepugs): MLTOOLS-3540
                    defaultCustomCallbacks.Get(),
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
            }

            ui32 approxDimension = NCB::GetApproxDimension(catBoostOptions, labelConverter, data->RawTargetData.GetTargetDimension());
            const TVector<THolder<IMetric>> metrics = CreateMetrics(
                catBoostOptions.MetricOptions,
                evalMetricDescriptor,
                approxDimension,
                data->MetaInfo.HasWeights
            );

            const TString& lossDescription = metrics[0]->GetDescription();
            double bestMetricValue = metricsAndTimeHistory.TestBestError[0][lossDescription]; //[testId][lossDescription]
            if (iterationIdx == 0) {
                // We guarantee to update the parameters on the first iteration
                bestParamsSetMetricValue = bestMetricValue + GetSignForMetricMinimization(metrics[0]);
                outputFileOptions.SetAllowWriteFiles(allowWriteFiles);
                if (allowWriteFiles) {
                    // Initialize Files Loggers
                    TOutputFiles outputFiles(outputFileOptions, "");
                    InitializeFilesLoggers(
                        metrics,
                        outputFiles,
                        gridIterator->GetTotalElementsCount(),
                        ELaunchMode::Train,
                        trainTestData.Test.ysize(),
                        parametersToken,
                        &logger
                    );
                }
                (*trainTestResult) = metricsAndTimeHistory;
            }
            bool isUpdateBest = SetBestParamsAndUpdateMetricValueIfNeeded(
                bestMetricValue,
                metrics,
                quantizationParamsSet,
                *modelParamsToBeTried,
                paramNames,
                quantizedFeaturesInfo,
                bestGridParams,
                &bestParamsSetMetricValue);
            if (isUpdateBest) {
                bestIterationIdx = iterationIdx;
                (*trainTestResult) = metricsAndTimeHistory;
            }
            TOneInterationLogger oneIterLogger(logger);
            oneIterLogger.OutputMetric(
                searchToken,
                TMetricEvalResult(
                    lossDescription,
                    bestMetricValue,
                    bestParamsSetMetricValue,
                    bestIterationIdx,
                    true
                )
            );
            if (allowWriteFiles) {
                //log metrics
                const auto& skipMetricOnTrain = GetSkipMetricOnTrain(metrics);
                auto& learnErrors = metricsAndTimeHistory.LearnBestError;
                auto& testErrors = metricsAndTimeHistory.TestBestError[0];
                for (auto metricIdx : xrange(metrics.size())) {
                    const auto& lossDescription = metrics[metricIdx]->GetDescription();
                    LogTrainTest(
                        lossDescription,
                        oneIterLogger,
                        skipMetricOnTrain[metricIdx] ? Nothing() :
                            MakeMaybe<double>(learnErrors.at(lossDescription)),
                        testErrors.at(lossDescription),
                        "learn",
                        "test",
                        metricIdx == 0
                    );
                }
                //log parameters
                LogParameters(
                    paramNames,
                    paramsSet,
                    parametersToken,
                    generalQuantizeParamsInfo,
                    oneIterLogger
                );
            }
            profile.FinishIterationBlock(1);
            oneIterLogger.OutputProfile(profile.GetProfileResults());
            iterationIdx++;
        }
        if (!foundValidParams) {
            ythrow TCatBoostException() << "All params in grid were invalid, last error message: " << paramsErrorString;
        }
        return bestParamsSetMetricValue;
    }
} // anonymous namespace

namespace NCB {
    void TBestOptionValuesWithCvResult::SetOptionsFromJson(
        const THashMap<TString, NJson::TJsonValue>& options,
        const TVector<TString>& optionsNames) {

        BestParams = NJson::TJsonValue(NJson::JSON_MAP);
        auto& bestParamsMap = BestParams.GetMapSafe();

        for (const auto& optionName : optionsNames) {
            bestParamsMap.emplace(optionName, options.at(optionName));
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
        TMetricsAndTimeLeftHistory* trainTestResult,
        bool isSearchUsingTrainTestSplit,
        bool returnCvStat,
        int verbose) {

        // CatBoost options
        NJson::TJsonValue jsonParams;
        NJson::TJsonValue outputJsonParams;
        NCatboostOptions::PlainJsonToOptions(modelJsonParams, &jsonParams, &outputJsonParams);
        ConvertParamsToCanonicalFormat(data.Get()->MetaInfo, &jsonParams);
        NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
        NCatboostOptions::TOutputFilesOptions outputFileOptions;
        outputFileOptions.Load(outputJsonParams);
        CB_ENSURE(!outputJsonParams["save_snapshot"].GetBoolean(), "Snapshots are not yet supported for GridSearchCV");

        InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric, &catBoostOptions.MetricOptions->EvalMetric);

        UpdateMetricPeriodOption(catBoostOptions, &outputFileOptions);

        auto trainerEnv = NCB::CreateTrainerEnv(catBoostOptions);
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

        for (auto gridEnumerator : xrange(paramGrids.size())) {
            auto grid = paramGrids[gridEnumerator];
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

            const ui64 cpuUsedRamLimit
                = ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get());

            double metricValue;
            if (verbose && paramGrids.size() > 1) {
                TSetLogging inThisScope(ELoggingLevel::Verbose);
                CATBOOST_NOTICE_LOG << "Grid #" << gridEnumerator << Endl;
            }
            if (isSearchUsingTrainTestSplit) {
                metricValue = TuneHyperparamsTrainTest(
                    paramNames,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    trainTestSplitParams,
                    generalQuantizeParamsInfo,
                    cpuUsedRamLimit,
                    data,
                    &gridIterator,
                    &modelParamsToBeTried,
                    &gridParams,
                    trainTestResult,
                    &localExecutor,
                    verbose
                );
            } else {
                metricValue = TuneHyperparamsCV(
                    paramNames,
                    objectiveDescriptor,
                    evalMetricDescriptor,
                    cvParams,
                    generalQuantizeParamsInfo,
                    cpuUsedRamLimit,
                    data,
                    &gridIterator,
                    &modelParamsToBeTried,
                    &gridParams,
                    &bestCvResult,
                    &localExecutor,
                    verbose
                );
            }

            if (metricValue < bestParamsSetMetricValue) {
                bestGridParams = gridParams;
                bestGridParams.QuantizationParamsSet.GeneralInfo = generalQuantizeParamsInfo;
                SetGridParamsToBestOptionValues(bestGridParams, bestOptionValuesWithCvResult);
            }
        }
        trainerEnv.Reset();
        if (returnCvStat || isSearchUsingTrainTestSplit) {
            if (isSearchUsingTrainTestSplit) {
                if (verbose) {
                    TSetLogging inThisScope(ELoggingLevel::Verbose);
                    CATBOOST_NOTICE_LOG << "Estimating final quality...\n";
                }
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
        TMetricsAndTimeLeftHistory* trainTestResult,
        bool isSearchUsingTrainTestSplit,
        bool returnCvStat,
        int verbose) {

        // CatBoost options
        NJson::TJsonValue jsonParams;
        NJson::TJsonValue outputJsonParams;
        NCatboostOptions::PlainJsonToOptions(modelJsonParams, &jsonParams, &outputJsonParams);
        ConvertParamsToCanonicalFormat(data.Get()->MetaInfo, &jsonParams);
        NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
        NCatboostOptions::TOutputFilesOptions outputFileOptions;
        outputFileOptions.Load(outputJsonParams);
        CB_ENSURE(!outputJsonParams["save_snapshot"].GetBoolean(), "Snapshots are not yet supported for RandomizedSearchCV");

        InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric, &catBoostOptions.MetricOptions->EvalMetric);

        UpdateMetricPeriodOption(catBoostOptions, &outputFileOptions);

        auto trainerEnv = NCB::CreateTrainerEnv(catBoostOptions);
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

        const ui64 cpuUsedRamLimit
            = ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get());


        TGridParamsInfo bestGridParams;
        TVector<TCVResult> cvResult;
        if (isSearchUsingTrainTestSplit) {
            TuneHyperparamsTrainTest(
                paramNames,
                objectiveDescriptor,
                evalMetricDescriptor,
                trainTestSplitParams,
                generalQuantizeParamsInfo,
                cpuUsedRamLimit,
                data,
                &gridIterator,
                &modelParamsToBeTried,
                &bestGridParams,
                trainTestResult,
                &localExecutor,
                verbose,
                randDistGenerators
            );
        } else {
            TuneHyperparamsCV(
                paramNames,
                objectiveDescriptor,
                evalMetricDescriptor,
                cvParams,
                generalQuantizeParamsInfo,
                cpuUsedRamLimit,
                data,
                &gridIterator,
                &modelParamsToBeTried,
                &bestGridParams,
                &cvResult,
                &localExecutor,
                verbose,
                randDistGenerators
            );
        }
        trainerEnv.Reset();
        bestGridParams.QuantizationParamsSet.GeneralInfo = generalQuantizeParamsInfo;
        SetGridParamsToBestOptionValues(bestGridParams, bestOptionValuesWithCvResult);
        if (returnCvStat || isSearchUsingTrainTestSplit) {
            if (isSearchUsingTrainTestSplit) {
                if (verbose) {
                    TSetLogging inThisScope(ELoggingLevel::Verbose);
                    CATBOOST_NOTICE_LOG << "Estimating final quality...\n";
                }
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
