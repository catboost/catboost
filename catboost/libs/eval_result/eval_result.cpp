#include "eval_result.h"

#include "eval_helpers.h"
#include "pool_printer.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/enum_helpers.h>

#include <util/generic/hash_set.h>
#include <util/stream/fwd.h>
#include <util/string/builder.h>
#include <util/string/cast.h>


using namespace NCB;


const TString BaselinePrefix = "Baseline#";

namespace {
    struct TFeatureDesc {
        size_t Index;
        bool IsCategorical;

        TFeatureDesc(size_t Index_, bool IsCategorical_)
            : Index(Index_)
            , IsCategorical(IsCategorical_)
        {
        }
    };

    using TFeatureIdToDesc = THashMap<TString, TFeatureDesc>;
}

static TFeatureIdToDesc GetFeatureIdToDesc(const TDataProvider& pool) {
    TFeatureIdToDesc res;

    const auto& poolFeaturesMetaInfo = pool.MetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo();

    for (size_t index = 0; index < poolFeaturesMetaInfo.size(); ++index) {
        res.emplace(
            poolFeaturesMetaInfo[index].Name,
            TFeatureDesc{index, poolFeaturesMetaInfo[index].Type == EFeatureType::Categorical}
        );
    }

    return res;
}

namespace NCB {

    TVector<TVector<TVector<double>>>& TEvalResult::GetRawValuesRef() {
        return RawValues;
    }

    const TVector<TVector<TVector<double>>>& TEvalResult::GetRawValuesConstRef() const {
        return RawValues;
    }

    size_t TEvalResult::GetEnsemblesCount() const {
        return EnsemblesCount;
    }

    void TEvalResult::ClearRawValues() {
        RawValues.clear();
        RawValues.resize(1);
    }

    void TEvalResult::SetRawValuesByMove(TVector<TVector<double>>& rawValues) {
        if (RawValues.size() < 1) {
            RawValues.resize(1);
        }
        RawValues[0] = std::move(rawValues);
    }

    void ValidateColumnOutput(
        const TVector<TVector<TString>>& outputColumns,
        const TDataProvider& pool,
        bool cvMode)
    {
        THashSet<TString> featureIds;
        for (const auto& featureMetaInfo : pool.MetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()) {
            featureIds.emplace(featureMetaInfo.Name);
        }
        THashSet<TString> auxiliaryColumnNames;
        if (pool.MetaInfo.ColumnsInfo.Defined()) {
            for (const auto& columnMetaInfo : pool.MetaInfo.ColumnsInfo->Columns) {
                if (columnMetaInfo.Type == EColumn::Auxiliary && columnMetaInfo.Id) {
                    auxiliaryColumnNames.insert(columnMetaInfo.Id);
                }
            }
        }

        bool notQuantizedPool = bool(dynamic_cast<TRawObjectsDataProvider*>(pool.ObjectsData.Get()));

        for (const auto& columns : outputColumns) {
            bool hasPrediction = false;

            for (const auto& name : columns) {
                EPredictionType predictionType;
                if (TryFromString<EPredictionType>(name, predictionType)) {
                    hasPrediction = true;
                    continue;
                }

                EColumn columnType;
                if (TryFromString<EColumn>(ToCanonicalColumnName(name), columnType)) {
                    switch (columnType) {
                        case (EColumn::Label):
                            CB_ENSURE(
                                pool.MetaInfo.TargetCount > 0,
                                "bad output column name " << name << " (No target/label info in pool)"
                            );
                            break;
                        case (EColumn::Baseline):
                            CB_ENSURE(
                                pool.MetaInfo.BaselineCount > 0,
                                "bad output column name " << name << " (No baseline info in pool)"
                            );
                            break;
                        case (EColumn::Weight):
                            CB_ENSURE(
                                pool.MetaInfo.HasWeights,
                                "bad output column name " << name << " (No WeightId info in pool)"
                            );
                            break;
                        case (EColumn::GroupId):
                            CB_ENSURE(
                                pool.MetaInfo.HasGroupId,
                                "bad output column name " << name << " (No GroupId info in pool)"
                            );
                            break;
                        case (EColumn::SubgroupId):
                            CB_ENSURE(
                                pool.MetaInfo.HasSubgroupIds,
                                "bad output column name " << name << " (No SubgroupIds info in pool)"
                            );
                            break;
                        case (EColumn::Timestamp):
                            CB_ENSURE(
                                pool.MetaInfo.HasTimestamp,
                                "bad output column name " << name << " (No Timestamp info in pool)"
                            );
                            break;
                        default:
                            CB_ENSURE(
                                columnType != EColumn::Auxiliary && !IsFactorColumn(columnType),
                                "bad output column type " << name
                            );
                            break;
                    }
                    continue;
                }

                if (!name.compare(0, BaselinePrefix.length(), BaselinePrefix)) {
                    size_t baselineInd = FromString<int>(name.substr(BaselinePrefix.length()));
                    CB_ENSURE(
                        baselineInd < pool.MetaInfo.BaselineCount,
                        "bad output column name " << name << ", Baseline columns count: " << pool.MetaInfo.BaselineCount
                    );
                    continue;
                }

                if (name[0] == '#') {
                    CB_ENSURE(notQuantizedPool, "Quantized pool, can't specify column index");
                    ui32 columnNumber;
                    TString columnName;
                    ParseOutputColumnByIndex(name, &columnNumber, &columnName);
                    CB_ENSURE(
                        columnNumber < pool.MetaInfo.FeaturesLayout->GetExternalFeatureCount(),
                        "column number " << columnNumber << " is out of range"
                    );
                } else {
                    if (auxiliaryColumnNames.contains(name)) {  // can add by Id
                        continue;
                    }
                    CB_ENSURE(featureIds.contains(name), "Pool doesn't has column with name `" << name << "`.");
                    CB_ENSURE(
                        notQuantizedPool,
                        "Raw feature values are not available for quantized pools"
                    );
                }
                CB_ENSURE(!cvMode, "can't output pool column in cross validation mode");
            }
            CB_ENSURE(hasPrediction, "No prediction type chosen in output-column header");
        }
    }

    TVector<THolder<IColumnPrinter>> InitializeColumnWriter(
        const TEvalColumnsInfo& evalColumnsInfo,
        NPar::ILocalExecutor* executor,
        const TVector<TVector<TString>>& outputColumns, // [modelIdx]
        const TDataProvider& pool,
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter,
        std::pair<int, int> testFileWhichOf,
        ui64 docIdOffset,
        bool* needColumnsPrinterPtr,
        TMaybe<std::pair<size_t, size_t>> evalParameters,
        double binClassLogitThreshold) {

        TFeatureIdToDesc featureIdToDesc = GetFeatureIdToDesc(pool);

        TVector<THolder<IColumnPrinter>> columnPrinter;

        const auto targetDim = pool.RawTargetData.GetTargetDimension();
        const bool isMultiTarget = targetDim > 1;
        const auto modelCount = evalColumnsInfo.Approxes.size();
        CB_ENSURE_INTERNAL(
            modelCount == outputColumns.size()
            && modelCount == evalColumnsInfo.LossFunctions.size()
            && modelCount == evalColumnsInfo.LabelHelpers.size(),
            "Invalid evalColumnsInfo"
        );

        *needColumnsPrinterPtr = false;

        for (auto modelIdx : xrange(modelCount)) {
            const auto& lossFunction = evalColumnsInfo.LossFunctions[modelIdx];
            const bool isMultiLabel = !lossFunction.empty() && IsMultiLabelObjective(lossFunction);
            TMaybe<TString> modelName;
            if (modelCount > 1) {
                modelName = TString("Model") + ToString(modelIdx);
            }
            const auto& approx = evalColumnsInfo.Approxes[modelIdx];
            const auto& labelHelper = evalColumnsInfo.LabelHelpers[modelIdx];
            for (const auto& outputColumn : outputColumns[modelIdx]) {
                EPredictionType type;
                if (TryFromString<EPredictionType>(outputColumn, type)) {
                    PushBackEvalPrinters(
                        approx.GetRawValuesConstRef(),
                        type,
                        lossFunction,
                        modelName,
                        isMultiTarget,
                        approx.GetEnsemblesCount(),
                        labelHelper,
                        evalParameters,
                        &columnPrinter,
                        executor,
                        binClassLogitThreshold
                    );
                    continue;
                }
                EColumn outputType;
                if (TryFromString<EColumn>(ToCanonicalColumnName(outputColumn), outputType)) {
                    if (outputType == EColumn::Label) {
                        const auto target = pool.RawTargetData.GetTarget().GetRef();
                        for (auto targetIdx : xrange(targetDim)) {
                            TStringBuilder header;
                            header << outputColumn;
                            if (targetDim > 1) {
                                if (isMultiLabel) {
                                    header << ":Class=" << labelHelper.GetVisibleClassNameFromClass(targetIdx);
                                } else {
                                    header << ":Dim=" << targetIdx;
                                }
                            }
                            if (const ITypedSequencePtr<float>* typedSequence
                                    = std::get_if<ITypedSequencePtr<float>>(&(target[targetIdx])))
                            {
                                columnPrinter.push_back(
                                    MakeHolder<TArrayPrinter<float>>(
                                        ToVector(**typedSequence),
                                        header
                                    )
                                );
                            } else {
                                columnPrinter.push_back(
                                    MakeHolder<TArrayPrinter<TString>>(
                                        std::get<TVector<TString>>(target[targetIdx]),
                                        header
                                    )
                                );
                            }
                        }
                        continue;
                    }
                    if (outputType == EColumn::SampleId) {
                        if (testFileWhichOf.second > 1) {
                            columnPrinter.push_back(
                                MakeHolder<TPrefixPrinter<TString>>(ToString(testFileWhichOf.first), "EvalSet", ":")
                            );
                        }
                        auto printer = MakeHolder<TDocIdPrinter>(poolColumnsPrinter, docIdOffset, outputColumn);
                        if (printer->NeedPrinterPtr()) {
                            *needColumnsPrinterPtr = true;
                        }
                        columnPrinter.emplace_back(printer.Release());
                        continue;
                    }
                    if (outputType == EColumn::Timestamp) {
                        columnPrinter.push_back(
                            MakeHolder<TArrayPrinter<ui64>>(*pool.ObjectsData->GetTimestamp(), outputColumn)
                        );
                        continue;
                    }
                    if (outputType == EColumn::Weight) {
                        columnPrinter.push_back(MakeHolder<TWeightsPrinter>(pool.RawTargetData.GetWeights(), outputColumn));
                        continue;
                    }
                    if ((outputType == EColumn::GroupId) || (outputType == EColumn::SubgroupId)) {
                        columnPrinter.push_back(
                            (THolder<IColumnPrinter>)MakeHolder<TColumnPrinter>(
                                poolColumnsPrinter,
                                outputType,
                                docIdOffset,
                                outputColumn
                            )
                        );
                        *needColumnsPrinterPtr = true;
                        continue;
                    }
                    if (outputType == EColumn::GroupWeight) {
                        columnPrinter.push_back(
                            MakeHolder<TWeightsPrinter>(pool.RawTargetData.GetGroupWeights(), outputColumn)
                        );
                        continue;
                    }
                    if (outputType == EColumn::Baseline) {
                        auto baseline = pool.RawTargetData.GetBaseline().GetRef();
                        for (size_t idx = 0; idx < baseline.size(); ++idx) {
                            TStringBuilder header;
                            header << "Baseline";
                            if (baseline.size() > 1) {
                                header << "#" << idx;
                            }
                            columnPrinter.push_back(MakeHolder<TArrayPrinter<float>>(baseline[idx], header));
                        }
                        continue;
                    }
                }
                if (!outputColumn.compare(0, BaselinePrefix.length(), BaselinePrefix)) {
                    int idx = FromString<int>(outputColumn.substr(BaselinePrefix.length()));
                    // NonOwning
                    columnPrinter.push_back(
                        MakeHolder<TArrayPrinter<float>>((*pool.RawTargetData.GetBaseline())[idx], outputColumn)
                    );
                    continue;
                }
                if (outputColumn[0] == '#') {
                    ui32 columnNumber;
                    TString columnName;
                    ParseOutputColumnByIndex(outputColumn, &columnNumber, &columnName);

                    columnPrinter.push_back(
                        MakeHolder<TFeatureColumnPrinter>(
                            poolColumnsPrinter,
                            columnNumber,
                            columnName,
                            docIdOffset
                        )
                    );
                    *needColumnsPrinterPtr = true;
                } else if (poolColumnsPrinter->ValidAuxiliaryColumn(outputColumn)) {
                    columnPrinter.push_back(
                        MakeHolder<TAuxiliaryColumnPrinter>(
                            poolColumnsPrinter,
                            outputColumn,
                            docIdOffset
                        )
                    );
                    *needColumnsPrinterPtr = true;
                } else {
                    auto it = featureIdToDesc.find(outputColumn);
                    CB_ENSURE(
                        it != featureIdToDesc.end(),
                        "output column [" << outputColumn << "] not found in featureIds"
                    );

                    const auto* rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(pool.ObjectsData.Get());
                    CB_ENSURE(
                        rawObjectsData,
                        "Raw feature values are not available for quantized pools"
                    );

                    auto internalIdx = pool.MetaInfo.FeaturesLayout->GetInternalFeatureIdx(it->second.Index);

                    if (it->second.IsCategorical) {
                        columnPrinter.push_back(
                            MakeHolder<TCatFeaturePrinter>(
                                (*rawObjectsData->GetCatFeature(internalIdx))->ExtractValues(executor),
                                rawObjectsData->GetCatFeaturesHashToString(internalIdx),
                                outputColumn
                            )
                        );
                    } else {
                        TMaybeOwningArrayHolder<float> extractedValues
                            = (*rawObjectsData->GetFloatFeature(internalIdx))->ExtractValues(executor);
                        columnPrinter.push_back(
                            MakeHolder<TArrayPrinter<float>>(
                                TMaybeOwningConstArrayHolder<float>::CreateOwningReinterpretCast(extractedValues),
                                outputColumn
                            )
                        );
                    }
                }
            }
        }
        return columnPrinter;
    }

    void OutputEvalResultToFile(
        const TEvalColumnsInfo& evalColumnsInfo,
        NPar::ILocalExecutor* executor,
        const TVector<TVector<TString>>& outputColumns, // [modelIdx]
        const TDataProvider& pool,
        IOutputStream* outputStream,
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter,
        std::pair<int, int> testFileWhichOf,
        bool writeHeader,
        ui64 docIdOffset,
        TMaybe<std::pair<size_t, size_t>> evalParameters, // evalPeriod, iterationsLimit
        double binClassLogitThreshold) {

        bool needPoolColumnsPrinter;
        TVector<THolder<IColumnPrinter>> columnPrinter = InitializeColumnWriter(
            evalColumnsInfo,
            executor,
            outputColumns,
            pool,
            poolColumnsPrinter,
            testFileWhichOf,
            docIdOffset,
            &needPoolColumnsPrinter,
            evalParameters,
            binClassLogitThreshold
        );

        if (writeHeader) {
            TString delimiter = "";
            for (auto& printer : columnPrinter) {
                *outputStream << delimiter;
                printer->OutputHeader(outputStream);
                delimiter = printer->GetAfterColumnDelimiter();
            }
            *outputStream << Endl;
        }
        for (ui32 docId = 0; docId < pool.ObjectsGrouping->GetObjectCount(); ++docId) {
            TString delimiter = "";
            for (auto& printer : columnPrinter) {
                *outputStream << delimiter;
                printer->OutputValue(outputStream, docId);
                delimiter = printer->GetAfterColumnDelimiter();
            }
            *outputStream << Endl;
        }
    }

    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        NPar::ILocalExecutor* const executor,
        const TVector<TString>& outputColumns,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TDataProvider& pool,
        IOutputStream* outputStream,
        const TPathWithScheme& testSetPath,
        std::pair<int, int> testFileWhichOf,
        const TDsvFormatOptions& testSetFormat,
        bool writeHeader,
        ui64 docIdOffset,
        double binClassLogitThreshold) {

        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter;
        if (testSetPath.Inited()) {
            poolColumnsPrinter = GetProcessor<IPoolColumnsPrinter>(
                testSetPath,
                TPoolColumnsPrinterPullArgs{testSetPath, testSetFormat, pool.MetaInfo.ColumnsInfo}
            ).Release();
        }
        const TEvalColumnsInfo evalColumnInfo{
            {evalResult},
            {visibleLabelsHelper},
            {lossFunctionName}
        };
        OutputEvalResultToFile(
            evalColumnInfo,
            executor,
            {outputColumns},
            pool,
            outputStream,
            poolColumnsPrinter,
            testFileWhichOf,
            writeHeader,
            docIdOffset,
            {},
            binClassLogitThreshold
        );
    }

} // namespace NCB
