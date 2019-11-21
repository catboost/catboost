#include "eval_result.h"

#include "eval_helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

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

    public:
        // no sane default-initialization
        TFeatureDesc() = delete;
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

    void ValidateColumnOutput(const TVector<TString>& outputColumns,
                              const TDataProvider& pool,
                              bool CV_mode)
    {
        THashSet<TString> featureIds;
        for (const auto& featureMetaInfo : pool.MetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()) {
            featureIds.emplace(featureMetaInfo.Name);
        }

        bool hasPrediction = false;

        for (const auto& name : outputColumns) {
            EPredictionType predictionType;
            if (TryFromString<EPredictionType>(name, predictionType)) {
                hasPrediction = true;
                continue;
            }

            EColumn columnType;
            if (TryFromString<EColumn>(ToCanonicalColumnName(name), columnType)) {
                switch (columnType) {
                    case (EColumn::Label):
                        CB_ENSURE(pool.MetaInfo.TargetCount > 0, "bad output column name " << name << " (No target/label info in pool)");
                        break;
                    case (EColumn::Baseline):
                        CB_ENSURE(pool.MetaInfo.BaselineCount > 0, "bad output column name " << name << " (No baseline info in pool)");
                        break;
                    case (EColumn::Weight):
                        CB_ENSURE(pool.MetaInfo.HasWeights, "bad output column name " << name << " (No WeightId info in pool)");
                        break;
                    case (EColumn::GroupId):
                        CB_ENSURE(pool.MetaInfo.ColumnsInfo.Defined(), "GroupId output is currently supported only for columnar pools");
                        CB_ENSURE(pool.MetaInfo.HasGroupId, "bad output column name " << name << " (No GroupId info in pool)");
                        break;
                    case (EColumn::SubgroupId):
                        CB_ENSURE(pool.MetaInfo.ColumnsInfo.Defined(), "SubgroupId output is currently supported only for columnar pools");
                        CB_ENSURE(pool.MetaInfo.HasSubgroupIds, "bad output column name " << name << " (No SubgroupIds info in pool)");
                        break;
                    case (EColumn::Timestamp):
                        CB_ENSURE(pool.MetaInfo.HasTimestamp, "bad output column name " << name << " (No Timestamp info in pool)");
                        break;
                    default:
                        CB_ENSURE(columnType != EColumn::Auxiliary && !IsFactorColumn(columnType), "bad output column type " << name);
                        break;
                }
                continue;
            }

            if (!name.compare(0, BaselinePrefix.length(), BaselinePrefix)) {
                size_t baselineInd = FromString<int>(name.substr(BaselinePrefix.length()));
                CB_ENSURE(baselineInd < pool.MetaInfo.BaselineCount, "bad output column name " << name << ", Baseline columns count: " << pool.MetaInfo.BaselineCount);
                continue;
            }

            if (name[0] == '#') {
                CB_ENSURE(pool.MetaInfo.ColumnsInfo.Defined(),
                          "Non-columnar pool, can't specify column index");
                ui32 columnNumber;
                TString columnName;
                ParseOutputColumnByIndex(name, &columnNumber, &columnName);
                CB_ENSURE(columnNumber < pool.MetaInfo.ColumnsInfo->Columns.size(),
                          "column number " << columnNumber << " is out of range");
            } else {
                CB_ENSURE(featureIds.contains(name), "bad output column name " << name);
                CB_ENSURE(
                    dynamic_cast<TRawObjectsDataProvider*>(pool.ObjectsData.Get()),
                    "Raw feature values are not available for quantized pools"
                );
            }
            CB_ENSURE(!CV_mode, "can't output pool column in cross validation mode");
        }
        CB_ENSURE(hasPrediction, "No prediction type chosen in output-column header");
    }

    TIntrusivePtr<IPoolColumnsPrinter> CreatePoolColumnPrinter(
        const TPathWithScheme& testSetPath,
        const TDsvFormatOptions& testSetFormat,
        const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo
    ) {
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter;
        if (testSetPath.Inited()) {
            if (testSetPath.Scheme.Contains("quantized")) {
                poolColumnsPrinter = TIntrusivePtr<IPoolColumnsPrinter>(new TQuantizedPoolColumnsPrinter(testSetPath));
            } else if (testSetPath.Scheme.Contains("dsv")) {
                poolColumnsPrinter = TIntrusivePtr<IPoolColumnsPrinter>(new TDSVPoolColumnsPrinter(testSetPath, testSetFormat, columnsMetaInfo));
            }
        }
        return poolColumnsPrinter;
    }

    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        NPar::TLocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TDataProvider& pool,
        IOutputStream* outputStream,
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter,
        std::pair<int, int> testFileWhichOf,
        bool writeHeader,
        ui64 docIdOffset,
        TMaybe<std::pair<size_t, size_t>> evalParameters) {

        TFeatureIdToDesc featureIdToDesc = GetFeatureIdToDesc(pool);

        TVector<THolder<IColumnPrinter>> columnPrinter;

        for (const auto& outputColumn : outputColumns) {
            EPredictionType type;
            if (TryFromString<EPredictionType>(outputColumn, type)) {
                columnPrinter.push_back(MakeHolder<TEvalPrinter>(executor, evalResult.GetRawValuesConstRef(), type, lossFunctionName,
                                                                 pool.RawTargetData.GetTargetDimension(), visibleLabelsHelper, evalParameters));
                continue;
            }
            EColumn outputType;
            if (TryFromString<EColumn>(ToCanonicalColumnName(outputColumn), outputType)) {
                if (outputType == EColumn::Label) {
                    const auto target = pool.RawTargetData.GetTarget().GetRef();
                    const auto targetDim = target.size();
                    for (auto targetIdx : xrange(targetDim)) {
                        TStringBuilder header;
                        header << outputColumn;
                        if (targetDim > 1) {
                            header << ":Dim=" << targetIdx;
                        }
                        if (const ITypedSequencePtr<float>* typedSequence
                                = GetIf<ITypedSequencePtr<float>>(&(target[targetIdx])))
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
                                    Get<TVector<TString>>(target[targetIdx]),
                                    header
                                )
                            );
                        }
                    }
                    continue;
                }
                if (outputType == EColumn::SampleId) {
                    if (testFileWhichOf.second > 1) {
                        columnPrinter.push_back(MakeHolder<TPrefixPrinter<TString>>(ToString(testFileWhichOf.first), "EvalSet", ":"));
                    }
                    columnPrinter.push_back(
                        (THolder<IColumnPrinter>)MakeHolder<TDocIdPrinter>(
                            poolColumnsPrinter,
                            docIdOffset,
                            outputColumn)
                    );
                    continue;
                }
                if (outputType == EColumn::Timestamp) {
                    columnPrinter.push_back(MakeHolder<TArrayPrinter<ui64>>(*pool.ObjectsData->GetTimestamp(), outputColumn));
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
                            outputColumn)
                    );
                    continue;
                }
                if (outputType == EColumn::Baseline) {
                    auto baseline = *pool.RawTargetData.GetBaseline();
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
                columnPrinter.push_back(MakeHolder<TArrayPrinter<float>>((*pool.RawTargetData.GetBaseline())[idx], outputColumn));
                continue;
            }
            if (outputColumn[0] == '#') {
                ui32 columnNumber;
                TString columnName;
                ParseOutputColumnByIndex(outputColumn, &columnNumber, &columnName);

                columnPrinter.push_back(
                    MakeHolder<TNumColumnPrinter>(
                        poolColumnsPrinter,
                        columnNumber,
                        columnName,
                        docIdOffset
                    )
                );
            } else {
                auto it = featureIdToDesc.find(outputColumn);
                CB_ENSURE(it != featureIdToDesc.end(),
                          "output column [" << outputColumn << "] not found in featureIds");

                const auto* rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(pool.ObjectsData.Get());
                CB_ENSURE(
                    rawObjectsData,
                    "Raw feature values are not available for quantized pools"
                );

                if (it->second.IsCategorical) {
                    columnPrinter.push_back(
                        MakeHolder<TCatFeaturePrinter>(
                            (*rawObjectsData->GetCatFeature(it->second.Index))->ExtractValues(executor),
                            rawObjectsData->GetCatFeaturesHashToString(it->second.Index),
                            outputColumn
                        )
                    );
                } else {
                    TMaybeOwningArrayHolder<float> extractedValues
                        = (*rawObjectsData->GetFloatFeature(it->second.Index))->ExtractValues(executor);
                    columnPrinter.push_back(
                        MakeHolder<TArrayPrinter<float>>(
                            TMaybeOwningConstArrayHolder<float>::CreateOwningReinterpretCast(extractedValues),
                            outputColumn
                        )
                    );
                }
            }
        }
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
        NPar::TLocalExecutor* const executor,
        const TVector<TString>& outputColumns,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TDataProvider& pool,
        IOutputStream* outputStream,
        const TPathWithScheme& testSetPath,
        std::pair<int, int> testFileWhichOf,
        const TDsvFormatOptions& testSetFormat,
        bool writeHeader,
        ui64 docIdOffset) {

        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter = CreatePoolColumnPrinter(
            testSetPath,
            testSetFormat,
            pool.MetaInfo.ColumnsInfo);

        OutputEvalResultToFile(
            evalResult,
            executor,
            outputColumns,
            lossFunctionName,
            visibleLabelsHelper,
            pool,
            outputStream,
            poolColumnsPrinter,
            testFileWhichOf,
            writeHeader,
            docIdOffset);
    }

} // namespace NCB
