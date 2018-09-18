#include "eval_result.h"
#include "pool_printer.h"
#include "column_printer.h"

#include <util/generic/hash_set.h>


const TString BaselinePrefix = "Baseline#";

namespace {
    struct TFeatureDesc {
        int Index;
        bool IsCategorical;

    public:
        // no sane default-initialization
        TFeatureDesc() = delete;
    };

    using TFeatureIdToDesc = THashMap<TString, TFeatureDesc>;
}

static TFeatureIdToDesc GetFeatureIdToDesc(const TPool& pool) {
    TFeatureIdToDesc res;

    THashSet<int> catFeatures(pool.CatFeatures.begin(), pool.CatFeatures.end());

    for (int index = 0; index < pool.FeatureId.ysize(); ++index) {
        res.emplace(pool.FeatureId[index], TFeatureDesc{index, catFeatures.has(index)});
    }

    return res;
}


static int GetColumnIndex(const TPoolColumnsMetaInfo& poolColumnsMetaInfo, EColumn columnType) {
    const auto& columns = poolColumnsMetaInfo.Columns;
    auto it = FindIf(columns.begin(), columns.end(),
                     [columnType](const TColumn& col) {
                         return col.Type == columnType;
                     });
    CB_ENSURE(it != columns.end(), "column " << columnType << " not found");
    return int(it - columns.begin());
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
                              const TPool& pool,
                              bool isPartOfFullTestSet,
                              bool CV_mode)
    {
        THashSet<TString> featureIds(pool.FeatureId.begin(), pool.FeatureId.end());

        bool hasPrediction = false;

        for (const auto& name : outputColumns) {
            EPredictionType predictionType;
            if (TryFromString<EPredictionType>(name, predictionType)) {
                hasPrediction = true;
                continue;
            }

            EColumn columnType;
            if (TryFromString<EColumn>(name, columnType)) {
                switch (columnType) {
                    case (EColumn::Baseline):
                        CB_ENSURE(pool.MetaInfo.BaselineCount > 0, "bad output column name " << name << " (No baseline info in pool)");
                        break;
                    case (EColumn::Weight):
                        CB_ENSURE(pool.MetaInfo.HasWeights, "bad output column name " << name << " (No WeightId info in pool)");
                        break;
                    case (EColumn::GroupId):
                        CB_ENSURE(pool.MetaInfo.ColumnsInfo.Defined(), "GroupId output is currently supported only for columnar pools");
                        CB_ENSURE(pool.MetaInfo.HasGroupId, "bad output column name " << name << " (No GroupId info in pool)");
                        CB_ENSURE(!isPartOfFullTestSet, "GroupId output is currently supported only for full pools, not pool parts");
                        break;
                    case (EColumn::SubgroupId):
                        CB_ENSURE(pool.MetaInfo.ColumnsInfo.Defined(), "SubgroupId output is currently supported only for columnar pools");
                        CB_ENSURE(pool.MetaInfo.HasSubgroupIds, "bad output column name " << name << " (No SubgroupIds info in pool)");
                        CB_ENSURE(!isPartOfFullTestSet, "SubgroupId output is currently supported only for full pools, not pool parts");
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
                CB_ENSURE(FromString<size_t>(name.substr(1)) < pool.MetaInfo.ColumnsInfo->Columns.size(),
                          "bad output column name " << name);
                CB_ENSURE(!isPartOfFullTestSet, "Column output by # is currently supported only for full pools, not pool parts");
            } else {
                CB_ENSURE(featureIds.has(name), "bad output column name " << name);
            }
            CB_ENSURE(!CV_mode, "can't output pool column in cross validation mode");
        }
        CB_ENSURE(hasPrediction, "No prediction type chosen in output-column header");
    }



    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        NPar::TLocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TPool& pool,
        bool isPartOfFullTestSet,
        IOutputStream* outputStream,
        const TPathWithScheme& testSetPath,
        std::pair<int, int> testFileWhichOf,
        const TDsvFormatOptions& testSetFormat,
        bool writeHeader,
        ui64 docIdOffset,
        TMaybe<std::pair<size_t, size_t>> evalParameters) {

        TFeatureIdToDesc featureIdToDesc = GetFeatureIdToDesc(pool);

        TVector<TString> convertedTarget = ConvertTargetToExternalName(pool.Docs.Target, visibleLabelsHelper);

        TVector<THolder<IColumnPrinter>> columnPrinter;

        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter;

        // lazy init
        auto getPoolColumnsPrinter = [&]() {
            /* there's a special case when poolColumnsPrinter can be empty:
              when we need only header output
            */
            if (testSetPath.Inited() && !poolColumnsPrinter) {
                if (testSetPath.Scheme == "quantized") {
                    poolColumnsPrinter = TIntrusivePtr<IPoolColumnsPrinter>(new TQuantizedPoolColumnsPrinter(testSetPath));
                } else {
                    CB_ENSURE(testSetPath.Scheme == "dsv", "OutputEvalResultToFile supports only quantized and dsv pool schema. Got: " << testSetPath.Scheme << ".");
                    poolColumnsPrinter = TIntrusivePtr<IPoolColumnsPrinter>(new TDSVPoolColumnsPrinter(testSetPath, testSetFormat, pool.MetaInfo.ColumnsInfo->Columns));
                }
            }
            return poolColumnsPrinter;
        };

        for (const auto& columnName : outputColumns) {
            EPredictionType type;
            if (TryFromString<EPredictionType>(columnName, type)) {
                columnPrinter.push_back(MakeHolder<TEvalPrinter>(executor, evalResult.GetRawValuesConstRef(), type, visibleLabelsHelper, evalParameters));
                continue;
            }
            EColumn outputType;
            if (TryFromString<EColumn>(columnName, outputType)) {
                if (outputType == EColumn::Label) {
                    if  (!pool.Docs.Target.empty()) {
                        columnPrinter.push_back(MakeHolder<TVectorPrinter<TString>>(convertedTarget, columnName));
                    }
                    continue;
                }
                if (outputType == EColumn::DocId) {
                    if (testFileWhichOf.second > 1) {
                        columnPrinter.push_back(MakeHolder<TPrefixPrinter<TString>>(ToString(testFileWhichOf.first), "EvalSet", ":"));
                    }
                    bool needToGenerate = true;
                    TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter;
                    try {
                        GetColumnIndex(*(pool.MetaInfo.ColumnsInfo), outputType);
                        poolColumnsPrinter = getPoolColumnsPrinter();
                        needToGenerate = false;
                    } catch (...) {
                    }
                    columnPrinter.push_back(
                          (THolder<IColumnPrinter>)MakeHolder<TDocIdPrinter>(
                              poolColumnsPrinter,
                              needToGenerate,
                              docIdOffset,
                              columnName
                          )
                    );
                    continue;
                }
                if (outputType == EColumn::Timestamp) {
                    columnPrinter.push_back(MakeHolder<TVectorPrinter<ui64>>(pool.Docs.Timestamp, columnName));
                    continue;
                }
                if (outputType == EColumn::Weight) {
                    columnPrinter.push_back(MakeHolder<TVectorPrinter<float>>(pool.Docs.Weight, columnName));
                    continue;
                }
                if ((outputType == EColumn::GroupId) || (outputType == EColumn::SubgroupId)) {
                    CB_ENSURE(pool.MetaInfo.ColumnsInfo.Defined(), "GroupId output is currently supported only for columnar pools");
                    CB_ENSURE(!isPartOfFullTestSet, "output for column " << columnName << "is currently supported only for full pools, not pool parts");

                    columnPrinter.push_back(
                        (outputType == EColumn::GroupId) ?
                          (THolder<IColumnPrinter>)MakeHolder<TGroupOrSubgroupIdPrinter<TGroupId>>(
                              getPoolColumnsPrinter(),
                              EColumn::GroupId,
                              columnName
                          )
                        : (THolder<IColumnPrinter>)MakeHolder<TGroupOrSubgroupIdPrinter<TSubgroupId>>(
                              getPoolColumnsPrinter(),
                              EColumn::SubgroupId,
                              columnName
                          )
                    );
                    continue;
                }
                if (outputType == EColumn::Baseline) {
                    for (int idx = 0; idx < pool.Docs.Baseline.ysize(); ++idx) {
                        TStringBuilder header;
                        header << "Baseline";
                        if (pool.Docs.Baseline.ysize() > 1) {
                            header << "#" << idx;
                        }
                        columnPrinter.push_back(MakeHolder<TVectorPrinter<double>>(pool.Docs.Baseline[idx], header));
                    }
                    continue;
                }
            }
            if (!columnName.compare(0, BaselinePrefix.length(), BaselinePrefix)) {
                int idx = FromString<int>(columnName.substr(BaselinePrefix.length()));
                columnPrinter.push_back(MakeHolder<TVectorPrinter<double>>(pool.Docs.Baseline[idx], columnName));
                continue;
            }
            if (columnName[0] == '#') {
                CB_ENSURE(!isPartOfFullTestSet, "Column output by # is currently supported only for full pools, not pool parts");

                columnPrinter.push_back(
                    MakeHolder<TNumColumnPrinter>(
                        getPoolColumnsPrinter(),
                        FromString<int>(columnName.substr(1))
                    )
                );
            } else {
                auto it = featureIdToDesc.find(columnName);
                CB_ENSURE(it != featureIdToDesc.end(),
                          "columnName [" << columnName << "] not found in featureIds");
                if (it->second.IsCategorical) {
                    columnPrinter.push_back(
                        MakeHolder<TCatFeaturePrinter>(
                            pool.Docs.Factors[it->second.Index],
                            pool.CatFeaturesHashToString,
                            columnName
                        )
                    );
                } else {
                    columnPrinter.push_back(
                        MakeHolder<TVectorPrinter<float>>(
                            pool.Docs.Factors[it->second.Index],
                            columnName
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
        for (size_t docId = 0; docId < pool.Docs.GetDocCount(); ++docId) {
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
        int threadCount,
        const TVector<TString>& outputColumns,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TPool& pool,
        bool isPartOfFullTestSet,
        IOutputStream* outputStream,
        const TPathWithScheme& testSetPath,
        std::pair<int, int> testFileWhichOf,
        const TDsvFormatOptions& testSetFormat,
        bool writeHeader,
        ui64 docIdOffset) {

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(threadCount - 1);
        OutputEvalResultToFile(
            evalResult,
            &executor,
            outputColumns,
            visibleLabelsHelper,
            pool,
            isPartOfFullTestSet,
            outputStream,
            testSetPath,
            testFileWhichOf,
            testSetFormat,
            writeHeader,
            docIdOffset);
    }

} // namespace NCB
