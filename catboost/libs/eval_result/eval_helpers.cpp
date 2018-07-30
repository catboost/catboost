#include "eval_helpers.h"

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/logging/logging.h>

#include <library/fast_exp/fast_exp.h>

#include <util/generic/hash_set.h>
#include <util/generic/ymath.h>

#include <functional>


using namespace NCB;


const TString BaselinePrefix = "Baseline#";

void CalcSoftmax(const TVector<double>& approx, TVector<double>* softmax) {
    double maxApprox = *MaxElement(approx.begin(), approx.end());
    for (int dim = 0; dim < approx.ysize(); ++dim) {
        (*softmax)[dim] = approx[dim] - maxApprox;
    }
    FastExpInplace(softmax->data(), softmax->ysize());
    double sumExpApprox = 0;
    for (auto curSoftmax : *softmax) {
        sumExpApprox += curSoftmax;
    }
    for (auto& curSoftmax : *softmax) {
        curSoftmax /= sumExpApprox;
    }
}

TVector<double> CalcSigmoid(const TVector<double>& approx) {
    TVector<double> probabilities(approx.size());
    for (int i = 0; i < approx.ysize(); ++i) {
        probabilities[i] = 1 / (1 + exp(-approx[i]));
    }
    return probabilities;
}

static TVector<TVector<double>> CalcSoftmax(const TVector<TVector<double>>& approx, NPar::TLocalExecutor* localExecutor) {
    TVector<TVector<double>> probabilities = approx;
    const int threadCount = localExecutor->GetThreadCount() + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    auto calcSoftmaxInBlock = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        TVector<double> line(approx.size());
        TVector<double> softmax(approx.size());
        for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
            for (int dim = 0; dim < approx.ysize(); ++dim) {
                line[dim] = approx[dim][lineInd];
            }
            CalcSoftmax(line, &softmax);
            for (int dim = 0; dim < approx.ysize(); ++dim) {
                probabilities[dim][lineInd] = softmax[dim];
            }
        }
    };
    localExecutor->ExecRange(calcSoftmaxInBlock, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    return probabilities;
}

static TVector<int> SelectBestClass(const TVector<TVector<double>>& approx, NPar::TLocalExecutor* localExecutor) {
    TVector<int> classApprox(approx[0].size());
    const int threadCount = localExecutor->GetThreadCount() + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    auto selectBestClassInBlock = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
            double maxApprox = approx[0][lineInd];
            int maxApproxId = 0;
            for (int dim = 1; dim < approx.ysize(); ++dim) {
                if (approx[dim][lineInd] > maxApprox) {
                    maxApprox = approx[dim][lineInd];
                    maxApproxId = dim;
                }
            }
            classApprox[lineInd] = maxApproxId;
        }
    };
    localExecutor->ExecRange(selectBestClassInBlock, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    return classApprox;
}

static bool IsMulticlass(const TVector<TVector<double>>& approx) {
    return approx.size() > 1;
}

static TVector<TVector<double>> MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TVisibleLabelsHelper& visibleLabelsHelper
) {
    const double inf = std::numeric_limits<double>::infinity();
    TVector<TVector<double>> externalApprox(visibleLabelsHelper.GetVisibleApproxDimension(),
                                            TVector<double>(internalApprox.back().ysize(), -inf));

    for (int classId = 0; classId < internalApprox.ysize(); ++classId) {
        int visibleId = visibleLabelsHelper.GetVisibleIndex(classId);

        for (int docId = 0; docId < externalApprox.back().ysize(); ++docId) {
            externalApprox[visibleId][docId] = internalApprox[classId][docId];
        }
    }
    return externalApprox;
}

static TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TVisibleLabelsHelper& visibleLabelsHelper
) {
    TVector<TString> convertedTarget(target.ysize());

    if (visibleLabelsHelper.IsInitialized()) {
        for (int targetIdx = 0; targetIdx < target.ysize(); ++targetIdx) {
            convertedTarget[targetIdx] = visibleLabelsHelper.GetVisibleClassNameFromLabel(target[targetIdx]);
        }
    } else {
        for (int targetIdx = 0; targetIdx < target.ysize(); ++targetIdx) {
            convertedTarget[targetIdx] = ToString<float>(target[targetIdx]);
        }
    }

    return convertedTarget;
}

TVector<TVector<double>> PrepareEval(const EPredictionType predictionType,
                                     const TVector<TVector<double>>& approx,
                                     int threadCount) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    return PrepareEval(predictionType, approx, &executor);
}

TVector<TVector<double>> PrepareEval(const EPredictionType predictionType,
                                     const TVector<TVector<double>>& approx,
                                     NPar::TLocalExecutor* localExecutor) {
    TVector<TVector<double>> result;
    switch (predictionType) {
        case EPredictionType::Probability:
            if (IsMulticlass(approx)) {
                result = CalcSoftmax(approx, localExecutor);
            } else {
                result = {CalcSigmoid(approx[0])};
            }
            break;
        case EPredictionType::Class:
            result.resize(1);
            result[0].reserve(approx.size());
            if (IsMulticlass(approx)) {
                TVector<int> predictions = {SelectBestClass(approx, localExecutor)};
                result[0].assign(predictions.begin(), predictions.end());
            } else {
                for (const double prediction : approx[0]) {
                    result[0].push_back(prediction > 0);
                }
            }
            break;
        case EPredictionType::RawFormulaVal:
            result = approx;
            break;
        default:
            Y_ASSERT(false);
    }
    return result;
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

namespace {
    class IColumnPrinter {
    public:
        virtual void OutputValue(IOutputStream* outstream, size_t docIndex) = 0;
        virtual void OutputHeader(IOutputStream* outstream) = 0;
        virtual TString GetAfterColumnDelimiter() const {
            return "\t";
        }
        virtual ~IColumnPrinter() = default;
    };

    template <typename T>
    class TVectorPrinter: public IColumnPrinter {
    public:
        TVectorPrinter(const TVector<T>& targetRef, const TString& header)
            : Ref(targetRef)
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << Ref[docIndex];
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

    private:
        const TVector<T>& Ref;
        const TString Header;
    };

    template <typename T>
    class TPrefixPrinter: public IColumnPrinter {
    public:
        TPrefixPrinter(const TString& prefix, const TString& header, const TString& delimiter)
            : Prefix(prefix)
            , Header(header)
            , Delimiter(delimiter)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            Y_UNUSED(docIndex);
            *outStream << Prefix;
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        TString GetAfterColumnDelimiter() const override {
            return Delimiter;
        }

    private:
        const TString Prefix;
        const TString Header;
        const TString Delimiter;
    };

    class TPoolColumnsPrinter : public TThrRefBase {
    public:
        TPoolColumnsPrinter(const TPathWithScheme& testSetPath,
                            const TDsvFormatOptions& format)
            : LineDataReader(GetLineDataReader(testSetPath, format))
            , Delimiter(format.Delimiter)
            , DocIndex(-1)
        {}

        void Output(IOutputStream* outStream, size_t docIndex, int colId) {
            *outStream << GetCell(docIndex, colId);
        }

        const TString& GetCell(size_t docIndex, int colId) {
            if (docIndex == DocIndex + 1) {
                DocIndex++;
                TString line;
                CB_ENSURE(LineDataReader->ReadLine(&line),
                          "there's no line in pool for " << DocIndex);
                Columns.clear();
                for (const auto& typeName : StringSplitter(line).Split(Delimiter)) {
                    Columns.push_back(FromString<TString>(typeName.Token()));
                }
            }
            CB_ENSURE(docIndex == DocIndex, "only serial lines possible to output");
            return Columns[colId];
        }

    private:
        THolder<ILineDataReader> LineDataReader;
        char Delimiter;
        size_t DocIndex;
        TVector<TString> Columns;
    };

    class TNumColumnPrinter: public IColumnPrinter {
    public:
        TNumColumnPrinter(TIntrusivePtr<TPoolColumnsPrinter> printerPtr, int colId)
            : PrinterPtr(printerPtr)
            , ColId(colId) {}

        void OutputValue(IOutputStream* outStream, size_t docIndex) override  {
            PrinterPtr->Output(outStream, docIndex, ColId);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << '#' << ColId;
        }

    private:
        TIntrusivePtr<TPoolColumnsPrinter> PrinterPtr;
        int ColId;
    };

    class TCatFeaturePrinter: public IColumnPrinter {
    public:
        TCatFeaturePrinter(const TVector<float>& hashedValues,
                           const THashMap<int, TString>& hashToString,
                           const TString& header)
            : HashedValues(hashedValues)
            , HashToString(hashToString)
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << HashToString.at(HashedValues[docIndex]);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

    private:
        const TVector<float>& HashedValues;
        const THashMap<int, TString>& HashToString;
        const TString Header;
    };

    class TEvalPrinter: public IColumnPrinter {
    public:
        TEvalPrinter(
            NPar::TLocalExecutor* executor,
            const TVector<TVector<TVector<double>>>& rawValues,
            const EPredictionType predictionType,
            const TVisibleLabelsHelper& visibleLabelsHelper,
            TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>())
            : VisibleLabelsHelper(visibleLabelsHelper) {
            int begin = 0;
            for (const auto& raws : rawValues) {
                CB_ENSURE(VisibleLabelsHelper.IsInitialized() == IsMulticlass(raws),
                          "Inappropriated usage of visible label helper: it MUST be initialized ONLY for multiclass problem");
                const auto& approx = VisibleLabelsHelper.IsInitialized() ? MakeExternalApprox(raws, VisibleLabelsHelper) : raws;
                Approxes.push_back(PrepareEval(predictionType, approx, executor));
                for (int classId = 0; classId < Approxes.back().ysize(); ++classId) {
                    TStringBuilder str;
                    str << predictionType;
                    if (Approxes.back().ysize() > 1) {
                        str << ":Class=" << VisibleLabelsHelper.GetVisibleClassNameFromClass(classId);
                    }
                    if (rawValues.ysize() > 1) {
                        str << ":TreesCount=[" << begin << "," << Min(begin + evalParameters->first, evalParameters->second) << ")";
                    }
                    Header.push_back(str);
                }
                if (evalParameters) {
                    begin += evalParameters->first;
                }
            }
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            TString delimiter = "";
            if (VisibleLabelsHelper.IsInitialized() && Approxes.back().ysize() == 1) { // class labels
                for (const auto& approxes : Approxes) {
                    for (const auto& approx : approxes) {
                        *outStream << delimiter
                                   << VisibleLabelsHelper.GetVisibleClassNameFromClass(static_cast<int>(approx[docIndex]));
                        delimiter = "\t";
                    }
                }
            } else {
                for (const auto& approxes : Approxes) {
                    for (const auto& approx : approxes) {
                        *outStream << delimiter << approx[docIndex];
                        delimiter = "\t";
                    }
                }
            }
        }

        void OutputHeader(IOutputStream* outStream) override {
            for (int idx = 0; idx < Header.ysize(); ++idx) {
                if (idx > 0) {
                    *outStream << "\t";
                }
                *outStream << Header[idx];
            }
        }

    private:
        TVector<TString> Header;
        TVector<TVector<TVector<double>>> Approxes;
        const TVisibleLabelsHelper& VisibleLabelsHelper;
    };

    template <typename TId>
    class TGroupOrSubgroupIdPrinter: public IColumnPrinter {
    public:
        TGroupOrSubgroupIdPrinter(TIntrusivePtr<TPoolColumnsPrinter> printerPtr,
                                  int columnId,
                                  const TVector<TId>& ref,
                                  std::function<TId(TStringBuf)> hashFunc,
                                  const TString& header)
            : PrinterPtr(printerPtr)
            , ColumnId(columnId)
            , Ref(ref)
            , HashFunc(hashFunc)
            , Header(header)
        {}

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            const TString& cell = PrinterPtr->GetCell(docIndex, ColumnId);
            Y_VERIFY(Ref[docIndex] == HashFunc(cell));
            *outStream << cell;
        }

    private:
        TIntrusivePtr<TPoolColumnsPrinter> PrinterPtr;
        int ColumnId;
        const TVector<TId>& Ref;
        std::function<TId(TStringBuf)> HashFunc;
        TString Header;
    };
}

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



void TEvalResult::OutputToFile(
    NPar::TLocalExecutor* executor,
    const TVector<TString>& outputColumns,
    const TVisibleLabelsHelper& visibleLabelsHelper,
    const TPool& pool,
    bool isPartOfFullTestSet,
    IOutputStream* outputStream,
    const TPathWithScheme& testSetPath,
    std::pair<int, int> testFileWhichOf,
    const TDsvFormatOptions& testSetFormat,
    bool writeHeader,
    TMaybe<std::pair<size_t, size_t>> evalParameters) {

    CB_ENSURE(!pool.IsQuantized(), "Not supported for quantized pools");
    TFeatureIdToDesc featureIdToDesc = GetFeatureIdToDesc(pool);

    TVector<TString> convertedTarget = ConvertTargetToExternalName(pool.Docs.Target, visibleLabelsHelper);

    TVector<THolder<IColumnPrinter>> columnPrinter;

    TIntrusivePtr<TPoolColumnsPrinter> poolColumnsPrinter;

    // lazy init
    auto getPoolColumnsPrinter = [&]() {
        /* there's a special case when poolColumnsPrinter can be empty:
          when we need only header output
        */
        if (testSetPath.Inited() && !poolColumnsPrinter) {
            poolColumnsPrinter = MakeIntrusive<TPoolColumnsPrinter>(testSetPath, testSetFormat);
        }
        return poolColumnsPrinter;
    };

    for (const auto& columnName : outputColumns) {
        EPredictionType type;
        if (TryFromString<EPredictionType>(columnName, type)) {
            columnPrinter.push_back(MakeHolder<TEvalPrinter>(executor, RawValues, type, visibleLabelsHelper, evalParameters));
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
                columnPrinter.push_back(MakeHolder<TVectorPrinter<TString>>(pool.Docs.Id, "DocId"));
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

                int columnIndex = GetColumnIndex(*(pool.MetaInfo.ColumnsInfo), outputType);

                columnPrinter.push_back(
                    (outputType == EColumn::GroupId) ?
                      (THolder<IColumnPrinter>)MakeHolder<TGroupOrSubgroupIdPrinter<TGroupId>>(
                          getPoolColumnsPrinter(),
                          columnIndex,
                          pool.Docs.QueryId,
                          CalcGroupIdFor,
                          columnName
                      )
                    : (THolder<IColumnPrinter>)MakeHolder<TGroupOrSubgroupIdPrinter<TSubgroupId>>(
                          getPoolColumnsPrinter(),
                          columnIndex,
                          pool.Docs.SubgroupId,
                          CalcSubgroupIdFor,
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

void TEvalResult::OutputToFile(
    int threadCount,
    const TVector<TString>& outputColumns,
    const TVisibleLabelsHelper& visibleLabelsHelper,
    const TPool& pool,
    bool isPartOfFullTestSet,
    IOutputStream* outputStream,
    const TPathWithScheme& testSetPath,
    std::pair<int, int> testFileWhichOf,
    const TDsvFormatOptions& testSetFormat,
    bool writeHeader) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    OutputToFile(&executor,
                 outputColumns,
                 visibleLabelsHelper,
                 pool,
                 isPartOfFullTestSet,
                 outputStream,
                 testSetPath,
                 testFileWhichOf,
                 testSetFormat,
                 writeHeader);
}

TVector<TVector<TVector<double>>>& TEvalResult::GetRawValuesRef() {
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
