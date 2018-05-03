#include "eval_helpers.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/ymath.h>

const TString BaselinePrefix = "Baseline#";

void CalcSoftmax(const TVector<double>& approx, TVector<double>* softmax) {
    double maxApprox = *MaxElement(approx.begin(), approx.end());
    double sumExpApprox = 0;
    for (int dim = 0; dim < approx.ysize(); ++dim) {
        double expApprox = exp(approx[dim] - maxApprox);
        (*softmax)[dim] = expApprox;
        sumExpApprox += expApprox;
    }
    for (auto& curSoftmax : *softmax) {
        curSoftmax /= sumExpApprox;
    }
}

static TVector<double> CalcSigmoid(const TVector<double>& approx) {
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

void ValidateColumnOutput(const TVector<TString>& outputColumns, const TPool& pool, bool CV_mode) {
    TMap<TString, int> featureId;
    for (int idx = 0; idx < pool.FeatureId.ysize(); idx++) {
        featureId[pool.FeatureId[idx]] = idx;
    }
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
                    CB_ENSURE(pool.MetaInfo.BaselineCount > 0, "bad output column name " << name << " (No baseline)");
                    break;
                case (EColumn::Weight):
                    CB_ENSURE(pool.MetaInfo.HasWeights, "bad output column name " << name << " (No WeightId in CD file)");
                    break;
                case (EColumn::GroupId):
                    CB_ENSURE(pool.MetaInfo.GroupIdColumn >= 0, "bad output column name " << name << " (No GroupId in CD file)");
                    break;
                case (EColumn::Timestamp):
                    CB_ENSURE(pool.MetaInfo.HasTimestamp, "bad output column name " << name << " (No Timestamp in CD file)");
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

        size_t index;
        if (name[0] == '#') {
            index = FromString<int>(name.substr(1));
        } else {
            CB_ENSURE(featureId.find(name) != featureId.end(), "bad output column name " << name);
            index = featureId[name];
        }
        CB_ENSURE(index < pool.MetaInfo.ColumnsCount, "bad output column name " << name);
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
        TPoolColumnsPrinter(const TString& testFilePath, char delimiter, bool hasHeader)
            : TestFile(testFilePath)
            , Delimiter(delimiter)
            , DocIndex(-1)
        {
            if (hasHeader) {
                TestFile.ReadLine();
            }
        }

        void OutputDoc(IOutputStream* outStream, int docId, size_t docIndex) {
            *outStream << GetCell(docIndex, docId);
        }

        const TString& GetCell(size_t docIndex, int colId) {
            if (docIndex == DocIndex + 1) {
                DocIndex++;
                TString line;
                TestFile.ReadLine(line);
                Factors.clear();
                for (const auto& typeName : StringSplitter(line).Split(Delimiter)) {
                    Factors.push_back(FromString<TString>(typeName.Token()));
                }
            }
            CB_ENSURE(docIndex == DocIndex, "only serial lines possible to output");
            return Factors[colId];
        }

    private:
        TIFStream TestFile;
        char Delimiter;
        size_t DocIndex;
        TVector<TString> Factors;
    };

    class TFactorPrinter: public IColumnPrinter {
    public:
        TFactorPrinter(TIntrusivePtr<TPoolColumnsPrinter> printerPtr, int factorId)
            : PrinterPtr(printerPtr)
            , FactorId(factorId) {}

        void OutputValue(IOutputStream* outStream, size_t docIndex) override  {
            PrinterPtr->OutputDoc(outStream, FactorId, docIndex);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << '#' << FactorId;
        }

    private:
        TIntrusivePtr<TPoolColumnsPrinter> PrinterPtr;
        int FactorId;
    };

    class TEvalPrinter: public IColumnPrinter {
    public:
        TEvalPrinter(
            NPar::TLocalExecutor* executor,
            const TVector<TVector<TVector<double>>>& rawValues,
            const EPredictionType predictionType,
            TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>()) {
            int begin = 0;
            for (const auto& raws : rawValues) {
                Approxes.push_back(PrepareEval(predictionType, raws, executor));
                for (int classId = 0; classId < Approxes.back().ysize(); ++classId) {
                    TStringBuilder str;
                    str << predictionType;
                    if (Approxes.back().ysize() > 1) {
                        str << ":Class=" << classId;
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
            for (const auto& approxes : Approxes) {
                for (const auto& approx : approxes) {
                    *outStream << delimiter << approx[docIndex];
                    delimiter = "\t";
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
    };

    class TGroupIdPrinter: public IColumnPrinter {
    public:
        TGroupIdPrinter(TIntrusivePtr<TPoolColumnsPrinter> printerPtr, int groupIdColumn, const TVector<TGroupId>& ref, const TString& header)
            : PrinterPtr(printerPtr)
            , GroupIdColumn(groupIdColumn)
            , Ref(ref)
            , Header(header) {}

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            const TString& cell = PrinterPtr->GetCell(docIndex, GroupIdColumn);
            Y_VERIFY(Ref[docIndex] == CalcGroupIdFor(cell));
            *outStream << cell;
        }

    private:
        TIntrusivePtr<TPoolColumnsPrinter> PrinterPtr;
        int GroupIdColumn;
        const TVector<TGroupId>& Ref;
        TString Header;
    };
}

void TEvalResult::OutputToFile(
    NPar::TLocalExecutor* executor,
    const TVector<TString>& outputColumns,
    const TPool& pool,
    IOutputStream* outputStream,
    const TString& testFile,
    std::pair<int, int> testFileWhichOf,
    char delimiter,
    bool hasHeader,
    bool writeHeader,
    TMaybe<std::pair<size_t, size_t>> evalParameters) {

    TVector<THolder<IColumnPrinter>> columnPrinter;
    TIntrusivePtr<TPoolColumnsPrinter> poolColumnsPrinter;
    if (!testFile.empty()) {
        poolColumnsPrinter = MakeIntrusive<TPoolColumnsPrinter>(testFile, delimiter, hasHeader);
    }

    TMap<TString, int> featureId;
    for (int idx = 0; idx < pool.FeatureId.ysize(); idx++) {
        featureId[pool.FeatureId[idx]] = idx;
    }

    for (const auto& columnName : outputColumns) {
        EPredictionType type;
        if (TryFromString<EPredictionType>(columnName, type)) {
            columnPrinter.push_back(MakeHolder<TEvalPrinter>(executor, RawValues, type, evalParameters));
            continue;
        }
        EColumn outputType;
        if (TryFromString<EColumn>(columnName, outputType)) {
            if (outputType == EColumn::Label) {
                if  (!pool.Docs.Target.empty()) {
                    columnPrinter.push_back(MakeHolder<TVectorPrinter<float>>(pool.Docs.Target, columnName));
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
            if (outputType == EColumn::GroupId) {
                columnPrinter.push_back(MakeHolder<TGroupIdPrinter>(poolColumnsPrinter, pool.MetaInfo.GroupIdColumn, pool.Docs.QueryId, columnName));
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
        int idx;
        if (columnName[0] == '#') {
            idx = FromString<int>(columnName.substr(1));
        } else {
            idx = featureId[columnName];
        }
        columnPrinter.push_back(MakeHolder<TFactorPrinter>(poolColumnsPrinter, idx));
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
    const TPool& pool,
    IOutputStream* outputStream,
    const TString& testFile,
    std::pair<int, int> testFileWhichOf,
    char delimiter,
    bool hasHeader,
    bool writeHeader) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    OutputToFile(&executor, outputColumns, pool, outputStream, testFile, testFileWhichOf, delimiter, hasHeader, writeHeader);
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
