#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/pool.h>
#include <library/threading/local_executor/local_executor.h>
#include <library/digest/crc32c/crc32c.h>

#include <util/string/builder.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/string/iterator.h>
#include <util/string/cast.h>

void CalcSoftmax(const TVector<double>& approx, TVector<double>* softmax);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    int threadCount);

class IColumnPrinter {
public:
    virtual void OutputValue(IOutputStream* outstream, size_t docIndex) = 0;
    virtual void OutputHeader(IOutputStream* outstream) = 0;
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

void ValidateColumnOutput(const TVector<TString>& outputColumns, const TPool& pool, bool CV_mode=false);

class TEvalResult {
public:
    TEvalResult() {
        RawValues.resize(1);
    }

    TVector<TVector<TVector<double>>>& GetRawValuesRef();
    void ClearRawValues();

    void OutputToFile(
        NPar::TLocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TPool& pool,
        IOutputStream* outputStream,
        const TString& testFile,
        char delimiter,
        bool hasHeader,
        bool writeHeader = true,
        TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>());
    void OutputToFile(
        int threadCount,
        const TVector<TString>& outputColumns,
        const TPool& pool,
        IOutputStream* outputStream,
        const TString& testFile,
        char delimiter,
        bool hasHeader,
        bool writeHeader = true);

private:
    TVector<TVector<TVector<double>>> RawValues; // [evalIter][dim][docIdx]
};

template <typename T>
ui32 CalcMatrixCheckSum(ui32 init, const TVector<TVector<T>>& matrix) {
    ui32 checkSum = init;
    for (const auto& row : matrix) {
        checkSum = Crc32cExtend(checkSum, row.data(), row.size() * sizeof(T));
    }
    return checkSum;
}
