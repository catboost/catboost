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

void ValidateColumnOutput(const TVector<TString>& outputColumns, const TPool& pool, bool CV_mode=false);

class TEvalResult {
public:
    TEvalResult() {
        RawValues.resize(1);
    }

    TVector<TVector<TVector<double>>>& GetRawValuesRef();
    void ClearRawValues();

    /// *Move* data from `rawValues` to `RawValues[0]`
    void SetRawValuesByMove(TVector<TVector<double>>& rawValues);

    void OutputToFile(
        NPar::TLocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TPool& pool,
        IOutputStream* outputStream,
        const TString& testFile,
        std::pair<int, int> testFileWhichOf,
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
        std::pair<int, int> testFileWhichOf,
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
