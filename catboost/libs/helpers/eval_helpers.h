#pragma once

#include <catboost/libs/options/enums.h>
#include <library/threading/local_executor/local_executor.h>
#include <library/digest/crc32c/crc32c.h>

#include <util/generic/vector.h>
#include <util/generic/maybe.h>
#include <util/stream/output.h>


void CalcSoftmax(const TVector<double>& approx, TVector<double>* softmax);

TVector<TVector<double>> PrepareEval(const EPredictionType predictionType,
                                     const TVector<TVector<double>>& approx,
                                     NPar::TLocalExecutor* localExecutor);

TVector<TVector<double>> PrepareEval(const EPredictionType predictionType,
                                     const TVector<TVector<double>>& approx,
                                     int threadCount);

class TEvalResult {
public:
    TEvalResult() {}
    ~TEvalResult() {}

    void SetPredictionTypes(const TVector<EPredictionType>& predictionTypes_);
    TVector<TVector<double>>& GetRawValuesRef();
    TVector<TVector<double>>& GetApproxesRef();
    void DropRawValues();

    void PostProcess(NPar::TLocalExecutor* executor, TMaybe<std::pair<int, int>> evalBorders=TMaybe<std::pair<int, int>>());
    void PostProcess(int threadCount);

    void OutputToFile(const TVector<TString>& docIds,
                      IOutputStream* outputStream, bool writeHeader=true,
                      const TVector<float>* targets=nullptr);



private:
    TVector<TVector<double>> RawValues;
    TVector<TVector<double>> Approxes;
    TVector<EPredictionType> PredictionTypes;
    TVector<TString> ColumnNames;
};

template <typename T>
ui32 CalcMatrixCheckSum(ui32 init, const TVector<TVector<T>>& matrix) {
    ui32 checkSum = init;
    for (const auto& row : matrix) {
        checkSum = Crc32cExtend(checkSum, row.data(), row.size() * sizeof(T));
    }
    return checkSum;
}
