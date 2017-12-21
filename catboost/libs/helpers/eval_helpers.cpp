#include "eval_helpers.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/string/builder.h>
#include <util/generic/ymath.h>
#include <util/generic/algorithm.h>

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


void TEvalResult::OutputToFile(const TVector<TString>& docIds,
                               IOutputStream* outputStream, bool writeHeader,
                               const TVector<float>* targets) {
    if (writeHeader) {
        *outputStream << "DocId";
        for (int i = 0; i < ColumnNames.ysize(); ++i) {
            *outputStream << "\t" << ColumnNames[i];
        }
        if (targets != nullptr) {
            *outputStream << "\tTarget";
        }
        *outputStream << Endl;
    }
    for (int docId = 0; docId < Approxes[0].ysize(); ++docId) {
        *outputStream << docIds[docId];
        for (int approxId = 0; approxId < Approxes.ysize(); ++ approxId) {
            *outputStream << "\t" << Approxes[approxId][docId];
        }
        if (targets != nullptr) {
            *outputStream << "\t" << (*targets)[docId];
        }
        *outputStream << Endl;
    }
}

void TEvalResult::SetPredictionTypes(const TVector<EPredictionType>& predictionTypes_) {
    PredictionTypes.clear();
    PredictionTypes.assign(predictionTypes_.begin(), predictionTypes_.end());
}

TVector<TVector<double>>& TEvalResult::GetRawValuesRef() {
    return RawValues;
}

TVector<TVector<double>>& TEvalResult::GetApproxesRef() {
    return Approxes;
}

void TEvalResult::ClearRawValues() {
    RawValues.clear();
}

void TEvalResult::ClearApproxes() {
    Approxes.clear();
}

void TEvalResult::PostProcess(NPar::TLocalExecutor* executor, TMaybe<std::pair<int, int>> evalBorders) {
    int classesCount = RawValues.ysize();
    for (auto predictionType: PredictionTypes) {
        int classId = 0;
        for (const auto &approx: PrepareEval(predictionType, RawValues, executor)) {
            Approxes.push_back(approx);
            TStringBuilder str;
            if (classesCount > 1) {
                str << "Class_" << classId << "_";
            }
            str << predictionType;
            if (evalBorders) {
                str << "[" << evalBorders->first << "," << evalBorders->second << ")";
            }
            ColumnNames.push_back(str);
            classId++;
        }
    }
}

void TEvalResult::PostProcess(int threadCount) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    PostProcess(&executor);
}

