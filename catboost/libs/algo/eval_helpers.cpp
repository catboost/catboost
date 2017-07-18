#include "eval_helpers.h"
#include "error_functions.h"

#include <util/generic/ymath.h>
#include <library/threading/local_executor/local_executor.h>


#include "../../../util/generic/vector.h"

static yvector<double> CalcSigmoid(const yvector<double>& approx) {
    yvector<double> probabilities(approx.size());
    for (int i = 0; i < approx.ysize(); ++i) {
        probabilities[i] = 1 / (1 + exp(-approx[i]));
    }
    return probabilities;
}

static yvector<yvector<double>> CalcSoftmax(const yvector<yvector<double>>& approx, NPar::TLocalExecutor* localExecutor) {
    yvector<yvector<double>> probabilities = approx;
    const int threadCount = localExecutor->GetThreadCount() + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    auto calcSoftmaxInBlock = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        yvector<double> line(approx.size());
        yvector<double> softmax(approx.size());
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

static yvector<int> SelectBestClass(const yvector<yvector<double>>& approx, NPar::TLocalExecutor* localExecutor) {
    yvector<int> classApprox(approx[0].size());
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

static bool IsMulticlass(const yvector<yvector<double>>& approx) {
    return approx.size() > 1;
}

yvector<yvector<double>> PrepareEval(const EPredictionType predictionType,
                                     const yvector<yvector<double>>& approx,
                                     NPar::TLocalExecutor* localExecutor) {
    yvector<yvector<double>> result;
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
                yvector<int> predictions = {SelectBestClass(approx, localExecutor)};
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

void OutputTestEval(const yvector<yvector<double>>& testApprox, const TString& testEvalFile, const yvector<TDocInfo>& docs,
                    const bool outputTarget) {
    TOFStream f(testEvalFile);
    for (int i = 0; i < testApprox[0].ysize(); ++i) {
        if (!docs[i].Id.empty()) {
            f << docs[i].Id << '\t';
        }
        for (int dim = 0; dim < testApprox.ysize(); ++dim) {
            f << testApprox[dim][i] << (dim + 1 < testApprox.ysize() || outputTarget ? "\t" : "\n");
        }
        if (outputTarget) {
            f << docs[i].Target << "\n";
        }
    }
}
