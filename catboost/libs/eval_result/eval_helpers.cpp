#include "eval_helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/labels/external_label_helper.h>

#include <util/generic/array_ref.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>

#include <cmath>
#include <limits>

template <typename Function>
static TVector<TVector<double>> CalcSomeSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* executor,
    Function func)
{
    TVector<TVector<double>> probabilities = approx;
    probabilities.resize(approx.size());
    ForEach(probabilities.begin(), probabilities.end(), [&](auto& v) { v.yresize(approx.front().size()); });
    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    const int threadCount = executorThreadCount + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    const auto calcSomeSoftmaxInBlock = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        TVector<double> line;
        line.yresize(approx.size());
        TVector<double> someSoftmax;
        someSoftmax.yresize(approx.size());
        for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
            for (int dim = 0; dim < approx.ysize(); ++dim) {
                line[dim] = approx[dim][lineInd];
            }
            func(line, someSoftmax);
            for (int dim = 0; dim < approx.ysize(); ++dim) {
                probabilities[dim][lineInd] = someSoftmax[dim];
            }
        }
    };
    if (executor) {
        executor->ExecRange(calcSomeSoftmaxInBlock, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        calcSomeSoftmaxInBlock(0);
    }
    return probabilities;
}

static TVector<TVector<double>> CalcSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* executor)
{
    return CalcSomeSoftmax(
        approx, executor,
        [](const TConstArrayRef<double> approx, TArrayRef<double> target) {
            CalcSoftmax(approx, target);
        });
}

static TVector<TVector<double>> CalcLogSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* executor)
{
    return CalcSomeSoftmax(
        approx, executor,
        [](const TConstArrayRef<double> approx, TArrayRef<double> target) {
            CalcLogSoftmax(approx, target);
        });
}

static TVector<int> SelectBestClass(
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* executor)
{
    TVector<int> classApprox;
    classApprox.yresize(approx.front().size());
    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    const int threadCount = executorThreadCount + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    const auto selectBestClassInBlock = [&](const int blockId) {
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
    if (executor) {
        executor->ExecRange(selectBestClassInBlock, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        selectBestClassInBlock(0);
    }
    return classApprox;
}

bool IsMulticlass(const TVector<TVector<double>>& approx) {
    return approx.size() > 1;
}

void MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper,
    TVector<TVector<double>>* resultApprox
) {
    resultApprox->resize(externalLabelsHelper.GetExternalApproxDimension());
    for (int classId = 0; classId < internalApprox.ysize(); ++classId) {
        int visibleId = externalLabelsHelper.GetExternalIndex(classId);
        (*resultApprox)[visibleId] = internalApprox[classId];
    }
}

TVector<TVector<double>> MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper
) {
    const double inf = std::numeric_limits<double>::infinity();
    TVector<TVector<double>> externalApprox(externalLabelsHelper.GetExternalApproxDimension(),
                                            TVector<double>(internalApprox.back().ysize(), -inf));
    MakeExternalApprox(internalApprox, externalLabelsHelper, &externalApprox);
    return externalApprox;
}


TVector<TVector<double>> PrepareEvalForInternalApprox(
    const EPredictionType predictionType,
    const TFullModel& model,
    const TVector<TVector<double>>& approx,
    int threadCount
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    return PrepareEvalForInternalApprox(predictionType, model, approx, &executor);
}

TVector<TVector<double>> PrepareEvalForInternalApprox(
    const EPredictionType predictionType,
    const TFullModel& model,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor
) {
    const TExternalLabelsHelper externalLabelsHelper(model);
    const auto& externalApprox
        = (externalLabelsHelper.IsInitialized() && (externalLabelsHelper.GetExternalApproxDimension() > 1)) ?
            MakeExternalApprox(approx, externalLabelsHelper)
            : approx;
    return PrepareEval(predictionType, model.GetLossFunctionName(), externalApprox, localExecutor);
}

TVector<TVector<double>> PrepareEval(const EPredictionType predictionType,
                                     const TString& lossFunctionName,
                                     const TVector<TVector<double>>& approx,
                                     int threadCount) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    return PrepareEval(predictionType, lossFunctionName, approx, &executor);
}


void PrepareEval(const EPredictionType predictionType,
                 const TString& lossFunctionName,
                 const TVector<TVector<double>>& approx,
                 NPar::TLocalExecutor* executor,
                 TVector<TVector<double>>* result) {

    switch (predictionType) {
        case EPredictionType::LogProbability:
        case EPredictionType::Probability:
            if (IsMulticlass(approx)) {
                if (lossFunctionName == "MultiClassOneVsAll") {
                    result->resize(approx.size());
                    if (predictionType == EPredictionType::Probability) {
                        for (auto dim : xrange(approx.size())) {
                            (*result)[dim] = CalcSigmoid(approx[dim]);
                        }
                    } else {
                        for (auto dim : xrange(approx.size())) {
                            (*result)[dim] = CalcLogSigmoid(approx[dim]);
                        }
                    }
                } else {
                    if (lossFunctionName.empty()) {
                        CATBOOST_WARNING_LOG << "Optimized loss function was not saved in the model. Probabilities will be calculated \
                                                 under the assumption that it is MultiClass loss function" << Endl;
                    }
                    if (predictionType == EPredictionType::Probability) {
                        *result = CalcSoftmax(approx, executor);
                    } else {
                        *result = CalcLogSoftmax(approx, executor);
                    }
                }
            } else {
                if (predictionType == EPredictionType::Probability) {
                    *result = {CalcSigmoid(approx[0])};
                } else {
                    *result = {CalcLogSigmoid(InvertSign(approx[0])), CalcLogSigmoid(approx[0])};
                }
            }
            break;
        case EPredictionType::Class:
            result->resize(1);
            (*result)[0].reserve(approx.size());
            if (IsMulticlass(approx)) {
                TVector<int> predictions = {SelectBestClass(approx, executor)};
                (*result)[0].assign(predictions.begin(), predictions.end());
            } else {
                for (const double prediction : approx[0]) {
                    (*result)[0].push_back(prediction > 0);
                }
            }
            break;
        case EPredictionType::Exponent:
            *result = {CalcExponent(approx[0])};
            break;
        case EPredictionType::RawFormulaVal:
            *result = approx;
            break;
        default:
            Y_ASSERT(false);
    }
}

TVector<TVector<double>> PrepareEval(const EPredictionType predictionType,
                                     const TString& lossFunctionName,
                                     const TVector<TVector<double>>& approx,
                                     NPar::TLocalExecutor* localExecutor) {
    TVector<TVector<double>> result;
    PrepareEval(predictionType, lossFunctionName, approx, localExecutor, &result);
    return result;
}
