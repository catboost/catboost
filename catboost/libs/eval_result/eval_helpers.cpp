#include "eval_helpers.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/labels/external_label_helper.h>
#include <catboost/private/libs/options/enum_helpers.h>

#include <util/generic/array_ref.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>

#include <cmath>
#include <limits>

template <typename Function>
static TVector<TVector<double>> CalcSomeSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor,
    Function func
) {
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

TVector<TVector<double>> CalcSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor
) {
    return CalcSomeSoftmax(
        approx, executor,
        [](const TConstArrayRef<double> approx, TArrayRef<double> target) {
            CalcSoftmax(approx, target);
        });
}

static TVector<TVector<double>> CalcLogSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor
) {
    return CalcSomeSoftmax(
        approx, executor,
        [](const TConstArrayRef<double> approx, TArrayRef<double> target) {
            CalcLogSoftmax(approx, target);
        });
}

static TVector<int> SelectBestClass(
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor
) {
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


// totalUncertainty = entropy of expected
// dataUncertainty = entropy of expected - expected entropy
static void CalcMulticlassUncertainty(
    const TVector<TVector<double>>& approx,
    TVector<double>* dataUncertaintyPtr,
    TVector<double>* totalUncertaintyPtr,
    size_t virtEnsemblesCount,
    size_t classCount,
    NPar::ILocalExecutor* executor
) {
    TVector<double>& dataUncertainty = *dataUncertaintyPtr;
    TVector<double>& totalUncertainty = *totalUncertaintyPtr;

    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    const int threadCount = executorThreadCount + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    const auto calcUncertaitny = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        int firstLineId = blockId * blockSize;
        if (firstLineId >= lastLineId) {
            return;
        }

        TVector<double> line(classCount);
        TVector<double> probs(classCount);
        for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
            TVector<double> meanProbs(classCount);
            double meanEntropy = 0;
            for (size_t idx = 0; idx < virtEnsemblesCount; ++idx) {
                for (size_t dim = 0; dim < classCount; ++dim) {
                    line[dim] = approx[idx * classCount + dim][lineInd];
                }
                CalcSoftmax(TConstArrayRef<double>(line), TArrayRef<double>(probs));
                double entropy = 0;
                for (size_t i = 0; i < classCount; ++i) {
                    entropy -= probs[i] * std::log(probs[i]);
                    meanProbs[i] += probs[i];
                }
                meanEntropy += entropy;
            }
            meanEntropy /= virtEnsemblesCount;
            double entropyOfMeans = 0;
            for (size_t i = 0; i < classCount; ++i) {
                meanProbs[i] /= virtEnsemblesCount;
                entropyOfMeans -= meanProbs[i] * std::log(meanProbs[i]);
            }

            dataUncertainty[lineInd] = meanEntropy;
            totalUncertainty[lineInd] = entropyOfMeans;
        }
    };
    if (executor) {
        executor->ExecRange(calcUncertaitny, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        calcUncertaitny(0);
    }
}


static void CalcClassificationUncertainty(
    const TVector<TVector<double>>& approx,
    TVector<double>* dataUncertaintyPtr,
    TVector<double>* totalUncertaintyPtr,
    size_t virtEnsemblesCount,
    NPar::ILocalExecutor* executor
) {
    TVector<double>& dataUncertainty = *dataUncertaintyPtr;
    TVector<double>& totalUncertainty = *totalUncertaintyPtr;

    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    const int threadCount = executorThreadCount + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    const auto calcUncertaitny = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        int firstLineId = blockId * blockSize;
        if (firstLineId >= lastLineId) {
            return;
        }
        for (size_t idx = 0; idx < approx.size(); ++idx) {
            TConstArrayRef<double> arrayRef = TConstArrayRef<double>(
                approx[idx].begin() + firstLineId,
                lastLineId - firstLineId);
            TVector<double> probability = CalcSigmoid(arrayRef);
            auto entropy = CalcEntropyFromProbabilities(probability);
            for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
                dataUncertainty[lineInd] += entropy[lineInd - firstLineId];
                totalUncertainty[lineInd] += probability[lineInd - firstLineId];
            }
        }
        for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
            dataUncertainty[lineInd] /= virtEnsemblesCount;
            totalUncertainty[lineInd] /= virtEnsemblesCount;
        }
    };
    if (executor) {
        executor->ExecRange(calcUncertaitny, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        calcUncertaitny(0);
    }
    totalUncertainty = CalcEntropyFromProbabilities(totalUncertainty);
}

static void CalcRegressionUncertaitny(
    const TVector<TVector<double>>& approx,
    TVector<double>* meanApproxPtr,
    TVector<double>* knowledgeUncertaintyPtr,
    TVector<double>* dataUncertaintyPtr,
    size_t virtEnsemblesCount,
    NPar::ILocalExecutor* executor
) {
    TVector<double>& meanApprox = *meanApproxPtr;
    TVector<double>& knowledgeUncertainty = *knowledgeUncertaintyPtr;
    size_t dimShift = 1;
    if (dataUncertaintyPtr) {
        dataUncertaintyPtr->resize(approx.front().size());
        dimShift = 2;
    }
    meanApprox.resize(approx.front().size());
    knowledgeUncertainty.resize(approx.front().size());
    const int executorThreadCount = executor ? executor->GetThreadCount() : 0;
    const int threadCount = executorThreadCount + 1;
    const int blockSize = (approx[0].ysize() + threadCount - 1) / threadCount;
    const auto calcUncertaitny = [&](const int blockId) {
        int lastLineId = Min((blockId + 1) * blockSize, approx[0].ysize());
        int firstLineId = blockId * blockSize;
        if (firstLineId >= lastLineId) {
            return;
        }
        for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
            double mean = 0;
            for (size_t dimIdx = 0; dimIdx < virtEnsemblesCount; ++dimIdx) {
                mean += approx[dimIdx * dimShift][lineInd];
            }
            mean /= virtEnsemblesCount;
            meanApprox[lineInd] = mean;
            double var = 0;
            for (size_t dimIdx = 0; dimIdx < virtEnsemblesCount; ++dimIdx) {
                var += Sqr(approx[dimIdx * dimShift][lineInd] - mean);
            }
            var /= virtEnsemblesCount;
            knowledgeUncertainty[lineInd] = var;
        }
        if (dataUncertaintyPtr) {
            for (size_t dimIdx = 0; dimIdx < virtEnsemblesCount; ++dimIdx) {
                TVector<double> tmp = TVector<double>(
                    approx[dimIdx * 2 + 1].begin() + firstLineId,
                    approx[dimIdx * 2 + 1].begin() + lastLineId);
                CalcSquaredExponentInplace(tmp);
                for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
                    (*dataUncertaintyPtr)[lineInd] += tmp[lineInd - firstLineId];
                }
            }
            for (int lineInd = blockId * blockSize; lineInd < lastLineId; ++lineInd) {
                (*dataUncertaintyPtr)[lineInd] /= virtEnsemblesCount;
            }
        }
    };
    if (executor) {
        executor->ExecRange(calcUncertaitny, 0, threadCount, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        calcUncertaitny(0);
    }
}

namespace NCB {
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
    TVector<TVector<double>> externalApprox(
        externalLabelsHelper.GetExternalApproxDimension(),
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
    NPar::ILocalExecutor* localExecutor
) {
    const TExternalLabelsHelper externalLabelsHelper(model);
    const auto& externalApprox
        = (externalLabelsHelper.IsInitialized() && (externalLabelsHelper.GetExternalApproxDimension() > 1)) ?
            MakeExternalApprox(approx, externalLabelsHelper)
            : approx;
    return PrepareEval(
        predictionType,
        /* ensemblesCount */ 1,
        model.GetLossFunctionName(),
        externalApprox,
        localExecutor,
        model.GetBinClassLogitThreshold());
}

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    size_t ensemblesCount,
    const TString& lossFunctionName,
    const TVector<TVector<double>>& approx,
    int threadCount,
    double binClassLogitThreshold
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    return PrepareEval(
        predictionType,
        ensemblesCount,
        lossFunctionName,
        approx,
        &executor,
        binClassLogitThreshold);
}


void PrepareEval(
    const EPredictionType predictionType,
    size_t ensemblesCount,
    const TString& lossFunctionName,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor,
    TVector<TVector<double>>* result,
    double binClassLogitThreshold
) {

    switch (predictionType) {
        case EPredictionType::LogProbability:
        case EPredictionType::Probability:
            if (IsMulticlass(approx)) {
                if (EqualToOneOf(lossFunctionName, "MultiClassOneVsAll", "MultiLogloss", "MultiCrossEntropy")) {
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
                        CATBOOST_WARNING_LOG << "Optimized loss function was not saved in the model. \
                            Probabilities will be calculated \
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
            if (EqualToOneOf(lossFunctionName, "MultiLogloss", "MultiCrossEntropy")) {
                result->resize(approx.ysize());
                for (auto dim = 0; dim < approx.ysize(); ++dim) {
                    (*result)[dim].reserve(approx[dim].ysize());
                    for (const double prediction: approx[dim]) {
                        (*result)[dim].push_back(prediction > binClassLogitThreshold);
                    }
                }
            } else {
                result->resize(1);
                (*result)[0].reserve(approx.size());
                if (IsMulticlass(approx)) {
                    TVector<int> predictions = {SelectBestClass(approx, executor)};
                    (*result)[0].assign(predictions.begin(), predictions.end());
                } else {
                    for (const double prediction : approx[0]) {
                        (*result)[0].push_back(prediction > binClassLogitThreshold);
                    }
                }
            }
            break;
        case EPredictionType::Exponent:
            *result = {CalcExponent(approx[0])};
            break;
        case EPredictionType::RMSEWithUncertainty:
            Y_ASSERT(approx.size() == 2);
            result->resize(2);
            (*result)[0] = approx[0];
            (*result)[1] = CalcSquaredExponent(approx[1]);
            break;
        case EPredictionType::VirtEnsembles: {
            auto lossFunction = FromString<ELossFunction>(lossFunctionName);
            if (IsRegressionMetric(lossFunction)) {
                *result = approx;
                if (lossFunction == ELossFunction::RMSEWithUncertainty) {
                    for (size_t idx = 1; idx < result->size(); idx += 2) {
                        CalcSquaredExponentInplace((*result)[idx]);
                    }
                }
            }
            else if (IsClassificationMetric(lossFunction)) {
                *result = approx;
            } else {
                CB_ENSURE(false, "uncertainty is not supported for " << lossFunction);
            }
            break;
        }
        case EPredictionType::TotalUncertainty: {
            auto lossFunction = FromString<ELossFunction>(lossFunctionName);
            if (lossFunction == ELossFunction::RMSEWithUncertainty) {
                result->resize(3);
                CalcRegressionUncertaitny(
                    approx,
                    &(result->at(0)),
                    &(result->at(1)),
                    &(result->at(2)),
                    approx.size() / 2,
                    executor);
            } else if (IsRegressionMetric(lossFunction)) {
                result->resize(2);
                CalcRegressionUncertaitny(
                    approx,
                    &(result->at(0)),
                    &(result->at(1)),
                    nullptr,
                    approx.size(),
                    executor);
            } else {
                CB_ENSURE(
                    IsClassificationMetric(lossFunction),
                    "unsupported loss function for uncertainty " << lossFunction);
                const size_t classCount = approx.size() / ensemblesCount;
                result->resize(2, TVector<double>(approx.front().size()));
                if (classCount < 2) {
                    CalcClassificationUncertainty(
                        approx,
                        &(result->at(0)),
                        &(result->at(1)),
                        ensemblesCount,
                        executor);
                } else {
                    CalcMulticlassUncertainty(
                        approx,
                        &(result->at(0)),
                        &(result->at(1)),
                        ensemblesCount,
                        classCount,
                        executor);
                }
            }
            break;
        }
        case EPredictionType::RawFormulaVal:
            *result = approx;
            break;
        default:
            Y_ASSERT(false);
    }
}

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    size_t ensemblesCount,
    const TString& lossFunctionName,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* localExecutor,
    double binClassLogitThreshold
) {
    TVector<TVector<double>> result;
    PrepareEval(
        predictionType,
        ensemblesCount,
        lossFunctionName,
        approx,
        localExecutor,
        &result,
        binClassLogitThreshold);
    return result;
}
}
