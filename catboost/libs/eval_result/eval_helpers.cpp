#include "eval_helpers.h"

#include <library/fast_exp/fast_exp.h>

#include <util/generic/ymath.h>

#include <functional>


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

bool IsMulticlass(const TVector<TVector<double>>& approx) {
    return approx.size() > 1;
}

TVector<TVector<double>> MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper
) {
    const double inf = std::numeric_limits<double>::infinity();
    TVector<TVector<double>> externalApprox(externalLabelsHelper.GetExternalApproxDimension(),
                                            TVector<double>(internalApprox.back().ysize(), -inf));

    for (int classId = 0; classId < internalApprox.ysize(); ++classId) {
        int visibleId = externalLabelsHelper.GetExternalIndex(classId);

        for (int docId = 0; docId < externalApprox.back().ysize(); ++docId) {
            externalApprox[visibleId][docId] = internalApprox[classId][docId];
        }
    }
    return externalApprox;
}

TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TExternalLabelsHelper& externalLabelsHelper
) {
    TVector<TString> convertedTarget(target.ysize());

    if (externalLabelsHelper.IsInitialized()) {
        for (int targetIdx = 0; targetIdx < target.ysize(); ++targetIdx) {
            convertedTarget[targetIdx] = externalLabelsHelper.GetVisibleClassNameFromLabel(target[targetIdx]);
        }
    } else {
        for (int targetIdx = 0; targetIdx < target.ysize(); ++targetIdx) {
            convertedTarget[targetIdx] = ToString<float>(target[targetIdx]);
        }
    }

    return convertedTarget;
}

TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TFullModel& model
) {
    const auto& externalLabelsHelper = BuildLabelsHelper<TExternalLabelsHelper>(model);
    return ConvertTargetToExternalName(target, externalLabelsHelper);
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
    const auto& externalLabelsHelper = BuildLabelsHelper<TExternalLabelsHelper>(model);
    CB_ENSURE(externalLabelsHelper.IsInitialized() == IsMulticlass(approx),
              "Inappropriated usage of visible label helper: it MUST be initialized ONLY for multiclass problem");
    const auto& externalApprox = externalLabelsHelper.IsInitialized() ?
                                 MakeExternalApprox(approx, externalLabelsHelper) : approx;
    return PrepareEval(predictionType, externalApprox, localExecutor);
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
