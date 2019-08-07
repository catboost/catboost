#include "auc_mu.h"

#include <catboost/libs/helpers/parallel_sort/parallel_sort.h>
#include <catboost/libs/index_range/index_range.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/maybe.h>

namespace {
    struct TElement {
        bool IsFirst;
        double Prediction;
        double Weight;
    };
}

static bool CompareByPrediction(const TElement& left, const TElement& right) {
    return std::tie(right.Prediction, left.IsFirst) < std::tie(left.Prediction, right.IsFirst);
}

static double CalcAucBetweenTwoClasses(
    const double signSum,
    TVector<TElement>* elements,
    NPar::TLocalExecutor* localExecutor
) {
    if (signSum == 0) {
        return 0.5;
    }
    if (signSum < 0) {
        for (auto& element : *elements) {
            element.Prediction = -element.Prediction;
        }
    }
    double result = 0;
    NCB::ParallelMergeSort(CompareByPrediction, elements, localExecutor);
    double firstWeightSum = 0;
    double secondWeightSum = 0;
    double currentFirstWeightSum = 0;
    double currentSecondWeightSum = 0;
    double deltaSum = 0;
    for (ui32 i = 0; i < elements->size(); ++i) {
        const auto& element = (*elements)[i];
        if (element.IsFirst) {
            result += element.Weight * secondWeightSum;
            firstWeightSum += element.Weight;
            currentFirstWeightSum += element.Weight;
        } else {
            secondWeightSum += element.Weight;
            currentSecondWeightSum += element.Weight;
        }
        if (i + 1 == elements->size() || (*elements)[i].Prediction != (*elements)[i + 1].Prediction) {
            deltaSum += currentFirstWeightSum * currentSecondWeightSum;
            currentFirstWeightSum = 0;
            currentSecondWeightSum = 0;
        }
    }
    result -= deltaSum / 2.0;
    return result / (firstWeightSum * secondWeightSum);
}

double CalcMuAuc(
    const TVector<TVector<double>>& approx,
    const TConstArrayRef<float>& target,
    const TConstArrayRef<float>& weight,
    NPar::TLocalExecutor* localExecutor,
    const TMaybe<TVector<TVector<double>>>& misclassCostMatrix
) {
    ui32 threadCount = Min((ui32)localExecutor->GetThreadCount() + 1, (ui32)target.size());
    NCB::TEqualRangesGenerator<ui32> generator({0, (ui32)target.size()}, threadCount);
    TVector<TVector<double>> dotProducts(approx);
    ui32 classCount = approx.size();
    NPar::ParallelFor(
        *localExecutor,
        0,
        threadCount,
        [&](int blockId) {
            for (ui32 i : generator.GetRange(blockId).Iter()) {
                if (misclassCostMatrix) {
                    for (ui32 row = 0; row < classCount; ++row) {
                        dotProducts[row][i] = 0;
                        for (ui32 k = 0; k < classCount; ++k) {
                            dotProducts[row][i] += (*misclassCostMatrix)[row][k] * approx[k][i];
                        }
                    }
                } else {
                    double sum = 0;
                    for (ui32 row = 0; row < classCount; ++row) {
                        sum += approx[row][i];
                    }
                    for (ui32 row = 0; row < classCount; ++row) {
                        dotProducts[row][i] = sum - approx[row][i];
                    }
                }
            }
        }
    );
    TVector<TVector<ui32>> indicesByTarget(classCount);
    for (ui32 i = 0; i < target.size(); ++i) {
        indicesByTarget[static_cast<ui32>(target[i])].emplace_back(i);
    }
    double result = 0;
    for (ui32 i = 0; i < classCount; ++i) {
        if (indicesByTarget[i].empty()) {
            continue;
        }
        for (ui32 j = i + 1; j < classCount; ++j) {
            if (indicesByTarget[j].empty()) {
                continue;
            }
            const auto realWeight = [&](ui32 index) {
                return weight.empty() ? 1.0 : weight[index];
            };
            TVector<TElement> elements;
            elements.reserve(indicesByTarget[i].size() + indicesByTarget[j].size());
            for (ui32 index : indicesByTarget[i]) {
                elements.emplace_back(TElement{true, dotProducts[i][index] - dotProducts[j][index], realWeight(index)});
            }
            for (ui32 index : indicesByTarget[j]) {
                elements.emplace_back(TElement{false, dotProducts[i][index] - dotProducts[j][index], realWeight(index)});
            }
            double signSum = 2;
            if (misclassCostMatrix) {
                signSum = (*misclassCostMatrix)[i][j] + (*misclassCostMatrix)[j][i];
            }
            result += CalcAucBetweenTwoClasses(signSum, &elements, localExecutor);
        }
    }
    return (2.0 * result) / (classCount * (classCount - 1));
}

double CalcMuAuc(
    const TVector<TVector<double>>& approx,
    const TConstArrayRef<float>& target,
    const TConstArrayRef<float>& weight,
    int threadCount,
    const TMaybe<TVector<TVector<double>>>& misclassCostMatrix
) {
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);
    return CalcMuAuc(approx, target, weight, &localExecutor, misclassCostMatrix);
}
