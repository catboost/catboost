#include "auc.h"
#include "auc_mu.h"

#include <catboost/libs/helpers/parallel_sort/parallel_sort.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/maybe.h>

using NMetrics::TBinClassSample;

double CalcMuAuc(
    const TVector<TVector<double>>& approx,
    const TConstArrayRef<float>& target,
    const TConstArrayRef<float>& weight,
    NPar::ILocalExecutor* localExecutor,
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
            double signSum = 2;
            if (misclassCostMatrix) {
                signSum = (*misclassCostMatrix)[i][j] + (*misclassCostMatrix)[j][i];
            }
            if (signSum == 0) {
                result += 0.5;
                continue;
            }
            const auto getApprox = [&](ui32 index) {
                return (signSum < 0 ? dotProducts[i][index] - dotProducts[j][index] : dotProducts[j][index] - dotProducts[i][index]);
            };
            TVector<TBinClassSample> positiveSamples;
            positiveSamples.reserve(indicesByTarget[i].size());
            for (ui32 index : indicesByTarget[i]) {
                positiveSamples.emplace_back(getApprox(index), realWeight(index));
            }
            TVector<TBinClassSample> negativeSamples;
            negativeSamples.reserve(indicesByTarget[j].size());
            for (ui32 index : indicesByTarget[j]) {
                negativeSamples.emplace_back(getApprox(index), realWeight(index));
            }
            result += CalcBinClassAuc(&positiveSamples, &negativeSamples, localExecutor);
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
    NPar::TLocalExecutor localExecutor; // TODO(espetrov): may be slow, if threadCount == 1
    localExecutor.RunAdditionalThreads(threadCount - 1);
    return CalcMuAuc(approx, target, weight, &localExecutor, misclassCostMatrix);
}
