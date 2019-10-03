#include "kappa.h"
#include "classification_utils.h"
#include "metric_holder.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/array_ref.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

static TVector<TVector<int>> GetWeights(EKappaMetricType type, int classCount) {
    TVector<TVector<int>> weights(classCount, TVector<int>(classCount));
    if (type == EKappaMetricType::Cohen) {
        for (int i = 0; i < classCount; ++i) {
            for (int j = 0; j < classCount; ++j) {
                weights[i][j] = (i == j ? 0 : 1);
            }
        }
        return weights;
    } else {
        for (int i = 0; i < classCount; ++i) {
            for (int j = 0; j < classCount; ++j) {
                weights[i][j] = (i - j) * (i - j);
            }
        }
        return weights;
    }
}

static TVector<TVector<double>> GetExpectedMatrix(TConstArrayRef<TVector<int>> matrix, int classCount) {
    TVector<TVector<double>> expected(classCount, TVector<double>(classCount));

    TVector<int> rows(classCount, 0);
    TVector<int> columns(classCount, 0);
    int all = 0;

    for (int i = 0; i < classCount; ++i) {
        for (int j = 0; j < classCount; ++j) {
            all += matrix[i][j];
            rows[i] += matrix[i][j];
            columns[j] += matrix[i][j];
        }
    }

    for (int i = 0; i < classCount; ++i) {
        for (int j = 0; j < classCount; ++j) {
            expected[i][j] = rows[i] * columns[j] / static_cast<double>(all);
        }
    }
    return expected;
}

static TVector<TVector<int>> UnzipKappaMatrix(TMetricHolder metric, int classCount) {
    TVector<TVector<int>> matrix(classCount, TVector<int>(classCount));
    for (int i = 0; i < classCount; ++i) {
        for (int j = 0; j < classCount; ++j) {
            matrix[i][j] = static_cast<int>(metric.Stats[i * classCount + j]);
        }
    }
    return matrix;
}

TMetricHolder CalcKappaMatrix(TConstArrayRef<TVector<double>> approx,
                              TConstArrayRef<float> target,
                              int begin,
                              int end,
                              double border) {
    int classCount = approx.size() == 1 ? 2 : static_cast<int>(approx.size());
    TMetricHolder metric(classCount * classCount);

    for (int i = begin; i < end; ++i) {
        const float targetVal = classCount > 1 ? target[i] : target[i] > border;
        metric.Stats[GetApproxClass(approx, i) * classCount + static_cast<int>(targetVal)] += 1;
    }
    return metric;
}

double CalcKappa(TMetricHolder metric, int classCount, EKappaMetricType type) {

    TVector<TVector<int>> matrix = UnzipKappaMatrix(metric, classCount);
    TVector<TVector<int>> weights = GetWeights(type, classCount);
    TVector<TVector<double>> expected = GetExpectedMatrix(matrix, classCount);

    double matrixSum = 0;
    double expectedSum = 0;

    for (int i = 0; i < classCount; ++i) {
        for (int j = 0; j < classCount; ++j) {
            matrixSum += matrix[i][j] * weights[i][j];
            expectedSum += expected[i][j] * weights[i][j];
        }
    }

    return expectedSum == 0 ? -1 : 1 - matrixSum / expectedSum;
}
