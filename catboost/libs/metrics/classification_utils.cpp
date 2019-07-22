#include "metric_holder.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

/* Classification helpers */

int GetApproxClass(TConstArrayRef<TVector<double>> approx, int docIdx) {
    if (approx.size() == 1) {
        return approx[0][docIdx] > 0.0;
    }
    double maxApprox = approx[0][docIdx];
    int maxApproxIndex = 0;

    for (size_t dim = 1; dim < approx.size(); ++dim) {
        if (approx[dim][docIdx] > maxApprox) {
            maxApprox = approx[dim][docIdx];
            maxApproxIndex = dim;
        }
    }
    return maxApproxIndex;
}

void GetPositiveStats(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double border,
        double* truePositive,
        double* targetPositive,
        double* approxPositive
) {
    double truePos = 0;
    double targetPos = 0;
    double approxPos = 0;
    const bool isMulticlass = approx.size() > 1;
    const int classesCount = isMulticlass ? approx.size() : 2;
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        const float targetVal = isMulticlass ? target[i] : target[i] > border;
        int targetClass = static_cast<int>(targetVal);
        Y_ASSERT(targetClass >= 0 && targetClass < classesCount);

        float w = weight.empty() ? 1 : weight[i];

        if (targetClass == positiveClass) {
            targetPos += w;
            if (approxClass == positiveClass) {
                truePos += w;
            }
        }
        if (approxClass == positiveClass) {
            approxPos += w;
        }
    }

    *truePositive = truePos;
    *targetPositive = targetPos;
    *approxPositive = approxPos;
}

void GetSpecificity(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double border,
        double* trueNegative,
        double* targetNegative
) {
    double trueNeg = 0;
    double targetNeg = 0;
    const bool isMulticlass = approx.size() > 1;
    const int classesCount = isMulticlass ? approx.size() : 2;
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        const float targetVal = isMulticlass ? target[i] : target[i] > border;
        int targetClass = static_cast<int>(targetVal);
        Y_ASSERT(targetClass >= 0 && targetClass < classesCount);

        float w = weight.empty() ? 1 : weight[i];

        if (targetClass != positiveClass) {
            targetNeg += w;
            if (approxClass != positiveClass) {
                trueNeg += w;
            }
        }
    }
    *trueNegative = trueNeg;
    *targetNegative = targetNeg;
}

void GetTotalPositiveStats(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        TVector<double>* truePositive,
        TVector<double>* targetPositive,
        TVector<double>* approxPositive,
        double border
) {
    const bool isMultiClass = approx.size() > 1;
    const int classesCount = isMultiClass ? approx.size() : 2;
    truePositive->assign(classesCount, 0);
    targetPositive->assign(classesCount, 0);
    approxPositive->assign(classesCount, 0);
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        int targetClass = isMultiClass ? static_cast<int>(target[i]) : target[i] > border;
        Y_ASSERT(targetClass >= 0 && targetClass < classesCount);

        float w = weight.empty() ? 1 : weight[i];

        if (approxClass == targetClass) {
            (*truePositive)[targetClass] += w;
        }
        (*targetPositive)[targetClass] += w;
        (*approxPositive)[approxClass] += w;
    }
}

TMetricHolder GetAccuracy(
        const TVector<TVector<double>>& approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        double border
) {
    TMetricHolder error(2);
    const bool isMulticlass = approx.size() > 1;
    const int classesCount = isMulticlass ? approx.size() : 2;
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i);
        const float targetVal = isMulticlass ? target[i] : target[i] > border;
        int targetClass = static_cast<int>(targetVal);
        Y_ASSERT(targetClass >= 0 && targetClass < classesCount);

        float w = weight.empty() ? 1 : weight[i];
        error.Stats[0] += approxClass == targetClass ? w : 0.0;
        error.Stats[1] += w;
    }
    return error;
}
