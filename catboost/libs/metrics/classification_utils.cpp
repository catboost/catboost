#include "classification_utils.h"

#include "metric_holder.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

/* Classification helpers */

template <typename TArrayLike>
int GetApproxClassImpl(TConstArrayRef<TArrayLike> approx, int docIdx, double predictionLogitBorder) {
    if (approx.size() == 1) {
        return approx[0][docIdx] > predictionLogitBorder;
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

int GetApproxClass(TConstArrayRef<TVector<double>> approx, int docIdx, double predictionLogitBorder) {
    return GetApproxClassImpl<TVector<double>>(approx, docIdx, predictionLogitBorder);
}

int GetApproxClass(TConstArrayRef<TConstArrayRef<double>> approx, int docIdx, double predictionLogitBorder) {
    return GetApproxClassImpl<TConstArrayRef<double>>(approx, docIdx, predictionLogitBorder);
}

void GetPositiveStats(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double targetBorder,
        double predictionLogitBorder,
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
        int approxClass = GetApproxClass(approx, i, predictionLogitBorder);
        const float targetVal = isMulticlass ? target[i] : target[i] > targetBorder;
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
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double targetBorder,
        double predictionLogitBorder,
        double* trueNegative,
        double* targetNegative
) {
    double trueNeg = 0;
    double targetNeg = 0;
    const bool isMulticlass = approx.size() > 1;
    const int classesCount = isMulticlass ? approx.size() : 2;
    for (int i = begin; i < end; ++i) {
        int approxClass = GetApproxClass(approx, i, predictionLogitBorder);
        const float targetVal = isMulticlass ? target[i] : target[i] > targetBorder;
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
