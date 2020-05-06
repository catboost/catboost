#pragma once

#include "ders_holder.h"
#include "hessian.h"

#include <util/generic/fwd.h>

struct TDers;
class THessianInfo;


struct TCustomObjectiveDescriptor {
    using TCalcDersRangePtr = void (*)(
        int count,
        const double* approxes,
        const float* targets,
        const float* weights,
        TDers* ders,
        void* customData);
    
    using TCalcDersMultiClassPtr = void (*)(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* ders,
        THessianInfo* der2,
        void* customData);

    using TCalcDersMultiRegressionPtr = void (*)(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* ders,
        THessianInfo* der2,
        void* customData);

public:
    void* CustomData = nullptr;
    TCalcDersRangePtr CalcDersRange = nullptr;
    TCalcDersMultiClassPtr CalcDersMultiClass = nullptr;
    TCalcDersMultiRegressionPtr CalcDersMultiRegression = nullptr;
};
