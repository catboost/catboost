#pragma once

#include "ders_holder.h"
#include "hessian.h"

#include <util/generic/fwd.h>

struct TCustomObjectiveDescriptor {
    using TCalcDersRangePtr = void (*)(
        int count,
        const double* approxes,
        const float* targets,
        const float* weights,
        TDers* ders,
        void* customData);
    using TCalcDersMultiPtr = void (*)(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* ders,
        THessianInfo* der2,
        void* customData);

public:
    void* CustomData = nullptr;
    TCalcDersRangePtr CalcDersRange = nullptr;
    TCalcDersMultiPtr CalcDersMulti = nullptr;
};
