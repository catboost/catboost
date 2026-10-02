#pragma once

#include "ders_holder.h"
#include "hessian.h"

#include <util/generic/fwd.h>
#include <util/generic/string.h>

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

    using TGpuCalcDersRangePtr = void(*)(
        TConstArrayRef<float> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<float> value,
        TConstArrayRef<float> der1Result,
        TConstArrayRef<float> der2Result,
        size_t length,
        void* customData,
        void* cudaStream,
        size_t blockSize,
        size_t numBlocks);

    using TCalcDersMultiClassPtr = void (*)(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* ders,
        THessianInfo* der2,
        void* customData);

    using TCalcDersMultiTargetPtr = void (*)(
        TConstArrayRef<double> approx,
        TConstArrayRef<float> target,
        float weight,
        TVector<double>* ders,
        THessianInfo* der2,
        void* customData);

public:
    // Metal per-object objective function body. It receives float approx,
    // target and weight, and returns float3(value, negative loss derivative,
    // positive curvature), with weights already applied. CUDA callbacks keep
    // their existing ABI and are never interpreted as Metal callbacks.
    TString MetalSource;
    void* CustomData = nullptr;
    TGpuCalcDersRangePtr GpuCalcDersRange = nullptr;
    TCalcDersRangePtr CalcDersRange = nullptr;
    TCalcDersMultiClassPtr CalcDersMultiClass = nullptr;
    TCalcDersMultiTargetPtr CalcDersMultiTarget = nullptr;
};
