#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    inline constexpr ui32 GetDotProductBlockSize() {
        return 512;
    }

    template <class T>
    struct TDotProductContext : public IKernelContext {
        T* PartResults = nullptr;
        ui64 PartResultSize = 0;
        ui64 NumBlocks = 0;
        ui64 Size = 0;
    };


    template<typename T>
    void DotProduct(const T *x, const T *y, TDotProductContext<T>& context, TCudaStream stream);

    template<typename T>
    void WeightedDotProduct(const T *x, const T *weights, const T *y, TDotProductContext<T>& context, TCudaStream stream);
}
