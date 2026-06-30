#include "pointwise_targets.cuh"

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>


namespace NKernel {


    struct TQuantileTarget  {
        float Alpha;

        __host__ __device__ __forceinline__ TQuantileTarget(float alpha = 0.5)
                : Alpha(alpha) {
        }

        __device__ __forceinline__ float Score(float target, float prediction) const {
            const float val = target - prediction;
            const float multiplier = (val > 0) ? Alpha : -(1 - Alpha);
            return multiplier * val;
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float val = target - prediction;
            return (val > 0) ? Alpha : -(1.0f - Alpha);
        }

         __device__ __forceinline__ float Der2(float, float) const {
            return 0;
        }
    };

    struct TTweedieTarget  {
        float VariancePower;

        __host__ __device__ __forceinline__ TTweedieTarget(float variancePower = 1.5)
                : VariancePower(variancePower) {
        }

        __device__ __forceinline__ float Score(float target, float prediction) const {
            const float val = -target * std::exp((1 - VariancePower) * prediction) / (1 - VariancePower);
            const float delta = std::exp((2 - VariancePower) * prediction) / (2 - VariancePower);
            return val + delta;
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float der = target * std::exp((1 - VariancePower) * prediction);
            const float delta = std::exp((2 - VariancePower) * prediction);
            return der - delta;
        }

        __device__ __forceinline__ float Der2(float target, float prediction) const {
            const float der2 = target * std::exp((1 - VariancePower) * prediction) * (1 - VariancePower);
            const float delta = std::exp((2 - VariancePower) * prediction) * (2 - VariancePower);
            return - der2 + delta;
        }
    };

    struct THuberTarget  {
        static constexpr double HUBER_DER2 = -1.0;

        float Delta;

        __host__ __device__ __forceinline__ THuberTarget(float delta)
                : Delta(delta) {
        }

        __device__ __forceinline__ float Score(float target, float prediction) const {
            const float targetMismatch = fabs(target - prediction);
            if (targetMismatch < Delta) {
                return 0.5 * targetMismatch * targetMismatch;
            } else {
                return Delta * (targetMismatch - 0.5 * Delta);
            }
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float diff = target - prediction;
            if (fabs(diff) < Delta) {
                return diff;
            } else {
                return diff > 0.0 ? Delta : -Delta;
            }
        }

        __device__ __forceinline__ float Der2(float target, float prediction) const {
            const float diff = target - prediction;
            if (fabs(diff) < Delta) {
                return -HUBER_DER2;
            } else {
                return 0.0;
            }
        }
    };

    struct TExpectileTarget  {
        float Alpha;

        __host__ __device__ __forceinline__ TExpectileTarget(float alpha = 0.5)
                : Alpha(alpha) {
        }

        __device__ __forceinline__ float Score(float target, float prediction) const {
            const float val = target - prediction;
            const float multiplier = (val > 0) ? Alpha : (1 - Alpha);
            return multiplier * val * val;
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float val = target - prediction;
            const float multiplier = (val > 0) ? Alpha : (1 - Alpha);
            return 2.0 * multiplier * val;
        }

        __device__ __forceinline__ float Der2(float target, float prediction) const {
            const float val = target - prediction;
            const float multiplier = (val > 0) ? Alpha : (1 - Alpha);
            return 2.0 * multiplier;
        }
    };

    struct TLogLinQuantileTarget {
        float Alpha;

        __host__ __device__ __forceinline__ TLogLinQuantileTarget(float alpha = 0.5)
                : Alpha(alpha) {
        }

         __device__ __forceinline__ float Score(float target, float prediction) const {
            const float val = target - __expf(prediction);
            const float multiplier = (val > 0) ? Alpha : -(1 - Alpha);
            return val * multiplier;
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float expPred = __expf(prediction);
            return (target - expPred > 0) ? Alpha * expPred : -(1 - Alpha) * expPred;
        }

        __device__ __forceinline__ float Der2(float, float) const {
            return 0;
        }
    };

    struct TMAPETarget  {

        __device__ __forceinline__ float Score(float target, float prediction) const {
            return abs(target - prediction) / max(1.f, abs(target));
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            return (target - prediction > 0) ? 1.0f / max(1.f, abs(target)) : -1.0f / max(1.f, abs(target));
        }

        __device__ __forceinline__  float Der2(float, float) const {
            return 0;
        }
    };

    struct TPoissonTarget  {

        __device__ __forceinline__ float Score(float target, float prediction) const {
            return (__expf(prediction) - target * prediction);
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float expPred = __expf(prediction);
            return target - expPred;
        }

       __device__ __forceinline__ float Der2(float, float prediction) const {
            return __expf(prediction);
        }
    };


    struct TRmseTarget  {

        __device__ __forceinline__ float Score(float target, float prediction) const {
            return  (target - prediction) * (target - prediction);
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            return target - prediction;
        }

        __device__ __forceinline__ float Der2(float, float prediction) const {
            return 1.0f;
        }
    };

    __forceinline__  __device__ float sign(float x) {
        return x > 0 ? 1.0f : -1.0f;
    }

    struct TLqTarget  {
        __host__ __device__ __forceinline__ TLqTarget(float q)
        : Q(q) {

        }

        __device__ __forceinline__ float Score(float target, float prediction) const {
            const float absLoss = abs(target - prediction);
            return __powf(absLoss, Q);
        }

        __device__ __forceinline__ float Der(float target, float prediction) const {
            const float absLoss = abs(target - prediction);
            float absLossQ = powf(absLoss, Q - 1);
            return Q * sign(target - prediction)  * absLossQ;
        }

        __device__ __forceinline__ float Der2(float target, float prediction) const {
            const float absLoss = abs(target - prediction);
            return  Q >= 2 ? Q * (Q - 1) * powf(absLoss, Q - 2) : 1.0f;
        }

        float Q = 2;
    };


    struct TNumErrorsMetric  {
        float K;

        __host__ __device__ __forceinline__ TNumErrorsMetric(float k)
                : K(k) {
        }

        __device__ __forceinline__ float Score(float target, float prediction) const {
            const float val = abs(target - prediction);
            return val > K ? 1 : 0;
        }

        __device__ __forceinline__ float Der(float, float) const {
            return 0;
        }

        __device__ __forceinline__ float Der2(float, float) const {
            return 0;
        }
    };


    template <class TTarget, int BLOCK_SIZE>
    __global__ void PointwiseTargetImpl(const float* relevs, const float* weights, ui32 size,
                                        const float* predictions,
                                        TTarget target,
                                        float* functionValue,
                                        float* der,
                                        float* der2) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float tmpScores[BLOCK_SIZE];

        const float val = i < size ? predictions[i] : 0;
        const float relev = i < size ? relevs[i] : 0;
        const float weight =  (weights && (i < size)) ? weights[i] : 1.0f;

        if (i < size) {
            if (der) {
                der[i] = weight * target.Der(relev, val);
            }
            if (der2) {
                der2[i] = weight * target.Der2(relev, val);
            }
        }

        if (functionValue) {
            tmpScores[threadIdx.x] = (i < size) ? -weight * target.Score(relev, val)  : 0;
            __syncthreads();
        }

        if (functionValue) {
            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);
            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }



    template <int BLOCK_SIZE>
    __global__ void MseImpl(const float* relevs, const float* weights, ui32 size,
                            const float* predictions,
                            float* functionValue,
                            float* der,
                            float* der2) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float tmpScores[BLOCK_SIZE];

        const float val = i < size ? predictions[i] : 0;
        const float relev = i < size ? relevs[i] : 0;
        const float direction = relev - val;
        const float weight =  (weights && (i < size)) ? weights[i] : 1.0f;

        if (i < size) {
            if (der) {
                der[i] = weight * direction;
            }
            if (der2) {
                der2[i] = weight;
            }
        }

        if (functionValue) {
            tmpScores[threadIdx.x] = (i < size) ? -weight * (val - relev) * (val - relev)  : 0;
            __syncthreads();
        }

        if (functionValue) {
            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);
            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }



    template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD, bool HAS_BORDER>
    __launch_bounds__(BLOCK_SIZE, CUDA_MAX_THREADS_PER_SM / BLOCK_SIZE)
    __global__ void CrossEntropyImpl(const float* targetClasses, const float* targetWeights, ui32 size,
                                     const float* predictions,
                                     float* functionValue, float* der, float* der2,
                                     float border) {
        ui32 tid = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + threadIdx.x;

        float tmpScore = 0;

        float direction[ELEMENTS_PER_THREAD];
        float weight[ELEMENTS_PER_THREAD];
        float scale[ELEMENTS_PER_THREAD];

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = tid + j * BLOCK_SIZE;
            direction[j] = idx < size ? predictions[idx] : 0;
            weight[j] = (targetWeights && (idx < size)) ? targetWeights[idx] : 1.0f;
            scale[j] = (idx < size) ? targetClasses[idx] : 1.0f;
        }

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = tid + j * BLOCK_SIZE;
            const float val = direction[j];
            const float targetClass = scale[j];

            const float expVal = idx < size ? __expf(val) : 0;
            const float p = max(min(isfinite(expVal) ? expVal / (1.0f + expVal) : 1.0f, 1.0f - 1e-40f), 1e-40f);
            const float c = HAS_BORDER ? targetClass > border : targetClass;

            direction[j] = c - p; //c * (1 - p) - (1-c) * p;
            scale[j] = p * (1.0f - p);

            if (functionValue) {
                const float logExpValPlusOne = isfinite(expVal) ? __logf(1 + expVal) : val;
                tmpScore += (idx < size) ? weight[j] * (c * val - logExpValPlusOne) : 0;
            }
        }

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = tid + j * BLOCK_SIZE;

            //we already classify this observations
            if (der && (idx < size)) {
                der[idx] = weight[j] * direction[j];
            }
            if (der2  && (idx < size)) {
               der2[idx] = weight[j] * scale[j];
            }
        }

        if (functionValue) {
            __shared__ float tmpScores[BLOCK_SIZE];
            tmpScores[threadIdx.x] = tmpScore;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }



    void CrossEntropyTargetKernel(const float* targetClasses, const float* targetWeights, ui32 size,
                                  const float* predictions,
                                  float* functionValue, float* der, float* der2,
                                  float border, bool useBorder, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 elementsPerThreads = 2;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }

        if (useBorder)
        {
            CrossEntropyImpl < blockSize, elementsPerThreads, true ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, targetWeights, size, predictions, functionValue, der, der2, border);
        } else {
            CrossEntropyImpl < blockSize, elementsPerThreads, false ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, targetWeights, size, predictions, functionValue, der, der2, border);
        }
    }


    void MseTargetKernel(const float* relevs, const float* weights, ui32 size,
                         const float* predictions,
                         float* functionValue, float* der, float* der2,
                         TCudaStream stream) {
        const ui32 blockSize = 1024;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        MseImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(relevs, weights, size, predictions, functionValue, der, der2);
    }




    template <int BLOCK_SIZE, class TTarget>
    void RunPointwiseTargetKernel(const float* relevs, const float* weights, ui32 size,
                                  TTarget target,
                                  const float* predictions,
                                  float* functionValue, float* der, float* der2,
                                  TCudaStream stream) {
        const ui32 numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        PointwiseTargetImpl<TTarget, BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, 0, stream>>>(relevs, weights, size, predictions, target, functionValue, der, der2);

    }


#define POINTWISE_TARGET() \
    RunPointwiseTargetKernel<blockSize>(relevs, weights, size, target, predictions, functionValue, der, der2, stream);


    void PointwiseTargetKernel(const float* relevs, const float* weights, ui32 size,
                               ELossFunction loss, float alpha,
                               const float* predictions,
                               float* functionValue, float* der, float* der2,
                               TCudaStream stream)
    {
        const ui32 blockSize = 1024;
        switch (loss)
        {
            case ELossFunction::Expectile:
            {
                TExpectileTarget target(alpha);
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::Quantile:
            case ELossFunction::MAE:
            {
                TQuantileTarget target(alpha);
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::LogLinQuantile:
            {
                TLogLinQuantileTarget target(alpha);
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::MAPE:
            {
                TMAPETarget target;
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::Poisson:
            {
                TPoissonTarget target;
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::Lq:
            {
                TLqTarget target(alpha);
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::RMSE:
            {
                TRmseTarget target;
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::NumErrors:
            {
                TNumErrorsMetric target(alpha);
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::Tweedie:
            {
                TTweedieTarget target(alpha);
                POINTWISE_TARGET()
                break;
            }
            case ELossFunction::Huber:
            {
                THuberTarget target(alpha);
                POINTWISE_TARGET()
                break;
            }
            default: {
                Y_ABORT_UNLESS(false, "Unknown target");
            }
        }
    }
}
