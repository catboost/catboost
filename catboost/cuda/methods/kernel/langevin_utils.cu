#include <catboost/cuda/methods/kernel/langevin_utils.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

namespace NKernel {

    template <int BLOCK_SIZE>
    __global__ void AddLangevinNoiseImpl(ui64* seeds,
                                         float* values,
                                         ui32 objectsCount,
                                         float coefficient) {
        ui64 i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        seeds += i;

        ui64 randomSeed = seeds[0];
        while (i < objectsCount) {
            values[i] += coefficient * NextNormal(&randomSeed);
            i += gridDim.x * BLOCK_SIZE;
        }
        seeds[0] = randomSeed;

        return;
    }

    void AddLangevinNoise(ui64* seeds,
                          float* values,
                          ui32 objectsCount,
                          float coefficient,
                          TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui64 blocksNum = min((objectsCount + blockSize - 1) / blockSize,
                                   (ui32)TArchProps::MaxBlockCount());

        AddLangevinNoiseImpl<blockSize> << < blocksNum, blockSize, 0, stream >> > (seeds,
                                                                                   values,
                                                                                   objectsCount,
                                                                                   coefficient);
    }
}
