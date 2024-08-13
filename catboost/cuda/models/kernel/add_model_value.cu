#include "add_model_value.cuh"

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>


namespace NKernel {

    //it may be faster to cache in shared memory binValues, but for fold + bin + depth > 10 it'll be slower and may fail on shared memory allocation
    //so current impl more conservative with const-load cache
    template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
    __launch_bounds__(BLOCK_SIZE, CUDA_MAX_THREADS_PER_SM / BLOCK_SIZE)
    __global__ void AddBinModelValueImpl(const float* binValues, ui32 binCount,
                                         const ui32* bins,
                                         ui32 size,
                                         const ui32* readIndices,
                                         const ui32* writeIndices,
                                         ui32 cursorDim, ui32 cursorAlignSize,
                                         float* cursor) {
        const ui32 i = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + threadIdx.x;

        ui32 writeIndicesLocal[ELEMENTS_PER_THREAD];
        ui32 binsLocal[ELEMENTS_PER_THREAD];

        #pragma unroll ELEMENTS_PER_THREAD
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = i + j * BLOCK_SIZE;
            const ui32 readIdx = idx < size ? (readIndices ? LdgWithFallback(readIndices, idx) : idx) : (ui32)-1;;
            writeIndicesLocal[j] = idx < size ? (writeIndices ? LdgWithFallback(writeIndices, idx) : idx) : (ui32)-1;
            binsLocal[j] = idx < size ? LdgWithFallback(bins, readIdx) : 0;
        }


        float binsValuesLocal[ELEMENTS_PER_THREAD];

        for (int dim = 0; dim < cursorDim; ++dim) {

            #pragma unroll ELEMENTS_PER_THREAD
            for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                const int idx = i + j * BLOCK_SIZE;
                binsValuesLocal[j] = idx < size ? __ldg(binValues + binsLocal[j] * cursorDim + dim) : 0;
            }

            #pragma unroll ELEMENTS_PER_THREAD
            for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                const int idx = i + j * BLOCK_SIZE;
                if (idx < size) {
                    cursor[writeIndicesLocal[j] + dim * cursorAlignSize] += binsValuesLocal[j];
                }
            }
        }
    }

    void AddBinModelValue(const float* binValues, ui32 binCount,
                          const ui32* bins,
                          const ui32* readIndices, const ui32* writeIndices,
                          ui32 size,
                          float* cursor,
                          ui32 cursorDim, ui32 cursorAlignSize,
                          TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize * elementsPerThreads);
        AddBinModelValueImpl<blockSize, elementsPerThreads> << <numBlocks, blockSize, 0, stream>>>(binValues, binCount, bins, size, readIndices, writeIndices, cursorDim, cursorAlignSize, cursor);
    }



    __global__ void AddObliviousTreeImpl(const TCFeature* features,
                                         const ui8* bins,
                                         const float* leaves, ui32 depth,
                                         const ui32* cindex,
                                         const ui32* readIndices,
                                         const ui32* writeIndices,
                                         ui32 size,
                                         float* cursor,
                                         ui32 cursorAlignSize,
                                         ui32 cursorDim) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ ui32 masksLocal[32];
        __shared__ ui32 valuesLocal[32];
        __shared__ ui64 offsetsLocal[32];
        __shared__ ui32 takeEqual[32];

        if (threadIdx.x < depth) {
            const int level = threadIdx.x;
            TCFeature feature = features[level];
            const ui32 value =(ui32)(bins[level]) << feature.Shift;
            const ui32 mask = feature.Mask << feature.Shift;

            masksLocal[level] = mask;
            valuesLocal[level] = value;
            takeEqual[level] = feature.OneHotFeature;
            offsetsLocal[level] = feature.Offset;
        }
        __syncthreads();

        while (tid < size) {
            ui32 bin = 0;
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;

            #pragma unroll 8
            for (ui32 level = 0; level < depth; ++level) {
                const ui32 value = valuesLocal[level];
                const ui32 mask = masksLocal[level];
                const ui32 featureVal = __ldg((cindex + offsetsLocal[level]) + loadIdx) & mask;
                const ui32 split = (takeEqual[level] ? (featureVal == value) : featureVal > value);
                bin |= split << level;
            }
            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;

            for (int dim = 0; dim < cursorDim; ++dim) {
                cursor[writeIdx + dim * cursorAlignSize] += __ldg(leaves + bin * cursorDim + dim);
            }
            tid += blockDim.x  * gridDim.x;
        }
    }


    __global__ void ComputeObliviousTreeBinsImpl(const TCFeature* features, const ui8* bins,  ui32 depth,
                                                 const ui32* cindex,
                                                 const ui32* readIndices,
                                                 const ui32* writeIndices,
                                                 ui32* cursor,
                                                 ui32 size) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ ui32 masksLocal[32];
        __shared__ ui32 valuesLocal[32];
        __shared__ ui64 offsetsLocal[32];
        __shared__ ui32 takeEqual[32];

        if (threadIdx.x < depth) {
            const int level = threadIdx.x;
            TCFeature feature = features[level];
            const ui32 value =(ui32)(bins[level]) << feature.Shift;
            const ui32 mask = feature.Mask << feature.Shift;

            masksLocal[level] = mask;
            valuesLocal[level] = value;
            takeEqual[level] = feature.OneHotFeature;
            offsetsLocal[level] = feature.Offset;
        }
        __syncthreads();

        while (tid < size) {
            ui32 bin = 0;
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;

            #pragma unroll 8
            for (ui32 level = 0; level < depth; ++level) {
                const ui32 value = valuesLocal[level];
                const ui32 mask = masksLocal[level];
                const ui32 featureVal = __ldg(cindex + offsetsLocal[level] + loadIdx) & mask;
                const ui32 split = (takeEqual[level] ? (featureVal == value) : featureVal > value);
                bin |= split << level;
            }
            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;
            cursor[writeIdx] = bin;
            tid += blockDim.x  * gridDim.x;
        }
    }


    //doc parallel routines
    void AddObliviousTree(const TCFeature* features, const ui8* bins, const float* leaves, ui32 depth,
                          const ui32* cindex,
                          const ui32* readIndices,
                          const ui32* writeIndices,
                          ui32 size,
                          float* cursor,
                          ui32 cursorDim, ui32 cursorAlignSize,
                          TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
        AddObliviousTreeImpl<< <numBlocks, blockSize, 0, stream>>>(features, bins, leaves, depth, cindex, readIndices, writeIndices, size, cursor, cursorAlignSize, cursorDim);
    }


    void ComputeObliviousTreeBins(const TCFeature* features, const ui8* bins, ui32 depth,
                                  const ui32* cindex,
                                  const ui32* readIndices,
                                  const ui32* writeIndices,
                                  ui32* cursor,
                                  ui32 size,
                                  TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
       ComputeObliviousTreeBinsImpl<< <numBlocks, blockSize, 0, stream>>>(features, bins, depth, cindex, readIndices, writeIndices, cursor, size);
    }



    __global__ void AddRegionImpl(const TCFeature* features,
                                  const TRegionDirection* splits,
                                  const float* leaves, ui32 depth,
                                  const ui32* cindex,
                                  const ui32* readIndices,
                                  const ui32* writeIndices,
                                  ui32 size,
                                  float* cursor,
                                  ui32 cursorAlignSize,
                                  ui32 cursorDim) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        const int maxDepth = 32;
        __shared__ ui32 masksLocal[maxDepth];
        __shared__ ui32 valuesLocal[maxDepth];
        __shared__ ui64 offsetsLocal[maxDepth];
        __shared__ ui32 takeEqualAndSplitDirection[maxDepth];


        if (threadIdx.x < depth) {
            const int level = threadIdx.x;
            TCFeature feature = features[level];
            TRegionDirection split = splits[level];
            const ui32 value =(ui32)(split.Bin) << feature.Shift;
            const ui32 mask = feature.Mask << feature.Shift;

            masksLocal[level] = mask;
            valuesLocal[level] = value;
            takeEqualAndSplitDirection[level] = (feature.OneHotFeature ? 1 : 0) | (split.Value << 1);
            offsetsLocal[level] = feature.Offset;
        }
        __syncthreads();

        while (tid < size) {
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;
            ui32 bin = 0;

            #pragma unroll 8
            for (ui32 level = 0; level < depth; ++level) {
                const ui32 value = valuesLocal[level];
                const ui32 mask = masksLocal[level];
                const ui32 featureVal = __ldg(cindex + offsetsLocal[level] + loadIdx) & mask;
                const ui32 tmp = takeEqualAndSplitDirection[level];
                const bool takeEqual = (tmp & 1);
                const ui32 split = (takeEqual ? (featureVal == value) : featureVal > value);
                const bool shouldContinue = split == (tmp >> 1);
                if (shouldContinue) {
                    ++bin;
                } else {
                    break;
                }
            }
            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;

            for (int dim = 0; dim < cursorDim; ++dim) {
                cursor[writeIdx + dim * cursorAlignSize] += __ldg(leaves + bin * cursorDim + dim);
            }
            tid += blockDim.x  * gridDim.x;
        }
    }


    __global__ void ComputeRegionBinsImpl(const TCFeature* features,
                                          const TRegionDirection* splits,
                                          ui32 depth,
                                          const ui32* cindex,
                                          const ui32* readIndices,
                                          const ui32* writeIndices,
                                          ui32* cursor,
                                          ui32 size) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        const int maxDepth = 32;
        __shared__ ui32 masksLocal[maxDepth];
        __shared__ ui32 valuesLocal[maxDepth];
        __shared__ ui64 offsetsLocal[maxDepth];
        __shared__ ui32 takeEqualAndSplitDirection[maxDepth];


        if (threadIdx.x < depth) {
            const int level = threadIdx.x;
            TCFeature feature = features[level];
            TRegionDirection split = splits[level];
            const ui32 value =(ui32)(split.Bin) << feature.Shift;
            const ui32 mask = feature.Mask << feature.Shift;

            masksLocal[level] = mask;
            valuesLocal[level] = value;
            takeEqualAndSplitDirection[level] = (feature.OneHotFeature ? 1 : 0) | (split.Value << 1);
            offsetsLocal[level] = feature.Offset;
        }
        __syncthreads();

        while (tid < size) {
            ui32 bin = 0;
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;

            #pragma unroll 8
            for (ui32 level = 0; level < depth; ++level) {
                const ui32 value = valuesLocal[level];
                const ui32 mask = masksLocal[level];
                const ui32 featureVal = __ldg(cindex + offsetsLocal[level] + loadIdx) & mask;
                const ui32 tmp = takeEqualAndSplitDirection[level];
                const bool takeEqual = (tmp & 1);
                const ui32 split = (takeEqual ? (featureVal == value) : featureVal > value);
                const bool shouldContinue = split == (tmp >> 1);
                if (shouldContinue) {
                    ++bin;
                } else {
                    break;
                }
            }
            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;
            cursor[writeIdx] = bin;
            tid += blockDim.x  * gridDim.x;
        }
    }


    //doc parallel routines
    void AddRegion(const TCFeature* features,
                   const TRegionDirection* splits,
                   const float* leaves, ui32 depth,
                   const ui32* cindex,
                   const ui32* readIndices,
                   const ui32* writeIndices,
                   ui32 size,
                   float* cursor,
                   ui32 cursorDim,
                   ui32 cursorAlignSize,
                   TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
        AddRegionImpl<< <numBlocks, blockSize, 0, stream>>>(features, splits, leaves, depth, cindex, readIndices, writeIndices, size, cursor, cursorAlignSize, cursorDim);
    }

    void ComputeRegionBins(const TCFeature* features, const TRegionDirection* bins, ui32 depth,
                           const ui32* cindex,
                           const ui32* readIndices,
                           const ui32* writeIndices,
                           ui32* cursor,
                           ui32 size,
                           TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
        ComputeRegionBinsImpl<< <numBlocks, blockSize, 0, stream>>>(features, bins, depth, cindex, readIndices, writeIndices, cursor, size);
    }


    __global__ void ComputeNonSymmetricDecisionTreeBinsImpl(const TCFeature* features,
                                                            const TTreeNode* nodes,
                                                            const ui32* cindex,
                                                            const ui32* readIndices,
                                                            const ui32* writeIndices,
                                                            ui32* cursor,
                                                            ui32 size) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;


        if  (tid < size) {
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;


            ui32 bin = 0;
            bool stop = nodes == nullptr;
            while (!stop) {
                TTreeNode node = Ldg(nodes);
                TCFeature feature = Ldg(features);

                const ui32 featureVal = (__ldg(cindex + feature.Offset  + loadIdx) >> feature.Shift) & feature.Mask;
                const bool split = (feature.OneHotFeature ? (featureVal == node.Bin) : featureVal > node.Bin);

                if (split) {
                    bin += node.LeftSubtree;
                    stop = node.RightSubtree == 1;
                    if (!stop) {
                        nodes += node.LeftSubtree;
                        features += node.LeftSubtree;
                    }
                } else {
                    stop = node.LeftSubtree == 1;
                    if (!stop) {
                        nodes += 1;
                        features += 1;
                    }
                }
            }

            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;
            cursor[writeIdx] = bin;
        }
    }


    void ComputeNonSymmetricDecisionTreeBins(const TCFeature* features,
                                             const TTreeNode* nodes,
                                             const ui32* cindex,
                                             const ui32* readIndices,
                                             const ui32* writeIndices,
                                             ui32* cursor,
                                             ui32 size,
                                             TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
        ComputeNonSymmetricDecisionTreeBinsImpl<< <numBlocks, blockSize, 0, stream>>>(features, nodes, cindex, readIndices, writeIndices, cursor, size);
    }



}
