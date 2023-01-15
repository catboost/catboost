#include <catboost/cuda/gpu_data/gpu_structures.h>

#include <catboost/cuda/cuda_util/kernel/compression.cuh>
#include <catboost/cuda/cuda_util/kernel/compression_helper.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

namespace NKernel {

    struct TBinSplitLoader {
        const ui32* CompressedIndex;
        const ui32* Indices;
        ui32 Value;
        ui32 Mask;
        bool TakeEqual;

        __forceinline__ __device__ TBinSplitLoader(const ui32* index,
                                                   const ui32* indices,
                                                   const ui32 value,
                                                   const ui32 mask,
                                                   bool takeEqual)
                : CompressedIndex(index)
                , Indices(indices)
                , Value(value)
                , Mask(mask)
                , TakeEqual(takeEqual) {

        }

        __forceinline__ __device__ ui32 operator()(ui32 offset) {
            const ui32 idx = Indices ? Indices[offset] : offset;
            const ui32 featureVal = CompressedIndex[idx] & Mask;
            return static_cast<ui32>(TakeEqual ? (featureVal == Value) : featureVal > Value);
        }
    };


    struct TFloatSplitLoader {

        const float* Values;
        const ui32* Indices;
        float Border;

        __device__ TFloatSplitLoader(const float* values,
                                     const ui32* indices,
                                     float border
        )
                : Values(values)
                , Indices(indices)
                , Border(border) {

        }

        __forceinline__ __device__ ui32 operator()(ui32 offset) {
            ui32 idx = Indices ? Indices[offset] : offset;
            return static_cast<ui32>(Values[idx] > Border);
        }
    };


    struct TBinUpdater {
        ui32* Bins;
        ui32 Depth;

        __forceinline__ __device__ TBinUpdater(ui32* bins, ui32 depth)
                : Bins(bins)
                , Depth(depth) {

        }

        __forceinline__ __device__ ui32 operator()(ui32 offset, ui32 bin) {
            return Bins[offset] |= bin << Depth;
        }
    };


    template <int BLOCK_SIZE>
    __global__ void WriteCompressedSplitImpl(TCFeature feature, ui32 binIdx,
                                             const ui32* compressedIndex,
                                             const ui32* indices, int size,
                                             ui64* compressedBits)
    {
        TCompressionHelper<ui64, BLOCK_SIZE> helper(1);

        if (indices) {
            indices += helper.KeysPerBlock() * blockIdx.x;
        } else {
            compressedIndex +=  helper.KeysPerBlock() * blockIdx.x;
        }
        size -= helper.KeysPerBlock() * blockIdx.x;

        compressedBits += BLOCK_SIZE * blockIdx.x;
        compressedIndex += feature.Offset;

        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;

        TBinSplitLoader loader(compressedIndex, indices, value, mask, feature.OneHotFeature);
        helper.CompressBlock(loader, size, compressedBits);
    }



    template <int BLOCK_SIZE>
    __global__ void WriteCompressedSplitFloatImpl(const float* values, float border,
                                                  const ui32* indices, int size,
                                                  ui64* compressedBits)
    {
        TCompressionHelper<ui64, BLOCK_SIZE> helper(1);

        if (indices) {
            indices += helper.KeysPerBlock() * blockIdx.x;
        } else {
            values += helper.KeysPerBlock() * blockIdx.x;
        }
        size -= helper.KeysPerBlock() * blockIdx.x;
        compressedBits += BLOCK_SIZE * blockIdx.x;

        TFloatSplitLoader loader(values, indices, border);
        helper.CompressBlock(loader, size, compressedBits);
    }


    template <int BLOCK_SIZE>
    __global__ void UpdateBinsImpl(const ui64* compressedBits,
                                   ui32 depth,
                                   ui32* bins, int size) {

        TCompressionHelper<ui64, BLOCK_SIZE> helper(1);

        bins += helper.KeysPerBlock() * blockIdx.x;
        size -= helper.KeysPerBlock() * blockIdx.x;
        compressedBits += BLOCK_SIZE * blockIdx.x;

        TBinUpdater writer(bins, depth);
        helper.DecompressBlock(writer, compressedBits, size);
    }

    void WriteCompressedSplit(TCFeature feature, ui32 binIdx,
                              const ui32* compressedIndex,
                              const ui32* indices, int size,
                              ui64* compressedBits,
                              TCudaStream stream) {

        constexpr int blockSize = CompressCudaBlockSize();
        const int numBlocks = CeilDivide(size, TCompressionHelper<ui64, blockSize>(1).KeysPerBlock());

        if (numBlocks) {
            WriteCompressedSplitImpl<blockSize> << < numBlocks, blockSize, 0, stream >> >(feature, binIdx, compressedIndex,
                    indices, size, compressedBits);
        }
    }

    void WriteCompressedSplitFloat(const float* values, float border,
                                   const ui32* indices, int size,
                                   ui64* compressedBits,
                                   TCudaStream stream) {
        constexpr int blockSize = CompressCudaBlockSize();
        const int numBlocks = CeilDivide(size, TCompressionHelper<ui64, blockSize>(1).KeysPerBlock());

        if (numBlocks) {
            WriteCompressedSplitFloatImpl<blockSize> << < numBlocks, blockSize, 0, stream >> >(values, border, indices, size, compressedBits);
        }
    }

    void UpdateBins(const ui64* compressedBits,
                    ui32 depth,
                    ui32* bins, int size,
                    TCudaStream stream) {

        constexpr int blockSize = CompressCudaBlockSize();
        const int numBlocks = CeilDivide(size, TCompressionHelper<ui64, blockSize>(1).KeysPerBlock());

        if (numBlocks) {
            UpdateBinsImpl<blockSize> << < numBlocks, blockSize, 0, stream >> >(compressedBits, depth, bins, size);
        }
    }


    __global__ void UpdateBinsFromCompressedIndexImpl(const ui32* compressedIndex,
                                                      const ui32* indices,
                                                      const int size,
                                                      const TCFeature feature,
                                                      const ui32 binIdx,
                                                      const ui32 depth,
                                                      ui32* bins)
    {

        compressedIndex += feature.Offset;
        int i =  blockIdx.x * blockDim.x + threadIdx.x;

        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;

        while (i < size) {
            const ui32 idx = indices ? __ldg(indices + i) : i;
            const ui32 featureVal = __ldg(compressedIndex + idx) & mask;
            const ui32 split = (feature.OneHotFeature ? (featureVal == value) : featureVal > value);
            bins[i] |= split << depth;
            i += blockDim.x * gridDim.x;
        }
    }

    void UpdateBinsFromCompressedIndex(const ui32* compressedIndex,
                                       const ui32* indices,
                                       const int size,
                                       const TCFeature feature,
                                       const ui32 binIdx,
                                       const ui32 depth,
                                       ui32* bins,
                                       TCudaStream stream) {

        constexpr int blockSize = 256;
        const int numBlocks = min(CeilDivide(size, blockSize), TArchProps::MaxBlockCount());

        if (numBlocks) {
            UpdateBinsFromCompressedIndexImpl << < numBlocks, blockSize, 0, stream >> >(compressedIndex, indices, size, feature, binIdx, depth, bins);
        }
    }

}
