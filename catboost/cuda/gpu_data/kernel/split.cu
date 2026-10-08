#include <catboost/cuda/gpu_data/gpu_structures.h>

#include <catboost/cuda/cuda_util/kernel/compression.cuh>
#include <catboost/cuda/cuda_util/kernel/compression_helper.cuh>
#include <library/cpp/cuda/wrappers/arch.h>

namespace NKernel {

    struct TBinSplitLoader {
        const ui32* CompressedIndex;
        const ui32* Indices;
#if defined(USE_HIP)
        ui32 BinIdx;
        ui32 FeatureMask;
        ui32 Shift;
#else
        ui32 Value;
        ui32 Mask;
#endif
        bool TakeEqual;

#if defined(USE_HIP)
        __forceinline__ __device__ TBinSplitLoader(const ui32* index,
                                                   const ui32* indices,
                                                   const ui32 binIdx,
                                                   const ui32 featureMask,
                                                   const ui32 shift,
                                                   bool takeEqual)
                : CompressedIndex(index)
                , Indices(indices)
                , BinIdx(binIdx)
                , FeatureMask(featureMask)
                , Shift(shift)
                , TakeEqual(takeEqual) {

        }

        __forceinline__ __device__ ui32 operator()(ui32 offset) {
            const ui32 idx = Indices ? Indices[offset] : offset;
            // Compare the extracted bin against binIdx directly. Shifting binIdx
            // up (binIdx << Shift) instead overflows ui32 when a feature sits at a
            // high shift (e.g. a 1-bit feature at Shift=31) and binIdx is out of
            // range for it, making an == split spuriously match -- the CPU
            // compares the unshifted bin, so match that. This is a pre-existing
            // overflow that catboost's BinBuilderTest exercises; HIP-only here so
            // the NVIDIA path is byte-identical.
            const ui32 featureVal = (CompressedIndex[idx] >> Shift) & FeatureMask;
            return static_cast<ui32>(TakeEqual ? (featureVal == BinIdx) : featureVal > BinIdx);
        }
#else
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
#endif
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

#if defined(USE_HIP)
        TBinSplitLoader loader(compressedIndex, indices, binIdx, feature.Mask, feature.Shift, feature.OneHotFeature);
#else
        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;

        TBinSplitLoader loader(compressedIndex, indices, value, mask, feature.OneHotFeature);
#endif
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
            WriteCompressedSplitImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>>(feature, binIdx, compressedIndex,
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
            WriteCompressedSplitFloatImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>>(values, border, indices, size, compressedBits);
        }
    }

    void UpdateBins(const ui64* compressedBits,
                    ui32 depth,
                    ui32* bins, int size,
                    TCudaStream stream) {

        constexpr int blockSize = CompressCudaBlockSize();
        const int numBlocks = CeilDivide(size, TCompressionHelper<ui64, blockSize>(1).KeysPerBlock());

        if (numBlocks) {
            UpdateBinsImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>>(compressedBits, depth, bins, size);
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

#if !defined(USE_HIP)
        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;
#endif

        while (i < size) {
            const ui32 idx = indices ? __ldg(indices + i) : i;
#if defined(USE_HIP)
            // Extract the bin and compare to binIdx directly; shifting binIdx up
            // overflows ui32 for a feature at a high Shift with an out-of-range
            // binIdx, so an == split spuriously matches (see TBinSplitLoader).
            // HIP-only so the NVIDIA path stays byte-identical.
            const ui32 featureVal = (__ldg(compressedIndex + idx) >> feature.Shift) & feature.Mask;
            const ui32 split = (feature.OneHotFeature ? (featureVal == binIdx) : featureVal > binIdx);
#else
            const ui32 featureVal = __ldg(compressedIndex + idx) & mask;
            const ui32 split = (feature.OneHotFeature ? (featureVal == value) : featureVal > value);
#endif
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
            UpdateBinsFromCompressedIndexImpl <<< numBlocks, blockSize, 0, stream >>>(compressedIndex, indices, size, feature, binIdx, depth, bins);
        }
    }

}
