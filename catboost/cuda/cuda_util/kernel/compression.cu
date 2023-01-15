#include "compression.cuh"
#include "compression_helper.cuh"
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <stdio.h>
#include <stdlib.h>



//TODO: if we'll need more memory, try full block bit compression instead of ui64 as storage for keys
namespace NKernel {

    template <class TStorageType, ui32 BLOCK_SIZE>
    __global__ void CompressImpl(const ui32* src, int size, TStorageType* dst, ui32 bitsPerKey, int blockCount) {

        int blockId = blockIdx.x;
        TCompressionHelper<TStorageType, BLOCK_SIZE> helper(bitsPerKey);
        while (blockId < blockCount) {
            TDirectLoader loader(src + helper.KeysPerBlock() * blockId);
            helper.CompressBlock(loader, size - helper.KeysPerBlock() * blockId, dst + BLOCK_SIZE * blockId);
            blockId += gridDim.x;
        }
    }


    template <class TStorageType, ui32 BLOCK_SIZE>
    __global__ void DecompressImpl(const TStorageType* src, ui32* dst, int size, ui32 bitsPerKey, int blockCount) {

        TCompressionHelper<TStorageType, BLOCK_SIZE> helper(bitsPerKey);

        int blockId = blockIdx.x;
        while (blockId < blockCount) {
            TDirectWriter writer(dst + helper.KeysPerBlock() * blockId);
            helper.DecompressBlock(writer, src +  BLOCK_SIZE * blockId, size - helper.KeysPerBlock() * blockId);
            blockId += gridDim.x;
        }
    }


    template <class TStorageType, ui32 BLOCK_SIZE>
    __global__ void GatherFromCompressedImpl(const TStorageType* src,
                                             const ui32* map, ui32 mapMask,
                                             ui32* dst, int size, ui32 bitsPerKey) {

        TCompressionHelper<TStorageType, BLOCK_SIZE> helper(bitsPerKey);
        ui32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        while (tid < size) {
            const ui32 value = helper.Read(src, map[tid] & mapMask);
            dst[tid] = value;
            tid += blockDim.x * gridDim.x;
        }
    }

    template <class TStorageType>
    void GatherFromCompressed(const TStorageType* src, const ui32* map, ui32 mapMask, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream) {

        constexpr ui32 compressedBlockSize = CompressCudaBlockSize();
        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);

        GatherFromCompressedImpl<TStorageType, compressedBlockSize> << < min(TArchProps::MaxBlockCount(), numBlocks), blockSize, 0, stream >> >(src, map, mapMask, dst, size, bitsPerKey);
    }


    template <class TStorageType>
    void Decompress(const TStorageType* src, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream) {

        constexpr ui32 blockSize = CompressCudaBlockSize();
        const ui32 numBlocks = CeilDivide((int)size, TCompressionHelper<TStorageType, blockSize>(bitsPerKey).KeysPerBlock());

        DecompressImpl<TStorageType, blockSize> << < min(TArchProps::MaxBlockCount(), numBlocks), blockSize, 0, stream >> >(src, dst, size, bitsPerKey, numBlocks);
    }

    template <class TStorageType>
    void Compress(const ui32* src, TStorageType* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream) {

        constexpr ui32 blockSize = CompressCudaBlockSize();
        const ui32 numBlocks = CeilDivide((int)size, TCompressionHelper<TStorageType, blockSize>(bitsPerKey).KeysPerBlock());
        CompressImpl<TStorageType, blockSize> << < min(TArchProps::MaxBlockCount(), numBlocks), blockSize, 0, stream >> >(src, size, dst, bitsPerKey, numBlocks);
    }

    #define COMPRESS(Type) \
    template void GatherFromCompressed<Type>(const Type* src, const ui32* map, ui32 mapMask, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream); \
    template void Compress<Type>(const ui32* src, Type* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream);\
    template void Decompress<Type>(const Type* src, ui32* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream);

    COMPRESS(ui32)
    COMPRESS(ui64)


}




