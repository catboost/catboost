#include "compression.cuh"
#include "compression_helper.cuh"
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


//TODO: if we'll need more memory, try full block bit compression instead of ui64 as storage for keys
namespace NKernel {

    template <class TStorageType, ui32 BLOCK_SIZE>
    __global__ void CompressImpl(const ui32* src, int size, TStorageType* dst, ui32 bitsPerKey) {

        TCompressionHelper<TStorageType, BLOCK_SIZE> helper(bitsPerKey);

        src += helper.KeysPerBlock() * blockIdx.x;
        size -= helper.KeysPerBlock() * blockIdx.x;
        dst += BLOCK_SIZE * blockIdx.x;

        TDirectLoader loader(src);
        helper.CompressBlock(loader, size, dst);
    }


    template <class TStorageType, ui32 BLOCK_SIZE>
    __global__ void DecompressImpl(const TStorageType* src, ui32* dst, int size, ui32 bitsPerKey) {

        TCompressionHelper<TStorageType, BLOCK_SIZE> helper(bitsPerKey);

        dst += helper.KeysPerBlock() * blockIdx.x;
        size -= helper.KeysPerBlock() * blockIdx.x;
        src += BLOCK_SIZE * blockIdx.x;
        TDirectWriter writer(dst);
        helper.DecompressBlock(writer, src, size);
    }


    template <class TStorageType, ui32 BLOCK_SIZE>
    __global__ void GatherFromCompressedImpl(const TStorageType* src,
                                             const ui32* map, ui32 mapMask,
                                             ui32* dst, int size, ui32 bitsPerKey) {

        TCompressionHelper<TStorageType, BLOCK_SIZE> helper(bitsPerKey);
        ui32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            const ui32 value = helper.Read(src, map[tid] & mapMask);
            dst[tid] = value;
        }
    }

    template <class TStorageType>
    void GatherFromCompressed(const TStorageType* src, const ui32* map, ui32 mapMask, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream) {

        constexpr ui32 compressedBlockSize = CompressCudaBlockSize();
        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);

        GatherFromCompressedImpl<TStorageType, compressedBlockSize> << < numBlocks, blockSize, 0, stream >> >(src, map, mapMask, dst, size, bitsPerKey);
    }


    template <class TStorageType>
    void Decompress(const TStorageType* src, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream) {

        constexpr ui32 blockSize = CompressCudaBlockSize();
        const ui32 numBlocks = CeilDivide((int)size, TCompressionHelper<TStorageType, blockSize>(bitsPerKey).KeysPerBlock());

        DecompressImpl<TStorageType, blockSize> << < numBlocks, blockSize, 0, stream >> >(src, dst, size, bitsPerKey);
    }

    template <class TStorageType>
    void Compress(const ui32* src, TStorageType* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream) {

        constexpr ui32 blockSize = CompressCudaBlockSize();
        const ui32 numBlocks = CeilDivide((int)size, TCompressionHelper<TStorageType, blockSize>(bitsPerKey).KeysPerBlock());

        CompressImpl<TStorageType, blockSize> << < numBlocks, blockSize, 0, stream >> >(src, size, dst, bitsPerKey);
    }

    template void Compress<ui32>(const ui32* src, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream);
    template void Compress<ui64>(const ui32* src, ui64* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream);

    template void Decompress<ui32>(const ui32* src, ui32* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream);
    template void Decompress<ui64>(const ui64* src, ui32* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream);

    template void GatherFromCompressed<ui64>(const ui64* src, const ui32* map, ui32 mapMask, ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream);

}




