#include "compression.cuh"
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


//TODO: if we'll need more memory, try full block bit compression instead of ui64 as storage for keys
namespace NKernel {

    struct TDirectLoader {
        const ui32* Src;

        __device__ TDirectLoader(const ui32* src)
                : Src(src) {

        }

        __forceinline__ __device__ ui32 operator()(ui32 offset) {
            return Src[offset];
        }
    };


    struct TDirectWriter {
        ui32* Dst;

        __device__ TDirectWriter(ui32* dst)
                : Dst(dst) {

        }

        __forceinline__ __device__ ui32 operator()(ui32 offset, ui32 bin) {
            return Dst[offset] = bin;
        }
    };



    template <class TStorageType, ui32 BLOCK_SIZE>
    struct TCompressionHelper {

        const int BitsPerKey;
        const int KeysPerStorageType;

        __host__ __device__ TCompressionHelper(int bitsPerKey)
                : BitsPerKey(bitsPerKey), KeysPerStorageType(sizeof(TStorageType) * 8 / bitsPerKey) {

        }

        __forceinline__ __device__ __host__ int KeysPerBlock() const {
            return KeysPerStorageType * BLOCK_SIZE;
        }

        __forceinline__ __device__ __host__ int CompressedSize(int size) {
            return CeilDivide(size, KeysPerStorageType);
        }

        __forceinline__ __device__ __host__ ui32 Mask() const {
            return (ui32)((1 << BitsPerKey) - 1);
        }

        __forceinline__ __device__ ui32 GetBlockIndex(ui32 key) const {
            return CeilDivide<ui32>(key, KeysPerBlock());
        }

        __forceinline__ __device__ ui32 Read(const TStorageType* __restrict data,
                                             ui32 key) const {
            const ui32 blockOffset = key / KeysPerBlock();
            key %= KeysPerBlock();
            const int id = key / BLOCK_SIZE;
            const int offset = key & (BLOCK_SIZE - 1);
            return (LdgWithFallback(data, blockOffset * BLOCK_SIZE + offset) >> (((KeysPerStorageType - id - 1) * BitsPerKey))) & Mask();
        }

        template <class TLoader>
        __forceinline__ __device__ void CompressBlock(TLoader&& loader,
                                                      int srcSize,
                                                      TStorageType* dst) {

            const int N = 4;
            const int tid = threadIdx.x;

            TStorageType compressedEntries[N];
#pragma unroll 4
            for (int i = 0; i < N; ++i) {
                compressedEntries[i] = 0;
            }


            for (int i = 0; i < KeysPerStorageType; i += N) {
#pragma unroll 4
                for (int j = 0; j < N; ++j) {
                    int id = i + j;
                    if (id < KeysPerStorageType) {
                        const ui32 offset = BLOCK_SIZE * id + tid;
                        const TStorageType key = offset < srcSize ? loader(offset) : 0;
                        compressedEntries[j] |= (key << ((KeysPerStorageType - id - 1) * BitsPerKey));
                    }
                }
            }

#pragma unroll 4
            for (int j = 1; j < N; ++j) {
                compressedEntries[0] |= compressedEntries[j];
            }

            if (tid < srcSize) {
                dst[tid] = compressedEntries[0];
            }
        }


        template <class TWriter>
        __forceinline__ __device__ void DecompressBlock(TWriter&& writer,
                                                        const TStorageType* src,
                                                        int dstSize) {
            const int N = 8;
            const int tid = threadIdx.x;
            const ui32 mask = Mask();
            const TStorageType compressedKeys = tid < dstSize ? src[tid] : 0;

            for (int i = 0; i < KeysPerStorageType; i += N) {
#pragma unroll 8
                for (int j = 0; j < N; ++j) {
                    int id = i + j;
                    if (id < KeysPerStorageType) {
                        int dstOffset = BLOCK_SIZE * id + tid;
                        if (dstOffset < dstSize) {
                            ui32 entry = (compressedKeys >> ((KeysPerStorageType - id - 1) * BitsPerKey)) & mask;
                            writer(dstOffset, entry);
                        }
                    }
                }
            }
        }
    };
}
