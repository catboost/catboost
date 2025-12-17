#include "gpu_input_factorize.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

namespace NKernel {
    namespace {
        // Device-side CityHash64 (v1) implementation, compatible with util/digest/city.cpp for lengths <= 32.
        // Used to compute CalcCatFeatureHash(ToString(value)) for integer categorical values on GPU.
        __device__ __forceinline__ ui64 CityRotate(ui64 val, int shift) {
            return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
        }

        __device__ __forceinline__ ui64 CityRotateByAtLeast1(ui64 val, int shift) {
            return (val >> shift) | (val << (64 - shift));
        }

        __device__ __forceinline__ ui64 CityShiftMix(ui64 val) {
            return val ^ (val >> 47);
        }

        __device__ __forceinline__ ui64 CityHash128to64(ui64 low, ui64 high) {
            // Murmur-inspired hashing (matches Hash128to64 in util/digest/city.h).
            const ui64 kMul = 0x9ddfea08eb382d69ULL;
            ui64 a = (low ^ high) * kMul;
            a ^= (a >> 47);
            ui64 b = (high ^ a) * kMul;
            b ^= (b >> 47);
            b *= kMul;
            return b;
        }

        __device__ __forceinline__ ui64 CityHashLen16(ui64 u, ui64 v) {
            return CityHash128to64(u, v);
        }

        __device__ __forceinline__ ui64 CityFetch64(const char* p) {
            // Little-endian unaligned load.
            ui64 result = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                result |= (static_cast<ui64>(static_cast<unsigned char>(p[i])) << (8 * i));
            }
            return result;
        }

        __device__ __forceinline__ ui32 CityFetch32(const char* p) {
            ui32 result = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                result |= (static_cast<ui32>(static_cast<unsigned char>(p[i])) << (8 * i));
            }
            return result;
        }

        __device__ __forceinline__ ui64 CityHashLen0to16(const char* s, ui32 len) {
            constexpr ui64 k2 = 0x9ae16a3b2f90404fULL;
            constexpr ui64 k3 = 0xc949d7c7509e6557ULL;

            if (len > 8) {
                const ui64 a = CityFetch64(s);
                const ui64 b = CityFetch64(s + len - 8);
                return CityHashLen16(a, CityRotateByAtLeast1(b + len, static_cast<int>(len))) ^ b;
            }
            if (len >= 4) {
                const ui64 a = CityFetch32(s);
                return CityHashLen16(static_cast<ui64>(len) + (a << 3), CityFetch32(s + len - 4));
            }
            if (len > 0) {
                const ui8 a = static_cast<ui8>(s[0]);
                const ui8 b = static_cast<ui8>(s[len >> 1]);
                const ui8 c = static_cast<ui8>(s[len - 1]);
                const ui32 y = static_cast<ui32>(a) + (static_cast<ui32>(b) << 8);
                const ui32 z = static_cast<ui32>(len) + (static_cast<ui32>(c) << 2);
                return CityShiftMix(static_cast<ui64>(y) * k2 ^ static_cast<ui64>(z) * k3) * k2;
            }
            return k2;
        }

        __device__ __forceinline__ ui64 CityHashLen17to32(const char* s, ui32 len) {
            constexpr ui64 k0 = 0xc3a5c85c97cb3127ULL;
            constexpr ui64 k1 = 0xb492b66fbe98f273ULL;
            constexpr ui64 k2 = 0x9ae16a3b2f90404fULL;
            constexpr ui64 k3 = 0xc949d7c7509e6557ULL;

            const ui64 a = CityFetch64(s) * k1;
            const ui64 b = CityFetch64(s + 8);
            const ui64 c = CityFetch64(s + len - 8) * k2;
            const ui64 d = CityFetch64(s + len - 16) * k0;
            return CityHashLen16(
                CityRotate(a - b, 43) + CityRotate(c, 30) + d,
                a + CityRotate(b ^ k3, 20) - c + len
            );
        }

        __device__ __forceinline__ ui64 CityHash64Len0to32(const char* s, ui32 len) {
            if (len <= 16) {
                return CityHashLen0to16(s, len);
            }
            return CityHashLen17to32(s, len);
        }

        __device__ __forceinline__ int CityU64ToDecString(ui64 x, char* out) {
            char tmp[32];
            int len = 0;
            do {
                const ui64 q = x / 10;
                const ui32 digit = static_cast<ui32>(x - q * 10);
                tmp[len++] = static_cast<char>('0' + digit);
                x = q;
            } while (x != 0);

            for (int i = 0; i < len; ++i) {
                out[i] = tmp[len - 1 - i];
            }
            return len;
        }

        __device__ __forceinline__ int CityI64ToDecString(i64 v, char* out) {
            ui64 x = 0;
            int pos = 0;
            if (v < 0) {
                out[pos++] = '-';
                // Avoid overflow for INT64_MIN.
                x = static_cast<ui64>(-(v + 1)) + 1;
            } else {
                x = static_cast<ui64>(v);
            }
            pos += CityU64ToDecString(x, out + pos);
            return pos;
        }

        template <typename T>
        __global__ void HashUniqueSignedImpl(const T* __restrict uniqueValues, ui32 uniqueCount, ui32* __restrict hashesOut) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < uniqueCount) {
                char buf[32];
                const int len = CityI64ToDecString(static_cast<i64>(uniqueValues[i]), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                hashesOut[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void HashUniqueUnsignedImpl(const T* __restrict uniqueValues, ui32 uniqueCount, ui32* __restrict hashesOut) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < uniqueCount) {
                char buf[32];
                const int len = CityU64ToDecString(static_cast<ui64>(uniqueValues[i]), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                hashesOut[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void CopyStridedImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            T* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                dst[i] = *ptr;
            }
        }

        __global__ void FillSequenceImpl(ui32 size, ui32* __restrict dst) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                dst[i] = i;
            }
        }

        template <typename T>
        __global__ void ComputeChangeFlagsImpl(
            const T* __restrict sorted,
            ui32 size,
            ui32* __restrict flags
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                if (i == 0) {
                    flags[i] = 0;
                } else {
                    flags[i] = (sorted[i] != sorted[i - 1]) ? 1u : 0u;
                }
            }
        }

        __global__ void ScatterRanksImpl(
            const ui32* __restrict sortedIdx,
            const ui32* __restrict runIdsSorted,
            ui32 size,
            ui32* __restrict ranksOut
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                ranksOut[sortedIdx[i]] = runIdsSorted[i];
            }
        }

        template <typename T>
        inline void LaunchCopyStrided(
            const void* src,
            ui64 strideBytes,
            ui32 size,
            T* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            CopyStridedImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                reinterpret_cast<const char*>(src),
                strideBytes,
                size,
                dst
            );
        }

        inline void LaunchFillSequence(ui32 size, ui32* dst, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            FillSequenceImpl<<<numBlocks, blockSize, 0, stream>>>(size, dst);
        }

        template <typename T>
        inline void LaunchComputeChangeFlags(const T* sorted, ui32 size, ui32* flags, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            ComputeChangeFlagsImpl<T><<<numBlocks, blockSize, 0, stream>>>(sorted, size, flags);
        }

        inline void LaunchScatterRanks(const ui32* sortedIdx, const ui32* runIdsSorted, ui32 size, ui32* ranksOut, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            ScatterRanksImpl<<<numBlocks, blockSize, 0, stream>>>(sortedIdx, runIdsSorted, size, ranksOut);
        }

        template <typename T>
        void FactorizeImpl(
            const void* src,
            ui64 strideBytes,
            ui32 size,
            ui32* ranksOut,
            T* uniqueValuesOut,
            ui32* countsOut,
            ui32* uniqueCountOut,
            TCudaStream stream
        ) {
            if (size == 0) {
                if (uniqueCountOut) {
                    CUDA_SAFE_CALL(cudaMemsetAsync(uniqueCountOut, 0, sizeof(ui32), stream));
                }
                return;
            }

            T* values = nullptr;
            ui32* indices = nullptr;
            T* valuesSorted = nullptr;
            ui32* indicesSorted = nullptr;
            ui32* flags = nullptr;
            ui32* runIdsSorted = nullptr;

            CUDA_SAFE_CALL(cudaMalloc(&values, static_cast<size_t>(size) * sizeof(T)));
            CUDA_SAFE_CALL(cudaMalloc(&indices, static_cast<size_t>(size) * sizeof(ui32)));
            CUDA_SAFE_CALL(cudaMalloc(&valuesSorted, static_cast<size_t>(size) * sizeof(T)));
            CUDA_SAFE_CALL(cudaMalloc(&indicesSorted, static_cast<size_t>(size) * sizeof(ui32)));
            CUDA_SAFE_CALL(cudaMalloc(&flags, static_cast<size_t>(size) * sizeof(ui32)));
            CUDA_SAFE_CALL(cudaMalloc(&runIdsSorted, static_cast<size_t>(size) * sizeof(ui32)));

            LaunchCopyStrided<T>(src, strideBytes, size, values, stream);
            LaunchFillSequence(size, indices, stream);

            size_t sortTmpBytes = 0;
            CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
                nullptr,
                sortTmpBytes,
                values,
                valuesSorted,
                indices,
                indicesSorted,
                static_cast<int>(size),
                /*begin_bit*/ 0,
                /*end_bit*/ static_cast<int>(sizeof(T) * 8),
                stream
            ));
            void* sortTmp = nullptr;
            CUDA_SAFE_CALL(cudaMalloc(&sortTmp, sortTmpBytes));
            CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
                sortTmp,
                sortTmpBytes,
                values,
                valuesSorted,
                indices,
                indicesSorted,
                static_cast<int>(size),
                /*begin_bit*/ 0,
                /*end_bit*/ static_cast<int>(sizeof(T) * 8),
                stream
            ));
            CUDA_SAFE_CALL(cudaFree(sortTmp));

            LaunchComputeChangeFlags<T>(valuesSorted, size, flags, stream);

            size_t scanTmpBytes = 0;
            CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
                nullptr,
                scanTmpBytes,
                flags,
                runIdsSorted,
                static_cast<int>(size),
                stream
            ));
            void* scanTmp = nullptr;
            CUDA_SAFE_CALL(cudaMalloc(&scanTmp, scanTmpBytes));
            CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
                scanTmp,
                scanTmpBytes,
                flags,
                runIdsSorted,
                static_cast<int>(size),
                stream
            ));
            CUDA_SAFE_CALL(cudaFree(scanTmp));

            LaunchScatterRanks(indicesSorted, runIdsSorted, size, ranksOut, stream);

            size_t rleTmpBytes = 0;
            CUDA_SAFE_CALL(cub::DeviceRunLengthEncode::Encode(
                nullptr,
                rleTmpBytes,
                valuesSorted,
                uniqueValuesOut,
                countsOut,
                uniqueCountOut,
                static_cast<int>(size),
                stream
            ));
            void* rleTmp = nullptr;
            CUDA_SAFE_CALL(cudaMalloc(&rleTmp, rleTmpBytes));
            CUDA_SAFE_CALL(cub::DeviceRunLengthEncode::Encode(
                rleTmp,
                rleTmpBytes,
                valuesSorted,
                uniqueValuesOut,
                countsOut,
                uniqueCountOut,
                static_cast<int>(size),
                stream
            ));
            CUDA_SAFE_CALL(cudaFree(rleTmp));

            CUDA_SAFE_CALL(cudaFree(values));
            CUDA_SAFE_CALL(cudaFree(indices));
            CUDA_SAFE_CALL(cudaFree(valuesSorted));
            CUDA_SAFE_CALL(cudaFree(indicesSorted));
            CUDA_SAFE_CALL(cudaFree(flags));
            CUDA_SAFE_CALL(cudaFree(runIdsSorted));
        }

        __global__ void MapRanksToBinsImpl(
            const ui32* __restrict ranks,
            ui32 size,
            const ui32* __restrict binsForRank,
            ui32* __restrict dstBins
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                dstBins[i] = binsForRank[ranks[i]];
            }
        }

        __global__ void GatherUi32BinsToUi8Impl(
            const ui32* __restrict srcBins,
            ui32 size,
            const ui32* __restrict gatherIndices,
            ui8* __restrict dstBins
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 srcIdx = gatherIndices ? gatherIndices[i] : i;
                dstBins[i] = static_cast<ui8>(srcBins[srcIdx]);
            }
        }
    }

    void FactorizeStridedGpuInputToUnique(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        ui32* ranksOut,
        void* uniqueValuesOut,
        ui32* countsOut,
        ui32* uniqueCountOut,
        TCudaStream stream
    ) {
        if (size == 0) {
            if (uniqueCountOut) {
                CUDA_SAFE_CALL(cudaMemsetAsync(uniqueCountOut, 0, sizeof(ui32), stream));
            }
            return;
        }

        switch (dtype) {
            case EGpuInputDType::Int8:
                FactorizeImpl<i8>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i8*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::Int16:
                FactorizeImpl<i16>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i16*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::Int32:
                FactorizeImpl<i32>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i32*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::Int64:
                FactorizeImpl<i64>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i64*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt8:
                FactorizeImpl<ui8>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui8*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt16:
                FactorizeImpl<ui16>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui16*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt32:
                FactorizeImpl<ui32>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui32*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt64:
                FactorizeImpl<ui64>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui64*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            default:
                // Not supported here; caller should validate dtype.
                CUDA_SAFE_CALL(cudaMemsetAsync(uniqueCountOut, 0, sizeof(ui32), stream));
                break;
        }
    }

    void MapRanksToBins(
        const ui32* ranks,
        ui32 size,
        const ui32* binsForRank,
        ui32* dstBins,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        MapRanksToBinsImpl<<<numBlocks, blockSize, 0, stream>>>(ranks, size, binsForRank, dstBins);
    }

    void GatherUi32BinsToUi8(
        const ui32* srcBins,
        ui32 size,
        const ui32* gatherIndices,
        ui8* dstBins,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        GatherUi32BinsToUi8Impl<<<numBlocks, blockSize, 0, stream>>>(srcBins, size, gatherIndices, dstBins);
    }

    void HashUniqueNumericToCatHash(
        const void* uniqueValues,
        ui32 uniqueCount,
        EGpuInputDType dtype,
        ui32* hashesOut,
        TCudaStream stream
    ) {
        if (uniqueCount == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (uniqueCount + blockSize - 1) / blockSize;

        switch (dtype) {
            case EGpuInputDType::Int8:
                HashUniqueSignedImpl<i8><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i8*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::Int16:
                HashUniqueSignedImpl<i16><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i16*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::Int32:
                HashUniqueSignedImpl<i32><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i32*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::Int64:
                HashUniqueSignedImpl<i64><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i64*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt8:
                HashUniqueUnsignedImpl<ui8><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui8*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt16:
                HashUniqueUnsignedImpl<ui16><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui16*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt32:
                HashUniqueUnsignedImpl<ui32><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui32*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt64:
                HashUniqueUnsignedImpl<ui64><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui64*>(uniqueValues), uniqueCount, hashesOut);
                break;
            default:
                // Not supported here; caller should validate dtype.
                break;
        }
    }

}
