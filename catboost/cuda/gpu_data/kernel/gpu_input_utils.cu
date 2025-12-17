#include "gpu_input_utils.cuh"

#include <catboost/cuda/cuda_lib/memcpy_tracker.h>

#include <cub/device/device_reduce.cuh>

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
        __global__ void CopyStridedCastToFloatImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            float* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                dst[i] = static_cast<float>(*ptr);
            }
        }

        template <typename T>
        __global__ void HashStridedSignedToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                char buf[32];
                const int len = CityI64ToDecString(static_cast<i64>(*ptr), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                dst[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void HashStridedUnsignedToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                char buf[32];
                const int len = CityU64ToDecString(static_cast<ui64>(*ptr), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                dst[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void MapStridedSignedCodesToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            const ui32* __restrict dict,
            ui32 dictSize,
            ui32 nullValue,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                const i64 code = static_cast<i64>(*ptr);
                if ((code < 0) || (static_cast<ui64>(code) >= static_cast<ui64>(dictSize))) {
                    dst[i] = nullValue;
                } else {
                    dst[i] = dict[static_cast<ui32>(code)];
                }
            }
        }

        template <typename T>
        __global__ void MapStridedUnsignedCodesToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            const ui32* __restrict dict,
            ui32 dictSize,
            ui32 nullValue,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                const ui64 code = static_cast<ui64>(*ptr);
                if (code >= static_cast<ui64>(dictSize)) {
                    dst[i] = nullValue;
                } else {
                    dst[i] = dict[static_cast<ui32>(code)];
                }
            }
        }

        template <typename T>
        inline void LaunchCopyStridedCastToFloat(
            const void* src,
            ui64 srcStrideBytes,
            ui32 size,
            float* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            CopyStridedCastToFloatImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                reinterpret_cast<const char*>(src),
                srcStrideBytes,
                size,
                dst
            );
        }

        template <typename T, bool IsSigned>
        inline void LaunchHashStridedToCatHash(
            const void* src,
            ui64 srcStrideBytes,
            ui32 size,
            ui32* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            if (IsSigned) {
                HashStridedSignedToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dst
                );
            } else {
                HashStridedUnsignedToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dst
                );
            }
        }

        template <typename T, bool IsSigned>
        inline void LaunchMapStridedCatCodesToCatHash(
            const void* src,
            ui64 srcStrideBytes,
            ui32 size,
            const ui32* dict,
            ui32 dictSize,
            ui32 nullValue,
            ui32* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            if (IsSigned) {
                MapStridedSignedCodesToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dict,
                    dictSize,
                    nullValue,
                    dst
                );
            } else {
                MapStridedUnsignedCodesToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dict,
                    dictSize,
                    nullValue,
                    dst
                );
            }
        }
    }

    void CopyStridedGpuInputToFloat(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        float* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        switch (dtype) {
            case EGpuInputDType::Float32: {
                CUDA_SAFE_CALL(cudaMemcpy2DAsync(
                    /*dst*/ dst,
                    /*dpitch*/ sizeof(float),
                    /*src*/ src,
                    /*spitch*/ srcStrideBytes,
                    /*width*/ sizeof(float),
                    /*height*/ size,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
                break;
            }
            case EGpuInputDType::Float64:
                LaunchCopyStridedCastToFloat<double>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int8:
                LaunchCopyStridedCastToFloat<i8>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int16:
                LaunchCopyStridedCastToFloat<i16>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int32:
                LaunchCopyStridedCastToFloat<i32>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int64:
                LaunchCopyStridedCastToFloat<i64>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt8:
                LaunchCopyStridedCastToFloat<ui8>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt16:
                LaunchCopyStridedCastToFloat<ui16>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt32:
                LaunchCopyStridedCastToFloat<ui32>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt64:
                LaunchCopyStridedCastToFloat<ui64>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Bool:
                LaunchCopyStridedCastToFloat<ui8>(src, srcStrideBytes, size, dst, stream);
                break;
        }
    }

    void HashStridedGpuInputToCatHash(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        ui32* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        switch (dtype) {
            case EGpuInputDType::Int8:
                LaunchHashStridedToCatHash<i8, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int16:
                LaunchHashStridedToCatHash<i16, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int32:
                LaunchHashStridedToCatHash<i32, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int64:
                LaunchHashStridedToCatHash<i64, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt8:
                LaunchHashStridedToCatHash<ui8, false>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt16:
                LaunchHashStridedToCatHash<ui16, false>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt32:
                LaunchHashStridedToCatHash<ui32, false>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt64:
                LaunchHashStridedToCatHash<ui64, false>(src, srcStrideBytes, size, dst, stream);
                break;
            default:
                // Caller is expected to validate categorical input dtype.
                break;
        }
    }

    void MapStridedCatCodesToCatHash(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        const ui32* dict,
        ui32 dictSize,
        ui32 nullValue,
        ui32* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        switch (dtype) {
            case EGpuInputDType::Int8:
                LaunchMapStridedCatCodesToCatHash<i8, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::Int16:
                LaunchMapStridedCatCodesToCatHash<i16, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::Int32:
                LaunchMapStridedCatCodesToCatHash<i32, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::Int64:
                LaunchMapStridedCatCodesToCatHash<i64, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt8:
                LaunchMapStridedCatCodesToCatHash<ui8, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt16:
                LaunchMapStridedCatCodesToCatHash<ui16, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt32:
                LaunchMapStridedCatCodesToCatHash<ui32, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt64:
                LaunchMapStridedCatCodesToCatHash<ui64, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            default:
                // Caller is expected to validate categorical input dtype.
                break;
        }
    }

    void ComputeMinMaxToHost(
        const float* values,
        ui32 size,
        float* minValue,
        float* maxValue,
        TCudaStream stream
    ) {
        if (size == 0) {
            if (minValue) {
                *minValue = 0.0f;
            }
            if (maxValue) {
                *maxValue = 0.0f;
            }
            return;
        }

        float* dMin = nullptr;
        float* dMax = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dMin, sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dMax, sizeof(float)));

        size_t tmpBytesMin = 0;
        size_t tmpBytesMax = 0;
        CUDA_SAFE_CALL(cub::DeviceReduce::Min(nullptr, tmpBytesMin, values, dMin, static_cast<int>(size), stream));
        CUDA_SAFE_CALL(cub::DeviceReduce::Max(nullptr, tmpBytesMax, values, dMax, static_cast<int>(size), stream));

        size_t tmpBytes = (tmpBytesMin > tmpBytesMax) ? tmpBytesMin : tmpBytesMax;
        void* tmp = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&tmp, tmpBytes));

        CUDA_SAFE_CALL(cub::DeviceReduce::Min(tmp, tmpBytes, values, dMin, static_cast<int>(size), stream));
        CUDA_SAFE_CALL(cub::DeviceReduce::Max(tmp, tmpBytes, values, dMax, static_cast<int>(size), stream));

        if (minValue) {
            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(minValue, dMin, sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_SAFE_CALL(cudaMemcpyAsync(minValue, dMin, sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (maxValue) {
            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(maxValue, dMax, sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_SAFE_CALL(cudaMemcpyAsync(maxValue, dMax, sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        cudaFree(tmp);
        cudaFree(dMin);
        cudaFree(dMax);
    }

}
