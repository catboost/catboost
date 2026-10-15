#pragma once



namespace NKernel {

// HIP's HIP_vector_type<unsigned,2> already provides these uint2 operators, so
// they would be ambiguous on HIP.
#if !defined(USE_HIP)
    __host__ __device__ __forceinline__ uint2 operator+(const uint2& left, const uint2& right) {
        uint2 res = left;
        res.x += right.x;
        res.y += right.y;
        return res;
    }

    __host__ __device__ __forceinline__ uint2 operator-(const uint2& left, const uint2& right) {
        uint2 res = left;
        res.x -= right.x;
        res.y -= right.y;
        return res;
    }

    __host__ __device__ __forceinline__ uint2 operator*(const uint2& left, const uint2& right) {
        uint2 res = left;
        res.x *= right.x;
        res.y *= right.y;
        return res;
    }
#endif


    template <class T>
    __host__ __device__ __forceinline__ T ZeroAwareDivide(const T& left, const T& right, bool skipZeroes) {
        return (skipZeroes && (left < 1e-15f && left > -1e-15f)) ? 0 :  left  / (right + ((T)1e-15f));
    }

    template <>
    __host__ __device__ __forceinline__ uint2 ZeroAwareDivide(const uint2& left, const uint2& right, bool skipZeroes) {
        uint2 res;
        res.x = ZeroAwareDivide(left.x, right.x, skipZeroes);
        res.y = ZeroAwareDivide(left.y, right.y, skipZeroes);
        return res;
    }





}
