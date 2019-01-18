#pragma once

#include <math_constants.h>


namespace NKernel {
    __forceinline__ __host__ __device__ ui64 AdvanceSeed(ui64* seed) {
        ui32 v = *seed >> 32;
        ui32 u = *seed & 0xFFFFFFFF;
        v = 36969 * (v & 0xFFFF) + (v >> 16);
        u = 18000 * (u & 0xFFFF) + (u >> 16);
        *seed = ((ui64) v << 32) | (ui64) u;
        return *seed;
    }

    __forceinline__ __host__ __device__ ui64 AdvanceSeed(ui64* seed, int k) {
        for (int i = 0; i < k; ++i) {
            AdvanceSeed(seed);
        }
        return *seed;
    }


    __forceinline__ __host__ __device__ double NextUniform(ui64* seed) {
        ui64 x = AdvanceSeed(seed);
        ui32 v = x >> 32;
        ui32 u = x & 0xFFFFFFFF;

        return ((v << 16) + u) * 2.328306435996595e-10;
    }

    __forceinline__ __device__ float NextUniformF(ui64* seed) {
        ui64 x = AdvanceSeed(seed);
        ui32 v = x >> 32;
        ui32 u = x & 0xFFFFFFFF;

        return ((v << 16) + u) * 2.328306435996595e-10f;
    }

    __forceinline__ __device__ ui32 AdvanceSeed32(ui32* seed) {
        (*seed) = 1664525 * (*seed) + 1013904223;
        return *seed;
    }

    __forceinline__ __device__ float NextUniformFloat32(ui32* seed) {
        ui32 v = AdvanceSeed32(seed);
        return v * 2.328306435996595e-10f;
    }

    __forceinline__ __host__ __device__ float NextNormal(ui64* seed) {
        float a = NextUniform(seed);
        float b = NextUniform(seed);
        return sqrtf(-2.0f * logf(a)) * cosf(2.0f * CUDART_PI_F * b);
    }


    __forceinline__ __device__ float NextPoisson(ui64* seed, float alpha) {
        if (alpha > 20) {
            float a = sqrtf(alpha) * NextNormal(seed) + alpha;
            while (a < 0) {
                a = sqrtf(alpha) * NextNormal(seed) + alpha;
            }
            return a;
        }
        float logp = 0.0f, L = -alpha;
        int k = 0;
        do {
            k++;
            logp += log(NextUniform(seed));
        } while (logp > L);
        return k - 1;
    }

    __forceinline__ __device__

    float NextGamma(ui64* seed, float shape, float scale) {
        float d, c, x, xsquared, v, u;
        float w = 0;
        float result = 0;
        float origScale = 0;
        if (shape < 1.0f) {
            w = NextUniform(seed);
            shape += 1.0f;
            origScale = scale;
            scale = 1.0f;
        }
        d = shape - 1.0f / 3.0f;
        c = 1.0f / sqrtf(9.0f * d);
        for (;;) {
            do {
                x = NextNormal(seed);
                v = 1.0f + c * x;
            } while (v <= 0.0f);
            v = v * v * v;
            u = NextUniform(seed);
            xsquared = x * x;
            if (u < 1.0f - 0.0331f * xsquared * xsquared || logf(u) < 0.5f * xsquared + d * (1.0f - v + logf(v))) {
                result = scale * d * v;
                break;
            }
        }
        return w > 0 ? (origScale * result * pow(w, 1.0f / (shape - 1.0f))) : result;
    }

    __forceinline__ __device__

    float NextBeta(ui64* seed, float alpha, float beta) {
        float u = NextGamma(seed, alpha, 1.0f);
        float v = NextGamma(seed, beta, 1.0f);
        return u / (u + v);
    }

}
