#pragma once

// Wave-size aware full lane mask for the *_sync warp shuffle intrinsics.
//
// HIP's __shfl_*_sync static_assert that sizeof(mask) == 8, so a 32-bit
// 0xffffffff literal fails to compile; a 64-bit all-ones mask is correct for
// both wave32 and wave64 (every lane that can participate is selected). On CUDA
// the value is the usual 32-bit all-ones mask, so the NVIDIA build is unchanged.
//
// This lives in its own header (rather than in cuda_to_hip.h, which is
// force-included on HIP translation units only) so the macro is reachable on
// both the CUDA and HIP compile paths: the kernel helpers that use it include
// this header directly, and cuda_to_hip.h includes it too for the HIP runtime
// surface.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#define CB_FULL_WARP_MASK 0xffffffffffffffffULL
#else
#define CB_FULL_WARP_MASK 0xffffffffu
#endif
