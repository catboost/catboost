#pragma once

// CatBoost CUDA-to-HIP compatibility shim.
//
// This is the single file that knows about HIP. On AMD it includes the HIP
// runtime and aliases the cuda* runtime surface CatBoost uses to the matching
// hip* spelling, so every other .cu/.cuh keeps its CUDA spelling and the
// NVIDIA build stays byte-for-byte unchanged. On NVIDIA it is a plain include
// of the CUDA runtime.
//
// It is force-included on every HIP translation unit by the cuda.cmake HIP
// branch (-include this header), so its defines precede any use regardless of
// per-file include order.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// Host C++ TUs in the GPU runtime layer (cuda_lib wrappers) are compiled by the
// plain host compiler, not hipcc, yet include <cuda_runtime.h> (shimmed here).
// Select HIP's AMD host path so <hip/hip_runtime.h> provides the host runtime
// declarations under a non-hipcc compiler too.
#if !defined(__HIP_PLATFORM_AMD__)
#define __HIP_PLATFORM_AMD__ 1
#endif

// libc host decls must win over HIP's __device__ memcpy/memset overloads in
// host code compiled as HIP: include them before the runtime.
// <cfloat> too: several kernels use FLT_MAX which CUDA pulls in transitively
// but hipcc does not.
#include <cstring>
#include <cstdlib>
#include <cfloat>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// HIP's device_library_decls.h does `#define __local __attribute__((address_space(3)))`,
// which collides with CatBoost's bundled libc++ that uses `__local` as an ordinary
// identifier (e.g. <deque>'s __local() helper) -> "expected ')'" parse errors in host
// TUs. The macro only backs HIP's internal __to_local intrinsic (already parsed by
// now), so undefining it is safe and unblocks every STL header.
#ifdef __local
#undef __local
#endif

// CatBoost's bundled libc++ (contrib/libs/cxxsupp/libcxxcuda11) declares
// placement operator new/delete as host-only inline functions; nvcc supplies an
// implicit __device__ placement new, but clang/hipcc does not, so rocPRIM/hipCUB
// device code ("reference to __host__ function 'operator new' in __device__
// function") fails to compile. Provide __device__ placement new/delete; clang's
// HIP target overloading lets these coexist with the host-only ones and resolves
// device-side calls here. Guarded so they only exist in the device pass.
#if defined(__HIP_DEVICE_COMPILE__)
#include <cstddef>
__device__ inline void* operator new(size_t, void* p) noexcept { return p; }
__device__ inline void* operator new[](size_t, void* p) noexcept { return p; }
__device__ inline void operator delete(void*, void*) noexcept {}
__device__ inline void operator delete[](void*, void*) noexcept {}
#endif

// CatBoost gates the cudaPointerAttributes.type field access on
// CUDART_VERSION >= 10000. hipPointerAttribute_t exposes .type, so present a
// recent CUDART_VERSION to select that path. (Defined only if HIP did not.)
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

// ---- runtime types -------------------------------------------------------
#define cudaError_t                       hipError_t
#define cudaError                         hipError_t
#define cudaStream_t                      hipStream_t
#define cudaEvent_t                       hipEvent_t
#define cudaGraph_t                       hipGraph_t
#define cudaGraphExec_t                   hipGraphExec_t
#define cudaDeviceProp                    hipDeviceProp_t
#define cudaPointerAttributes             hipPointerAttribute_t
#define cudaMemcpyKind                    hipMemcpyKind
#define cudaMemoryType                    hipMemoryType

// ---- status codes ---------------------------------------------------------
#define cudaSuccess                       hipSuccess
#define cudaErrorNotReady                 hipErrorNotReady
#define cudaErrorCudartUnloading          hipErrorDeinitialized
#define cudaErrorUnknown                  hipErrorUnknown
#define cudaErrorNotYetImplemented        hipErrorNotSupported

// ---- memcpy / memory-type enums ------------------------------------------
#define cudaMemcpyHostToHost              hipMemcpyHostToHost
#define cudaMemcpyHostToDevice            hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost            hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice          hipMemcpyDeviceToDevice
#define cudaMemcpyDefault                 hipMemcpyDefault
#define cudaMemoryTypeHost                hipMemoryTypeHost
#define cudaMemoryTypeDevice              hipMemoryTypeDevice
#define cudaMemoryTypeManaged             hipMemoryTypeManaged

// ---- stream / event flags -------------------------------------------------
#define cudaStreamNonBlocking             hipStreamNonBlocking
#define cudaStreamPerThread               hipStreamPerThread
#define cudaStreamCaptureModeThreadLocal  hipStreamCaptureModeThreadLocal
#define cudaEventBlockingSync             hipEventBlockingSync
#define cudaEventDisableTiming            hipEventDisableTiming
#define cudaHostAllocPortable             hipHostMallocPortable

// ---- memory ---------------------------------------------------------------
#define cudaMalloc                        hipMalloc
#define cudaMallocManaged                 hipMallocManaged
#define cudaFree                          hipFree
#define cudaFreeHost                      hipHostFree
#define cudaHostAlloc                     hipHostMalloc
#define cudaMemcpy                        hipMemcpy
#define cudaMemcpyAsync                   hipMemcpyAsync
#define cudaMemsetAsync                   hipMemsetAsync
#define cudaMemGetInfo                    hipMemGetInfo
#define cudaPointerGetAttributes          hipPointerGetAttributes

// ---- device ---------------------------------------------------------------
#define cudaGetDevice                     hipGetDevice
#define cudaGetDeviceCount                hipGetDeviceCount
#define cudaSetDevice                     hipSetDevice
#define cudaGetDeviceProperties           hipGetDeviceProperties
#define cudaDeviceSynchronize             hipDeviceSynchronize
#define cudaDeviceGetStreamPriorityRange  hipDeviceGetStreamPriorityRange
#define cudaDeviceCanAccessPeer           hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess        hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess       hipDeviceDisablePeerAccess

// ---- errors ---------------------------------------------------------------
#define cudaGetLastError                  hipGetLastError
#define cudaGetErrorString                hipGetErrorString

// ---- streams --------------------------------------------------------------
#define cudaStreamCreate                  hipStreamCreate
#define cudaStreamCreateWithFlags         hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority      hipStreamCreateWithPriority
#define cudaStreamDestroy                 hipStreamDestroy
#define cudaStreamSynchronize             hipStreamSynchronize
#define cudaStreamWaitEvent               hipStreamWaitEvent
#define cudaStreamBeginCapture            hipStreamBeginCapture
#define cudaStreamEndCapture              hipStreamEndCapture

// ---- events ---------------------------------------------------------------
#define cudaEventCreate                   hipEventCreate
#define cudaEventCreateWithFlags          hipEventCreateWithFlags
#define cudaEventRecord                   hipEventRecord
#define cudaEventQuery                    hipEventQuery
#define cudaEventSynchronize              hipEventSynchronize
#define cudaEventDestroy                  hipEventDestroy

// ---- graphs ---------------------------------------------------------------
#define cudaGraphInstantiate              hipGraphInstantiate
#define cudaGraphLaunch                   hipGraphLaunch
#define cudaGraphDestroy                  hipGraphDestroy
#define cudaGraphExecDestroy              hipGraphExecDestroy

// hipCUB / rocThrust: route cub:: to hipcub:: . The <cub/...> include paths are
// handled by forwarding shim headers on the HIP include path (see hip_compat/),
// which include the hipCUB umbrella; this define maps the namespace token.
#define cub hipcub

#else  // CUDA

#include <cuda_runtime.h>

#endif

// CB_FULL_WARP_MASK lives in its own header so it is reachable on both the CUDA
// and HIP compile paths (this shim is force-included on HIP TUs only).
#include "warp_mask.cuh"
