// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_H_
#define _CUDA_PTX_GENERATED_FENCE_H_

/*
// fence.sem.scope; // 1. PTX ISA 60, SM_70
// .sem       = { .sc }
// .scope     = { .cta, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_sc_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 600
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_70__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void fence(::cuda::ptx::sem_sc_t, ::cuda::ptx::scope_t<_Scope> __scope)
{
  // __sem == sem_sc (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_gpu || __scope == scope_sys, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.sc.cta; // 1." : : : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile("fence.sc.gpu; // 1." : : : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile("fence.sc.sys; // 1." : : : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 600

/*
// fence.sem.scope; // 2. PTX ISA 78, SM_90
// .sem       = { .sc }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence(
  cuda::ptx::sem_sc_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence(::cuda::ptx::sem_sc_t, ::cuda::ptx::scope_cluster_t)
{
// __sem == sem_sc (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("fence.sc.cluster; // 2." : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// fence.sem.scope; // 1. PTX ISA 60, SM_70
// .sem       = { .acq_rel }
// .scope     = { .cta, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_acq_rel_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 600
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_70__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void fence(::cuda::ptx::sem_acq_rel_t, ::cuda::ptx::scope_t<_Scope> __scope)
{
  // __sem == sem_acq_rel (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_gpu || __scope == scope_sys, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.acq_rel.cta; // 1." : : : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile("fence.acq_rel.gpu; // 1." : : : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile("fence.acq_rel.sys; // 1." : : : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 600

/*
// fence.sem.scope; // 2. PTX ISA 78, SM_90
// .sem       = { .acq_rel }
// .scope     = { .cluster }
template <typename = void>
__device__ static inline void fence(
  cuda::ptx::sem_acq_rel_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void fence(::cuda::ptx::sem_acq_rel_t, ::cuda::ptx::scope_cluster_t)
{
// __sem == sem_acq_rel (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("fence.acq_rel.cluster; // 2." : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// fence.sem.scope; // PTX ISA 86, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void fence(::cuda::ptx::sem_acquire_t, ::cuda::ptx::scope_t<_Scope> __scope)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.acquire.cta;" : : : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm volatile("fence.acquire.cluster;" : : : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile("fence.acquire.gpu;" : : : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile("fence.acquire.sys;" : : : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// fence.sem.scope; // PTX ISA 86, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void fence(::cuda::ptx::sem_release_t, ::cuda::ptx::scope_t<_Scope> __scope)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__scope == scope_cta)
  {
    asm volatile("fence.release.cta;" : : : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm volatile("fence.release.cluster;" : : : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile("fence.release.gpu;" : : : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile("fence.release.sys;" : : : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_FENCE_H_
