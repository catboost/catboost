// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TENSORMAP_CP_FENCEPROXY_H_
#define _CUDA_PTX_GENERATED_TENSORMAP_CP_FENCEPROXY_H_

/*
// tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.sem.scope.sync.aligned  [dst], [src], size; // PTX ISA
83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void tensormap_cp_fenceproxy(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  void* dst,
  const void* src,
  cuda::ptx::n32_t<N32> size);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_cp_fenceproxy_is_not_supported_before_SM_90__();
template <int _N32, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void tensormap_cp_fenceproxy(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  void* __dst,
  const void* __src,
  ::cuda::ptx::n32_t<_N32> __size)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__scope == scope_cta)
  {
    asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned  [%0], [%1], %2;"
      :
      : "l"(__as_ptr_gmem(__dst)), "r"(__as_ptr_smem(__src)), "n"(__size.value)
      : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned  [%0], [%1], %2;"
      :
      : "l"(__as_ptr_gmem(__dst)), "r"(__as_ptr_smem(__src)), "n"(__size.value)
      : "memory");
  }
  else if constexpr (__scope == scope_gpu)
  {
    asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned  [%0], [%1], %2;"
      :
      : "l"(__as_ptr_gmem(__dst)), "r"(__as_ptr_smem(__src)), "n"(__size.value)
      : "memory");
  }
  else if constexpr (__scope == scope_sys)
  {
    asm volatile(
      "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned  [%0], [%1], %2;"
      :
      : "l"(__as_ptr_gmem(__dst)), "r"(__as_ptr_smem(__src)), "n"(__size.value)
      : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_cp_fenceproxy_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

#endif // _CUDA_PTX_GENERATED_TENSORMAP_CP_FENCEPROXY_H_
