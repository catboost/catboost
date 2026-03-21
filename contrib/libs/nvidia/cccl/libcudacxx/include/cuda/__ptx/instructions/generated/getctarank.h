// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_GETCTARANK_H_
#define _CUDA_PTX_GENERATED_GETCTARANK_H_

/*
// getctarank.space.u32 dest, addr; // PTX ISA 78, SM_90
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline uint32_t getctarank(
  cuda::ptx::space_cluster_t,
  const void* addr);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_getctarank_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t getctarank(::cuda::ptx::space_cluster_t, const void* __addr)
{
// __space == space_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("getctarank.shared::cluster.u32 %0, %1;" : "=r"(__dest) : "r"(__as_ptr_smem(__addr)) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_getctarank_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

#endif // _CUDA_PTX_GENERATED_GETCTARANK_H_
