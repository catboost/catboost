// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_
#define _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_

/*
// fence.proxy.alias; // 4. PTX ISA 75, SM_70
template <typename = void>
__device__ static inline void fence_proxy_alias();
*/
#if __cccl_ptx_isa >= 750
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_alias_is_not_supported_before_SM_70__();
template <typename = void>
_CCCL_DEVICE static inline void fence_proxy_alias()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm volatile("fence.proxy.alias; // 4." : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_fence_proxy_alias_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 750

#endif // _CUDA_PTX_GENERATED_FENCE_PROXY_ALIAS_H_
