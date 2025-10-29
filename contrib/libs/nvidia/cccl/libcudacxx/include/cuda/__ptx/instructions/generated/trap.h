// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TRAP_H_
#define _CUDA_PTX_GENERATED_TRAP_H_

/*
// trap; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline void trap();
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_trap_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline void trap()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm volatile("trap;" : : :);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_trap_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 100

#endif // _CUDA_PTX_GENERATED_TRAP_H_
