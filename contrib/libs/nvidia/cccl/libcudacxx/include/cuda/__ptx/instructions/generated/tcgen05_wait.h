// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_

/*
// tcgen05.wait::ld.sync.aligned; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
template <typename = void>
__device__ static inline void tcgen05_wait_ld();
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_wait_ld_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename = void>
_CCCL_DEVICE static inline void tcgen05_wait_ld()
{
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_wait_ld_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.wait::st.sync.aligned; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
template <typename = void>
__device__ static inline void tcgen05_wait_st();
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_wait_st_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename = void>
_CCCL_DEVICE static inline void tcgen05_wait_st()
{
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_wait_st_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_
