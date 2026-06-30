// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_

/*
// tcgen05.shift.cta_group.down [taddr]; // PTX ISA 86, SM_100a, SM_103a, SM_110a
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_shift_down(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_shift_down_is_only_supported_on_SM_100a_103a_110a_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void
tcgen05_shift_down(::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100)))
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" : : "r"(__taddr) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.shift.cta_group::2.down [%0];" : : "r"(__taddr) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_shift_down_is_only_supported_on_SM_100a_103a_110a_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_SHIFT_H_
