// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.b64 [smem_bar]; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_commit_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void
tcgen05_commit(::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar))
                 : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar))
                 : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_commit_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem_bar], ctaMask; // PTX ISA
86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_commit_multicast(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint64_t* smem_bar,
  uint16_t ctaMask);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_commit_multicast_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_commit_multicast(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint64_t* __smem_bar, ::cuda::std::uint16_t __ctaMask)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "h"(__ctaMask)
                 : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :
                 : "r"(__as_ptr_dsmem(__smem_bar)), "h"(__ctaMask)
                 : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_commit_multicast_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_COMMIT_H_
