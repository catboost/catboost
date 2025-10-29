// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_EXPECT_TX_H_
#define _CUDA_PTX_GENERATED_MBARRIER_EXPECT_TX_H_

/*
// mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // 1. PTX ISA 80, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_expect_tx(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  uint32_t txCount);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_expect_tx_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void mbarrier_expect_tx(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __txCount)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1; // 1."
        :
        : "r"(__as_ptr_smem(__addr)), "r"(__txCount)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.expect_tx.relaxed.cluster.shared::cta.b64 [%0], %1; // 1."
        :
        : "r"(__as_ptr_smem(__addr)), "r"(__txCount)
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_expect_tx_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.expect_tx.sem.scope.space.b64 [addr], txCount; // 2. PTX ISA 80, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_expect_tx(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  uint32_t txCount);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_expect_tx_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void mbarrier_expect_tx(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint32_t __txCount)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
// __space == space_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.expect_tx.relaxed.cta.shared::cluster.b64 [%0], %1; // 2."
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [%0], %1; // 2."
        :
        : "r"(__as_ptr_dsmem(__addr)), "r"(__txCount)
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_expect_tx_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_MBARRIER_EXPECT_TX_H_
