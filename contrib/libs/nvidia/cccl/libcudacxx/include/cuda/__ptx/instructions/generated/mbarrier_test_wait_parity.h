// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_TEST_WAIT_PARITY_H_
#define _CUDA_PTX_GENERATED_MBARRIER_TEST_WAIT_PARITY_H_

/*
// mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3. PTX
ISA 71, SM_80 template <typename = void>
__device__ static inline bool mbarrier_test_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 710
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline bool
mbarrier_test_wait_parity(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __phaseParity)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __waitComplete;
  asm("{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.test_wait.parity.shared.b64 P_OUT, [%1], %2;                                     // 3. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
      : "memory");
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_80__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 710

/*
// mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX
ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_test_wait_parity(
  ::cuda::ptx::sem_acquire_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P_OUT, [%1], %2;                  // 4. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P_OUT, [%1], %2;                  // 4. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.test_wait.parity.sem.scope.shared::cta.b64 waitComplete, [addr], phaseParity; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait_parity(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_test_wait_parity(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __phaseParity)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.parity.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.parity.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "r"(__phaseParity)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_MBARRIER_TEST_WAIT_PARITY_H_
