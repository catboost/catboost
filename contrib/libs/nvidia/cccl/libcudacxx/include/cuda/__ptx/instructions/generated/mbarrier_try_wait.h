// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_TRY_WAIT_H_
#define _CUDA_PTX_GENERATED_MBARRIER_TRY_WAIT_H_

/*
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.
PTX ISA 78, SM_90 template <typename = void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline bool mbarrier_try_wait(::cuda::std::uint64_t* __addr, const ::cuda::std::uint64_t& __state)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  asm("{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.shared::cta.b64         P_OUT, [%1], %2;                                      // 5a. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)), "l"(__state)
      : "memory");
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b. PTX
ISA 78, SM_90 template <typename = void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  ::cuda::std::uint64_t* __addr, const ::cuda::std::uint64_t& __state, const ::cuda::std::uint32_t& __suspendTimeHint)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  asm("{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.shared::cta.b64         P_OUT, [%1], %2, %3;                    // 5b. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)), "l"(__state), "r"(__suspendTimeHint)
      : "memory");
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.sem.scope.shared::cta.b64         waitComplete, [addr], state;                        // 6a.  PTX
ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  ::cuda::ptx::sem_acquire_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint64_t& __state)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cta.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.try_wait.sem.scope.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.  PTX
ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  ::cuda::ptx::sem_acquire_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint64_t& __state,
  const ::cuda::std::uint32_t& __suspendTimeHint)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cta.shared::cta.b64         P_OUT, [%1], %2 , %3;      // 6b. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], %2 , %3;      // 6b. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state), "r"(__suspendTimeHint)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.try_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state, suspendTimeHint; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint64_t& __state,
  const ::cuda::std::uint32_t& __suspendTimeHint)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2, %3;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state), "r"(__suspendTimeHint)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2, %3;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state), "r"(__suspendTimeHint)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.try_wait.sem.scope.shared::cta.b64 waitComplete, [addr], state; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint64_t& __state)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __waitComplete;
  if constexpr (__scope == scope_cta)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.relaxed.cta.shared::cta.b64 P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.relaxed.cluster.shared::cta.b64 P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)), "l"(__state)
        : "memory");
  }
  return static_cast<bool>(__waitComplete);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_MBARRIER_TRY_WAIT_H_
