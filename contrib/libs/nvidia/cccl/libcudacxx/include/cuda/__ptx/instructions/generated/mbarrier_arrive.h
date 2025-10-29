// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_H_
#define _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_H_

/*
// mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.  PTX ISA 70, SM_80
template <typename = void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(::cuda::std::uint64_t* __addr)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __state;
  asm("mbarrier.arrive.shared.b64                                  %0,  [%1];           // 1. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr))
      : "memory");
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.  PTX ISA 78, SM_90
template <typename = void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t
mbarrier_arrive(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __count)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __state;
  asm("mbarrier.arrive.shared::cta.b64                             %0,  [%1], %2;    // 2. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.arrive.sem.scope.space.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.release.cta.shared::cta.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.release.cta.shared::cta.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64                   _, [addr];                // 4a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_release_t, ::cuda::ptx::scope_cluster_t, ::cuda::ptx::space_cluster_t, ::cuda::std::uint64_t* __addr)
{
// __sem == sem_release (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
// __space == space_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [%0];                // 4a. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64                   _, [addr], count;         // 4b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_release_t,
  ::cuda::ptx::scope_cluster_t,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
// __sem == sem_release (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
// __space == space_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [%0], %1;         // 4b. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)), "r"(__count)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.sem.scope.space.b64 state, [addr], count; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.relaxed.cluster.shared::cta.b64 %0, [%1], %2;"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)), "r"(__count)
        : "memory");
  }
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.b64 state, [addr]; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::space_shared_t,
  ::cuda::std::uint64_t* __addr)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __state;
  if constexpr (__scope == scope_cta)
  {
    asm("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  else if constexpr (__scope == scope_cluster)
  {
    asm("mbarrier.arrive.relaxed.cluster.shared::cta.b64 %0, [%1];"
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.b64 _, [addr], count; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t,
  ::cuda::ptx::scope_cluster_t,
  ::cuda::ptx::space_cluster_t,
  ::cuda::std::uint64_t* __addr,
  const ::cuda::std::uint32_t& __count)
{
// __sem == sem_relaxed (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
// __space == space_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// mbarrier.arrive.sem.scope.space.b64 _, [addr]; // PTX ISA 86, SM_90
// .sem       = { .relaxed }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename = void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  ::cuda::ptx::sem_relaxed_t, ::cuda::ptx::scope_cluster_t, ::cuda::ptx::space_cluster_t, ::cuda::std::uint64_t* __addr)
{
// __sem == sem_relaxed (due to parameter type constraint)
// __scope == scope_cluster (due to parameter type constraint)
// __space == space_cluster (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" : : "r"(__as_ptr_smem(__addr)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_H_
