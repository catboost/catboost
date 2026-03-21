// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MULTIMEM_RED_H_
#define _CUDA_PTX_GENERATED_MULTIMEM_RED_H_

/*
// multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  uint32_t* addr,
  uint32_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t* __addr,
  ::cuda::std::uint32_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.min.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  uint64_t* addr,
  uint64_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint64_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.min.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  int32_t* addr,
  int32_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  ::cuda::std::int32_t* __addr,
  ::cuda::std::int32_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.min.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  int64_t* addr,
  int64_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  ::cuda::std::int64_t* __addr,
  ::cuda::std::int64_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.min.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  uint32_t* addr,
  uint32_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t* __addr,
  ::cuda::std::uint32_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.max.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  uint64_t* addr,
  uint64_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint64_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.max.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  int32_t* addr,
  int32_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  ::cuda::std::int32_t* __addr,
  ::cuda::std::int32_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.max.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  int64_t* addr,
  int64_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  ::cuda::std::int64_t* __addr,
  ::cuda::std::int64_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.max.s64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  uint32_t* addr,
  uint32_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t* __addr,
  ::cuda::std::uint32_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  uint64_t* addr,
  uint64_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint64_t* __addr,
  ::cuda::std::uint64_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  int32_t* addr,
  int32_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  ::cuda::std::int32_t* __addr,
  ::cuda::std::int32_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.add.s32 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  int64_t* addr,
  int64_t val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  ::cuda::std::int64_t* __addr,
  ::cuda::std::int64_t __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.add.u64 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "l"(__val) : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_and_op_t,
  B32* addr,
  B32 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_and_op_t,
  _B32* __addr,
  _B32 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.and.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_or_op_t,
  B32* addr,
  B32 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_or_op_t,
  _B32* __addr,
  _B32 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.or.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_xor_op_t,
  B32* addr,
  B32 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_xor_op_t,
  _B32* __addr,
  _B32 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.xor.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .and }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_and_op_t,
  B64* addr,
  B64 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_and_op_t,
  _B64* __addr,
  _B64 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.and.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .or }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_or_op_t,
  B64* addr,
  B64 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_or_op_t,
  _B64* __addr,
  _B64 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.or.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .xor }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_red(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_xor_op_t,
  B64* addr,
  B64 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void multimem_red(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_xor_op_t,
  _B64* __addr,
  _B64 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.red.relaxed.cta.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.red.relaxed.cluster.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.red.relaxed.gpu.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.red.relaxed.sys.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.red.release.cta.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.red.release.cluster.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.red.release.gpu.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.red.release.sys.global.xor.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_red_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

#endif // _CUDA_PTX_GENERATED_MULTIMEM_RED_H_
