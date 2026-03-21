// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MULTIMEM_LD_REDUCE_H_
#define _CUDA_PTX_GENERATED_MULTIMEM_LD_REDUCE_H_

/*
// multimem.ld_reduce.sem.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .min }
template <typename = void>
__device__ static inline uint32_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_min_t,
  const uint32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_min_t, const ::cuda::std::uint32_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("multimem.ld_reduce.weak.global.min.u32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline uint32_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  const uint32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint32_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  const ::cuda::std::uint32_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.min.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .min }
template <typename = void>
__device__ static inline uint64_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_min_t,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_min_t, const ::cuda::std::uint64_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  asm("multimem.ld_reduce.weak.global.min.u64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  const ::cuda::std::uint64_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .min }
template <typename = void>
__device__ static inline int32_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_min_t,
  const int32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int32_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_min_t, const ::cuda::std::int32_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int32_t __dest;
  asm("multimem.ld_reduce.weak.global.min.s32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline int32_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  const int32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::int32_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  const ::cuda::std::int32_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.min.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .min }
template <typename = void>
__device__ static inline int64_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_min_t,
  const int64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int64_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_min_t, const ::cuda::std::int64_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int64_t __dest;
  asm("multimem.ld_reduce.weak.global.min.s64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .min }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline int64_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_min_t,
  const int64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::int64_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_min_t,
  const ::cuda::std::int64_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.min.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .max }
template <typename = void>
__device__ static inline uint32_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_max_t,
  const uint32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_max_t, const ::cuda::std::uint32_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("multimem.ld_reduce.weak.global.max.u32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline uint32_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  const uint32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint32_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  const ::cuda::std::uint32_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.max.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .max }
template <typename = void>
__device__ static inline uint64_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_max_t,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_max_t, const ::cuda::std::uint64_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  asm("multimem.ld_reduce.weak.global.max.u64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  const ::cuda::std::uint64_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.max.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .max }
template <typename = void>
__device__ static inline int32_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_max_t,
  const int32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int32_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_max_t, const ::cuda::std::int32_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int32_t __dest;
  asm("multimem.ld_reduce.weak.global.max.s32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline int32_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  const int32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::int32_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  const ::cuda::std::int32_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.max.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .max }
template <typename = void>
__device__ static inline int64_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_max_t,
  const int64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int64_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_max_t, const ::cuda::std::int64_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int64_t __dest;
  asm("multimem.ld_reduce.weak.global.max.s64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .max }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline int64_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_max_t,
  const int64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::int64_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_max_t,
  const ::cuda::std::int64_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.max.s64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .add }
template <typename = void>
__device__ static inline uint32_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_add_t,
  const uint32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_add_t, const ::cuda::std::uint32_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("multimem.ld_reduce.weak.global.add.u32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline uint32_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  const uint32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint32_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  const ::cuda::std::uint32_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.add.u32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .add }
template <typename = void>
__device__ static inline uint64_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_add_t,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_add_t, const ::cuda::std::uint64_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  asm("multimem.ld_reduce.weak.global.add.u64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::uint64_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  const ::cuda::std::uint64_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .add }
template <typename = void>
__device__ static inline int32_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_add_t,
  const int32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int32_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_add_t, const ::cuda::std::int32_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int32_t __dest;
  asm("multimem.ld_reduce.weak.global.add.s32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline int32_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  const int32_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::int32_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  const ::cuda::std::int32_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.add.s32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .add }
template <typename = void>
__device__ static inline int64_t multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_add_t,
  const int64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::int64_t
multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_add_t, const ::cuda::std::int64_t* __addr)
{
// __sem == sem_weak (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int64_t __dest;
  asm("multimem.ld_reduce.weak.global.add.u64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .add }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline int64_t multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_add_t,
  const int64_t* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_sem _Sem, ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline ::cuda::std::int64_t multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem,
  ::cuda::ptx::scope_t<_Scope> __scope,
  ::cuda::ptx::op_add_t,
  const ::cuda::std::int64_t* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::int64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.add.u64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_and_op_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_and_op_t, const _B32* __addr)
{
  // __sem == sem_weak (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("multimem.ld_reduce.weak.global.and.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline B32 multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_and_op_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline _B32 multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, ::cuda::ptx::op_and_op_t, const _B32* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.and.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_or_op_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_or_op_t, const _B32* __addr)
{
  // __sem == sem_weak (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("multimem.ld_reduce.weak.global.or.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline B32 multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_or_op_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline _B32 multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, ::cuda::ptx::op_or_op_t, const _B32* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.or.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_xor_op_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_xor_op_t, const _B32* __addr)
{
  // __sem == sem_weak (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  asm("multimem.ld_reduce.weak.global.xor.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline B32 multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_xor_op_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline _B32 multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, ::cuda::ptx::op_xor_op_t, const _B32* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint32_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.xor.b32 %0, [%1];"
        : "=r"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .and }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_and_op_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_and_op_t, const _B64* __addr)
{
  // __sem == sem_weak (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  asm("multimem.ld_reduce.weak.global.and.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .and }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline B64 multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_and_op_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline _B64 multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, ::cuda::ptx::op_and_op_t, const _B64* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.and.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .or }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_or_op_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_or_op_t, const _B64* __addr)
{
  // __sem == sem_weak (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  asm("multimem.ld_reduce.weak.global.or.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .or }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline B64 multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_or_op_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline _B64 multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, ::cuda::ptx::op_or_op_t, const _B64* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.or.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .weak }
// .op        = { .xor }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 multimem_ld_reduce(
  cuda::ptx::sem_weak_t,
  cuda::ptx::op_xor_op_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 multimem_ld_reduce(::cuda::ptx::sem_weak_t, ::cuda::ptx::op_xor_op_t, const _B64* __addr)
{
  // __sem == sem_weak (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  asm("multimem.ld_reduce.weak.global.xor.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
// .op        = { .xor }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline B64 multimem_ld_reduce(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::op_xor_op_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline _B64 multimem_ld_reduce(
  ::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, ::cuda::ptx::op_xor_op_t, const _B64* __addr)
{
  static_assert(__sem == sem_relaxed || __sem == sem_acquire, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  ::cuda::std::uint64_t __dest;
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.relaxed.cta.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.relaxed.cluster.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.relaxed.gpu.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.relaxed.sys.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cta)
  {
    asm("multimem.ld_reduce.acquire.cta.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_cluster)
  {
    asm("multimem.ld_reduce.acquire.cluster.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_gpu)
  {
    asm("multimem.ld_reduce.acquire.gpu.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  else if constexpr (__sem == sem_acquire && __scope == scope_sys)
  {
    asm("multimem.ld_reduce.acquire.sys.global.xor.b64 %0, [%1];"
        : "=l"(__dest)
        : "l"(__as_ptr_gmem(__addr))
        : "memory");
  }
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_ld_reduce_is_not_supported_before_SM_90__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 810

#endif // _CUDA_PTX_GENERATED_MULTIMEM_LD_REDUCE_H_
