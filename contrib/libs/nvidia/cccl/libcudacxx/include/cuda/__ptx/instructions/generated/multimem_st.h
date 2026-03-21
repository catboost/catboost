// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MULTIMEM_ST_H_
#define _CUDA_PTX_GENERATED_MULTIMEM_ST_H_

/*
// multimem.st.sem.global.b32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .weak }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void multimem_st(
  cuda::ptx::sem_weak_t,
  B32* addr,
  B32 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void multimem_st(::cuda::ptx::sem_weak_t, _B32* __addr, _B32 __val)
{
  // __sem == sem_weak (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("multimem.st.weak.global.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_st(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  B32* addr,
  B32 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void
multimem_st(::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, _B32* __addr, _B32 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.st.relaxed.cta.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.st.relaxed.cluster.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.st.relaxed.gpu.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.st.relaxed.sys.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.st.release.cta.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.st.release.cluster.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.st.release.gpu.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.st.release.sys.global.b32 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.st.sem.global.b64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .weak }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void multimem_st(
  cuda::ptx::sem_weak_t,
  B64* addr,
  B64 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void multimem_st(::cuda::ptx::sem_weak_t, _B64* __addr, _B64 __val)
{
  // __sem == sem_weak (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("multimem.st.weak.global.b64 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
// .sem       = { .relaxed, .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void multimem_st(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope,
  B64* addr,
  B64 val);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
template <typename _B64,
          ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true,
          ::cuda::ptx::dot_sem _Sem,
          ::cuda::ptx::dot_scope _Scope>
_CCCL_DEVICE static inline void
multimem_st(::cuda::ptx::sem_t<_Sem> __sem, ::cuda::ptx::scope_t<_Scope> __scope, _B64* __addr, _B64 __val)
{
  static_assert(__sem == sem_relaxed || __sem == sem_release, "");
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__sem == sem_relaxed && __scope == scope_cta)
  {
    asm("multimem.st.relaxed.cta.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_cluster)
  {
    asm("multimem.st.relaxed.cluster.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_gpu)
  {
    asm("multimem.st.relaxed.gpu.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_relaxed && __scope == scope_sys)
  {
    asm("multimem.st.relaxed.sys.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cta)
  {
    asm("multimem.st.release.cta.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_cluster)
  {
    asm("multimem.st.release.cluster.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_gpu)
  {
    asm("multimem.st.release.gpu.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
  else if constexpr (__sem == sem_release && __scope == scope_sys)
  {
    asm("multimem.st.release.sys.global.b64 [%0], %1;"
        :
        : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__val))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_multimem_st_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

#endif // _CUDA_PTX_GENERATED_MULTIMEM_ST_H_
