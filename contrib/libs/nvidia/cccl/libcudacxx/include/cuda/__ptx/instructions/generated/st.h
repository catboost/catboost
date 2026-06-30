// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_ST_H_
#define _CUDA_PTX_GENERATED_ST_H_

/*
// st.space.b8 [addr], src; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_st_is_not_supported_before_SM_50__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st(::cuda::ptx::space_global_t, _B8* __addr, _B8 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm("st.global.b8 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// st.space.b16 [addr], src; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_st_is_not_supported_before_SM_50__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st(::cuda::ptx::space_global_t, _B16* __addr, _B16 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm("st.global.b16 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// st.space.b32 [addr], src; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_st_is_not_supported_before_SM_50__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st(::cuda::ptx::space_global_t, _B32* __addr, _B32 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm("st.global.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// st.space.b64 [addr], src; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_st_is_not_supported_before_SM_50__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st(::cuda::ptx::space_global_t, _B64* __addr, _B64 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm("st.global.b64 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// st.space.b128 [addr], src; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st(::cuda::ptx::space_global_t, _B128* __addr, _B128 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.b128 [%0], B128_src;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.v4.b64 [addr], src; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st(::cuda::ptx::space_global_t, _B256* __addr, _B256 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.v4.b64 [%0], {%1, %2, %3, %4};"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L2_cache_hint(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void
st_L2_cache_hint(::cuda::ptx::space_global_t, _B8* __addr, _B8 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L2::cache_hint.b8 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)), "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L2_cache_hint(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void
st_L2_cache_hint(::cuda::ptx::space_global_t, _B16* __addr, _B16 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L2::cache_hint.b16 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L2_cache_hint(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
st_L2_cache_hint(::cuda::ptx::space_global_t, _B32* __addr, _B32 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L2::cache_hint.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L2_cache_hint(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void
st_L2_cache_hint(::cuda::ptx::space_global_t, _B64* __addr, _B64 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L2::cache_hint.b64 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L2_cache_hint(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void
st_L2_cache_hint(::cuda::ptx::space_global_t, _B128* __addr, _B128 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L2::cache_hint.b128 [%0], B128_src, %3;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L2_cache_hint(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void
st_L2_cache_hint(::cuda::ptx::space_global_t, _B256* __addr, _B256 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L2::cache_hint.v4.b64 [%0], {%1, %2, %3, %4}, %5;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L2_cache_hint_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L1::evict_first.b8 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L1_evict_first(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first(::cuda::ptx::space_global_t, _B8* __addr, _B8 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_first.b8 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.b16 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L1_evict_first(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first(::cuda::ptx::space_global_t, _B16* __addr, _B16 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_first.b16 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.b32 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L1_evict_first(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first(::cuda::ptx::space_global_t, _B32* __addr, _B32 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_first.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.b64 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L1_evict_first(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first(::cuda::ptx::space_global_t, _B64* __addr, _B64 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_first.b64 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.b128 [addr], src; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L1_evict_first(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first(::cuda::ptx::space_global_t, _B128* __addr, _B128 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L1::evict_first.b128 [%0], B128_src;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L1::evict_first.v4.b64 [addr], src; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L1_evict_first(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first(::cuda::ptx::space_global_t, _B256* __addr, _B256 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L1::evict_first.v4.b64 [%0], {%1, %2, %3, %4};"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L1::evict_first.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B8* __addr, _B8 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_first.L2::cache_hint.b8 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)), "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B16* __addr, _B16 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_first.L2::cache_hint.b16 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B32* __addr, _B32 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_first.L2::cache_hint.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B64* __addr, _B64 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_first.L2::cache_hint.b64 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_first.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B128* __addr, _B128 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L1::evict_first.L2::cache_hint.b128 [%0], B128_src, %3;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L1::evict_first.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B256* __addr, _B256 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L1::evict_first.L2::cache_hint.v4.b64 [%0], {%1, %2, %3, %4}, %5;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L1::evict_last.b8 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L1_evict_last(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last(::cuda::ptx::space_global_t, _B8* __addr, _B8 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_last.b8 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.b16 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L1_evict_last(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last(::cuda::ptx::space_global_t, _B16* __addr, _B16 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_last.b16 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.b32 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L1_evict_last(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last(::cuda::ptx::space_global_t, _B32* __addr, _B32 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_last.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.b64 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L1_evict_last(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last(::cuda::ptx::space_global_t, _B64* __addr, _B64 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::evict_last.b64 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.b128 [addr], src; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L1_evict_last(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last(::cuda::ptx::space_global_t, _B128* __addr, _B128 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L1::evict_last.b128 [%0], B128_src;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L1::evict_last.v4.b64 [addr], src; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L1_evict_last(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last(::cuda::ptx::space_global_t, _B256* __addr, _B256 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L1::evict_last.v4.b64 [%0], {%1, %2, %3, %4};"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L1::evict_last.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B8* __addr, _B8 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_last.L2::cache_hint.b8 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)), "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B16* __addr, _B16 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_last.L2::cache_hint.b16 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B32* __addr, _B32 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_last.L2::cache_hint.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B64* __addr, _B64 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::evict_last.L2::cache_hint.b64 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::evict_last.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B128* __addr, _B128 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L1::evict_last.L2::cache_hint.b128 [%0], B128_src, %3;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L1::evict_last.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B256* __addr, _B256 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L1::evict_last.L2::cache_hint.v4.b64 [%0], {%1, %2, %3, %4}, %5;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L1::no_allocate.b8 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L1_no_allocate(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate(::cuda::ptx::space_global_t, _B8* __addr, _B8 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::no_allocate.b8 [%0], %1;" : : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.b16 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L1_no_allocate(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate(::cuda::ptx::space_global_t, _B16* __addr, _B16 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::no_allocate.b16 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.b32 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L1_no_allocate(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate(::cuda::ptx::space_global_t, _B32* __addr, _B32 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::no_allocate.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.b64 [addr], src; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L1_no_allocate(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate(::cuda::ptx::space_global_t, _B64* __addr, _B64 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("st.global.L1::no_allocate.b64 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.b128 [addr], src; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L1_no_allocate(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate(::cuda::ptx::space_global_t, _B128* __addr, _B128 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L1::no_allocate.b128 [%0], B128_src;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_70__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L1::no_allocate.v4.b64 [addr], src; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L1_no_allocate(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate(::cuda::ptx::space_global_t, _B256* __addr, _B256 __src)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L1::no_allocate.v4.b64 [%0], {%1, %2, %3, %4};"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// st.space.L1::no_allocate.L2::cache_hint.b8 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  B8* addr,
  B8 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B8* __addr, _B8 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::no_allocate.L2::cache_hint.b8 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)), "r"(__b8_as_u32(__src)), "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.L2::cache_hint.b16 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  B16* addr,
  B16 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B16* __addr, _B16 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::no_allocate.L2::cache_hint.b16 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.L2::cache_hint.b32 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  B32* addr,
  B32 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B32* __addr, _B32 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::no_allocate.L2::cache_hint.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.L2::cache_hint.b64 [addr], src, cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  B64* addr,
  B64 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B64* __addr, _B64 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("st.global.L1::no_allocate.L2::cache_hint.b64 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__src)),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// st.space.L1::no_allocate.L2::cache_hint.b128 [addr], src, cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  B128* addr,
  B128 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B128* __addr, _B128 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("{\n\t .reg .b128 B128_src; \n\t"
      "mov.b128 B128_src, {%1, %2}; \n"
      "st.global.L1::no_allocate.L2::cache_hint.b128 [%0], B128_src, %3;\n\t"
      "}"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<longlong2*>(&__src)).x),
        "l"((*reinterpret_cast<longlong2*>(&__src)).y),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// st.space.L1::no_allocate.L2::cache_hint.v4.b64 [addr], src, cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline void st_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  B256* addr,
  B256 src,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline void st_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, _B256* __addr, _B256 __src, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.global.L1::no_allocate.L2::cache_hint.v4.b64 [%0], {%1, %2, %3, %4}, %5;"
      :
      : "l"(__as_ptr_gmem(__addr)),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).x),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).y),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).z),
        "l"((*reinterpret_cast<::cuda::ptx::longlong4_32a*>(&__src)).w),
        "l"(__cache_policy)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 880

#endif // _CUDA_PTX_GENERATED_ST_H_
