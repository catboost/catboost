// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_LD_H_
#define _CUDA_PTX_GENERATED_LD_H_

/*
// ld.space.b8 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_is_not_supported_before_SM_50__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint32_t __dest;
  asm("ld.global.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_is_not_supported_before_SM_50__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.b16 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_is_not_supported_before_SM_50__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint16_t __dest;
  asm("ld.global.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_is_not_supported_before_SM_50__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.b32 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_is_not_supported_before_SM_50__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint32_t __dest;
  asm("ld.global.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_is_not_supported_before_SM_50__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.b64 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_is_not_supported_before_SM_50__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint64_t __dest;
  asm("ld.global.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_is_not_supported_before_SM_50__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_L2_cache_hint(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_L2_cache_hint(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_L2_cache_hint(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_L2_cache_hint(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256
ld_L2_cache_hint(::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L1::evict_first.b8 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_first(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_evict_first(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.b16 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_first(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_evict_first(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_first.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_first.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.b32 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_first(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_evict_first(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.b64 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_first(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_evict_first(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_first.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_first.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_first(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_evict_first(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_first.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_first.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_first.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_first.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L1_evict_first(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_L1_evict_first(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L1::evict_first.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L1::evict_first.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_first.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_first.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_first.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_first.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_first.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256
ld_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L1::evict_first.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L1::evict_last.b8 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_last(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_evict_last(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.b16 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_last(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_evict_last(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_last.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_last.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.b32 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_last(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_evict_last(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.b64 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_last(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_evict_last(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_last.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_last.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_last(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_evict_last(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_last.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_last.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_last.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_last.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L1_evict_last(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_L1_evict_last(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L1::evict_last.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L1::evict_last.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::evict_last.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_last.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_last.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::evict_last.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::evict_last.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256
ld_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L1::evict_last.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L1::no_allocate.b8 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_no_allocate(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.b16 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_no_allocate(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::no_allocate.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::no_allocate.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.b32 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_no_allocate(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.b64 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_no_allocate(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::no_allocate.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::no_allocate.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_no_allocate(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::no_allocate.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::no_allocate.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::no_allocate.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::no_allocate.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_L1_no_allocate(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L1::no_allocate.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.L1::no_allocate.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.L1::no_allocate.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::no_allocate.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::no_allocate.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.L1::no_allocate.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.L1::no_allocate.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256
ld_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.L1::no_allocate.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.b8 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.nc.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.b16 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.nc.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.b32 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.nc.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.b64 dest, [addr]; // PTX ISA 10, SM_50
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_is_not_supported_before_SM_50__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// ld.space.nc.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_nc_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_nc_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_nc_L2_cache_hint(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_nc_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_nc_L2_cache_hint(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_nc_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_nc_L2_cache_hint(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_nc_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_nc_L2_cache_hint(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128
ld_nc_L2_cache_hint_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256
ld_nc_L2_cache_hint(::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L1::evict_first.b8 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_first(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_evict_first(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.b16 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_first(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_first(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_first.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.b32 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_first(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_first(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.b64 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_first(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_first(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_first.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_first(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_first(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_first.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_first.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_first_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_first_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_first.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_first.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L1_evict_first(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc_L1_evict_first(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L1::evict_first.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_nc_L1_evict_first_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_first.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_first_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_first.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_first.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L1_evict_first_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc_L1_evict_first_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L1::evict_first.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_first_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L1::evict_last.b8 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_last(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_evict_last(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.b16 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_last(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_last(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_last.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.b32 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_last(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_last(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.b64 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_last(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_last(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_last.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_last(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_last(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_last.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_last.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_last_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_last_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_last.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_last.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L1_evict_last(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc_L1_evict_last(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L1::evict_last.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_nc_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16
ld_nc_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32
ld_nc_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64
ld_nc_L1_evict_last_L2_cache_hint(::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_last.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_evict_last_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::evict_last.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L1_evict_last_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc_L1_evict_last_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_evict_last_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L1::no_allocate.b8 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_no_allocate(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B8* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::256B.b8 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.b16 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_no_allocate(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::no_allocate.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B16* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::256B.b16 %0, [%1];" : "=h"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.b32 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_no_allocate(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B32* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::256B.b32 %0, [%1];" : "=r"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.b64 dest, [addr]; // PTX ISA 74, SM_70
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_no_allocate(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::no_allocate.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B64* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::256B.b64 %0, [%1];" : "=l"(__dest) : "l"(__as_ptr_gmem(__addr)) : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.b128 dest, [addr]; // PTX ISA 83, SM_70
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_no_allocate(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::no_allocate.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_70__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::no_allocate.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_no_allocate_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_no_allocate_L2_256B(::cuda::ptx::space_global_t, const _B128* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::no_allocate.L2::256B.b128 B128_dest, [%2];\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::no_allocate.v4.b64 dest, [addr]; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L1_no_allocate(
  cuda::ptx::space_global_t,
  const B256* addr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc_L1_no_allocate(::cuda::ptx::space_global_t, const _B256* __addr)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L1::no_allocate.v4.b64 {%0, %1, %2, %3}, [%4];"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr))
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8
ld_nc_L1_no_allocate_L2_cache_hint(::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b8 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B8, enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B8* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B8, ::cuda::std::enable_if_t<sizeof(_B8) == 1, bool> = true>
_CCCL_DEVICE static inline _B8 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B8* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B8) == 1, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b8 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return __u32_as_b8<_B8>(__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B8*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b16 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B16* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B16* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint16_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b16 %0, [%1], %2;"
      : "=h"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b32 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B32* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B32* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint32_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
      : "=r"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b64 dest, [addr], cache_policy; // PTX ISA 74, SM_80
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B64* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 740
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B64* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  ::cuda::std::uint64_t __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b64 %0, [%1], %2;"
      : "=l"(__dest)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  ::cuda::std::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 740

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::no_allocate.L2::cache_hint.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.L2::256B.b128 dest, [addr], cache_policy; // PTX ISA 83, SM_80
// .space     = { .global }
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  cuda::ptx::space_global_t,
  const B128* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B128 ld_nc_L1_no_allocate_L2_cache_hint_L2_256B(
  ::cuda::ptx::space_global_t, const _B128* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B128) == 16, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  longlong2 __dest;
  asm("{\n\t .reg .b128 B128_dest; \n\t"
      "ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.b128 B128_dest, [%2], %3;\n\t"
      "mov.b128 {%0, %1}, B128_dest; \n"
      "}"
      : "=l"(__dest.x), "=l"(__dest.y)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B128*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_L2_256B_is_not_supported_before_SM_80__();
  longlong2 __err_out_var{0, 0};
  return *reinterpret_cast<_B128*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// ld.space.nc.L1::no_allocate.L2::cache_hint.v4.b64 dest, [addr], cache_policy; // PTX ISA 88, SM_100
// .space     = { .global }
template <typename B256, enable_if_t<sizeof(B256) == 32, bool> = true>
__device__ static inline B256 ld_nc_L1_no_allocate_L2_cache_hint(
  cuda::ptx::space_global_t,
  const B256* addr,
  uint64_t cache_policy);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_100__();
template <typename _B256, ::cuda::std::enable_if_t<sizeof(_B256) == 32, bool> = true>
_CCCL_DEVICE static inline _B256 ld_nc_L1_no_allocate_L2_cache_hint(
  ::cuda::ptx::space_global_t, const _B256* __addr, ::cuda::std::uint64_t __cache_policy)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B256) == 32, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  ::cuda::ptx::longlong4_32a __dest;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
      : "=l"(__dest.x), "=l"(__dest.y), "=l"(__dest.z), "=l"(__dest.w)
      : "l"(__as_ptr_gmem(__addr)), "l"(__cache_policy)
      : "memory");
  return *reinterpret_cast<_B256*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_nc_L1_no_allocate_L2_cache_hint_is_not_supported_before_SM_100__();
  ::cuda::ptx::longlong4_32a __err_out_var{0, 0, 0, 0};
  return *reinterpret_cast<_B256*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 880

#endif // _CUDA_PTX_GENERATED_LD_H_
