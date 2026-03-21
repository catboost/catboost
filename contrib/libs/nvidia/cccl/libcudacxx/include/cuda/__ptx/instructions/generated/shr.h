// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_SHR_H_
#define _CUDA_PTX_GENERATED_SHR_H_

/*
// shr.b16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 shr(
  B16 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_shr_is_not_supported_before_SM_50__();
template <typename _B16, _CUDA_VSTD::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline _B16 shr(_B16 __a_reg, _CUDA_VSTD::uint32_t __b_reg)
{
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B16) == 2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint16_t __dest;
  asm("shr.b16 %0, %1, %2;"
      : "=h"(__dest)
      : "h"(/*as_b16*/ *reinterpret_cast<const _CUDA_VSTD::int16_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_B16*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_shr_is_not_supported_before_SM_50__();
  _CUDA_VSTD::uint16_t __err_out_var = 0;
  return *reinterpret_cast<_B16*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.b32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 shr(
  B32 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_shr_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _B32 shr(_B32 __a_reg, _CUDA_VSTD::uint32_t __b_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("shr.b32 %0, %1, %2;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_B32*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_shr_is_not_supported_before_SM_50__();
  _CUDA_VSTD::uint32_t __err_out_var = 0;
  return *reinterpret_cast<_B32*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.b64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 shr(
  B64 a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_shr_is_not_supported_before_SM_50__();
template <typename _B64, _CUDA_VSTD::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline _B64 shr(_B64 __a_reg, _CUDA_VSTD::uint32_t __b_reg)
{
  static_assert(sizeof(_B64) == 8, "");
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint64_t __dest;
  asm("shr.b64 %0, %1, %2;"
      : "=l"(__dest)
      : "l"(/*as_b64*/ *reinterpret_cast<const _CUDA_VSTD::int64_t*>(&__a_reg)), "r"(__b_reg)
      :);
  return *reinterpret_cast<_B64*>(&__dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_shr_is_not_supported_before_SM_50__();
  _CUDA_VSTD::uint64_t __err_out_var = 0;
  return *reinterpret_cast<_B64*>(&__err_out_var);
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.s16 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline int16_t shr(
  int16_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_shr_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::int16_t shr(_CUDA_VSTD::int16_t __a_reg, _CUDA_VSTD::uint32_t __b_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::int16_t __dest;
  asm("shr.s16 %0, %1, %2;" : "=h"(__dest) : "h"(__a_reg), "r"(__b_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_shr_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.s32 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline int32_t shr(
  int32_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_shr_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::int32_t shr(_CUDA_VSTD::int32_t __a_reg, _CUDA_VSTD::uint32_t __b_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::int32_t __dest;
  asm("shr.s32 %0, %1, %2;" : "=r"(__dest) : "r"(__a_reg), "r"(__b_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_shr_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 100

/*
// shr.s64 dest, a_reg, b_reg; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline int64_t shr(
  int64_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_shr_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::int64_t shr(_CUDA_VSTD::int64_t __a_reg, _CUDA_VSTD::uint32_t __b_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::int64_t __dest;
  asm("shr.s64 %0, %1, %2;" : "=l"(__dest) : "l"(__a_reg), "r"(__b_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_shr_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 100

#endif // _CUDA_PTX_GENERATED_SHR_H_
