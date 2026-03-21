// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_PRMT_H_
#define _CUDA_PTX_GENERATED_PRMT_H_

/*
// prmt.b32 dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32 %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prmt.b32.f4e dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt_f4e(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_f4e_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt_f4e(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32.f4e %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_f4e_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prmt.b32.b4e dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt_b4e(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_b4e_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt_b4e(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32.b4e %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_b4e_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prmt.b32.rc8 dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt_rc8(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_rc8_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt_rc8(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32.rc8 %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_rc8_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prmt.b32.ecl dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt_ecl(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_ecl_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt_ecl(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32.ecl %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_ecl_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prmt.b32.ecr dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt_ecr(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_ecr_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt_ecr(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32.ecr %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_ecr_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// prmt.b32.rc16 dest, a_reg, b_reg, c_reg; // PTX ISA 20, SM_50
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline uint32_t prmt_rc16(
  B32 a_reg,
  B32 b_reg,
  uint32_t c_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_prmt_rc16_is_not_supported_before_SM_50__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t prmt_rc16(_B32 __a_reg, _B32 __b_reg, _CUDA_VSTD::uint32_t __c_reg)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("prmt.b32.rc16 %0, %1, %2, %3;"
      : "=r"(__dest)
      : "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__a_reg)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t*>(&__b_reg)),
        "r"(__c_reg)
      :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_prmt_rc16_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

#endif // _CUDA_PTX_GENERATED_PRMT_H_
