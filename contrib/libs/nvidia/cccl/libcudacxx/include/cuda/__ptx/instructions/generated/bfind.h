// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_BFIND_H_
#define _CUDA_PTX_GENERATED_BFIND_H_

/*
// bfind.u32 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind(
  uint32_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind(_CUDA_VSTD::uint32_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.u32 %0, %1;" : "=r"(__dest) : "r"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.u32 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind_shiftamt(
  uint32_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind_shiftamt(_CUDA_VSTD::uint32_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.shiftamt.u32 %0, %1;" : "=r"(__dest) : "r"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.u64 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind(
  uint64_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind(_CUDA_VSTD::uint64_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.u64 %0, %1;" : "=r"(__dest) : "l"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.u64 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind_shiftamt(
  uint64_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind_shiftamt(_CUDA_VSTD::uint64_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.shiftamt.u64 %0, %1;" : "=r"(__dest) : "l"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.s32 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind(
  int32_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind(_CUDA_VSTD::int32_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.s32 %0, %1;" : "=r"(__dest) : "r"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.s32 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind_shiftamt(
  int32_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind_shiftamt(_CUDA_VSTD::int32_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.shiftamt.s32 %0, %1;" : "=r"(__dest) : "r"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.s64 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind(
  int64_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind(_CUDA_VSTD::int64_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.s64 %0, %1;" : "=r"(__dest) : "l"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

/*
// bfind.shiftamt.s64 dest, a_reg; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline uint32_t bfind_shiftamt(
  int64_t a_reg);
*/
#if __cccl_ptx_isa >= 200
extern "C" _CCCL_DEVICE void __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t bfind_shiftamt(_CUDA_VSTD::int64_t __a_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  _CUDA_VSTD::uint32_t __dest;
  asm("bfind.shiftamt.s64 %0, %1;" : "=r"(__dest) : "l"(__a_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bfind_shiftamt_is_not_supported_before_SM_50__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 200

#endif // _CUDA_PTX_GENERATED_BFIND_H_
