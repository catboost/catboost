// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_ELECT_SYNC_H_
#define _CUDA_PTX_GENERATED_ELECT_SYNC_H_

/*
// elect.sync _|is_elected, membermask; // PTX ISA 80, SM_90
template <typename = void>
__device__ static inline bool elect_sync(
  const uint32_t& membermask);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_elect_sync_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline bool elect_sync(const _CUDA_VSTD::uint32_t& __membermask)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  _CUDA_VSTD::uint32_t __is_elected;
  asm volatile(
    "{\n\t .reg .pred P_OUT; \n\t"
    "elect.sync _|P_OUT, %1;\n\t"
    "selp.b32 %0, 1, 0, P_OUT; \n"
    "}"
    : "=r"(__is_elected)
    : "r"(__membermask)
    :);
  return static_cast<bool>(__is_elected);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_elect_sync_is_not_supported_before_SM_90__();
  return false;
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_ELECT_SYNC_H_
