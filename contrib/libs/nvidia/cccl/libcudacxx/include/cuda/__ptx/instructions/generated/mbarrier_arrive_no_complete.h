// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_NO_COMPLETE_H_
#define _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_NO_COMPLETE_H_

/*
// mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.  PTX ISA 70, SM_80
template <typename = void>
__device__ static inline uint64_t mbarrier_arrive_no_complete(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t
mbarrier_arrive_no_complete(_CUDA_VSTD::uint64_t* __addr, const _CUDA_VSTD::uint32_t& __count)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  _CUDA_VSTD::uint64_t __state;
  asm("mbarrier.arrive.noComplete.shared.b64                       %0,  [%1], %2;    // 5. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
  return __state;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 700

#endif // _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_NO_COMPLETE_H_
