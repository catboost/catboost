// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_NOINC_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_NOINC_H_

/*
// cp.async.mbarrier.arrive.noinc.b64 [addr]; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline void cp_async_mbarrier_arrive_noinc(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_mbarrier_arrive_noinc_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_mbarrier_arrive_noinc(_CUDA_VSTD::uint64_t* __addr)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 800
  asm("cp.async.mbarrier.arrive.noinc.b64 [%0];" : : "r"(__as_ptr_smem(__addr)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_mbarrier_arrive_noinc_is_not_supported_before_SM_80__();
#  endif
}
#endif // __cccl_ptx_isa >= 700

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_NOINC_H_
