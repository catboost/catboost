// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_F16_H_
#define _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_F16_H_

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f16 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  __half* dstMem,
  const __half* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  __half* __dstMem,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_f16 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16  [%0], [%1], %2; // 4."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f16 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  __half* dstMem,
  const __half* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  __half* __dstMem,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_f16 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16  [%0], [%1], %2; // 4."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.noftz.type  [dstMem], [srcMem], size; // 5. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .f16 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  __half* dstMem,
  const __half* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  __half* __dstMem,
  const __half* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_f16 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16  [%0], [%1], %2; // 5."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_F16_H_
