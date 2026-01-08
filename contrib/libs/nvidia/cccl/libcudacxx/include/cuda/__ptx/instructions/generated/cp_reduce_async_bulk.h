// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_H_
#define _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_H_

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .b32 }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_and_op_t,
  B32* dstMem,
  const B32* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_and_op_t,
  _B32* __dstMem,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_b32 (due to parameter type constraint)
// __op == op_and_op (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.and.b32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .b32 }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_or_op_t,
  B32* dstMem,
  const B32* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_or_op_t,
  _B32* __dstMem,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_b32 (due to parameter type constraint)
// __op == op_or_op (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.or.b32 [%0], [%1], %2, [%3]; // 1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .b32 }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_xor_op_t,
  B32* dstMem,
  const B32* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_xor_op_t,
  _B32* __dstMem,
  const _B32* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_b32 (due to parameter type constraint)
// __op == op_xor_op (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.xor.b32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.u32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.u32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .inc }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_inc_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_inc_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_inc (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.inc.u32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u32 }
// .op        = { .dec }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_dec_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_dec_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_dec (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.dec.u32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::int32_t* __dstMem,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s32 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::int32_t* __dstMem,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s32 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.s32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::int32_t* __dstMem,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.s32 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.type [dstMem], [srcMem], size, [rdsmem_bar]; // 1. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .u64 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint64_t* __dstMem,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u64 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 [%0], [%1], %2, [%3]; // "
      "1."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.u64 [dstMem], [srcMem], size, [rdsmem_bar]; // 2. PTX
ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .shared::cta }
// .type      = { .s64 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  int64_t* dstMem,
  const int64_t* srcMem,
  uint32_t size,
  uint64_t* rdsmem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::int64_t* __dstMem,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size,
  ::cuda::std::uint64_t* __rdsmem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s64 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 [%0], [%1], %2, [%3]; // "
      "2."
      :
      : "r"(__as_ptr_remote_dsmem(__dstMem)),
        "r"(__as_ptr_smem(__srcMem)),
        "r"(__size),
        "r"(__as_ptr_remote_dsmem(__rdsmem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 3. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .b32, .b64 }
// .op        = { .and }
template <typename Type>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_and_op_t,
  Type* dstMem,
  const Type* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_and_op_t,
  _Type* __dstMem,
  const _Type* __srcMem,
  ::cuda::std::uint32_t __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
// __op == op_and_op (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (sizeof(_Type) == 4)
  {
    asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b32  [%0], [%1], %2; // 3."
        :
        : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
        : "memory");
  }
  else if constexpr (sizeof(_Type) == 8)
  {
    asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b64  [%0], [%1], %2; // 3."
        :
        : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 3. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .b32, .b64 }
// .op        = { .or }
template <typename Type>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_or_op_t,
  Type* dstMem,
  const Type* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_or_op_t,
  _Type* __dstMem,
  const _Type* __srcMem,
  ::cuda::std::uint32_t __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
// __op == op_or_op (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (sizeof(_Type) == 4)
  {
    asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b32  [%0], [%1], %2; // 3."
        :
        : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
        : "memory");
  }
  else if constexpr (sizeof(_Type) == 8)
  {
    asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b64  [%0], [%1], %2; // 3."
        :
        : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 3. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .b32, .b64 }
// .op        = { .xor }
template <typename Type>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_xor_op_t,
  Type* dstMem,
  const Type* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename _Type>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_xor_op_t,
  _Type* __dstMem,
  const _Type* __srcMem,
  ::cuda::std::uint32_t __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_Type) == 4 || sizeof(_Type) == 8, "");
// __op == op_xor_op (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (sizeof(_Type) == 4)
  {
    asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32  [%0], [%1], %2; // 3."
        :
        : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
        : "memory");
  }
  else if constexpr (sizeof(_Type) == 8)
  {
    asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64  [%0], [%1], %2; // 3."
        :
        : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
        : "memory");
  }
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
// .type      = { .u32 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u32  [%0], [%1], %2; // 4."
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
// .type      = { .u32 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u32  [%0], [%1], %2; // 4."
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
// .type      = { .u32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32  [%0], [%1], %2; // 4."
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
// .type      = { .u32 }
// .op        = { .inc }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_inc_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_inc_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_inc (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32  [%0], [%1], %2; // 4."
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
// .type      = { .u32 }
// .op        = { .dec }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_dec_t,
  uint32_t* dstMem,
  const uint32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_dec_t,
  ::cuda::std::uint32_t* __dstMem,
  const ::cuda::std::uint32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u32 (due to parameter type constraint)
// __op == op_dec (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.dec.u32  [%0], [%1], %2; // 4."
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
// .type      = { .s32 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::int32_t* __dstMem,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s32 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32  [%0], [%1], %2; // 4."
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
// .type      = { .s32 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::int32_t* __dstMem,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s32 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32  [%0], [%1], %2; // 4."
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
// .type      = { .s32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  int32_t* dstMem,
  const int32_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::int32_t* __dstMem,
  const ::cuda::std::int32_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32  [%0], [%1], %2; // 4."
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
// .type      = { .u64 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::uint64_t* __dstMem,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u64 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u64  [%0], [%1], %2; // 4."
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
// .type      = { .u64 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::uint64_t* __dstMem,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u64 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u64  [%0], [%1], %2; // 4."
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
// .type      = { .u64 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  uint64_t* dstMem,
  const uint64_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::uint64_t* __dstMem,
  const ::cuda::std::uint64_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_u64 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64  [%0], [%1], %2; // 4."
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
// .type      = { .s64 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  int64_t* dstMem,
  const int64_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  ::cuda::std::int64_t* __dstMem,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s64 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s64  [%0], [%1], %2; // 4."
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
// .type      = { .s64 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  int64_t* dstMem,
  const int64_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  ::cuda::std::int64_t* __dstMem,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s64 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s64  [%0], [%1], %2; // 4."
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
// .type      = { .f32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  float* dstMem,
  const float* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  float* __dstMem,
  const float* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_f32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32  [%0], [%1], %2; // 4."
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
// .type      = { .f64 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  double* dstMem,
  const double* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  double* __dstMem,
  const double* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_f64 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64  [%0], [%1], %2; // 4."
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
// cp.reduce.async.bulk.dst.src.bulk_group.op.u64  [dstMem], [srcMem], size; // 6. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .s64 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  int64_t* dstMem,
  const int64_t* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  ::cuda::std::int64_t* __dstMem,
  const ::cuda::std::int64_t* __srcMem,
  ::cuda::std::uint32_t __size)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
// __type == type_s64 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64  [%0], [%1], %2; // 6."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_H_
