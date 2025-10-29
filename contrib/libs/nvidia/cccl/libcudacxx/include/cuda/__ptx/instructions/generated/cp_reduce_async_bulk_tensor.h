// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_TENSOR_H_
#define _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_TENSOR_H_

/*
// cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1a. PTX ISA 80,
SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.bulk_group [%0, {%1}], [%2]; // 1a."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1b. PTX ISA 80,
SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2}], [%3]; // 1b."
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1c. PTX ISA 80,
SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 1c."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1d. PTX ISA 80,
SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 1d."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 1e. PTX ISA 80,
SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 1e."
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_reduce_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_TENSOR_H_
