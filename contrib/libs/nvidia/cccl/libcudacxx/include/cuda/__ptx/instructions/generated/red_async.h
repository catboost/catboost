// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_RED_ASYNC_H_
#define _CUDA_PTX_GENERATED_RED_ASYNC_H_

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .inc }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_inc_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_inc_t,
          ::cuda::std::uint32_t* __dest,
          const ::cuda::std::uint32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_u32 (due to parameter type constraint)
// __op == op_inc (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .dec }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_dec_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_dec_t,
          ::cuda::std::uint32_t* __dest,
          const ::cuda::std::uint32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_u32 (due to parameter type constraint)
// __op == op_dec (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .min }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_min_t,
          ::cuda::std::uint32_t* __dest,
          const ::cuda::std::uint32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_u32 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .max }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_max_t,
          ::cuda::std::uint32_t* __dest,
          const ::cuda::std::uint32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_u32 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_add_t,
          ::cuda::std::uint32_t* __dest,
          const ::cuda::std::uint32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_u32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .min }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_min_t,
          ::cuda::std::int32_t* __dest,
          const ::cuda::std::int32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_s32 (due to parameter type constraint)
// __op == op_min (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .max }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_max_t,
          ::cuda::std::int32_t* __dest,
          const ::cuda::std::int32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_s32 (due to parameter type constraint)
// __op == op_max (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .add }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_add_t,
          ::cuda::std::int32_t* __dest,
          const ::cuda::std::int32_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_s32 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.s32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "r"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .and }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void red_async(
  cuda::ptx::op_and_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_and_op_t, _B32* __dest, const _B32& __value, ::cuda::std::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_and_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .or }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void red_async(
  cuda::ptx::op_or_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_or_op_t, _B32* __dest, const _B32& __value, ::cuda::std::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_or_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .xor }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void red_async(
  cuda::ptx::op_xor_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_xor_op_t, _B32* __dest, const _B32& __value, ::cuda::std::uint64_t* __remote_bar)
{
  // __type == type_b32 (due to parameter type constraint)
  // __op == op_xor_op (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type  [dest], value, [remote_bar];  // PTX
ISA 81, SM_90
// .type      = { .u64 }
// .op        = { .add }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint64_t* dest,
  const uint64_t& value,
  uint64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_add_t,
          ::cuda::std::uint64_t* __dest,
          const ::cuda::std::uint64_t& __value,
          ::cuda::std::uint64_t* __remote_bar)
{
// __type == type_u64 (due to parameter type constraint)
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [%0], %1, [%2]; "
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "l"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

/*
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.u64  [dest], value, [remote_bar]; // .u64
intentional PTX ISA 81, SM_90
// .op        = { .add }
template <typename = void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int64_t* dest,
  const int64_t& value,
  int64_t* remote_bar);
*/
#if __cccl_ptx_isa >= 810
extern "C" _CCCL_DEVICE void __cuda_ptx_red_async_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void
red_async(::cuda::ptx::op_add_t,
          ::cuda::std::int64_t* __dest,
          const ::cuda::std::int64_t& __value,
          ::cuda::std::int64_t* __remote_bar)
{
// __op == op_add (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u64  [%0], %1, [%2]; // .u64 "
      "intentional"
      :
      : "r"(__as_ptr_remote_dsmem(__dest)), "l"(__value), "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_red_async_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 810

#endif // _CUDA_PTX_GENERATED_RED_ASYNC_H_
