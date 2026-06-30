// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TENSORMAP_REPLACE_H_
#define _CUDA_PTX_GENERATED_TENSORMAP_REPLACE_H_

/*
// tensormap.replace.tile.global_address.space.b1024.b64 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void tensormap_replace_global_address(
  cuda::ptx::space_global_t,
  void* tm_addr,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_global_address_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void
tensormap_replace_global_address(::cuda::ptx::space_global_t, void* __tm_addr, _B64 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_global_address_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_address.space.b1024.b64 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void tensormap_replace_global_address(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_global_address_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void
tensormap_replace_global_address(::cuda::ptx::space_shared_t, void* __tm_addr, _B64 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_global_address_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.rank.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_rank(
  cuda::ptx::space_global_t,
  void* tm_addr,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_rank_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_rank(::cuda::ptx::space_global_t, void* __tm_addr, _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.rank.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_rank_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.rank.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_rank(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_rank_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_rank(::cuda::ptx::space_shared_t, void* __tm_addr, _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_rank_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.box_dim.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_box_dim(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_box_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tensormap_replace_box_dim(::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.box_dim.global.b1024.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_box_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.box_dim.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_box_dim(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_box_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tensormap_replace_box_dim(::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_box_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_dim.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_global_dim(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_global_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_global_dim(
  ::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_global_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_dim.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_global_dim(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_global_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_global_dim(
  ::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_global_dim_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_stride.space.b1024.b64 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32, typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void tensormap_replace_global_stride(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_global_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_global_stride(
  ::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B64 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord.value),
        "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_global_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.global_stride.space.b1024.b64 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32, typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void tensormap_replace_global_stride(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B64 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_global_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B64, ::cuda::std::enable_if_t<sizeof(_B64) == 8, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_global_stride(
  ::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B64 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B64) == 8, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord.value),
        "l"(/*as_b64*/ *reinterpret_cast<const ::cuda::std::int64_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_global_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.element_stride.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_element_stride(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_element_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_element_stride(
  ::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.element_stride.global.b1024.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_element_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.element_stride.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_element_stride(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_element_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_element_stride(
  ::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_element_stride_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.element_stride.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_element_size(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_element_size_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_element_size(
  ::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.element_stride.global.b1024.b32 [%0], %1, %2;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_element_size_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.element_stride.space.b1024.b32 [tm_addr], ord, new_val; // PTX ISA 83, SM_90a, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tensormap_replace_element_size(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> ord,
  B32 new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_element_size_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tensormap_replace_element_size(
  ::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __ord, _B32 __new_val)
{
  // __space == space_shared (due to parameter type constraint)
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__tm_addr)),
        "n"(__ord.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__new_val))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_element_size_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.elemtype.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_elemtype(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_elemtype_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_elemtype(::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.elemtype.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_elemtype_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.elemtype.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_elemtype(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_elemtype_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_elemtype(::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_elemtype_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_interleave_layout(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_interleave_layout_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_interleave_layout(::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.interleave_layout.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_interleave_layout_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.interleave_layout.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_interleave_layout(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_interleave_layout_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_interleave_layout(::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_interleave_layout_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.swizzle_mode.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_mode(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_swizzle_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_swizzle_mode(::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_swizzle_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.swizzle_mode.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_mode(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_swizzle_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_swizzle_mode(::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_swizzle_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.fill_mode.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_fill_mode(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_fill_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_fill_mode(::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.fill_mode.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_fill_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.fill_mode.space.b1024.b32 [tm_addr], new_val; // PTX ISA 83, SM_90a, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_fill_mode(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_fill_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_fill_mode(::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900)))   \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_fill_mode_is_only_supported_on_SM_90a_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 830

/*
// tensormap.replace.tile.swizzle_atomicity.space.b1024.b32 [tm_addr], new_val; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .global }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_atomicity(
  cuda::ptx::space_global_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_swizzle_atomicity_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_swizzle_atomicity(::cuda::ptx::space_global_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.swizzle_atomicity.global.b1024.b32 [%0], %1;"
      :
      : "l"(__as_ptr_gmem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_swizzle_atomicity_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tensormap.replace.tile.swizzle_atomicity.space.b1024.b32 [tm_addr], new_val; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f, SM_120a, SM_120f, SM_121a, SM_121f
// .space     = { .shared::cta }
template <int N32>
__device__ static inline void tensormap_replace_swizzle_atomicity(
  cuda::ptx::space_shared_t,
  void* tm_addr,
  cuda::ptx::n32_t<N32> new_val);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tensormap_replace_swizzle_atomicity_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
template <int _N32>
_CCCL_DEVICE static inline void
tensormap_replace_swizzle_atomicity(::cuda::ptx::space_shared_t, void* __tm_addr, ::cuda::ptx::n32_t<_N32> __new_val)
{
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))) \
    || (defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210))
  asm("tensormap.replace.tile.swizzle_atomicity.shared::cta.b1024.b32 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__tm_addr)), "n"(__new_val.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tensormap_replace_swizzle_atomicity_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_120a_120f_121a_121f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TENSORMAP_REPLACE_H_
