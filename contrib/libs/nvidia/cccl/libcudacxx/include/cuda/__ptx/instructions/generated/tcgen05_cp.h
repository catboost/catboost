// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_CP_H_
#define _CUDA_PTX_GENERATED_TCGEN05_CP_H_

/*
// tcgen05.cp.cta_group.128x256b [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_128x256b(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_128x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_128x256b(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.128x256b [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.128x256b [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_128x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.4x256b [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_4x256b(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_4x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_4x256b(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.4x256b [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.4x256b [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_4x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.128x128b [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_128x128b(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_128x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_128x128b(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.128x128b [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_128x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.64x128b.warpx2::02_13 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_64x128b_warpx2_02_13(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_64x128b_warpx2_02_13_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_64x128b_warpx2_02_13(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_64x128b_warpx2_02_13_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.64x128b.warpx2::01_23 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_64x128b_warpx2_01_23(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_64x128b_warpx2_01_23_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_64x128b_warpx2_01_23(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_64x128b_warpx2_01_23_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.32x128b.warpx4 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_32x128b_warpx4(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_32x128b_warpx4_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_32x128b_warpx4(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_32x128b_warpx4_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.128x256b.b8x16.b6x16_p32 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_128x256b_b8x16_b6x16_p32(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_128x256b_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_128x256b_b8x16_b6x16_p32(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_128x256b_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.4x256b.b8x16.b6x16_p32 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_4x256b_b8x16_b6x16_p32(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_4x256b_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_4x256b_b8x16_b6x16_p32(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_4x256b_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.128x128b.b8x16.b6x16_p32 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_128x128b_b8x16_b6x16_p32(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_128x128b_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_128x128b_b8x16_b6x16_p32(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_128x128b_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b6x16_p32 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b6x16_p32 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_64x128b_warpx2_02_13_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b6x16_p32 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b6x16_p32 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_64x128b_warpx2_01_23_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_32x128b_warpx4_b8x16_b6x16_p32_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.128x256b.b8x16.b4x16_p64 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_128x256b_b8x16_b4x16_p64(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_128x256b_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_128x256b_b8x16_b4x16_p64(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_128x256b_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.4x256b.b8x16.b4x16_p64 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_4x256b_b8x16_b4x16_p64(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_4x256b_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_4x256b_b8x16_b4x16_p64(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_4x256b_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.128x128b.b8x16.b4x16_p64 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_128x128b_b8x16_b4x16_p64(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_128x128b_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_128x128b_b8x16_b4x16_p64(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_128x128b_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13.b8x16.b4x16_p64 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13.b8x16.b4x16_p64 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_64x128b_warpx2_02_13_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23.b8x16.b4x16_p64 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23.b8x16.b4x16_p64 [%0], %1;"
        :
        : "r"(__taddr), "l"(__s_desc)
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_64x128b_warpx2_01_23_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.cp.cta_group.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64(
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  uint32_t taddr,
  uint64_t s_desc);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64(
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group, ::cuda::std::uint32_t __taddr, ::cuda::std::uint64_t __s_desc)
{
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  if constexpr (__cta_group == cta_group_1)
  {
    asm("tcgen05.cp.cta_group::1.32x128b.warpx4.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b4x16_p64 [%0], %1;" : : "r"(__taddr), "l"(__s_desc) : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_cp_32x128b_warpx4_b8x16_b4x16_p64_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_CP_H_
