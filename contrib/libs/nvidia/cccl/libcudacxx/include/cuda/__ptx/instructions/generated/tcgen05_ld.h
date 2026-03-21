// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_LD_H_
#define _CUDA_PTX_GENERATED_TCGEN05_LD_H_

/*
// tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x2.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[1],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%0}, [%1], %2;"
      : "=r"(__out[0])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[1],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[1], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 {%0}, [%1], %2;"
      : "=r"(__out[0])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {%0, %1}, [%2], %3;"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[2], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 {%0, %1}, [%2], %3;"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x4.b32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[4], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[8], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15}, [%16], %17;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[16], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16], %17;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[32], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64], %65;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[64], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64], %65;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128], %129;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[128], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128], %129;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_LD_H_
