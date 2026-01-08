// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_ST_H_
#define _CUDA_PTX_GENERATED_TCGEN05_ST_H_

/*
// tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], {%1};"
      :
      : "r"(__taddr), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [%0], {%1};"
      :
      : "r"(__taddr), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x2.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x4.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x64b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x64b.x64.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x64b.x128.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x1.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x128b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x128b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x128b.x64.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x256b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x256b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
      :
      : "r"(__taddr), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [%0], {%1};"
      :
      : "r"(__taddr), "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.32x32b.x128.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(::cuda::std::uint32_t __taddr, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x1.b32 [%0], %1, {%2};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [%0], %1, {%2};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x2.b32 [%0], %1, {%2, %3};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [%0], %1, {%2, %3};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x4.b32 [%0], %1, {%2, %3, %4, %5};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x8.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x16.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC)                                                                                      \
    || (defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))) \
    || (defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))) \
    || (defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))) \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))                            \
    || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
  asm("tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x32bx2.x32.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[32])
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
    "tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x32bx2.x64.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[64])
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
    "tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x32bx2.x128.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
    "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
    "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, "
    "%59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, "
    "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, "
    "%103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, "
    "%122, %123, %124, %125, %126, %127, %128, %129};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_110a, SM_110f template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <int _N32, typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x32bx2_unpack_16b(
  ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[128])
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
    "tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128, %129};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[0])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[1])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[2])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[3])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[4])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[5])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[6])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[7])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[8])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[9])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[10])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[11])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[12])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[13])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[14])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[15])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[16])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[17])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[18])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[19])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[20])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[21])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[22])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[23])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[24])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[25])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[26])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[27])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[28])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[29])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[30])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[31])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[32])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[33])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[34])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[35])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[36])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[37])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[38])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[39])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[40])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[41])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[42])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[43])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[44])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[45])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[46])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[47])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[48])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[49])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[50])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[51])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[52])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[53])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[54])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[55])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[56])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[57])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[58])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[59])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[60])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[61])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[62])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[63])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[64])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[65])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[66])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[67])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[68])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[69])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[70])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[71])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[72])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[73])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[74])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[75])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[76])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[77])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[78])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[79])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[80])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[81])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[82])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[83])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[84])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[85])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[86])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[87])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[88])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[89])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[90])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[91])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[92])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[93])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[94])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[95])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[96])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[97])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[98])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[99])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[100])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[101])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[102])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[103])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[104])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[105])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[106])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[107])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[108])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[109])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[110])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[111])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[112])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[113])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[114])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[115])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[116])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[117])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[118])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[119])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[120])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[121])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[122])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[123])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[124])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[125])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[126])),
      "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_ST_H_
