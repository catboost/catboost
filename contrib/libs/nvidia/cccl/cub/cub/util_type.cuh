/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Common type manipulation (metaprogramming) utilities
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh>
#include <cub/detail/uninitialized_copy.cuh>

#include <thrust/iterator/detail/any_assign.h>

#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/type_traits>

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVBF16()

// cuda_fp8.h resets default for C4127, so we have to guard the inclusion
#if _CCCL_HAS_NVFP8()
_CCCL_DIAG_PUSH
#  include <cuda_fp8.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP8()

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Conditional types
 ******************************************************************************/

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// the following iterator helpers are not named iter_value_t etc, like the C++20 facilities, because they are defined in
// terms of C++17 iterator_traits and not the new C++20 indirectly_readable trait etc. This allows them to detect nested
// value_type, difference_type and reference aliases, which the new C+20 traits do not consider (they only consider
// specializations of iterator_traits). Also, a value_type of void remains supported (needed by some output iterators).

template <typename It>
using it_value_t = typename ::cuda::std::iterator_traits<It>::value_type;

template <typename It>
using it_reference_t = typename ::cuda::std::iterator_traits<It>::reference;

template <typename It>
using it_difference_t = typename ::cuda::std::iterator_traits<It>::difference_type;

template <typename It>
using it_pointer_t = typename ::cuda::std::iterator_traits<It>::pointer;

// use this whenever you need to lazily evaluate a trait. E.g., as an alternative in replace_if_use_default.
template <template <typename...> typename Trait, typename... Args>
struct lazy_trait
{
  using type = Trait<Args...>;
};

template <typename It, typename FallbackT, bool = ::cuda::std::is_void_v<::cuda::std::remove_pointer_t<It>>>
struct non_void_value_impl
{
  using type = FallbackT;
};

template <typename It, typename FallbackT>
struct non_void_value_impl<It, FallbackT, false>
{
  // we consider thrust::discard_iterator's value_type (`any_assign`) as `void` as well, so users can switch from
  // cub::DiscardInputIterator to thrust::discard_iterator.
  using type = ::cuda::std::_If<::cuda::std::is_void_v<it_value_t<It>>
                                  || ::cuda::std::is_same_v<it_value_t<It>, THRUST_NS_QUALIFIER::detail::any_assign>,
                                FallbackT,
                                it_value_t<It>>;
};

/**
 * The output value type
 * type = (if IteratorT's value type is void) ?
 * ... then the FallbackT,
 * ... else the IteratorT's value type
 */
template <typename It, typename FallbackT>
using non_void_value_t = typename non_void_value_impl<It, FallbackT>::type;
} // namespace detail

/******************************************************************************
 * Static math
 ******************************************************************************/

/**
 * \brief Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
  /// Static logarithm value
  enum
  {
    VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE
  }; // Inductive case
};

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
  enum
  {
    VALUE = (1 << (COUNT - 1) < N) ? // Base case
              COUNT
                                   : COUNT - 1
  };
};

#  endif // _CCCL_DOXYGEN_INVOKED

/**
 * \brief Statically determine if N is a power-of-two
 */
template <int N>
struct PowerOfTwo
{
  enum
  {
    VALUE = ((N & (N - 1)) == 0)
  };
};

#endif // _CCCL_DOXYGEN_INVOKED

/******************************************************************************
 * Marker types
 ******************************************************************************/

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/**
 * \brief A simple "null" marker type
 */
struct NullType
{
  using value_type = NullType;

  NullType() = default;

  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE explicit NullType(const T&)
  {}

  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE NullType& operator=(const T&)
  {
    return *this;
  }

  friend _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const NullType&, const NullType&)
  {
    return true;
  }

  friend _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const NullType&, const NullType&)
  {
    return false;
  }
};

namespace detail
{

template <bool Value>
inline constexpr auto bool_constant_v = ::cuda::std::bool_constant<Value>{};

template <auto Value>
using constant_t = ::cuda::std::integral_constant<decltype(Value), Value>;

template <auto Value>
inline constexpr auto constant_v = constant_t<Value>{};

} // namespace detail

/**
 * \brief Allows algorithms that take a value as input to take a future value that is not computed yet at launch time.
 *
 * Note that it is user's responsibility to ensure that the result will be ready before use via external synchronization
 * or stream-ordering dependencies.
 *
 * \code
 * int *d_intermediate_result;
 * allocator.DeviceAllocate((void **)&d_intermediate_result, sizeof(int));
 * compute_intermediate_result<<<blocks, threads>>>(
 *     d_intermediate_result,  // output
 *     arg1,                   // input
 *     arg2);                  // input
 * cub::FutureValue<int> init_value(d_intermediate_result);
 * cub::DeviceScan::ExclusiveScan(
 *     d_temp_storage,
 *     temp_storage_bytes,
 *     d_in,
 *     d_out,
 *     cuda::std::plus<>{},
 *     init_value,
 *     num_items);
 * allocator.DeviceFree(d_intermediate_result);
 * \endcode
 */
template <typename T, typename IterT = T*>
struct FutureValue
{
  using value_type    = T;
  using iterator_type = IterT;

  explicit _CCCL_HOST_DEVICE _CCCL_FORCEINLINE FutureValue(IterT iter)
      : m_iter(iter)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator T() const noexcept
  {
    return *m_iter;
  }

private:
  IterT m_iter;
};

template <typename IterT>
FutureValue(IterT) -> FutureValue<detail::it_value_t<IterT>, IterT>;

namespace detail
{

/**
 * \brief Allows algorithms to instantiate a single kernel to support both immediate value and future value.
 */
template <typename T, typename IterT = T*>
struct InputValue
{
  using value_type    = T;
  using iterator_type = IterT;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator T()
  {
    if (m_is_future)
    {
      return m_future_value;
    }
    return m_immediate_value;
  }
  explicit _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InputValue(T immediate_value)
      : m_is_future(false)
      , m_immediate_value(immediate_value)
  {}
  explicit _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InputValue(FutureValue<T, IterT> future_value)
      : m_is_future(true)
      , m_future_value(future_value)
  {}
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InputValue(const InputValue& other)
      : m_is_future(other.m_is_future)
  {
    if (m_is_future)
    {
      m_future_value = other.m_future_value;
    }
    else
    {
      detail::uninitialized_copy_single(&m_immediate_value, other.m_immediate_value);
    }
  }

private:
  bool m_is_future;
  union
  {
    FutureValue<T, IterT> m_future_value;
    T m_immediate_value;
  };
};

} // namespace detail

/******************************************************************************
 * Size and alignment
 ******************************************************************************/

/// Structure alignment
template <typename T>
struct AlignBytes
{
  /// The "true CUDA" alignment of T in bytes
  static constexpr unsigned ALIGN_BYTES = alignof(T);

  /// The "truly aligned" type
  using Type = T;
};

// Specializations where host C++ compilers (e.g., 32-bit Windows) may disagree
// with device C++ compilers (EDG) on types passed as template parameters through
// kernel functions

#  define __CUB_ALIGN_BYTES(t, b)                                                                  \
    template <>                                                                                    \
    struct AlignBytes<t>                                                                           \
    {                                                                                              \
      static constexpr unsigned ALIGN_BYTES = b;                                                   \
                                                                                                   \
      typedef __align__(b) t Type;                                                                 \
      /* TODO(bgruber): rewriting the above to using Type __align__(b) = t; does not compile :S */ \
    };

__CUB_ALIGN_BYTES(short4, 8)
__CUB_ALIGN_BYTES(ushort4, 8)
__CUB_ALIGN_BYTES(int2, 8)
__CUB_ALIGN_BYTES(uint2, 8)
__CUB_ALIGN_BYTES(long long, 8)
__CUB_ALIGN_BYTES(unsigned long long, 8)
__CUB_ALIGN_BYTES(float2, 8)
__CUB_ALIGN_BYTES(double, 8)
#  ifdef _WIN32
__CUB_ALIGN_BYTES(long2, 8)
__CUB_ALIGN_BYTES(ulong2, 8)
#  else
__CUB_ALIGN_BYTES(long2, 16)
__CUB_ALIGN_BYTES(ulong2, 16)
#  endif
__CUB_ALIGN_BYTES(int4, 16)
__CUB_ALIGN_BYTES(uint4, 16)
__CUB_ALIGN_BYTES(float4, 16)
_CCCL_SUPPRESS_DEPRECATED_PUSH
__CUB_ALIGN_BYTES(long4, 16)
__CUB_ALIGN_BYTES(ulong4, 16)
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
__CUB_ALIGN_BYTES(long4_16a, 16)
__CUB_ALIGN_BYTES(long4_32a, 32)
__CUB_ALIGN_BYTES(ulong4_16a, 16)
__CUB_ALIGN_BYTES(ulong4_32a, 32)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
__CUB_ALIGN_BYTES(longlong2, 16)
__CUB_ALIGN_BYTES(ulonglong2, 16)
__CUB_ALIGN_BYTES(double2, 16)
_CCCL_SUPPRESS_DEPRECATED_PUSH
__CUB_ALIGN_BYTES(longlong4, 16)
__CUB_ALIGN_BYTES(ulonglong4, 16)
__CUB_ALIGN_BYTES(double4, 16)
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
__CUB_ALIGN_BYTES(longlong4_16a, 16)
__CUB_ALIGN_BYTES(longlong4_32a, 32)
__CUB_ALIGN_BYTES(ulonglong4_16a, 16)
__CUB_ALIGN_BYTES(ulonglong4_32a, 32)
__CUB_ALIGN_BYTES(double4_16a, 16)
__CUB_ALIGN_BYTES(double4_32a, 32)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

// clang-format off
template <typename T> struct AlignBytes<volatile T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const volatile T> : AlignBytes<T> {};
// clang-format on

/// Unit-words of data movement
template <typename T>
struct UnitWord
{
  static constexpr auto ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES;

  template <typename Unit>
  struct IsMultiple
  {
    static constexpr auto UNIT_ALIGN_BYTES = AlignBytes<Unit>::ALIGN_BYTES;
    static constexpr bool IS_MULTIPLE =
      (sizeof(T) % sizeof(Unit) == 0) && (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0);
  };

  /// Biggest shuffle word that T is a whole multiple of and is not larger than the alignment of T
  using ShuffleWord =
    ::cuda::std::_If<IsMultiple<int>::IS_MULTIPLE,
                     unsigned int,
                     ::cuda::std::_If<IsMultiple<short>::IS_MULTIPLE, unsigned short, unsigned char>>;

  /// Biggest volatile word that T is a whole multiple of and is not larger than the alignment of T
  using VolatileWord = ::cuda::std::_If<IsMultiple<long long>::IS_MULTIPLE, unsigned long long, ShuffleWord>;

  /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
  using DeviceWord = ::cuda::std::_If<IsMultiple<longlong2>::IS_MULTIPLE, ulonglong2, VolatileWord>;

  /// Biggest texture reference word that T is a whole multiple of and is not larger than the alignment of T
  using TextureWord = ::cuda::std::
    _If<IsMultiple<int4>::IS_MULTIPLE, uint4, ::cuda::std::_If<IsMultiple<int2>::IS_MULTIPLE, uint2, ShuffleWord>>;
};

// float2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord<float2>
{
  using ShuffleWord  = int;
  using VolatileWord = unsigned long long;
  using DeviceWord   = unsigned long long;
  using TextureWord  = float2;
};

// float4 specialization workaround (for SM10-SM13)
template <>
struct UnitWord<float4>
{
  using ShuffleWord  = int;
  using VolatileWord = unsigned long long;
  using DeviceWord   = ulonglong2;
  using TextureWord  = float4;
};

// char2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord<char2>
{
  using ShuffleWord  = unsigned short;
  using VolatileWord = unsigned short;
  using DeviceWord   = unsigned short;
  using TextureWord  = unsigned short;
};

// clang-format off
template <typename T> struct UnitWord<volatile T> : UnitWord<T> {};
template <typename T> struct UnitWord<const T> : UnitWord<T> {};
template <typename T> struct UnitWord<const volatile T> : UnitWord<T> {};
// clang-format on

/******************************************************************************
 * Vector type inference utilities.
 ******************************************************************************/

/**
 * \brief Exposes a member alias \p Type that names the corresponding CUDA vector type if one exists.  Otherwise \p
 * Type refers to the CubVector structure itself, which will wrap the corresponding \p x, \p y, etc. vector fields.
 */
template <typename T, int vec_elements>
struct CubVector
{
  static_assert(!sizeof(T), "CubVector can only have 1-4 elements");
};

/// The maximum number of elements in CUDA vector types
inline constexpr int MAX_VEC_ELEMENTS = 4;

/**
 * Generic vector-1 type
 */
template <typename T>
struct CubVector<T, 1>
{
  T x;

  using BaseType = T;
  using Type     = CubVector;
};

/**
 * Generic vector-2 type
 */
template <typename T>
struct CubVector<T, 2>
{
  T x;
  T y;

  using BaseType = T;
  using Type     = CubVector;
};

/**
 * Generic vector-3 type
 */
template <typename T>
struct CubVector<T, 3>
{
  T x;
  T y;
  T z;

  using BaseType = T;
  using Type     = CubVector;
};

/**
 * Generic vector-4 type
 */
template <typename T>
struct CubVector<T, 4>
{
  T x;
  T y;
  T z;
  T w;

  using BaseType = T;
  using Type     = CubVector;
};

// TODO(bgruber): should CubVectorType support (and how?) the new type4_16a and type4_32a vector types from CTK 13?
/**
 * Macro for expanding partially-specialized built-in vector types
 */
#  define CUB_DEFINE_VECTOR_TYPE(base_type, short_type)                                     \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 1> : short_type##1                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##1;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };                                                                                      \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 2> : short_type##2                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##2;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        retval.y = y + other.y;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        retval.y = y - other.y;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };                                                                                      \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 3> : short_type##3                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##3;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        retval.y = y + other.y;                                                             \
        retval.z = z + other.z;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        retval.y = y - other.y;                                                             \
        retval.z = z - other.z;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };                                                                                      \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 4> : short_type##4                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##4;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        retval.y = y + other.y;                                                             \
        retval.z = z + other.z;                                                             \
        retval.w = w + other.w;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        retval.y = y - other.y;                                                             \
        retval.z = z - other.z;                                                             \
        retval.w = w - other.w;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };

// Expand CUDA vector types for built-in primitives
// clang-format off
CUB_DEFINE_VECTOR_TYPE(char,               char)
CUB_DEFINE_VECTOR_TYPE(signed char,        char)
CUB_DEFINE_VECTOR_TYPE(short,              short)
CUB_DEFINE_VECTOR_TYPE(int,                int)
_CCCL_SUPPRESS_DEPRECATED_PUSH
CUB_DEFINE_VECTOR_TYPE(long,               long)
CUB_DEFINE_VECTOR_TYPE(long long,          longlong)
_CCCL_SUPPRESS_DEPRECATED_POP
CUB_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
CUB_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
CUB_DEFINE_VECTOR_TYPE(unsigned int,       uint)
_CCCL_SUPPRESS_DEPRECATED_PUSH
CUB_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
CUB_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
_CCCL_SUPPRESS_DEPRECATED_POP
CUB_DEFINE_VECTOR_TYPE(float,              float)
_CCCL_SUPPRESS_DEPRECATED_PUSH
CUB_DEFINE_VECTOR_TYPE(double,             double)
_CCCL_SUPPRESS_DEPRECATED_POP
CUB_DEFINE_VECTOR_TYPE(bool,               uchar)
// clang-format on

#  undef CUB_DEFINE_VECTOR_TYPE

/******************************************************************************
 * Wrapper types
 ******************************************************************************/

/**
 * \brief A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions
 */
template <typename T>
struct Uninitialized
{
  /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
  using DeviceWord = typename UnitWord<T>::DeviceWord;

  static constexpr ::cuda::std::size_t DATA_SIZE = sizeof(T);
  static constexpr ::cuda::std::size_t WORD_SIZE = sizeof(DeviceWord);
  static constexpr ::cuda::std::size_t WORDS     = DATA_SIZE / WORD_SIZE;

  /// Backing storage
  DeviceWord storage[WORDS];

  /// Alias
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T& Alias()
  {
    return reinterpret_cast<T&>(*this);
  }
};

/**
 * \brief A key identifier paired with a corresponding value
 */
template <typename _Key, typename _Value>
struct KeyValuePair
{
  using Key   = _Key; ///< Key data type
  using Value = _Value; ///< Value data type

  Key key; ///< Item key
  Value value; ///< Item value

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair(Key const& key, Value const& value)
      : key(key)
      , value(value)
  {}

  /// Inequality operator
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const KeyValuePair& b)
  {
    return (value != b.value) || (key != b.key);
  }
};

/**
 * \brief Double-buffer storage wrapper for multi-pass stream transformations that require more than one storage array
 * for streaming intermediate results back and forth.
 *
 * Many multi-pass computations require a pair of "ping-pong" storage
 * buffers (e.g., one for reading from and the other for writing to, and then
 * vice-versa for the subsequent pass).  This structure wraps a set of device
 * buffers and a "selector" member to track which is "current".
 */
template <typename T>
struct DoubleBuffer
{
  /// Pair of device buffer pointers
  T* d_buffers[2]{};

  ///  Selector into \p d_buffers (i.e., the active/valid buffer)
  int selector = 0;

  /// \brief Constructor
  DoubleBuffer() = default;

  /// \brief Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE DoubleBuffer(T* d_current, ///< The currently valid buffer
                                                   T* d_alternate) ///< Alternate storage buffer of the same size as \p
                                                                   ///< d_current
      : d_buffers{d_current, d_alternate}
  {}

  /// \brief Return pointer to the currently valid buffer
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T* Current()
  {
    return d_buffers[selector];
  }

  /// \brief Return pointer to the currently invalid buffer
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T* Alternate()
  {
    return d_buffers[selector ^ 1];
  }
};

/******************************************************************************
 * Typedef-detection
 ******************************************************************************/

/**
 * \brief Defines a structure \p detector_name that is templated on type \p T.  The \p detector_name struct exposes a
 * constant member \p value indicating whether or not parameter \p T exposes a nested type \p nested_type_name
 */
#  define CUB_DEFINE_DETECT_NESTED_TYPE(detector_name, nested_type_name)                                \
    template <typename T, typename = void>                                                              \
    struct detector_name : ::cuda::std::false_type                                                      \
    {};                                                                                                 \
    template <typename T>                                                                               \
    struct detector_name<T, ::cuda::std::void_t<typename T::nested_type_name>> : ::cuda::std::true_type \
    {};

/******************************************************************************
 * Typedef-detection
 ******************************************************************************/

/**
 * \brief Determine whether or not BinaryOp's functor is of the form <tt>bool operator()(const T& a, const T&b)</tt> or
 * <tt>bool operator()(const T& a, const T&b, unsigned int idx)</tt>
 */
template <typename T, typename BinaryOp, typename = void>
struct BinaryOpHasIdxParam : ::cuda::std::false_type
{};

template <typename T, typename BinaryOp>
struct BinaryOpHasIdxParam<T,
                           BinaryOp,
                           ::cuda::std::void_t<decltype(::cuda::std::declval<BinaryOp>()(
                             ::cuda::std::declval<T>(), ::cuda::std::declval<T>(), int{}))>> : ::cuda::std::true_type
{};

/******************************************************************************
 * Simple type traits utilities.
 ******************************************************************************/

/**
 * \brief Basic type traits categories
 */
enum Category
{
  NOT_A_NUMBER,
  SIGNED_INTEGER,
  UNSIGNED_INTEGER,
  FLOATING_POINT
};

namespace detail
{
struct is_primitive_impl;

template <Category _CATEGORY, bool _PRIMITIVE, typename _UnsignedBits, typename T>
struct BaseTraits
{
private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = _PRIMITIVE;
};

template <typename _UnsignedBits, typename T>
struct BaseTraits<UNSIGNED_INTEGER, true, _UnsignedBits, T>
{
  static_assert(::cuda::std::numeric_limits<T>::is_specialized,
                "Please also specialize cuda::std::numeric_limits for T");

  using UnsignedBits                       = _UnsignedBits;
  static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key;
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    UnsignedBits retval_bits = MAX_KEY;
    T retval;
    memcpy(&retval, &retval_bits, sizeof(T));
    return retval;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Lowest()
  {
    UnsignedBits retval_bits = LOWEST_KEY;
    T retval;
    memcpy(&retval, &retval_bits, sizeof(T));
    return retval;
  }

private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = true;
};

template <typename _UnsignedBits, typename T>
struct BaseTraits<SIGNED_INTEGER, true, _UnsignedBits, T>
{
  static_assert(::cuda::std::numeric_limits<T>::is_specialized,
                "Please also specialize cuda::std::numeric_limits for T");

  using UnsignedBits = _UnsignedBits;

  static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    UnsignedBits retval = MAX_KEY;
    return reinterpret_cast<T&>(retval);
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Lowest()
  {
    UnsignedBits retval = LOWEST_KEY;
    return reinterpret_cast<T&>(retval);
  }

private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = true;
};

template <typename _UnsignedBits, typename T>
struct BaseTraits<FLOATING_POINT, true, _UnsignedBits, T>
{
  static_assert(::cuda::std::numeric_limits<T>::is_specialized,
                "Please also specialize cuda::std::numeric_limits for T");
  static_assert(::cuda::is_floating_point<T>::value, "Please also specialize cuda::is_floating_point for T");
  static_assert(::cuda::is_floating_point_v<T>, "Please also specialize cuda::is_floating_point_v for T");

  using UnsignedBits = _UnsignedBits;

  static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(-1);
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
    return key ^ mask;
  };

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
    return key ^ mask;
  };

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    return ::cuda::std::numeric_limits<T>::max();
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Lowest()
  {
    return ::cuda::std::numeric_limits<T>::lowest();
  }

private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = true;
};
} // namespace detail

//! Use this class as base when specializing \ref NumericTraits for primitive signed/unsigned integers or floating-point
//! types.
template <Category _CATEGORY, bool _PRIMITIVE, typename _UnsignedBits, typename T>
using BaseTraits = detail::BaseTraits<_CATEGORY, _PRIMITIVE, _UnsignedBits, T>;

//! Numeric type traits for radix sort key operations, decoupled lookback and tuning. You can specialize this template
//! for your own types if:
//! * There is an unsigned integral type of equal size
//! * The size of the type is smaller than 64bits
//! * The arithmetic throughput of the type is similar to other built-in types of the same size
//! For other types, if you want to use them with radix sort, please use the decomposer interface of the radix sort.
// clang-format off
template <typename T> struct NumericTraits :            BaseTraits<NOT_A_NUMBER, false, T, T> {};

template <> struct NumericTraits<NullType> :            BaseTraits<NOT_A_NUMBER, false, NullType, NullType> {};

template <> struct NumericTraits<char> :                BaseTraits<(::cuda::std::numeric_limits<char>::is_signed) ? SIGNED_INTEGER : UNSIGNED_INTEGER, true, unsigned char, char> {};
template <> struct NumericTraits<signed char> :         BaseTraits<SIGNED_INTEGER, true, unsigned char, signed char> {};
template <> struct NumericTraits<short> :               BaseTraits<SIGNED_INTEGER, true, unsigned short, short> {};
template <> struct NumericTraits<int> :                 BaseTraits<SIGNED_INTEGER, true, unsigned int, int> {};
template <> struct NumericTraits<long> :                BaseTraits<SIGNED_INTEGER, true, unsigned long, long> {};
template <> struct NumericTraits<long long> :           BaseTraits<SIGNED_INTEGER, true, unsigned long long, long long> {};

template <> struct NumericTraits<unsigned char> :       BaseTraits<UNSIGNED_INTEGER, true, unsigned char, unsigned char> {};
template <> struct NumericTraits<unsigned short> :      BaseTraits<UNSIGNED_INTEGER, true, unsigned short, unsigned short> {};
template <> struct NumericTraits<unsigned int> :        BaseTraits<UNSIGNED_INTEGER, true, unsigned int, unsigned int> {};
template <> struct NumericTraits<unsigned long> :       BaseTraits<UNSIGNED_INTEGER, true, unsigned long, unsigned long> {};
template <> struct NumericTraits<unsigned long long> :  BaseTraits<UNSIGNED_INTEGER, true, unsigned long long, unsigned long long> {};
// clang-format on

#  if _CCCL_HAS_INT128()
template <>
struct NumericTraits<__uint128_t>
{
  using T            = __uint128_t;
  using UnsignedBits = __uint128_t;

  static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key;
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    return MAX_KEY;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Lowest()
  {
    return LOWEST_KEY;
  }

private:
  friend struct detail::is_primitive_impl;

  static constexpr bool is_primitive = false;
};

template <>
struct NumericTraits<__int128_t>
{
  using T            = __int128_t;
  using UnsignedBits = __uint128_t;

  static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    UnsignedBits retval = MAX_KEY;
    return reinterpret_cast<T&>(retval);
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()")
  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Lowest()
  {
    UnsignedBits retval = LOWEST_KEY;
    return reinterpret_cast<T&>(retval);
  }

private:
  friend struct detail::is_primitive_impl;

  static constexpr bool is_primitive = false;
};
#  endif // _CCCL_HAS_INT128()

// clang-format off
template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, unsigned int, float> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, unsigned long long, double> {};
#  if _CCCL_HAS_NVFP16()
    template <> struct NumericTraits<__half> :          BaseTraits<FLOATING_POINT, true, unsigned short, __half> {};
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    template <> struct NumericTraits<__nv_bfloat16> :   BaseTraits<FLOATING_POINT, true, unsigned short, __nv_bfloat16> {};
#  endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8()
    template <> struct NumericTraits<__nv_fp8_e4m3> :   BaseTraits<FLOATING_POINT, true, __nv_fp8_storage_t, __nv_fp8_e4m3> {};
    template <> struct NumericTraits<__nv_fp8_e5m2> :   BaseTraits<FLOATING_POINT, true, __nv_fp8_storage_t, __nv_fp8_e5m2> {};
#endif // _CCCL_HAS_NVFP8()

template <> struct NumericTraits<bool> :                BaseTraits<UNSIGNED_INTEGER, true, typename UnitWord<bool>::VolatileWord, bool> {};
// clang-format on

namespace detail
{
template <typename T>
struct Traits : NumericTraits<::cuda::std::remove_cv_t<T>>
{};
} // namespace detail

//! \brief Query type traits for radix sort key operations, decoupled lookback and tunings. To add support for your own
//! primitive types please specialize \ref NumericTraits.
template <typename T>
using Traits = detail::Traits<T>;

namespace detail
{
// we cannot befriend is_primitive on GCC < 11, since it's a template (bug)
struct is_primitive_impl
{
  // must be a struct instead of an alias, so the access of Traits<T>::is_primitive happens in the context of this class
  template <typename T>
  struct is_primitive : ::cuda::std::bool_constant<Traits<T>::is_primitive>
  {};
};
// This trait serves two purposes:
// 1. It is used for tunings to detect whether we have a build-in arithmetic type for which we can expect certain
// arithmetic throughput. E.g.: we expect all primitive types of the same size to show roughly similar performance.
// 2. Decoupled lookback uses this trait to determine whether there is a machine word twice the size of T which can be
// loaded/stored with a single instruction.
// TODO(bgruber): for 2. we should probably just check whether sizeof(T) * 2 <= sizeof(int128) (or 256-bit on SM100)
// Users must be able to hook into both scenarios with their custom types, so this trait must depend on cub::Traits
template <typename T>
struct is_primitive : is_primitive_impl::is_primitive<T>
{};

template <typename T>
inline constexpr bool is_primitive_v = is_primitive<T>::value;
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
